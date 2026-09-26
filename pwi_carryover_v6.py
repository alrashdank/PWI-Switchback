"""
Carry-over extension of the PWI testbed (v6: stationary initialisation; pilot evaluation designs).

Three persistent states are added to the shift-level model, scaled by a
carry-over strength c in [0, 1] (c = 0 recovers the shift-local testbed):
  * finished-goods stock F_t: a fraction c of surplus production is kept and
    consumed against the next shift's demand;
  * work-in-process W_t = (1 - phi) W*(B_t, Q_t) eps_t + phi W_{t-1}, phi = 0.7c,
    so a buffer or batch change moves WIP only gradually;
  * quality state zeta_t = 0.8 (1 - 0.5 c K_{t-1}) zeta_{t-1} + noise, so a tight
    trigger today lowers drift tomorrow.
Because the state depends on past levers, the exogenous paths cannot be
pre-computed; innovations are drawn once (common random numbers) and the model is
simulated sequentially for each lever sequence.

Estimand under carry-over: the long-run effect, Gamma_LR, obtained by holding each
of the 12 combinations fixed for the whole horizon (steady state).

Sweep: block length b in {1, 2, 4, 8} x carry-over strength c in {0, .25, .5, .75, 1}
x three estimators, replicated over seeds. Policy check: myopic total-cost OLS
trained on switchback data of block 2 and 8, versus best fixed setting, siloed
prescription and baseline, under carry-over.

Run: python pwi_carryover.py --reps 30   (writes results_v6/carryover_*.csv)
"""
import argparse, itertools, json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pwi_study_v6 import (BASE, COMBOS, BASELINE, BASE_IDX, CONTRASTS, WASTES, TARGETS, NAME,
                          SHIFT_MIN, RATE, switchback, gamma_true, pol_silo, pol_pwi, pol_best_fixed)


def innovations(n, seed, P):
    """Same draw order as pwi_study_v6.exogenous (stationary t = 0 states first), so
    that c = 0 reproduces it exactly."""
    r = np.random.default_rng(seed)
    d0 = r.normal(0, 7 / np.sqrt(1 - 0.25)); h0 = r.normal(0, P["bsd"] / np.sqrt(1 - 0.36))
    z0 = r.normal(0, P["zsd"] / np.sqrt(1 - 0.64))
    ed, eh, ez = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        ed[t] = r.normal(0, 7); eh[t] = r.normal(0, P["bsd"]); ez[t] = r.normal(0, P["zsd"])
    return dict(d0=d0, h0=h0, z0=z0, ed=ed, eh=eh, ez=ez, u1=r.lognormal(0, 0.1, n), u2=r.normal(0, 2, n))


def simulate_dyn(I, B, Q, K, P, c):
    """Sequential simulation with carry-over strength c. Returns the same columns as
    pwi_study_v5.simulate plus the state variables."""
    n = len(I["ed"]); B, Q, K = (np.asarray(x, int) for x in (B, Q, K))
    sB, topup = np.asarray(P["sB"]), np.asarray(P["topup"])
    phi = 0.7 * c
    d, h, zeta = I["d0"], I["h0"], I["z0"]; F = 0.0; W = 12 * (1 + P["wipB"] * B[0]) * (1 + P["wipQ"] * Q[0]); Kprev = 0
    b_lag, z_lag = float(np.clip(30 * np.exp(h), 0, 150)), max(zeta, 0.0)
    rows = []
    for t in range(n):
        d = 0.5 * d + I["ed"][t]
        h = 0.6 * h + I["eh"][t]
        zeta = 0.8 * (1 - 0.5 * c * Kprev) * zeta + I["ez"][t]
        D = float(np.clip(100 + d, 70, 125)); b = float(np.clip(30 * np.exp(h), 0, 150)); z = max(zeta, 0.0)
        Bt, Qt, Kt = B[t], Q[t], K[t]
        p = 0.02 + z * (1 - P["trig"] * Kt)
        stops = Kt * (P["stop0"] + P["stopz"] * z)
        lost = b * sB[Bt]
        cap = RATE * max(SHIFT_MIN - lost - stops, 0)
        ND = max(D - F, 0.0)                                   # net demand after stock
        target = ND / (1 - (0.02 + z_lag)) + topup[Bt]
        out = min(target, cap)
        defects = out * p; good = out - defects
        surplus = max(good - ND, 0.0); short = max(ND - good, 0.0)
        F_next = c * (surplus + max(F - D, 0.0))
        Wstar = 12 * (1 + P["wipB"] * Bt) * (1 + P["wipQ"] * Qt) * I["u1"][t]
        W = (1 - phi) * Wstar + phi * W
        w = dict(overproduction=surplus, waiting=lost + stops + 6 + P["batch_wait"] * Qt, defects=defects,
                 inventory=W, transport=np.ceil(out / (10.0 if Qt == 0 else 25.0)) * 0.4 + defects * 0.4,
                 motion=max(20 + P["motw"] * W + P["motK"] * Kt + I["u2"][t], 0.0),
                 over_processing=out * (0.2 + P["insp"] * Kt) + defects * 3.0)
        cost = {"c_" + k: v * P["kappa"][k] for k, v in w.items()}
        cost["shortage"] = short * P["short"]
        cost["objective"] = sum(cost.values())
        rows.append(dict(demand=D, breakdown=b, drift=z, breakdown_lag=b_lag, drift_lag=z_lag, stock=F, wip_state=W,
                         B=Bt, Q=Qt, K=Kt, K_lag=Kprev, B_lag=B[t - 1] if t else Bt, Q_lag=Q[t - 1] if t else Qt,
                         **w, **cost))
        F, Kprev, b_lag, z_lag = F_next, Kt, b, z
    return pd.DataFrame(rows)


def cube_dyn(I, P, c):
    n = len(I["ed"])
    return np.stack([simulate_dyn(I, *(np.full(n, v) for v in a), P, c)[TARGETS].values for a in COMBOS])


# ------------------------------------------------------------ estimators
def _lev(df, suffix=""):
    return pd.DataFrame({"B1" + suffix: (df["B" + suffix].values == 1) * 1., "B2" + suffix: (df["B" + suffix].values == 2) * 1.,
                         "Q" + suffix: df["Q" + suffix].values * 1., "K" + suffix: df["K" + suffix].values * 1.})


def est_shift(df, drop_first=None, block=2, lagged=False, exog_only=False):
    """Additive OLS. drop_first: drop the first `drop_first` shifts of each block (washout).
    lagged: add previous-shift lever dummies and report current + lagged coefficient.
    exog_only: covariates demand and breakdown_lag only (drift_lag is a mediator of past K)."""
    if drop_first:
        keep = (np.arange(len(df)) % block) >= drop_first
        df = df[keep]
    X = _lev(df)
    if lagged:
        X = pd.concat([X, _lev(df, "_lag")], axis=1)
    X["d"] = (df.demand.values - 100) / 10; X["bl"] = df.breakdown_lag.values / 30
    if not exog_only:
        X["zl"] = df.drift_lag.values * 50
    X = sm.add_constant(X, has_constant="add")
    G = np.zeros((len(CONTRASTS), len(WASTES)))
    for j, w in enumerate(WASTES):
        f = sm.OLS(df["c_" + w].values, X).fit()
        for i, (lev, hi, _) in enumerate(CONTRASTS):
            nm = NAME[(lev, hi)]
            G[i, j] = f.params[nm] + (f.params[nm + "_lag"] if lagged else 0.0)
    return G


ESTIMATORS = {
    # Main comparison: exogenous covariates only (demand, lagged breakdown). Under carry-over the lagged
    # drift reading depends on earlier trigger settings, i.e. it is post-treatment.
    "shift_level": dict(exog_only=True),
    "washout_1": dict(drop_first=1, exog_only=True),
    "lag1_exog": dict(lagged=True, exog_only=True),
    "block_mean": "block_mean",
    # For comparison only: the same estimators with the post-treatment drift reading included.
    "shift_level_drift": dict(),
    "washout_1_drift": dict(drop_first=1),
}


def est_block_mean(df, block):
    """Block-level regression (Bojinov-style): mean cost per block on the block's lever
    dummies and block-mean exogenous covariates; the first shift of each block is a burn-in."""
    nb = len(df) // block
    d = df.iloc[: nb * block].copy(); d["blk"] = np.repeat(np.arange(nb), block)
    d = d[(np.arange(len(d)) % block) >= (1 if block > 1 else 0)]
    g = d.groupby("blk").mean(numeric_only=True)
    X = pd.DataFrame({"B1": (g.B.values == 1) * 1., "B2": (g.B.values == 2) * 1., "Q": g.Q.values, "K": g.K.values,
                      "d": (g.demand.values - 100) / 10, "bl": g.breakdown_lag.values / 30})
    X = sm.add_constant(X, has_constant="add")
    G = np.zeros((len(CONTRASTS), len(WASTES)))
    for j, w in enumerate(WASTES):
        f = sm.OLS(g["c_" + w].values, X).fit()
        for i, (lev, hi, _) in enumerate(CONTRASTS):
            G[i, j] = f.params[NAME[(lev, hi)]]
    return G


# ---------------------------------------------------------------- sweep
def run_sweep(reps, n=600, blocks=(1, 2, 4, 8), cs=(0.0, 0.25, 0.5, 0.75, 1.0), P=BASE):
    out = []
    for rep in range(reps):
        I = innovations(n, 500 + rep, P)
        for c in cs:
            G = gamma_true(cube_dyn(I, P, c))
            mat = np.abs(G) >= 0.5
            for b in blocks:
                sw = simulate_dyn(I, *switchback(n, 600 + rep, block=b), P, c)
                for name, kw in ESTIMATORS.items():
                    if kw == "block_mean":
                        if b == 1:
                            continue
                        Ghat = est_block_mean(sw, b)
                    else:
                        kw2 = dict(kw); kw2["block"] = b
                        if kw2.get("drop_first") and b == 1:
                            continue
                        Ghat = est_shift(sw, **kw2)
                    err = Ghat - G
                    out.append(dict(rep=rep, c=c, block=b, estimator=name, bias=err.mean(),
                                    rmse=np.sqrt((err ** 2).mean()),
                                    rev=float(np.mean(np.sign(Ghat[mat]) != np.sign(G[mat]))) if mat.any() else np.nan,
                                    k_def_true=G[3, 2], k_def_hat=Ghat[3, 2], b2_wait_true=G[0, 1], b2_wait_hat=Ghat[0, 1],
                                    b2_inv_true=G[0, 3], b2_inv_hat=Ghat[0, 3]))
        print(f"rep {rep} done", flush=True)
    return pd.DataFrame(out)


# ------------------------------------------------------- policy check
def design_dyn(df):
    X = _lev(df)
    s = pd.DataFrame({"bl": df.breakdown_lag.values / 30, "zl": df.drift_lag.values * 50, "d": (df.demand.values - 100) / 10})
    for a, cc in itertools.product(list(X.columns), list(s.columns)):
        X[f"{a}:{cc}"] = X[a] * s[cc]
    return sm.add_constant(pd.concat([X, s], axis=1), has_constant="add")


def run_policy(reps, c_values=(0.0, 0.5, 1.0), n_train=600, n_test=300, P=BASE):
    """Myopic policies under carry-over. Each policy is simulated sequentially: at
    every shift the policy sees the current state (which depends on its own past
    choices) and picks a setting from the predictions."""
    out = []
    for rep in range(reps):
        I_tr, I_te = innovations(n_train, 700 + rep, P), innovations(n_test, 800 + rep, P)
        for c in c_values:
            for b in (2, 8):
                sw = simulate_dyn(I_tr, *switchback(n_train, 900 + rep, block=b), P, c)
                Xtr = design_dyn(sw)
                betas = np.column_stack([np.linalg.lstsq(Xtr.values, sw[y].values, rcond=None)[0] for y in TARGETS])
                bf = COMBOS[pol_best_fixed(sw)]
                res = {}
                for pname in ("baseline", "best_fixed", "silo", "pwi"):
                    if pname == "baseline":
                        df = simulate_dyn(I_te, *(np.full(n_test, v) for v in BASELINE), P, c)
                    elif pname == "best_fixed":
                        df = simulate_dyn(I_te, *(np.full(n_test, v) for v in bf), P, c)
                    else:
                        df = simulate_policy(I_te, betas, P, c, pname)
                    res[pname] = df
                Jb = res["baseline"]["objective"].mean()
                for pname, df in res.items():
                    out.append(dict(rep=rep, c=c, train_block=b, policy=pname,
                                    red=100 * (Jb - df["objective"].mean()) / Jb, shortage=df["shortage"].mean()))
        print(f"policy rep {rep} done", flush=True)
    return pd.DataFrame(out)


def _design_row(a, D, b_lag, z_lag):
    """Single-row design in the column order of design_dyn (const, levers, lever x state, state)."""
    lev = np.array([a[0] == 1, a[0] == 2, a[1], a[2]], float)
    st = np.array([b_lag / 30, z_lag * 50, (D - 100) / 10])
    return np.concatenate([[1.0], lev, np.outer(lev, st).ravel(), st])


def simulate_modes(I, modes, betas, P, c, fixed=BASELINE):
    """Closed-loop simulation under carry-over with a per-shift mode:
    0 = fixed setting `fixed`, 1 = siloed prescription, 2 = total-cost prescription.
    Decisions use predictions from `betas` given (D_t, b_{t-1}, z_{t-1})."""
    n = len(I["ed"]); sB, topup = np.asarray(P["sB"]), np.asarray(P["topup"]); phi = 0.7 * c
    d, h, zeta = I["d0"], I["h0"], I["z0"]; F = 0.0; W = 12 * (1 + P["wipB"]) * (1 + P["wipQ"]); Kprev = 0
    b_lag, z_lag = float(np.clip(30 * np.exp(h), 0, 150)), max(zeta, 0.0)
    J = np.empty(n); short_v = np.empty(n)
    for t in range(n):
        d = 0.5 * d + I["ed"][t]; h = 0.6 * h + I["eh"][t]; zeta = 0.8 * (1 - 0.5 * c * Kprev) * zeta + I["ez"][t]
        D = float(np.clip(100 + d, 70, 125)); b = float(np.clip(30 * np.exp(h), 0, 150)); z = max(zeta, 0.0)
        m = modes[t]
        if m == 0:
            Bt, Qt, Kt = fixed
        else:
            pred = np.stack([_design_row(a, D, b_lag, z_lag) @ betas for a in COMBOS])[:, None, :]
            idx = pol_pwi(pred)[0] if m == 2 else pol_silo(pred)[0][0]
            Bt, Qt, Kt = COMBOS[idx]
        p = 0.02 + z * (1 - P["trig"] * Kt); stops = Kt * (P["stop0"] + P["stopz"] * z); lost = b * sB[Bt]
        cap = RATE * max(SHIFT_MIN - lost - stops, 0); ND = max(D - F, 0.0)
        out = min(ND / (1 - (0.02 + z_lag)) + topup[Bt], cap); defects = out * p; good = out - defects
        surplus = max(good - ND, 0.0); short = max(ND - good, 0.0); F_next = c * (surplus + max(F - D, 0.0))
        W = (1 - phi) * 12 * (1 + P["wipB"] * Bt) * (1 + P["wipQ"] * Qt) * I["u1"][t] + phi * W
        w = dict(overproduction=surplus, waiting=lost + stops + 6 + P["batch_wait"] * Qt, defects=defects, inventory=W,
                 transport=np.ceil(out / (10.0 if Qt == 0 else 25.0)) * 0.4 + defects * 0.4,
                 motion=max(20 + P["motw"] * W + P["motK"] * Kt + I["u2"][t], 0.0),
                 over_processing=out * (0.2 + P["insp"] * Kt) + defects * 3.0)
        short_v[t] = short * P["short"]
        J[t] = sum(v * P["kappa"][k] for k, v in w.items()) + short_v[t]
        F, Kprev, b_lag, z_lag = F_next, Kt, b, z
    return J, short_v


def simulate_policy(I, betas, P, c, mode):
    """Closed-loop myopic policy ('pwi' or 'silo') for the whole horizon."""
    n = len(I["ed"]); J, sh = simulate_modes(I, np.full(n, 2 if mode == "pwi" else 1), betas, P, c)
    return pd.DataFrame({"objective": J, "shortage": sh})


# ------------------------------------------------------- pilot evaluation
DESIGNS = {  # name: (shifts per randomisation unit, blocks per arm per unit, washout shifts per block)
    "daily": (12, 2, 0),        # one-day blocks, two per arm per week, no washout (v3 protocol)
    "two_day": (12, 1, 1),      # two-day blocks, one per arm per week, first shift discarded
    "four_day": (24, 1, 1),     # four-day blocks, one per arm per fortnight, first shift discarded
}


def _unit_diffs(Yb, arm_b, unit_b, n_units):
    return np.array([Yb[(unit_b == u) & (arm_b == 2)].mean() - Yb[(unit_b == u) & (arm_b == 1)].mean()
                     for u in range(n_units)])


def _tstat(d):
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)


def randomisation_p(Yb, arm_b, unit_b, n_units, rng, mc_perms=5000):
    """Two-sided randomisation p-value for total-cost vs siloed with baseline blocks held
    fixed: within each unit the TC/silo labels are re-assigned among that unit's
    non-baseline blocks in every way the design allows (C(2m, m) ways for m blocks per
    arm). The statistic is the studentised mean of the within-unit differences. All
    arrangements are enumerated when there are at most 2^14 of them (for one block per
    arm per unit this is a sign-flip test); otherwise `mc_perms` random arrangements."""
    tables = []
    for u in range(n_units):
        nb = np.where((unit_b == u) & (arm_b != 0))[0]
        m = len(nb) // 2
        opts = [np.array(s) for s in itertools.combinations(range(len(nb)), m)]
        tables.append(np.array([Yb[nb[s]].mean() - Yb[np.setdiff1d(nb, nb[s])].mean() for s in opts]))
    d_obs = _unit_diffs(Yb, arm_b, unit_b, n_units); t_obs = _tstat(d_obs)
    sizes = [len(tb) for tb in tables]
    if np.prod(sizes, dtype=float) <= 2 ** 14:
        grid = np.array(list(itertools.product(*[range(s) for s in sizes])))
    else:
        grid = np.column_stack([rng.integers(0, s, mc_perms) for s in sizes])
    D = np.column_stack([tables[u][grid[:, u]] for u in range(n_units)])
    t = D.mean(axis=1) / (D.std(axis=1, ddof=1) / np.sqrt(n_units) + 1e-12)
    hits = np.abs(t) >= abs(t_obs) - 1e-9
    if np.prod(sizes, dtype=float) <= 2 ** 14:
        return float(np.mean(hits)), float(d_obs.mean())          # exact: all arrangements enumerated
    return float((1 + hits.sum()) / (1 + len(hits))), float(d_obs.mean())   # Monte Carlo, +1 correction


def pilot_eval(P, c, reps, weeks=12, seed0=31, het_sd=9.0, explore_block=4, designs=None):
    """Per replication: train OLS models on 144 balanced switchback shifts (blocks of
    `explore_block`, first shift of each block discarded), then simulate the evaluation
    phase under each design with (i) the total-cost policy in its slots (alternative),
    (ii) the siloed policy in the total-cost slots (sharp null of no difference), and
    (iii) the sharp null plus block-level effects N(0, het_sd^2) on total-cost blocks
    (a symmetric weak null, under which the sign-flip test is exact by construction), and
    (iv) zero-mean, right-skewed effects that spill into the following block (a weak null
    that can break the test). The long-run contrast is
    total-cost minus siloed, each run continuously on the same innovations."""
    n = weeks * 12; out = []
    for rep in range(reps):
        rng = np.random.default_rng(seed0 + 1000 * rep)
        I_ex = innovations(144, seed0 + 1000 * rep + 1, P)
        ex = simulate_dyn(I_ex, *switchback(144, seed0 + 1000 * rep + 2, block=explore_block), P, c)
        ex = ex[(np.arange(144) % explore_block) != 0]
        Xtr = design_dyn(ex)
        betas = np.column_stack([np.linalg.lstsq(Xtr.values, ex[y].values, rcond=None)[0] for y in TARGETS])
        I = innovations(n, seed0 + 1000 * rep + 3, P)
        truth = simulate_modes(I, np.full(n, 2), betas, P, c)[0].mean() - simulate_modes(I, np.full(n, 1), betas, P, c)[0].mean()
        for dname, (U, m, wash) in DESIGNS.items():
            if designs and dname not in designs:
                continue
            k = U // (3 * m); n_units = n // U
            arm_block = np.concatenate([rng.permutation(np.repeat([0, 1, 2], m)) for _ in range(n_units)])
            unit_block = np.repeat(np.arange(n_units), 3 * m)
            arm_shift = np.repeat(arm_block, k); blk = np.arange(n) // k; keep = (np.arange(n) % k) >= wash
            J_alt = simulate_modes(I, arm_shift, betas, P, c)[0]
            J_nul = simulate_modes(I, np.where(arm_shift == 2, 1, arm_shift), betas, P, c)[0]
            # Skewed weak null with spillover: zero-mean, right-skewed effects on the
            # total-cost blocks, half of which carries into the first two shifts of the next block.
            # Drawn from a separate stream so the other scenarios reproduce earlier runs exactly.
            rng2 = np.random.default_rng(seed0 + 1000 * rep + 7 + 100 * list(DESIGNS).index(dname))
            J_skew = J_nul.copy()
            delta = rng2.exponential(het_sd, len(arm_block)) - het_sd
            for bb in np.where(arm_block == 2)[0]:
                J_skew[blk == bb] += delta[bb]
                nxt = np.where(blk == bb + 1)[0][:2]
                J_skew[nxt] += 0.5 * delta[bb]
            for scen, J in (("alt", J_alt), ("sharp", J_nul), ("weak", J_nul), ("weak_skew", J_skew)):
                Yb = np.array([J[(blk == b) & keep].mean() for b in range(len(arm_block))])
                if scen == "weak":
                    Yb = Yb + (arm_block == 2) * rng.normal(0, het_sd, len(Yb))
                # the new scenario draws its re-randomisations from rng2, leaving the main stream untouched
                p, est = randomisation_p(Yb, arm_block, unit_block, n_units, rng2 if scen == "weak_skew" else rng)
                out.append(dict(rep=rep, c=c, design=dname, scenario=scen, p=p, est=est, truth=truth, n_units=n_units))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--policy-reps", type=int, default=10)
    ap.add_argument("--part", choices=["sweep", "policy", "both", "pilot"], default="both")
    ap.add_argument("--c", type=float, default=0.0)
    ap.add_argument("--pilot-reps", type=int, default=300)
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--rep-start", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("results_v6", exist_ok=True)
    if args.part == "pilot":
        df = pilot_eval(BASE, args.c, args.pilot_reps, weeks=args.weeks)
        df["weeks"] = args.weeks
        df.to_csv(f"results_v6/pilot_c{args.c:.2f}_w{args.weeks}.csv", index=False)
        print(df.groupby(["design", "scenario"]).agg(reject=("p", lambda x: (x < 0.05).mean()),
                                                     est=("est", "mean"), truth=("truth", "mean")).round(3))
        raise SystemExit(0)
    if args.part in ("sweep", "both"):
        df = run_sweep(args.reps)
        df.to_csv("results_v6/carryover_sweep.csv", index=False)
        print(df.groupby(["c", "block", "estimator"])[["bias", "rmse", "rev"]].mean().round(2).to_string())
    if args.part in ("policy", "both"):
        dp = run_policy(args.policy_reps)
        dp.to_csv("results_v6/carryover_policy.csv", index=False)
        print(dp.groupby(["c", "train_block", "policy"])[["red", "shortage"]].mean().round(2).to_string())
