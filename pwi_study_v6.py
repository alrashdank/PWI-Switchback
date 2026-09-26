"""
PWI study, v6 (third revision): latent processes start from their stationary laws.

v5 notes: PWI study, v5 (second revision).

Changes from v4: balanced switchback (exact counts per combination); causal oracle
samples exact conditional latent states (truncated stationary laws) with 400 draws
and a convergence check; aggregate erosion ratio over all shifts; pilot power
simulation with day-level randomisation and the protocol regression; --all pipeline.

v4 notes:

Changes from v3
- Historical regimes: 'logged' and 'unlogged' confounders, each with and without
  overlap (random deviations on all levers vs on the buffer only).
- Benchmarks: direct-J OLS (shown identical to summed OLS components), direct-J
  GBM, best fixed setting chosen from switchback data, service-constrained PWI.
- Oracles: perfect-foresight (current-shift disturbances known) and causal
  (lagged information only; expectation over the conditional disturbance law).
- Erosion ratio with prospective target and shortage included in spillover.
- Estimator diagnostics: bias, RMSE, HAC-CI coverage, sign reversal at three
  materiality thresholds.
- Pilot-size variants: models trained on the first 72 and 144 switchback shifts.
- Monte Carlo ranges extended to near-zero coupling.

Run:  python pwi_study_v4.py --chunk k   (worlds [40k, 40k+40))
      python pwi_study_v4.py --base      (base case only)
Requires numpy, pandas, statsmodels, scikit-learn.
"""
import argparse, copy, itertools, json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

WASTES = ["overproduction", "waiting", "defects", "inventory", "transport", "motion", "over_processing"]
TARGETS = ["c_" + w for w in WASTES] + ["shortage"]
COMBOS = list(itertools.product(range(3), range(2), range(2)))      # (B, Q, K)
BASELINE = (1, 1, 0)
BASE_IDX = COMBOS.index(BASELINE)
CONTRASTS = [("B", 2, 0), ("B", 1, 0), ("Q", 1, 0), ("K", 1, 0)]
SHIFT_MIN, RATE = 480.0, 0.26
N_TRAIN, N_TEST, N_ORACLE_DRAWS = 600, 300, 400
THRESHOLDS = (0.25, 0.5, 1.0)

BASE = dict(sB=[1.0, 0.60, 0.35], topup=[0.0, 4.0, 9.0], wipB=0.8, wipQ=0.6,
            batch_wait=14.0, trig=0.6, stop0=8.0, stopz=400.0, insp=0.25,
            motw=0.6, motK=8.0, bsd=0.5, zsd=0.012, flip=0.10, short=4.0,
            kappa=dict(overproduction=1.2, waiting=0.5, defects=3.0, inventory=0.4,
                       transport=0.8, motion=0.3, over_processing=0.35))


# ----------------------------------------------------------------- model
def exogenous(n, seed, P):
    """Demand (AR1), breakdown minutes (log-AR1), quality drift (censored AR1).
    Latent states start from their stationary laws at t = 0 and the lag readings of
    the first shift are the t = 0 readings, so every shift, including the first, is
    drawn from the stationary process and the causal oracle's conditional laws are
    exact at every t."""
    r = np.random.default_rng(seed)
    d = r.normal(0, 7 / np.sqrt(1 - 0.25))
    h = r.normal(0, P["bsd"] / np.sqrt(1 - 0.36))
    z = r.normal(0, P["zsd"] / np.sqrt(1 - 0.64))
    b0, z0 = float(np.clip(30 * np.exp(h), 0, 150)), max(z, 0.0)
    D, Hh, Z = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        d = 0.5 * d + r.normal(0, 7)
        h = 0.6 * h + r.normal(0, P["bsd"])
        z = 0.8 * z + r.normal(0, P["zsd"])
        D[t], Hh[t], Z[t] = d, h, z
    ex = pd.DataFrame({"demand": np.clip(100 + D, 70, 125),
                       "breakdown": np.clip(30 * np.exp(Hh), 0, 150),
                       "drift": np.maximum(Z, 0)})
    ex["breakdown_lag"] = ex["breakdown"].shift(1).fillna(b0)
    ex["drift_lag"] = ex["drift"].shift(1).fillna(z0)
    ex["u1"] = r.lognormal(0, 0.1, n)         # inventory noise, LogNormal(0, 0.1^2)
    ex["u2"] = r.normal(0, 2, n)              # motion noise, N(0, 2^2) minutes
    return ex


def simulate(ex, B, Q, K, P):
    B, Q, K = (np.asarray(x, int) for x in (B, Q, K))
    D, b, z = ex["demand"].values, ex["breakdown"].values, ex["drift"].values
    sB, topup = np.asarray(P["sB"]), np.asarray(P["topup"])
    p = 0.02 + z * (1 - P["trig"] * K)
    stops = K * (P["stop0"] + P["stopz"] * z)
    lost = b * sB[B]
    cap = RATE * np.maximum(SHIFT_MIN - lost - stops, 0)
    target = D / (1 - (0.02 + ex["drift_lag"].values)) + topup[B]
    out = np.minimum(target, cap)
    defects = out * p
    good = out - defects
    wip = 12 * (1 + P["wipB"] * B) * (1 + P["wipQ"] * Q) * ex["u1"].values
    w = pd.DataFrame({
        "overproduction": np.maximum(good - D, 0),
        "waiting": lost + stops + 6 + P["batch_wait"] * Q,
        "defects": defects,
        "inventory": wip,
        "transport": np.ceil(out / np.array([10.0, 25.0])[Q]) * 0.4 + defects * 0.4,
        "motion": np.maximum(20 + P["motw"] * wip + P["motK"] * K + ex["u2"].values, 0),
        "over_processing": out * (0.2 + P["insp"] * K) + defects * 3.0,
    })
    c = w.mul(pd.Series(P["kappa"]))
    c.columns = ["c_" + x for x in c.columns]
    c["shortage"] = np.maximum(D - good, 0) * P["short"]
    c["objective"] = c[TARGETS].sum(axis=1)
    return pd.concat([ex[["demand", "breakdown", "drift", "breakdown_lag", "drift_lag"]].reset_index(drop=True),
                      pd.DataFrame({"B": B, "Q": Q, "K": K}), w, c], axis=1)


def cube(ex, P):
    """True cost of every target for every lever combination: (12, n, 8)."""
    return np.stack([simulate(ex, *(np.full(len(ex), v) for v in a), P)[TARGETS].values for a in COMBOS])


def expected_cube(ex, P, seed, draws=N_ORACLE_DRAWS):
    """Expected cost per combination given the policies' information set
    (D_t, b_{t-1}, z_{t-1}) only, not the full history of readings: the Bayes
    decision rule for that restricted information set under the true model. With
    stationary initialisation (see exogenous) the conditional laws below are exact
    at every t.
    The latent states are sampled from their exact conditional laws given the
    lagged readings, assuming stationarity of the latent AR(1) processes:
      h_{t-1} = log(b_{t-1}/30) when b_{t-1} < 150; when the reading is capped at
                150, h_{t-1} ~ N(0, bsd^2/(1-0.36)) truncated to h >= log(5);
      zeta_{t-1} = z_{t-1} when z_{t-1} > 0; when the reading is censored at 0,
                zeta_{t-1} ~ N(0, zsd^2/(1-0.64)) truncated to zeta <= 0.
    Then h_t = 0.6 h_{t-1} + N(0, bsd^2), zeta_t = 0.8 zeta_{t-1} + N(0, zsd^2)."""
    r = np.random.default_rng(seed)
    n = len(ex)
    bl, zl = np.repeat(ex["breakdown_lag"].values, draws), np.repeat(ex["drift_lag"].values, draws)
    sd_h, sd_z = P["bsd"] / np.sqrt(1 - 0.36), P["zsd"] / np.sqrt(1 - 0.64)
    # truncated normals by inverse CDF
    from scipy.stats import norm
    u = r.random(n * draws)
    lo_h = norm.cdf(np.log(5.0) / sd_h)
    h_prev = np.where(bl >= 150 - 1e-9, sd_h * norm.ppf(lo_h + u * (1 - lo_h)), np.log(np.maximum(bl, 1e-9) / 30))
    u2 = r.random(n * draws)
    hi_z = norm.cdf(0.0)                                      # = 0.5
    z_prev = np.where(zl <= 0, sd_z * norm.ppf(u2 * hi_z), zl)
    h = 0.6 * h_prev + r.normal(0, P["bsd"], n * draws)
    zeta = 0.8 * z_prev + r.normal(0, P["zsd"], n * draws)
    ex_big = pd.DataFrame({"demand": np.repeat(ex["demand"].values, draws),
                           "breakdown": np.clip(30 * np.exp(h), 0, 150), "drift": np.maximum(zeta, 0),
                           "breakdown_lag": bl, "drift_lag": zl,
                           "u1": r.lognormal(0, 0.1, n * draws), "u2": r.normal(0, 2, n * draws)})
    big = cube(ex_big, P)
    return big.reshape(len(COMBOS), n, draws, len(TARGETS)).mean(axis=2)


def supervisor(ex, seed, P, unlogged, overlap):
    """Reactive supervisor. overlap=True: random deviations on all levers;
    overlap=False: deviations on the buffer only (Q and K deterministic)."""
    r = np.random.default_rng(seed)
    bl = ex["breakdown" if unlogged else "breakdown_lag"].values
    zl = ex["drift" if unlogged else "drift_lag"].values
    n = len(ex)
    B = np.where(bl > 45, 2, np.where(bl > 25, 1, 0))
    K = (zl > 0.012).astype(int)
    Q = (ex["demand"].values > 100).astype(int)
    B = np.where(r.random(n) < P["flip"], r.integers(0, 3, n), B)
    if overlap:
        Q = np.where(r.random(n) < P["flip"], r.integers(0, 2, n), Q)
        K = np.where(r.random(n) < P["flip"], r.integers(0, 2, n), K)
    return B, Q, K


def switchback(n, seed, block=2):
    """Balanced switchback: within every super-block of 12 x `block` shifts, each
    of the 12 combinations is run once, in random order, for `block` consecutive
    shifts. Counts per combination are therefore exact (n / 12 for n a multiple of 24)."""
    r = np.random.default_rng(seed)
    order = np.concatenate([r.permutation(len(COMBOS)) for _ in range(n // (block * len(COMBOS)) + 1)])
    idx = np.repeat(order, block)[:n]
    a = np.array([COMBOS[i] for i in idx])
    return a[:, 0], a[:, 1], a[:, 2]


# ------------------------------------------------------------ estimation
def _state(df):
    return pd.DataFrame({"bl": df.breakdown_lag.values / 30, "zl": df.drift_lag.values * 50,
                         "d": (df.demand.values - 100) / 10})


def _levers(df):
    return pd.DataFrame({"B1": (df.B.values == 1) * 1., "B2": (df.B.values == 2) * 1.,
                         "Q": df.Q.values * 1., "K": df.K.values * 1.})


def design_full(df):
    X, s = _levers(df), _state(df)
    for a, c in itertools.product(list(X.columns), list(s.columns)):
        X[f"{a}:{c}"] = X[a] * s[c]
    return sm.add_constant(pd.concat([X, s], axis=1), has_constant="add")


def design_additive(df):
    return sm.add_constant(pd.concat([_levers(df), _state(df)], axis=1), has_constant="add")


NAME = {("B", 2): "B2", ("B", 1): "B1", ("Q", 1): "Q", ("K", 1): "K"}


def gamma_true(C):
    arr = np.array(COMBOS)
    G = np.zeros((len(CONTRASTS), len(WASTES)))
    for i, (lev, hi, lo) in enumerate(CONTRASTS):
        col = "BQK".index(lev)
        G[i] = (C[arr[:, col] == hi].mean(axis=(0, 1)) - C[arr[:, col] == lo].mean(axis=(0, 1)))[:len(WASTES)]
    return G


def gamma_naive(df):
    G = np.full((len(CONTRASTS), len(WASTES)), np.nan)
    for i, (lev, hi, lo) in enumerate(CONTRASTS):
        a, b = df[df[lev] == hi], df[df[lev] == lo]
        if len(a) and len(b):
            G[i] = (a[["c_" + w for w in WASTES]].mean() - b[["c_" + w for w in WASTES]].mean()).values
    return G


def gamma_adjusted(df, ci=False):
    """Additive OLS on lever dummies + lagged state; HAC(4) 95% CI if requested."""
    X = design_additive(df)
    G = np.zeros((len(CONTRASTS), len(WASTES)))
    LO, HI = np.zeros_like(G), np.zeros_like(G)
    for j, w in enumerate(WASTES):
        f = sm.OLS(df["c_" + w].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4}) if ci else \
            sm.OLS(df["c_" + w].values, X).fit()
        for i, (lev, hi, _) in enumerate(CONTRASTS):
            G[i, j] = f.params[NAME[(lev, hi)]]
            if ci:
                LO[i, j], HI[i, j] = f.conf_int().loc[NAME[(lev, hi)]]
    return (G, LO, HI) if ci else G


def predict_cube_ols(train, ex, direct=False):
    Xtr = design_full(train)
    if direct:                                                   # single model of total J
        beta = np.linalg.lstsq(Xtr.values, train["objective"].values, rcond=None)[0]
        return np.stack([design_full(ex.assign(B=a[0], Q=a[1], K=a[2])).values @ beta for a in COMBOS])
    betas = {y: np.linalg.lstsq(Xtr.values, train[y].values, rcond=None)[0] for y in TARGETS}
    return np.stack([np.column_stack([design_full(ex.assign(B=a[0], Q=a[1], K=a[2])).values @ betas[y]
                                      for y in TARGETS]) for a in COMBOS])


def predict_cube_gbm(train, ex, seed=0, direct=False):
    feats = lambda d: np.column_stack([d.B.values, d.Q.values, d.K.values, _state(d).values])
    Xtr = feats(train)
    mk = lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                               min_samples_leaf=15, random_state=seed)
    if direct:
        m = mk().fit(Xtr, train["objective"].values)
        return np.stack([m.predict(feats(ex.assign(B=a[0], Q=a[1], K=a[2]))) for a in COMBOS])
    models = {y: mk().fit(Xtr, train[y].values) for y in TARGETS}
    return np.stack([np.column_stack([models[y].predict(feats(ex.assign(B=a[0], Q=a[1], K=a[2])))
                                      for y in TARGETS]) for a in COMBOS])


# --------------------------------------------------------------- policies
def pol_pwi(pred):                       # pred: (12, n, 8) or (12, n) total
    tot = pred.sum(axis=2) if pred.ndim == 3 else pred
    return tot.argmin(axis=0)


def pol_pwi_service(pred, margin=0.0):
    """Minimise predicted J among settings whose predicted shortage does not exceed
    the baseline's predicted shortage plus margin; fall back to baseline."""
    tot, sh = pred.sum(axis=2), pred[:, :, -1]
    ok = sh <= sh[BASE_IDX][None, :] + margin
    tot = np.where(ok, tot, np.inf)
    return tot.argmin(axis=0)


def pol_silo(pred):
    tgt = pred[BASE_IDX, :, :len(WASTES)].argmax(axis=1)
    vals = pred[:, np.arange(pred.shape[1]), tgt] + 1e-9 * (np.arange(len(COMBOS)) != BASE_IDX)[:, None]
    return vals.argmin(axis=0), tgt


def pol_best_fixed(train):
    """Conventional benchmark: the single setting with the lowest mean realised J in
    the switchback training data, applied on every shift."""
    m = train.groupby(["B", "Q", "K"])["objective"].mean()
    return COMBOS.index(tuple(int(v) for v in m.idxmin()))


def realise(C, idx):
    return C[idx, np.arange(C.shape[1]), :]


def prospective_target(pred, idx):
    """Target waste = largest *predicted* reduction of the chosen setting vs baseline
    (uses the same predictions the policy used, so it is fixed before the shift)."""
    d = pred[idx, np.arange(len(idx)), :len(WASTES)] - pred[BASE_IDX, :, :len(WASTES)]
    return d.argmin(axis=1)


def erosion(R, Rb, tgt):
    """Aggregate erosion ratio rho = sum_t S_t / sum_t G_t over ALL shifts, with the
    prospective target tgt fixed before each shift and shortage included in the
    spillover S. Identity: sum_t dJ_t = sum_t G_t (rho - 1). Returns rho (nan if the
    aggregate targeted gain is not positive) and, as a simulation-only descriptive,
    the share of shifts with non-positive realised targeted gain."""
    d = R - Rb
    t = np.arange(len(d))
    gain = -d[t, tgt]
    spill = d.sum(axis=1) + gain
    G = gain.sum()
    rho = spill.sum() / G if G > 1e-9 else np.nan
    return rho, float((gain <= 1e-9).mean())


def block_ci(d, block=10, reps=2000, seed=0):
    d = np.asarray(d); r, n = np.random.default_rng(seed), len(d)
    starts = r.integers(0, n - block + 1, (reps, n // block))   # every start, including the last
    boots = d[(starts[:, :, None] + np.arange(block)).reshape(reps, -1)].mean(axis=1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


# ------------------------------------------------------- pilot power
def run_world(P, seed, keep=False):
    ex_tr, ex_te = exogenous(N_TRAIN, seed, P), exogenous(N_TEST, seed + 10_000, P)
    C_tr, C_te = cube(ex_tr, P), cube(ex_te, P)
    E_te = expected_cube(ex_te, P, seed + 20_000)
    sw = simulate(ex_tr, *switchback(N_TRAIN, seed + 1), P)
    hist = {f"{'unlogged' if u else 'logged'}_{'overlap' if o else 'nooverlap'}":
            simulate(ex_tr, *supervisor(ex_tr, seed + 2, P, u, o), P)
            for u in (False, True) for o in (True, False)}
    G = gamma_true(C_tr)

    # ---- estimator diagnostics
    est = {"naive_" + k: gamma_naive(v) for k, v in hist.items()}
    adj = {"adjusted_" + k: gamma_adjusted(v, ci=True) for k, v in hist.items()}
    adj["switchback"] = gamma_adjusted(sw, ci=True)
    est.update({k: v[0] for k, v in adj.items()})
    res = {"J_base": float(C_te[BASE_IDX].sum(axis=1).mean())}
    for k, v in est.items():
        err = v - G
        res[f"bias_{k}"] = float(np.nanmean(err)); res[f"rmse_{k}"] = float(np.sqrt(np.nanmean(err ** 2)))
        for th in THRESHOLDS:
            m = np.abs(G) >= th
            res[f"rev{th}_{k}"] = float(np.nanmean(np.sign(v[m]) != np.sign(G[m]))) if m.any() else np.nan
    for k, (g, lo, hi) in adj.items():
        res[f"cover_{k}"] = float(np.mean((G >= lo) & (G <= hi)))

    # ---- predictions and policies on the test horizon
    pred = {"ols_sw": predict_cube_ols(sw, ex_te), "gbm_sw": predict_cube_gbm(sw, ex_te, seed),
            "ols_hist_LO": predict_cube_ols(hist["logged_overlap"], ex_te),
            "ols_hist_LN": predict_cube_ols(hist["logged_nooverlap"], ex_te),
            "ols_hist_UO": predict_cube_ols(hist["unlogged_overlap"], ex_te),
            "ols_sw72": predict_cube_ols(sw.iloc[:72], ex_te), "ols_sw144": predict_cube_ols(sw.iloc[:144], ex_te)}
    directJ_ols = predict_cube_ols(sw, ex_te, direct=True)
    res["directJ_equals_sum"] = float(np.max(np.abs(directJ_ols - pred["ols_sw"].sum(axis=2))))
    directJ_gbm = predict_cube_gbm(sw, ex_te, seed, direct=True)

    pol, tg = {}, {}
    pol["pf_oracle"] = C_te.sum(axis=2).argmin(axis=0);   tg["pf_oracle"] = prospective_target(C_te, pol["pf_oracle"])
    pol["causal_oracle"] = E_te.sum(axis=2).argmin(axis=0); tg["causal_oracle"] = prospective_target(E_te, pol["causal_oracle"])
    pol["silo_true"], tg["silo_true"] = pol_silo(E_te)
    pol["silo_est"], tg["silo_est"] = pol_silo(pred["ols_sw"])
    pol["silo_est72"], tg["silo_est72"] = pol_silo(pred["ols_sw72"])
    pol["silo_est144"], tg["silo_est144"] = pol_silo(pred["ols_sw144"])
    bf = pol_best_fixed(sw); pol["best_fixed"] = np.full(N_TEST, bf); tg["best_fixed"] = prospective_target(pred["ols_sw"], pol["best_fixed"])
    for k in ["ols_sw", "gbm_sw", "ols_hist_LO", "ols_hist_LN", "ols_hist_UO", "ols_sw72", "ols_sw144"]:
        pol["pwi_" + k] = pol_pwi(pred[k]); tg["pwi_" + k] = prospective_target(pred[k], pol["pwi_" + k])
    pol["pwi_svc_ols_sw"] = pol_pwi_service(pred["ols_sw"]); tg["pwi_svc_ols_sw"] = prospective_target(pred["ols_sw"], pol["pwi_svc_ols_sw"])
    pol["pwi_directJ_gbm_sw"] = pol_pwi(directJ_gbm); tg["pwi_directJ_gbm_sw"] = prospective_target(pred["gbm_sw"], pol["pwi_directJ_gbm_sw"])

    Rb = C_te[BASE_IDX]; Jb = Rb.sum(axis=1)
    for k, idx in pol.items():
        R = realise(C_te, idx); J = R.sum(axis=1)
        res[f"red_{k}"] = float(100 * (Jb.mean() - J.mean()) / Jb.mean())
        res[f"short_{k}"] = float(R[:, -1].mean())
        res[f"ero_{k}"], res[f"nogain_{k}"] = (float(x) for x in erosion(R, Rb, tg[k]))

    num = den = 0.0
    for row in G:
        t = row.argmin()
        if row[t] < 0:
            den += -row[t]; num += np.clip(np.delete(row, t), 0, None).sum()
    res["coupling_intensity"] = num / den if den > 0 else np.nan
    if keep:
        return res, dict(G=G, est=est, adj=adj, sw=sw, hist=hist, C_te=C_te, E_te=E_te, pol=pol, tg=tg, ex_te=ex_te)
    return res


def sample_world(r):
    """Ranges extended to include near-zero coupling (off-target effects near 0)."""
    P = copy.deepcopy(BASE)
    s1 = r.uniform(0.45, 0.85)
    P["sB"] = [1.0, s1, r.uniform(0.15, s1 - 0.1)]
    hi = r.uniform(0, 15)
    P["topup"] = [0.0, hi * r.uniform(0.3, 0.6), hi]
    P.update(wipB=r.uniform(0.0, 1.2), wipQ=r.uniform(0.0, 1.0), batch_wait=r.uniform(0, 25),
             trig=r.uniform(0.3, 0.9), stop0=r.uniform(0, 15), stopz=r.uniform(0, 650),
             insp=r.uniform(0.0, 0.4), motw=r.uniform(0.0, 1.0), motK=r.uniform(0, 15),
             bsd=r.uniform(0.3, 0.7), zsd=r.uniform(0.006, 0.02), flip=r.uniform(0.05, 0.3))
    P["kappa"] = {k: v * float(np.exp(r.normal(0, 0.35))) for k, v in BASE["kappa"].items()}
    return P


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PWI study v5. Use --all for the full pipeline (base case + Monte Carlo).")
    ap.add_argument("--worlds", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=None, help="run one chunk of worlds (for time-limited environments)")
    ap.add_argument("--chunk-size", type=int, default=30)
    ap.add_argument("--base", action="store_true", help="base case, oracle convergence check and pilot power simulation")
    ap.add_argument("--all", action="store_true", help="base case followed by every Monte Carlo chunk")
    args = ap.parse_args()
    os.makedirs("results_v6", exist_ok=True)
    if not (args.base or args.all or args.chunk is not None):
        ap.print_help(); raise SystemExit(0)

    if args.base or args.all:
        res, D = run_world(BASE, seed=1, keep=True)
        G = D["G"]
        rows = []
        for i, (lev, hi, lo) in enumerate(CONTRASTS):
            for j, w in enumerate(WASTES):
                row = dict(contrast=f"{lev}{hi} vs {lev}{lo}", waste=w, true=G[i, j])
                for k, v in D["est"].items():
                    row[k] = v[i, j]
                row["sw_lo"], row["sw_hi"] = D["adj"]["switchback"][1][i, j], D["adj"]["switchback"][2][i, j]
                rows.append(row)
        pd.DataFrame(rows).round(3).to_csv("results_v6/base_gamma.csv", index=False)
        C_te, Rb = D["C_te"], D["C_te"][BASE_IDX]
        prow = [dict(policy="baseline", waste=Rb[:, :7].sum(axis=1).mean(), shortage=Rb[:, 7].mean(),
                     J=Rb.sum(axis=1).mean(), dJ=0, dJ_lo=0, dJ_hi=0, erosion=np.nan, nogain=np.nan)]
        for k, idx in D["pol"].items():
            R = realise(C_te, idx)
            m, lo, hi = block_ci(R.sum(axis=1) - Rb.sum(axis=1))
            e, ng = erosion(R, Rb, D["tg"][k])
            prow.append(dict(policy=k, waste=R[:, :7].sum(axis=1).mean(), shortage=R[:, 7].mean(),
                             J=R.sum(axis=1).mean(), dJ=m, dJ_lo=lo, dJ_hi=hi, erosion=e, nogain=ng))
        pd.DataFrame(prow).round(3).to_csv("results_v6/base_policies.csv", index=False)
        json.dump({k: float(v) for k, v in res.items()}, open("results_v6/base_summary.json", "w"), indent=1)
        # oracle convergence: expected J of the causal-oracle choice at increasing draw counts
        conv = {}
        for dr in (100, 400, 1600):
            E = expected_cube(D["ex_te"], BASE, seed=20_001, draws=dr)
            idx = E.sum(axis=2).argmin(axis=0)
            conv[dr] = dict(realised_J=float(realise(C_te, idx).sum(axis=1).mean()),
                            agree_with_400=float(np.mean(idx == D["pol"]["causal_oracle"])))
        json.dump(conv, open("results_v6/oracle_convergence.json", "w"), indent=1)
        Jb = Rb.sum(axis=1); ac = [float(np.corrcoef(Jb[:-k], Jb[k:])[0, 1]) for k in range(1, 6)]
        json.dump(dict(sd_J=float(Jb.std()), acf=ac), open("results_v6/base_acf.json", "w"), indent=1)
        print(pd.DataFrame(prow).round(2).to_string(index=False)); print(conv)

    if args.all or args.chunk is not None:
        r = np.random.default_rng(2026)
        Ps = [sample_world(r) for _ in range(args.worlds)]
        chunks = range((args.worlds + args.chunk_size - 1) // args.chunk_size) if args.all else [args.chunk]
        for ch in chunks:
            lo, hi = args.chunk_size * ch, min(args.chunk_size * (ch + 1), args.worlds)
            out = []
            for w in range(lo, hi):
                rw = run_world(Ps[w], seed=100 + 7 * w); rw["world"] = w
                rw.update({f"p_{k}": (v if not isinstance(v, (list, dict)) else json.dumps(v)) for k, v in Ps[w].items()})
                out.append(rw)
            pd.DataFrame(out).to_csv(f"results_v6/mc_chunk{ch}.csv", index=False)
            print(f"chunk {ch}: worlds {lo}-{hi-1} done", flush=True)
