"""Does decision rule H1 discriminate confounding from noise?  Baseline phase = 48 shifts (weeks 1-4) run by the
reactive supervisor (base-case parameters, deviation prob 0.10, overlap); switchback estimate from 144 exploration
shifts in two-shift blocks with HAC(4) intervals.  Control: the same 48 lever settings shuffled across shifts, which
keeps the lever marginals but breaks the link to conditions, i.e. no confounding."""
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
import pwi_study_v6 as p
warnings.filterwarnings("ignore")
P = p.BASE
ALPHA = 0.05

def naive_with_ci(df):
    """Marginal difference in means per lever (the paper's naive contrast) with HAC(4) 95% CI."""
    G = np.full((4, 7), np.nan); LO = np.full((4, 7), np.nan); HI = np.full((4, 7), np.nan)
    for i, (lev, hi, lo) in enumerate(p.CONTRASTS):
        if (df[lev] == hi).sum() == 0 or (df[lev] == lo).sum() == 0: continue
        X = sm.add_constant(pd.DataFrame({f"{lev}{l}": (df[lev].values == l) * 1. for l in sorted(df[lev].unique()) if l != lo}), has_constant="add")
        for j, w in enumerate(p.WASTES):
            f = sm.OLS(df["c_" + w].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
            G[i, j] = f.params[f"{lev}{hi}"]; LO[i, j], HI[i, j] = f.conf_int(alpha=ALPHA).loc[f"{lev}{hi}"]
    return G, LO, HI

def one_rep(seed, regime):
    r = np.random.default_rng(seed)
    ex48 = p.exogenous(48, seed + 1, P)
    B, Q, K = p.supervisor(ex48, seed + 2, P, unlogged=(regime == "unlogged"), overlap=True)
    if regime == "shuffled":
        B, Q, K = p.supervisor(ex48, seed + 2, P, unlogged=True, overlap=True)
        perm = r.permutation(48); B, Q, K = B[perm], Q[perm], K[perm]
    hist = p.simulate(ex48, B, Q, K, P)
    Gn, LOn, HIn = naive_with_ci(hist)
    ex144 = p.exogenous(144, seed + 3, P)
    sw = p.simulate(ex144, *p.switchback(144, seed + 4, block=2), P)
    Gs, LOs, HIs = p.gamma_adjusted(sw, ci=True)
    material = (np.abs(Gs) >= 0.5) & ((LOs > 0) | (HIs < 0))
    opp = material & np.isfinite(Gn) & (np.sign(Gn) != np.sign(Gs))
    opp_sig = opp & ((LOn > 0) | (HIn < 0))               # naive interval excludes 0 on the wrong side
    undefined = int(np.isnan(Gn[:, 0]).sum())               # levers with a level absent in 48 shifts
    return dict(n_material=int(material.sum()), h1_as_written=bool(opp.any()), n_opp=int(opp.sum()),
                h1_strict=bool(opp_sig.any()), undefined_levers=undefined)

rows = []
for regime in ["unlogged", "logged", "shuffled"]:
    for k in range(300):
        rows.append(dict(regime=regime, **one_rep(50_000 + 100 * k, regime)))
d = pd.DataFrame(rows)
d.to_csv("results_v6/h1_check.csv", index=False)
print(d.groupby("regime").agg(reps=("h1_as_written", "size"), material_pairs=("n_material", "mean"),
                              H1_as_written=("h1_as_written", "mean"), opp_pairs_mean=("n_opp", "mean"),
                              H1_strict=("h1_strict", "mean"), undefined_lever_share=("undefined_levers", lambda x: (x > 0).mean())).round(3).to_string())
