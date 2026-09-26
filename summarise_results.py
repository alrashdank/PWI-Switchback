"""Summarise the result files in results_v6/ into results_summary.json: the values reported in the paper, computed from the simulation outputs."""
import json, pandas as pd, numpy as np
R = "results_v6/"
mc = pd.read_csv(R + "mc_worlds.csv"); g = pd.read_csv(R + "base_gamma.csv"); pol = pd.read_csv(R + "base_policies.csv")
acf = json.load(open(R + "base_acf.json")); conv = json.load(open(R + "oracle_convergence.json"))
co = pd.read_csv(R + "carryover_sweep.csv"); cp = pd.read_csv(R + "carryover_policy.csv")
import glob
def f(x, d=2, sign=False):
    s = f"{x:.{d}f}".replace("-", "−"); return ("+" + s) if (sign and x > 0) else s
def q(s): return dict(med=s.median(), lo=s.quantile(.25), hi=s.quantile(.75))
POL = [("silo_est", "Siloed, estimated model"), ("silo_true", "Siloed, true expected costs"), ("best_fixed", "Best fixed setting (switchback data)"),
       ("pwi_ols_hist_UO", "Total-cost OLS, historical, unlogged, overlap"), ("pwi_ols_hist_LN", "Total-cost OLS, historical, logged, no overlap"),
       ("pwi_ols_hist_LO", "Total-cost OLS, historical, logged, overlap"), ("pwi_ols_sw72", "Total-cost OLS, switchback, 72 shifts"),
       ("pwi_ols_sw144", "Total-cost OLS, switchback, 144 shifts"), ("pwi_ols_sw", "Total-cost OLS, switchback, 600 shifts"),
       ("pwi_gbm_sw", "Component GBM, switchback"), ("pwi_directJ_gbm_sw", "Direct-J GBM, switchback"), ("pwi_svc_ols_sw", "Service-constrained OLS, switchback"),
       ("causal_oracle", "Causal oracle (lagged information)"), ("pf_oracle", "Perfect-foresight oracle")]
MAIN = ["silo_est", "best_fixed", "pwi_ols_hist_UO", "pwi_ols_hist_LO", "pwi_ols_sw", "pwi_svc_ols_sw", "causal_oracle", "pf_oracle"]
names = dict(POL)
gi = g.set_index(["contrast", "waste"])
sel = [("B2 vs B0", "waiting", "Buffer high vs low", "Waiting"), ("B2 vs B0", "inventory", "Buffer high vs low", "Inventory"),
       ("B2 vs B0", "overproduction", "Buffer high vs low", "Overproduction"), ("K1 vs K0", "defects", "Tight vs standard trigger", "Defects"),
       ("K1 vs K0", "over_processing", "Tight vs standard trigger", "Over-processing"), ("K1 vs K0", "waiting", "Tight vs standard trigger", "Waiting"),
       ("Q1 vs Q0", "transport", "Large vs small batch", "Transport"), ("Q1 vs Q0", "waiting", "Large vs small batch", "Waiting")]
t3 = []
for c, w, cl, wl in sel:
    r = gi.loc[(c, w)]
    t3.append([cl, wl, f(r.true, 2, True), f(r.naive_logged_overlap, 2, True), f(r.naive_unlogged_overlap, 2, True), f(r.adjusted_logged_overlap, 2, True),
               f(r.adjusted_unlogged_overlap, 2, True), f"{f(r.switchback, 2, True)} [{f(r.sw_lo)}, {f(r.sw_hi)}]"])
mat = g.true.abs() >= 0.5
base = dict(n_cover=int(((g.true >= g.sw_lo) & (g.true <= g.sw_hi)).sum()), n_pairs=int(len(g)), n_material=int(mat.sum()),
            rev_naiveLO=int((g.naive_logged_overlap[mat] * g.true[mat] < 0).sum()), rev_naiveUO=int((g.naive_unlogged_overlap[mat] * g.true[mat] < 0).sum()),
            rev_adjLO=int((g.adjusted_logged_overlap[mat] * g.true[mat] < 0).sum()), rev_adjUO=int((g.adjusted_unlogged_overlap[mat] * g.true[mat] < 0).sum()),
            rev_sw=int((g.switchback[mat] * g.true[mat] < 0).sum()),
            b2_wait_true=f(gi.loc[("B2 vs B0", "waiting")].true, 1), b2_wait_naiveU=f(gi.loc[("B2 vs B0", "waiting")].naive_unlogged_overlap, 1, True),
            b2_wait_adjU=f(gi.loc[("B2 vs B0", "waiting")].adjusted_unlogged_overlap, 1, True),
            k_def_true=f(gi.loc[("K1 vs K0", "defects")].true, 1), k_def_naiveU=f(gi.loc[("K1 vs K0", "defects")].naive_unlogged_overlap, 1, True),
            k_def_adjU=f(gi.loc[("K1 vs K0", "defects")].adjusted_unlogged_overlap, 1, True))
pi = pol.set_index("policy")
def polrow(k):
    r = pi.loc[k]
    return ["Fixed baseline" if k == "baseline" else names[k], f(r.waste), f(r.shortage), f(r.J),
            "–" if k == "baseline" else f"{f(r.dJ)} [{f(r.dJ_lo)}, {f(r.dJ_hi)}]", "–" if k == "baseline" else f(r.erosion)]
t4 = [polrow("baseline")] + [polrow(k) for k in MAIN]; t4_full = [polrow("baseline")] + [polrow(k) for k, _ in POL]
b = pi.loc
basepol = dict(red_pwi=f(-b["pwi_ols_sw"].dJ, 1), red_bf=f(-b["best_fixed"].dJ, 1), red_silo=f(-b["silo_est"].dJ, 1), red_co=f(-b["causal_oracle"].dJ, 1),
               red_pf=f(-b["pf_oracle"].dJ, 1), red_svc=f(-b["pwi_svc_ols_sw"].dJ, 1), sh_base=f(b["baseline"].shortage), sh_pwi=f(b["pwi_ols_sw"].shortage),
               sh_svc=f(b["pwi_svc_ols_sw"].shortage), sh_bf=f(b["best_fixed"].shortage), sh_histU=f(b["pwi_ols_hist_UO"].shortage),
               red_histU=f(-b["pwi_ols_hist_UO"].dJ, 1), ero_silo=f(b["silo_est"].erosion), ero_pwi=f(b["pwi_ols_sw"].erosion), ero_bf=f(b["best_fixed"].erosion),
               ero_svc=f(b["pwi_svc_ols_sw"].erosion), red_sw72=f(-b["pwi_ols_sw72"].dJ, 1), red_sw144=f(-b["pwi_ols_sw144"].dJ, 1),
               red_silo72=f(-b["silo_est72"].dJ, 1), sh_silo72=f(b["silo_est72"].shortage), nogain_svc=f"{100 * b['pwi_svc_ols_sw'].nogain:.0f}")
EST = [("naive_logged_overlap", "Naive, historical, logged"), ("naive_unlogged_overlap", "Naive, historical, unlogged"),
       ("adjusted_logged_nooverlap", "Adjusted, historical, logged, no overlap"), ("adjusted_logged_overlap", "Adjusted, historical, logged, overlap"),
       ("adjusted_unlogged_overlap", "Adjusted, historical, unlogged, overlap"), ("switchback", "Adjusted, switchback")]
t5 = []; est = {}
for k, l in EST:
    cov = f"{100 * mc['cover_' + k].mean():.0f}%" if 'cover_' + k in mc else "–"
    row = [l, f(mc['bias_' + k].mean(), 2, True), f(mc['rmse_' + k].median()), f"{100 * mc['rev0.25_' + k].mean():.1f}%", f"{100 * mc['rev0.5_' + k].mean():.1f}%",
           f"{100 * mc['rev1.0_' + k].mean():.1f}%", f"{100 * (mc['rev0.5_' + k] > 0).mean():.1f}%", cov]
    t5.append(row); est[k] = dict(bias=row[1], rmse=row[2], rev5=row[4], any5=row[6], cov=cov)
t6 = []; t6_full = []; mcp = {}
for k, l in POL:
    Q = q(mc['red_' + k]); row = [l, f"{Q['med']:.1f} [{Q['lo']:.1f}, {Q['hi']:.1f}]", f(mc['short_' + k].median()), f(mc['ero_' + k].median()), f"{100 * mc['nogain_' + k].median():.0f}%"]
    t6_full.append(row); mcp[k] = dict(med=f"{Q['med']:.1f}", lo=f"{Q['lo']:.1f}", hi=f"{Q['hi']:.1f}", short=row[2], ero=row[3], nogain=row[4])
    if k in MAIN: t6.append(row[:4])
def diff(a, bb): d = mc['red_' + a] - mc['red_' + bb]; return dict(med=f"{d.median():.1f}", lo=f"{d.quantile(.25):.1f}", hi=f"{d.quantile(.75):.1f}", pos=f"{100 * (d > 0).mean():.1f}")
x = mc.coupling_intensity; y = mc.red_pwi_ols_sw - mc.red_silo_est
lo = x <= x.quantile(1 / 3); hi = x >= x.quantile(2 / 3); lowc = x < 1.0
gb = mc.red_pwi_gbm_sw - mc.red_pwi_ols_sw; dg = mc.red_pwi_directJ_gbm_sw - mc.red_pwi_gbm_sw
dgo = mc.red_pwi_directJ_gbm_sw - mc.red_pwi_ols_sw
mcs = dict(n=int(len(mc)), pwi_silo=diff("pwi_ols_sw", "silo_est"), pwi_bf=diff("pwi_ols_sw", "best_fixed"), pwi_co=diff("pwi_ols_sw", "causal_oracle"),
           sw_histU=diff("pwi_ols_sw", "pwi_ols_hist_UO"), sw_histL=diff("pwi_ols_sw", "pwi_ols_hist_LO"), histLO_LN=diff("pwi_ols_hist_LO", "pwi_ols_hist_LN"),
           sw_72=diff("pwi_ols_sw", "pwi_ols_sw72"), sw_144=diff("pwi_ols_sw", "pwi_ols_sw144"), sw144_72=diff("pwi_ols_sw144", "pwi_ols_sw72"),
           svc=diff("pwi_svc_ols_sw", "pwi_ols_sw"), gbm=dict(med=f(gb.median()), lo=f(gb.quantile(.25)), hi=f(gb.quantile(.75)), pos=f"{100 * (gb > 0).mean():.1f}"),
           dgbm=dict(med=f(dg.median()), pos=f"{100 * (dg > 0).mean():.1f}", vs_ols=f(dgo.median())),
           regret_ols=f"{(mc.red_causal_oracle - mc.red_pwi_ols_sw).median():.1f}", regret_pf=f"{(mc.red_pf_oracle - mc.red_pwi_ols_sw).median():.1f}",
           ci5=f"{x.quantile(.05):.1f}", ci50=f"{x.quantile(.5):.1f}", ci95=f"{x.quantile(.95):.1f}", cimin=f"{x.min():.2f}", rho=f"{x.corr(y, method='spearman'):.2f}",
           adv_low=f"{y[lo].median():.1f}", pos_low=f"{100 * (y[lo] > 0).mean():.0f}", adv_high=f"{y[hi].median():.1f}", n_lowc=int(lowc.sum()),
           adv_lowc=f"{y[lowc].median():.1f}", pos_lowc=f"{100 * (y[lowc] > 0).mean():.0f}", ybf_lowc=f"{(mc.red_pwi_ols_sw - mc.red_best_fixed)[lowc].median():.1f}",
           short_silo=mcp['silo_est']['short'], short_pwi=mcp['pwi_ols_sw']['short'], short_svc=mcp['pwi_svc_ols_sw']['short'], short_histU=mcp['pwi_ols_hist_UO']['short'],
           short_bf=mcp['best_fixed']['short'], sd_J=f"{acf['sd_J']:.1f}", acf=[f"{a:.2f}" for a in acf['acf']], neff_factor=f"{1 + 2 * sum(acf['acf']):.2f}",
           silo72_med=f"{mc.red_silo_est72.median():.1f}", silo144_med=f"{mc.red_silo_est144.median():.1f}", silo600_med=f"{mc.red_silo_est.median():.1f}")
# oracle convergence
oc = {k: dict(J=f"{v['realised_J']:.2f}", agree=f"{100 * v['agree_with_400']:.0f}") for k, v in conv.items()}
# pilot evaluation (pairwise randomisation test; carry-over model)
def _pct(x):
    v = int(round(100 * x)); return "0" if v == 0 else (f"{v:d}" if v > 0 else f"−{-v:d}")
pil = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "pilot_c[0-9]*_w*.csv"))], ignore_index=True)
szf = pd.read_csv(R + "sizecheck_four_day_c0.00_w24.csv"); szt = pd.read_csv(R + "sizecheck_two_day_c0.00_w24.csv")
def _cl(d, col):
    """Rejection rate (%) and clustered s.e. (by simulated path) for a fixed-path size check."""
    r = (d[col] < 0.05).groupby(d.rep).mean()
    return 100 * (d[col] < 0.05).mean(), 100 * r.std(ddof=1) / np.sqrt(len(r)), int(d.rep.nunique()), int(len(d) // d.rep.nunique())
sizechk = []
for dz, lab in (("two_day", "Two-day blocks"), ("four_day", "Four-day blocks")):
    for cc in ("0.00", "0.50"):
        d = pd.read_csv(R + f"sizecheck_{dz}_c{cc}_w24.csv")
        (rs, ss, npth, ndr), (rk, sk, _, _) = _cl(d, "p_sym"), _cl(d, "p_skew")
        sizechk.append([lab, cc[:3], f"{npth} × {ndr}", f"{rs:.2f} ({ss:.2f})", f"{rk:.2f} ({sk:.2f})"])
VN = {("exp", "two_half", 9): "Exponential, SD 9, half into next two shifts (as in Table S6)",
      ("exp", "two_half", 18): "Exponential, SD 18, half into next two shifts",
      ("lognorm", "two_half", 9): "Log-normal (skewness 6.2), SD 9, half into next two shifts",
      ("exp", "geom", 9): "Exponential, SD 9, geometric spill (factor 0.7 per shift) over the next block",
      ("lognorm", "geom", 9): "Log-normal, SD 9, geometric spill over the next block"}
variants = []
for (kind, spill, sd), lab in VN.items():
    row = [lab]
    for dz in ("two_day", "four_day"):
        d = pd.read_csv(R + f"sizevariant_{dz}_{kind}_{spill}_sd{sd}_w24.csv")
        r = (d.p < 0.05).groupby(d.rep).mean()
        row.append(f"{100 * (d.p < 0.05).mean():.1f} ({100 * r.std(ddof=1) / np.sqrt(len(r)):.1f})")
    variants.append(row)
daily_sym = {}
for wk in (12, 24):
    d = pd.read_csv(R + f"sizevariant_daily_sym_None_sd9_w{wk}.csv"); r = (d.p < 0.05).groupby(d.rep).mean()
    daily_sym[wk] = (f"{100 * (d.p < 0.05).mean():.1f}", f"{100 * r.std(ddof=1) / np.sqrt(len(r)):.1f}")
h95 = pd.read_csv(R + "h1_check.csv"); h99 = pd.read_csv(R + "h1_check_alpha01.csv")
h1 = {}
for reg in ("unlogged", "logged", "shuffled"):
    a95 = h95[h95.regime == reg]; a99 = h99[h99.regime == reg]
    h1[reg] = dict(written=f"{100 * a95.h1_as_written.mean():.0f}", strict95=f"{100 * a95.h1_strict.mean():.0f}",
                   strict99=f"{100 * a99.h1_strict.mean():.0f}", reps=int(len(a95)))
agg = pil.groupby(["weeks", "c", "design", "scenario"]).agg(rej=("p", lambda x: (x < 0.05).mean()), est=("est", "mean"),
                                                           truth=("truth", "mean"), reps=("rep", "nunique"), units=("n_units", "first")).reset_index()
A = agg.set_index(["weeks", "c", "design", "scenario"])
DN = {"daily": "One-day blocks, no washout", "two_day": "Two-day blocks, 1-shift washout", "four_day": "Four-day blocks, 1-shift washout"}
tpil = []
for wk in (12, 24):
    for c in sorted(agg[agg.weeks == wk].c.unique()):
        for dz in ("daily", "two_day", "four_day"):
            a, sh, wk_, sk = (A.loc[(wk, c, dz, x)] for x in ("alt", "sharp", "weak", "weak_skew"))
            tpil.append([str(wk), f"{c:.1f}", DN[dz], str(int(a.reps)), f"{100 * a.rej:.0f}", f(a.est, 1), f(a.truth, 1),
                         _pct(1 - a.est / a.truth), f"{100 * sh.rej:.1f}", f"{100 * wk_.rej:.1f}", f"{100 * sk.rej:.1f}"])
big = agg[(agg.units >= 12)]
sharp_big = big[big.scenario == "sharp"].rej
fd12 = agg[(agg.design == "four_day") & (agg.weeks == 12) & (agg.scenario == "sharp")].rej
skew_all = agg[agg.scenario == "weak_skew"].rej
def att(wk, c, dz): a = A.loc[(wk, c, dz, "alt")]; return _pct(1 - a.est / a.truth)
def pwr(wk, c, dz): return f"{100 * A.loc[(wk, c, dz, 'alt')].rej:.0f}"
pilot = dict(two24_c0=pwr(24, 0.0, "two_day"), two24_c05=pwr(24, 0.5, "two_day"), two24_att_c05=att(24, 0.5, "two_day"),
             daily12_c0=pwr(12, 0.0, "daily"), daily12_c05=pwr(12, 0.5, "daily"), daily12_c1=pwr(12, 1.0, "daily"),
             daily_att_c05=att(12, 0.5, "daily"), daily_att_c1=att(12, 1.0, "daily"), two12_c0=pwr(12, 0.0, "two_day"),
             two12_att_c1=att(12, 1.0, "two_day"), four12_att_c1=att(12, 1.0, "four_day"), four24_c0=pwr(24, 0.0, "four_day"),
             truth_c0=f(A.loc[(24, 0.0, "two_day", "alt")].truth, 1), truth_c05=f(A.loc[(24, 0.5, "two_day", "alt")].truth, 1),
             sharp_lo=f"{100 * sharp_big.min():.1f}", sharp_hi=f"{100 * sharp_big.max():.1f}",
             fd12_lo=f"{100 * fd12.min():.1f}", fd12_hi=f"{100 * fd12.max():.1f}",
             skew_lo=f"{100 * skew_all.min():.1f}", skew_hi=f"{100 * skew_all.max():.1f}",
             fd_sym=f"{100 * (szf.p_sym < 0.05).mean():.2f}", fd_skew=f"{100 * (szf.p_skew < 0.05).mean():.2f}", fd_n=int(len(szf)),
             td_sym=f"{100 * (szt.p_sym < 0.05).mean():.2f}", td_skew=f"{100 * (szt.p_skew < 0.05).mean():.2f}", td_n=int(len(szt)),
             var_td_max=max(float(r[1].split()[0]) for r in variants), var_td_min=min(float(r[1].split()[0]) for r in variants),
             var_fd_max=max(float(r[2].split()[0]) for r in variants),
             daily_sym12=daily_sym[12][0], daily_sym24=daily_sym[24][0],
             four24_weak300=f"{100 * A.loc[(24, 0.0, 'four_day', 'weak')].rej:.1f}",
             reps12=int(A.loc[(12, 0.0, 'two_day', 'alt')].reps),
             reps24=int(A.loc[(24, 0.0, 'two_day', 'alt')].reps))
# carry-over
cs = co.groupby(["c", "block", "estimator"])[["bias", "rmse", "rev"]].mean()
co["bi_err"] = co.b2_inv_hat - co.b2_inv_true; co["kd_err"] = co.k_def_hat - co.k_def_true
ce = co.groupby(["c", "block", "estimator"])[["bi_err", "kd_err"]].mean()
tco = []   # Table: RMSE by c x block for shift_level, washout_1, lag1_exog, block_mean
for c in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for bl in [1, 2, 4, 8]:
        row = [f"{c:.2f}", str(bl)]
        for e in ["shift_level", "washout_1", "lag1_exog", "block_mean", "shift_level_drift", "washout_1_drift"]:
            row.append(f(cs.loc[(c, bl, e)].rmse) if (c, bl, e) in cs.index else "–")
        tco.append(row)
truth_kd = co.groupby("c").k_def_true.mean(); truth_bi = co.groupby("c").b2_inv_true.mean()
carry = dict(reps=int(co.rep.nunique()),
             rmse_c0_b2=f(cs.loc[(0.0, 2, "shift_level")].rmse), rmse_c1_b1=f(cs.loc[(1.0, 1, "shift_level")].rmse), rmse_c1_b2=f(cs.loc[(1.0, 2, "shift_level")].rmse),
             rmse_c1_b8=f(cs.loc[(1.0, 8, "shift_level")].rmse), rmse_c1_b8_wash=f(cs.loc[(1.0, 8, "washout_1")].rmse), rmse_c1_b8_lag=f(cs.loc[(1.0, 8, "lag1_exog")].rmse),
             rmse_c1_b8_blk=f(cs.loc[(1.0, 8, "block_mean")].rmse), rmse_c025_b2=f(cs.loc[(0.25, 2, "shift_level")].rmse), rmse_c05_b2=f(cs.loc[(0.5, 2, "shift_level")].rmse),
             rev_cellmax=f"{100 * cs.rev.max():.1f}", bi_true=f(truth_bi.loc[1.0], 1), bi_err_c1_b2=f(ce.loc[(1.0, 2, "shift_level")].bi_err, 1),
             bi_err_c1_b8=f(ce.loc[(1.0, 8, "shift_level")].bi_err, 1), bi_err_c1_b8_lag=f(ce.loc[(1.0, 8, "lag1_exog")].bi_err, 1),
             kd_true=f(truth_kd.loc[1.0], 1), kd_err_c1_b2=f(ce.loc[(1.0, 2, "shift_level")].kd_err, 1), kd_err_c1_b4_lag=f(ce.loc[(1.0, 4, "lag1_exog")].kd_err, 2),
             kd_err_c1_b8_lag=f(ce.loc[(1.0, 8, "lag1_exog")].kd_err, 2), kd_err_c1_b8=f(ce.loc[(1.0, 8, "shift_level")].kd_err, 1),
             kd_share_c1_b2=f"{100 * (1 + ce.loc[(1.0, 2, 'shift_level')].kd_err / truth_kd.loc[1.0]):.0f}", bi_share_c1_b2=f"{100 * (1 - (-ce.loc[(1.0, 2, 'shift_level')].bi_err) / truth_bi.loc[1.0]):.0f}",
             bi_share_c1_b8_lag=f"{100 * (1 - (-ce.loc[(1.0, 8, 'lag1_exog')].bi_err) / truth_bi.loc[1.0]):.0f}")
# recovery share = mean estimate / mean true effect in each cell (ratio of means, used throughout)
_g = co.groupby(["c", "block", "estimator"])[["k_def_hat", "k_def_true", "b2_inv_hat", "b2_inv_true"]].mean()
rc = pd.DataFrame({"kd_rec": _g.k_def_hat / _g.k_def_true, "bi_rec": _g.b2_inv_hat / _g.b2_inv_true})
carry.update(kd_rec_sl_b2=f"{100 * rc.loc[(1.0, 2, 'shift_level')].kd_rec:.0f}", bi_rec_sl_b1=f"{100 * rc.loc[(1.0, 1, 'shift_level')].bi_rec:.0f}",
             kd_rec_lag_b4=f"{100 * rc.loc[(1.0, 4, 'lag1_exog')].kd_rec:.0f}", kd_rec_wash_b8=f"{100 * rc.loc[(1.0, 8, 'washout_1')].kd_rec:.0f}",
             kd_rec_sldrift_b8=f"{100 * rc.loc[(1.0, 8, 'shift_level_drift')].kd_rec:.0f}", kd_rec_washdrift_b8=f"{100 * rc.loc[(1.0, 8, 'washout_1_drift')].kd_rec:.0f}",
             kd_rec_sl_b8=f"{100 * rc.loc[(1.0, 8, 'shift_level')].kd_rec:.0f}")
pp = cp.groupby(["c", "train_block", "policy"])[["red", "shortage"]].mean()
tcp = []
for c in [0.0, 0.5, 1.0]:
    for bl in [2, 8]:
        tcp.append([f"{c:.1f}", str(bl)] + [f"{pp.loc[(c, bl, p)].red:.1f}" for p in ["silo", "best_fixed", "pwi"]] + [f(pp.loc[(c, bl, "pwi")].shortage)])
carry.update(pol_reps=int(cp.rep.nunique()), pwi_c0_b2=f"{pp.loc[(0.0, 2, 'pwi')].red:.1f}", pwi_c1_b2=f"{pp.loc[(1.0, 2, 'pwi')].red:.1f}", pwi_c1_b8=f"{pp.loc[(1.0, 8, 'pwi')].red:.1f}",
             bf_c1_b2=f"{pp.loc[(1.0, 2, 'best_fixed')].red:.1f}", silo_c1_b2=f"{pp.loc[(1.0, 2, 'silo')].red:.1f}", pwi_c05_b2=f"{pp.loc[(0.5, 2, 'pwi')].red:.1f}",
             bf_c0_b2=f"{pp.loc[(0.0, 2, 'best_fixed')].red:.1f}")
json.dump(dict(t3=t3, t4=t4, t4_full=t4_full, t5=t5, t6=t6, t6_full=t6_full, tco=tco, tcp=tcp, base=base, basepol=basepol, est=est, mcp=mcp, mc=mcs,
               oc=oc, pilot=pilot, tpil=tpil, carry=carry, sizechk=sizechk, variants=variants, h1=h1), open("results_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(dict(basepol=basepol, mc=mcs, carry=carry, pilot=pilot, oc=oc), indent=1, ensure_ascii=False)); print(json.dumps(est, ensure_ascii=False))
