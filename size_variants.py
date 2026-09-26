"""Fixed-path size estimates: (a) one-day design (two blocks per arm per unit) under the symmetric weak
null, where the sign-flip argument does not apply; (b) harder weak-null variants for the two-day and
four-day designs: heavier-tailed (log-normal, skewness 6.2) effects, geometric spill over the whole next
block (persistence 0.7, i.e. the model's own WIP dynamics), and effect SD 18 instead of 9."""
import sys, numpy as np, pandas as pd
import pwi_carryover_v6 as co, pwi_study_v6 as p

def path(design, rep, weeks, c, seed0):
    U, m, wash = co.DESIGNS[design]; n = weeks*12; k = U//(3*m); nu = n//U
    rng = np.random.default_rng(seed0 + 1000*rep)
    I_ex = co.innovations(144, seed0+1000*rep+1, p.BASE)
    ex = co.simulate_dyn(I_ex, *co.switchback(144, seed0+1000*rep+2, block=4), p.BASE, c); ex = ex[(np.arange(144)%4)!=0]
    X = co.design_dyn(ex); betas = np.column_stack([np.linalg.lstsq(X.values, ex[y].values, rcond=None)[0] for y in co.TARGETS])
    I = co.innovations(n, seed0+1000*rep+3, p.BASE)
    # baseline positions: m of the 3m slots per unit
    modes_block = np.ones(3*m*nu, int)
    for u in range(nu):
        pos = rng.choice(3*m, m, replace=False); modes_block[3*m*u + pos] = 0
    J = co.simulate_modes(I, np.repeat(modes_block, k), betas, p.BASE, c)[0]
    return rng, J, modes_block, k, m, nu, wash

def draw_labels(rng, modes_block, m, nu):
    arm = modes_block.copy()
    for u in range(nu):
        idx = np.array([b for b in range(3*m*u, 3*m*u+3*m) if modes_block[b]==1])
        arm[rng.choice(idx, m, replace=False)] = 2
    return arm

def effects(rng, kind, nb, sd):
    if kind == "exp":  return rng.exponential(sd, nb) - sd
    if kind == "lognorm":                       # mean 0, SD sd, skewness (e+2)sqrt(e-1) = 6.18
        z = rng.normal(0, 1, nb); return sd*(np.exp(z) - np.exp(0.5))/np.sqrt(np.e*(np.e-1))
    raise ValueError

def run(design, variant, outer, inner, weeks=24, c=0.0, seed0=777001, mc=2000):
    kind, spill, sd = variant
    rows = []
    for rep in range(outer):
        rng, J, modes_block, k, m, nu, wash = path(design, rep, weeks, c, seed0)
        n = len(J); blk = np.arange(n)//k; keep = (np.arange(n)%k) >= wash; unit = np.repeat(np.arange(nu), 3*m)
        for j in range(inner):
            arm = draw_labels(rng, modes_block, m, nu)
            if kind == "sym":
                Y = np.array([J[(blk==b)&keep].mean() for b in range(3*m*nu)]) + (arm==2)*rng.normal(0, sd, 3*m*nu)
            else:
                Js = J.copy(); delta = effects(rng, kind, 3*m*nu, sd)
                for bb in np.where(arm==2)[0]:
                    Js[blk==bb] += delta[bb]
                    nxt = np.where(blk==bb+1)[0]
                    if spill == "two_half": Js[nxt[:2]] += 0.5*delta[bb]
                    elif spill == "geom":   Js[nxt] += delta[bb]*0.7**np.arange(1, len(nxt)+1)
                Y = np.array([Js[(blk==b)&keep].mean() for b in range(3*m*nu)])
            pval, _ = co.randomisation_p(Y, arm, unit, nu, rng, mc_perms=mc)
            rows.append(dict(design=design, variant=str(variant), rep=rep, p=pval))
    d = pd.DataFrame(rows); r = (d.p<0.05).groupby(d.rep).mean()
    d["weeks"] = weeks; d["c"] = c
    d.to_csv(f"results_v6/sizevariant_{design}_{variant[0]}_{variant[1]}_sd{int(variant[2])}_w{weeks}.csv", index=False)
    return 100*(d.p<0.05).mean(), 100*r.std(ddof=1)/np.sqrt(outer), len(d)

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "daily":
        for weeks, outer, inner in [(24, 30, 100), (12, 30, 100)]:
            rate, se, n = run("daily", ("sym", None, 9.0), outer, inner, weeks=weeks)
            print(f"one-day design, {weeks} weeks, symmetric weak null (not a sign-flip test): {rate:.2f}% (clustered s.e. {se:.2f}, {n} draws)")
    else:
        design = which
        outer, inner = (40, 200) if design == "four_day" else (40, 100)
        allv = [("exp","two_half",9.0), ("exp","two_half",18.0), ("lognorm","two_half",9.0), ("exp","geom",9.0), ("lognorm","geom",9.0)]
        pick = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(len(allv))
        for variant in [allv[i] for i in pick]:
            rate, se, n = run(design, variant, outer, inner)
            print(f"{design}: effects={variant[0]:8s} spill={variant[1]:8s} SD={variant[2]:4.0f} -> reject {rate:.2f}% (clustered s.e. {se:.2f}, {n} draws)", flush=True)
