"""Precise size check of the pilot randomisation test under the two weak nulls (24 weeks).
For each outer replication a process path is simulated once with the siloed policy in every
non-baseline block; because that path depends only on the baseline positions, the total-cost and
siloed labels can then be re-drawn many times without re-simulating. Symmetric weak null: N(0, 9^2)
effects added to total-cost block means. Skewed weak null: centred exponential effects with half
spilling into the first two shifts of the next block."""
import sys, numpy as np, pandas as pd
import pwi_carryover_v6 as co, pwi_study_v6 as p
def run(design, c, outer=100, inner=300, weeks=24, seed0=555001, het_sd=9.0):
    U, m, wash = co.DESIGNS[design]; n = weeks * 12; k = U // (3 * m); nu = n // U
    assert m == 1, "size_check_v6 handles designs with one block per arm per unit; use size_variants.py for the one-day design"
    blk = np.arange(n) // k; keep = (np.arange(n) % k) >= wash; rows = []
    for rep in range(outer):
        rng = np.random.default_rng(seed0 + 1000 * rep)
        I_ex = co.innovations(144, seed0 + 1000 * rep + 1, p.BASE)
        ex = co.simulate_dyn(I_ex, *co.switchback(144, seed0 + 1000 * rep + 2, block=4), p.BASE, c)
        ex = ex[(np.arange(144) % 4) != 0]; X = co.design_dyn(ex)
        betas = np.column_stack([np.linalg.lstsq(X.values, ex[y].values, rcond=None)[0] for y in co.TARGETS])
        I = co.innovations(n, seed0 + 1000 * rep + 3, p.BASE)
        base_pos = rng.integers(0, 3 * m, nu)                       # m = 1 designs only
        modes_block = np.ones(3 * nu, int); modes_block[np.arange(nu) * 3 + base_pos] = 0
        J = co.simulate_modes(I, np.repeat(modes_block, k), betas, p.BASE, c)[0]
        for j in range(inner):
            arm = modes_block.copy()
            for u in range(nu):
                idx = [b for b in range(3 * u, 3 * u + 3) if modes_block[b] == 1]
                arm[idx[rng.integers(0, 2)]] = 2
            unit = np.repeat(np.arange(nu), 3)
            Ysym = np.array([J[(blk == b) & keep].mean() for b in range(3 * nu)]) + (arm == 2) * rng.normal(0, het_sd, 3 * nu)
            Js = J.copy(); delta = rng.exponential(het_sd, 3 * nu) - het_sd
            for bb in np.where(arm == 2)[0]:
                Js[blk == bb] += delta[bb]; Js[np.where(blk == bb + 1)[0][:2]] += 0.5 * delta[bb]
            Yskew = np.array([Js[(blk == b) & keep].mean() for b in range(3 * nu)])
            ps, _ = co.randomisation_p(Ysym, arm, unit, nu, rng, mc_perms=2000)
            pk, _ = co.randomisation_p(Yskew, arm, unit, nu, rng, mc_perms=2000)
            rows.append(dict(design=design, c=c, rep=rep, p_sym=ps, p_skew=pk))
    return pd.DataFrame(rows)
if __name__ == "__main__":
    design, c, outer, inner = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    d = run(design, c, outer, inner)
    d.to_csv(f"results_v6/sizecheck_{design}_c{c:.2f}_w24.csv", index=False)
    n = len(d); se = np.sqrt(0.05 * 0.95 / n)
    print(design, c, n, "symmetric %.2f%%  skewed+spill %.2f%%  (MC s.e. about %.2f points, ignoring clustering)"
          % (100 * (d.p_sym < 0.05).mean(), 100 * (d.p_skew < 0.05).mean(), 100 * se))
