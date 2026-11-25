"""Cavity solver, production run (DistributionFormulas, 'Cavity solver: production run, pre-registered', 24 Nov 20:26).
No arguments; settings below. Output: printed and written to results/cav_run_<date_time>.txt.
Generation 0: child records with the closure's tables (run U), parent records with a deterministic grandparent.
Generations 1..NGEN: child records from the previous ones; parent records from the previous ones and the new child
records; output vertices; mu by Euler-Lotka (E) and by the ratio equations (R); mu in the designation weights updated
to (R) after each generation. REPLICAS independent replicas. Single process."""
import os, time, datetime
import numpy as np
import cavity as C, nb_closure as nb, credit_sim as cs

NREC = 300000          # child records and parent records per generation
NV = 300000            # output vertices per generation
NGEN = 8               # generations after generation 0
LAST = (6, 7, 8)       # generations averaged for each replica's value
REPLICAS = 4
KEYMAX, MAXC, TCAP = 11, 64, 24.0
MU_REF = 0.565723
SEED0 = 20261124

rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(rd, exist_ok=True)
OUT = os.path.join(rd, f"cav_run_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}.txt")
def log(s):
    print(s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")

z = np.load(os.path.join(rd, C.TABLES)) if os.path.exists(os.path.join(rd, C.TABLES)) else np.load(C.TABLES)
aw = float(z['AGE_W']); DR = int(z['D_R'])

def resample_par(rng, pool, logw):
    (d0, off, gt, gr, ts, grid, W, sh, rdoff, rdgt, rdts, rdgrid, rdW) = pool
    w = np.exp(logw - logw.max()); w /= w.sum(); ess = 1.0 / np.sum(w * w) / w.size
    pick = rng.choice(w.size, w.size, p=w)
    def csr(o, a):
        n = np.diff(o)[pick]; no = np.concatenate([[0], np.cumsum(n)])
        starts = o[pick]; idx = np.repeat(starts - no[:-1], n) + np.arange(n.sum())
        return no, a[idx]
    off2, gt2 = csr(off, gt); _, gr2 = csr(off, gr); rdoff2, rdgt2 = csr(rdoff, rdgt)
    return (d0[pick], off2, gt2, gr2, ts[pick], grid[pick], W[pick], sh[pick], rdoff2, rdgt2, rdts[pick], rdgrid[pick], rdW[pick]), ess

def resample_ch(rng, recs, logw):
    d0r, off, gt, gr, tstar, grid, W = recs
    keys = np.minimum(d0r, KEYMAX); pick = np.empty(d0r.size, np.int64); ess = {}
    for kk in np.unique(keys):
        idx = np.nonzero(keys == kk)[0]
        w = np.exp(logw[idx] - logw[idx].max()); w /= w.sum()
        ess[int(kk)] = 1.0 / np.sum(w * w) / idx.size
        pick[idx] = idx[rng.choice(idx.size, idx.size, p=w)]
    n = np.diff(off)[pick]; no = np.concatenate([[0], np.cumsum(n)])
    sel = np.repeat(off[pick] - no[:-1], n) + np.arange(n.sum())
    return (d0r[pick], no, gt[sel], gr[sel], tstar[pick], grid[pick], W[pick]), ess

def mu_el(lws, births, tsv, tI, tdI, mu_used):
    w = np.exp(lws - lws.max()); w /= w.sum()
    b = np.where(births >= 0, births, np.inf)
    def f(m):
        s = np.exp(-m * b).sum(1)
        s += np.where(tsv >= 0, np.exp(-m * np.maximum(tsv, 0)) * (tI + tdI * (m - mu_used)), 0.0)
        return (w * s).sum() - 1.0
    lo, hi = 0.40, 0.80
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi), 1.0 / np.sum(w * w) / w.size

def mu_ratio(Wq, WCq, mu0, c_prev):
    tot = Wq.sum()
    cR = {(l, k): WCq[l, k] / Wq[l, k] for l in range(C.DSTAR) for k in range(1, C.DSTAR)
          if 2 <= l + k < C.DSTAR and Wq[l, k] > 1e-7 * tot}
    q = cs.solve_q(mu0, c_prev)
    c, lam = nb.table_from(cR, mu0, q)
    return nb.mu_full(c), cR, lam, c

def replica(rep):
    rng = np.random.default_rng(SEED0 + 1000 * rep)
    sd = SEED0 + 1000 * rep
    mu = float(z['mu']); c_prev = z['c']
    cum, sl, sk = nb.parent_law(z['q'], mu)
    t0 = time.time()
    chp = C.gen0_children(NREC, sd + 1, cum, sl, sk, z['ctabA'], z['mtab'], z['c'], aw, DR, C.H, NREC * C.DSTAR)[:7]
    order, keyoff = C.key_index(chp[0], KEYMAX)
    G0 = (np.zeros(1), np.zeros(2, np.int64), np.zeros(0), np.zeros(0), -np.ones(1), np.zeros((1, C.NGRID + 1), np.float32),
          np.zeros(1), np.zeros(1), np.zeros(2, np.int64), np.zeros(0), -np.ones(1), np.zeros((1, C.NGRID + 1), np.float32), np.zeros(1))
    pp, lwp = C.gen_parents(NREC, sd + 2, mu, C.H, TCAP, MAXC, 0, cum, sl, sk, *G0, *chp, order, keyoff, KEYMAX)
    pp, essp = resample_par(rng, pp, lwp)
    log(f"replica {rep}, generation 0 ({time.time()-t0:.0f}s): parent ESS {essp:.2f}")
    rows = []
    for gen in range(1, NGEN + 1):
        t1 = time.time()
        cum, sl, sk = nb.parent_law(cs.solve_q(mu, c_prev), mu)
        res = C.gen_children(NREC, sd + 10 * gen + 3, cum, sl, sk, C.H, MAXC, *chp, order, keyoff, KEYMAX)
        chp, essc = resample_ch(rng, res[:7], res[7])
        order, keyoff = C.key_index(chp[0], KEYMAX)
        pp2, lwp = C.gen_parents(NREC, sd + 10 * gen + 4, mu, C.H, TCAP, MAXC, 1, cum, sl, sk, *pp, *chp, order, keyoff, KEYMAX)
        pp, essp = resample_par(rng, pp2, lwp)
        lws, births, tsv, tI, tdI, Wq, WCq = C.output(NV, sd + 10 * gen + 5, mu, C.H, TCAP, MAXC, *pp, *chp, order, keyoff, KEYMAX)
        mE, essv = mu_el(lws, births, tsv, tI, tdI, mu)
        mR, cR, lam, c_new = mu_ratio(Wq, WCq, mu, c_prev)
        essmin = min(min(essc.values()), essp, essv)
        log(f"replica {rep}, generation {gen} ({time.time()-t1:.0f}s): ESS children min {min(essc.values()):.2f}, parents {essp:.2f}, "
            f"output {essv:.2f}; mu (E) {mE:.6f}, (R) {mR:.6f}; c(1,1) {cR[(1,1)]:.4f}, c(2,1) {cR[(2,1)]:.4f}, c(0,2) {cR[(0,2)]:.4f}, "
            f"c(1,2) {cR[(1,2)]:.4f}, c(0,3) {cR[(0,3)]:.4f}; lambda above degree 49 {lam:.4f}")
        rows.append((gen, mE, mR, essmin, cR[(1,1)], cR[(0,2)], cR[(1,2)], cR[(0,3)]))
        mu = mR; c_prev = c_new
    return rows

if __name__ == "__main__":
    log(f"=== cav_run started {datetime.datetime.now().isoformat(timespec='seconds')} NREC={NREC} NV={NV} NGEN={NGEN} LAST={LAST} "
        f"REPLICAS={REPLICAS} DSTAR={C.DSTAR} H={C.H} TCAP={TCAP} KEYMAX={KEYMAX} tables {C.TABLES} ===")
    log("Pre-registered (DistributionFormulas, 'Cavity solver: production run, pre-registered', 20:26 BRT): K2, K3, K4 below; K5 in a later run.")
    T0 = time.time(); allrows = []
    for rep in range(REPLICAS):
        allrows.append(replica(rep))
    E = np.array([[r[1] for r in rows if r[0] in LAST] for rows in allrows])
    R = np.array([[r[2] for r in rows if r[0] in LAST] for rows in allrows])
    essmin = min(r[3] for rows in allrows for r in rows)
    eR, rR = E.mean(1), R.mean(1)
    for name, v, M in (("E", eR, E), ("R", rR, R)):
        se = v.std(ddof=1) / np.sqrt(v.size)
        k2 = all(abs(M[i, j] - v[i]) < 2 * v.std(ddof=1) for i in range(M.shape[0]) for j in range(M.shape[1]))
        zz = (v.mean() - MU_REF) / se
        log(f"({name}) replica values " + ", ".join(f"{x:.6f}" for x in v) + f"; mean {v.mean():.6f}, s.e. {se:.6f}")
        log(f"PRE-REGISTERED K2 ({name}): generations {LAST} within 2 replica s.d. of each replica's mean -> {'PASS' if k2 else 'FAIL'}")
        log(f"PRE-REGISTERED K4 ({name}): mu - {MU_REF} = {v.mean()-MU_REF:+.6f}, z = {zz:+.2f} -> {'agreement' if abs(zz) <= 2 else 'no agreement'}")
    seE = eR.std(ddof=1) / 2; seR = rR.std(ddof=1) / 2
    zER = (eR.mean() - rR.mean()) / np.sqrt(seE**2 + seR**2)
    log(f"PRE-REGISTERED K4 (E against R): difference {eR.mean()-rR.mean():+.6f}, z = {zER:+.2f} -> {'agree' if abs(zER) <= 2 else 'do not agree'}")
    log(f"PRE-REGISTERED K3: minimum effective sample size {essmin:.2f} -> {'PASS' if essmin >= 0.10 else 'FAIL (run not usable)'}")
    log(f"done ({time.time()-T0:.0f}s)")
