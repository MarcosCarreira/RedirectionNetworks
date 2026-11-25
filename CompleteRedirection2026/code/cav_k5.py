"""Cavity solver, settings run K5 (DistributionFormulas, 'Cavity solver: settings run K5, pre-registered', 20:49-20:50).
No arguments; settings below. Variants (B) base d* = 50, H = 12; (D) d* = 100; (H) H = 16; each in its own process
with its own numba cache. Output: printed and written to results/cav_k5_<date_time>.txt."""
import os, sys, time, datetime, json, subprocess
import numpy as np

VARIANTS = {"B": (50, 12.0, 300000), "D": (100, 12.0, 200000), "H": (50, 16.0, 300000)}   # (d*, H, records)
NBATCH = 4                 # output batches per generation, each the size of the record pool
NGEN = 5                   # generations after generation 0
VALGEN = (2, 3, 4, 5)      # generations in each variant's value
REPLICAS = 2
KEYMAX, MAXC, TCAP = 11, 64, 24.0
MU_REF = 0.565723
SEED0 = 20261125
HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "results")


def child(variant, logfile, sumfile):
    import cavity as C, nb_closure as nb, credit_sim as cs
    dstar, H, NREC = VARIANTS[variant]
    assert C.DSTAR == dstar and C.H == H
    z = np.load(os.path.join(RD, C.TABLES)); aw = float(z['AGE_W']); DR = int(z['D_R'])
    def log(s):
        print(s, flush=True)
        with open(logfile, "a") as f:
            f.write(s + "\n")

    def resample_par(rng, pool, logw):
        (d0, off, gt, gr, ts, grid, W, sh, rdoff, rdgt, rdts, rdgrid, rdW) = pool
        w = np.exp(logw - logw.max()); w /= w.sum(); ess = 1.0 / np.sum(w * w) / w.size
        pick = rng.choice(w.size, w.size, p=w)
        def csr(o, a):
            n = np.diff(o)[pick]; no = np.concatenate([[0], np.cumsum(n)])
            return no, a[np.repeat(o[pick] - no[:-1], n) + np.arange(n.sum())]
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

    out = {}
    for rep in range(REPLICAS):
        rng = np.random.default_rng(SEED0 + 1000 * rep); sd = SEED0 + 1000 * rep
        mu = float(z['mu']); c_prev = z['c']
        cum, sl, sk = nb.parent_law(z['q'], mu)
        t0 = time.time()
        chp = C.gen0_children(NREC, sd + 1, cum, sl, sk, z['ctabA'], z['mtab'], z['c'], aw, DR, C.H, NREC * C.DSTAR)[:7]
        order, keyoff = C.key_index(chp[0], KEYMAX)
        G0 = (np.zeros(1), np.zeros(2, np.int64), np.zeros(0), np.zeros(0), -np.ones(1), np.zeros((1, C.NGRID + 1), np.float32),
              np.zeros(1), np.zeros(1), np.zeros(2, np.int64), np.zeros(0), -np.ones(1), np.zeros((1, C.NGRID + 1), np.float32), np.zeros(1))
        pp, lwp = C.gen_parents(NREC, sd + 2, mu, C.H, TCAP, MAXC, 0, cum, sl, sk, *G0, *chp, order, keyoff, KEYMAX)
        pp, essp = resample_par(rng, pp, lwp)
        log(f"[{variant}] replica {rep}, generation 0 ({time.time()-t0:.0f}s): parent ESS {essp:.2f}")
        rows = []
        for gen in range(1, NGEN + 1):
            t1 = time.time()
            cum, sl, sk = nb.parent_law(cs.solve_q(mu, c_prev), mu)
            res = C.gen_children(NREC, sd + 10 * gen + 3, cum, sl, sk, C.H, MAXC, *chp, order, keyoff, KEYMAX)
            chp, essc = resample_ch(rng, res[:7], res[7])
            order, keyoff = C.key_index(chp[0], KEYMAX)
            pp2, lwp = C.gen_parents(NREC, sd + 10 * gen + 4, mu, C.H, TCAP, MAXC, 1, cum, sl, sk, *pp, *chp, order, keyoff, KEYMAX)
            pp, essp = resample_par(rng, pp2, lwp)
            Wq = 0.0; WCq = 0.0; Sw = Sw2 = A0 = A1 = A2 = 0.0
            for bt in range(NBATCH):
                lws, births, tsv, tI, tdI, Wb, WCb = C.output(NREC, sd + 10 * gen + 100 * bt + 5, mu, C.H, TCAP, MAXC, *pp, *chp, order, keyoff, KEYMAX)
                Wq = Wq + Wb; WCq = WCq + WCb
                w = np.exp(np.minimum(lws, 50.0))
                b = np.where(births >= 0, births, np.inf)
                e = np.exp(-mu * b)
                bf_ = np.where(np.isfinite(b), b, 0.0)   # empty birth slots: e = 0 there (fixed 25 Nov 06:35: no inf * 0)
                s0 = e.sum(1); s1 = (bf_ * e).sum(1); s2 = (bf_ * bf_ * e).sum(1)
                tl = tsv >= 0; tsn = np.maximum(tsv, 0); et = np.where(tl, np.exp(-mu * tsn), 0.0)
                s0 = s0 + et * tI
                s1 = s1 + et * (tsn * tI - tdI)           # minus the first derivative in mu
                s2 = s2 + et * (tsn * tsn * tI - 2 * tsn * tdI)
                Sw += w.sum(); Sw2 += (w * w).sum(); A0 += (w * s0).sum(); A1 += (w * s1).sum(); A2 += (w * s2).sum()
            # Euler-Lotka: A0 - x A1 + x^2/2 A2 = Sw, x = m - mu
            aa, bb, cc = A2 / 2, -A1, A0 - Sw
            x = (-bb - np.sqrt(max(bb * bb - 4 * aa * cc, 0.0))) / (2 * aa) if aa != 0 else -cc / bb
            mE = mu + x; essv = Sw * Sw / Sw2 / (NBATCH * NREC)
            tot = Wq.sum()
            cR = {(l, k): WCq[l, k] / Wq[l, k] for l in range(C.DSTAR) for k in range(1, C.DSTAR)
                  if 2 <= l + k < C.DSTAR and Wq[l, k] > 1e-7 * tot}
            c_new, lam = nb.table_from(cR, mu, cs.solve_q(mu, c_prev))
            mR = nb.mu_full(c_new)
            essmin = min(min(essc.values()), essp, essv)
            log(f"[{variant}] replica {rep}, generation {gen} ({time.time()-t1:.0f}s): ESS children min {min(essc.values()):.2f}, "
                f"parents {essp:.2f}, output {essv:.2f}; mu (E) {mE:.6f}, (R) {mR:.6f}; c(1,1) {cR[(1,1)]:.4f}, c(0,2) {cR[(0,2)]:.4f}, "
                f"c(1,2) {cR[(1,2)]:.4f}, c(0,3) {cR[(0,3)]:.4f}; lambda above the tallied states {lam:.4f}")
            rows.append([gen, mE, mR, essmin])
            mu = mR; c_prev = c_new
        out[rep] = rows
    with open(sumfile, "w") as f:
        json.dump(out, f)


def value(rows_by_rep, col):
    vals = np.array([[r[col] for r in rows if r[0] in VALGEN] for rows in rows_by_rep.values()])
    v = vals.mean()
    se = max(vals.std(ddof=1) / np.sqrt(vals.size), vals.mean(1).std(ddof=1) / np.sqrt(vals.shape[0]))
    g = np.array(VALGEN, float); m = vals.mean(0)
    A = np.vstack([g, np.ones_like(g)]).T
    coef, res, *_ = np.linalg.lstsq(A, m, rcond=None)
    resid = m - A @ coef; s2 = resid @ resid / max(len(g) - 2, 1)
    se_slope = np.sqrt(s2 / np.sum((g - g.mean()) ** 2))
    return v, se, coef[0], se_slope


if __name__ == "__main__":
    if len(sys.argv) == 4:
        child(sys.argv[1], sys.argv[2], sys.argv[3]); sys.exit(0)
    os.makedirs(RD, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    LOG = os.path.join(RD, f"cav_k5_{stamp}.txt")
    def log(s):
        print(s, flush=True)
        with open(LOG, "a") as f:
            f.write(s + "\n")
    log(f"=== cav_k5 started {datetime.datetime.now().isoformat(timespec='seconds')} VARIANTS={VARIANTS} NBATCH={NBATCH} NGEN={NGEN} "
        f"VALGEN={VALGEN} REPLICAS={REPLICAS} ===")
    log("Pre-registered (DistributionFormulas, 'Cavity solver: settings run K5, pre-registered'): K2', K3, K4, K5 below.")
    T0 = time.time(); summ = {}
    for vname, (dstar, H, nrec) in VARIANTS.items():
        env = dict(os.environ, CAV_DSTAR=str(dstar), CAV_H=str(H), NUMBA_CACHE_DIR=os.path.join(HERE, f"numba_cache_k5_{vname}"))
        sumfile = os.path.join(RD, f"cav_k5_{stamp}_{vname}.json")
        subprocess.run([sys.executable, os.path.abspath(__file__), vname, LOG, sumfile], env=env, check=True)
        with open(sumfile) as f:
            summ[vname] = json.load(f)
    res = {}
    for vname in VARIANTS:
        for col, est in ((1, "E"), (2, "R")):
            v, se, sl_, ses = value(summ[vname], col)
            res[(vname, est)] = (v, se)
            log(f"[{vname}] ({est}) value {v:.6f}, s.e. {se:.6f}; slope over generations {VALGEN} {sl_:+.2e} (s.e. {ses:.2e}) -> "
                f"PRE-REGISTERED K2': {'PASS' if abs(sl_) <= 2 * ses else 'FAIL'}")
    essmin = min(r[3] for s in summ.values() for rows in s.values() for r in rows)
    log(f"PRE-REGISTERED K3: minimum effective sample size {essmin:.2f} -> {'PASS' if essmin >= 0.10 else 'FAIL (not usable)'}")
    for est in ("E", "R"):
        v, se = res[("B", est)]; zz = (v - MU_REF) / se
        log(f"PRE-REGISTERED K4 [B] ({est}): {v:.6f} - {MU_REF} = {v-MU_REF:+.6f}, z = {zz:+.2f} -> {'agreement' if abs(zz) <= 2 else 'no agreement'}")
    (vE, sE), (vR, sR) = res[("B", "E")], res[("B", "R")]
    log(f"PRE-REGISTERED K4 [B] (E against R): {vE-vR:+.6f}, z = {(vE-vR)/np.hypot(sE, sR):+.2f} -> {'agree' if abs(vE-vR) <= 2*np.hypot(sE, sR) else 'do not agree'}")
    for vname in ("D", "H"):
        for est in ("E", "R"):
            (v, se), (vb, seb) = res[(vname, est)], res[("B", est)]
            lim = 2 * np.hypot(se, seb)
            log(f"PRE-REGISTERED K5 [{vname} vs B] ({est}): {v-vb:+.6f} against {lim:.6f} -> {'PASS' if abs(v-vb) <= lim else 'FAIL'}")
    log(f"done ({time.time()-T0:.0f}s)")
