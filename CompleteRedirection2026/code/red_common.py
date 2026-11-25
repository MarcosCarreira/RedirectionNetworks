def G0():
    return (np.zeros(1), np.zeros(2, np.int64), np.zeros(0), np.zeros(0), -np.ones(1), np.zeros((1, C.NGRID + 1), np.float32),
            np.zeros(1), np.zeros(1), np.zeros(2, np.int64), np.zeros(0), -np.ones(1), np.zeros((1, C.NGRID + 1), np.float32), np.zeros(1))
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
def el_sums(lws, births, tsv, tI, tdI, mu):
    w = np.exp(np.minimum(lws, 50.0)); b = np.where(births >= 0, births, np.inf); e = np.exp(-mu * b)
    fb = np.isfinite(b)
    s0 = e.sum(1); s1 = np.where(fb, b * e, 0).sum(1); s2 = np.where(fb, b * b * e, 0).sum(1)
    tsn = np.maximum(tsv, 0); et = np.where(tsv >= 0, np.exp(-mu * tsn), 0.0)
    s0 = s0 + et * tI; s1 = s1 + et * (tsn * tI - tdI); s2 = s2 + et * (tsn * tsn * tI - 2 * tsn * tdI)
    return np.array([w.sum(), (w * w).sum(), (w * s0).sum(), (w * s1).sum(), (w * s2).sum()])
def el_root(acc, mu):
    Sw, Sw2, A0, A1, A2 = acc
    aa, bb, cc = A2 / 2, -A1, A0 - Sw
    return mu + (-bb - np.sqrt(max(bb * bb - 4 * aa * cc, 0.0))) / (2 * aa)
def ratio(Wq, WCq, mu0, c_prev):
    tot = Wq.sum()
    cR = {(l, k): WCq[l, k] / Wq[l, k] for l in range(C.DSTAR) for k in range(1, C.DSTAR)
          if 2 <= l + k < C.DSTAR and Wq[l, k] > 1e-7 * tot}
    c, lam = nb.table_from(cR, mu0, cs.solve_q(mu0, c_prev))
    return nb.mu_full(c), cR, lam, c
