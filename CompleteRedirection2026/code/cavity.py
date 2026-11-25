"""Cavity solver on degree paths (DistributionFormulas, 'Cavity solver: design and pre-registered checks', 24 Nov 18:29).
Chunk 1: path helpers and generation 0 of the child records (a child's own children credited from the closure's tables)."""
import os
import numpy as np
from numba import njit

H = float(os.environ.get('CAV_H', 12.0))        # horizon in age (CAV_H: K5 variant; each variant has its own numba cache)
DSTAR = int(os.environ.get('CAV_DSTAR', 50))   # degree from which the path follows the large-vertex model (note, 19:58)
DGRID, NGRID = 0.05, 80   # grid of the model path above DSTAR (s up to 4)
LAM_BIG = float(os.environ.get('CAV_LAM', 0.230))   # credit per non-leaf neighbour in the band 51-200 (run U): drift u' = 1 - LAM_BIG (CAV_LAM: run P)
TABLES = "nb_closure_2026-11-24_150442_tables.npz"


@njit(cache=True)
def deg_at(d0, gt, i0, n, tdet, ddet, t):
    """Degree at time t of a path that starts at degree d0 at time 0, gains at times gt[i0:i0+n] (increasing),
    and grows as ddet * exp(t - tdet) from tdet on (tdet < 0: never)."""
    if tdet >= 0.0 and t >= tdet:
        return ddet * np.exp(t - tdet)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if gt[i0 + mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return float(d0 + lo)


@njit(cache=True)
def inv_int(d0, gt, i0, n, tdet, ddet, T):
    """Integral of 1/degree over [0, T] for such a path."""
    s = 0.0; t = 0.0; d = float(d0)
    tend = T if (tdet < 0.0 or T < tdet) else tdet
    for j in range(n):
        tj = gt[i0 + j]
        if tj >= tend:
            break
        s += (tj - t) / d
        t = tj; d += 1.0
    s += (tend - t) / d
    if tdet >= 0.0 and T > tdet:
        s += (1.0 - np.exp(-(T - tdet))) / ddet
    return s


@njit(cache=True)
def log_ratio(gt, gr, i0, n, T,
              a_d0, a_gt, a_i0, a_n, a_tdet, a_ddet,
              b_d0, b_gt, b_i0, b_n, b_tdet, b_ddet):
    """log of the density ratio of a record (gain times gt, rates without the parent gr) under parent path A versus
    parent path B, over [0, T] (proposition 'Reweighting a child record')."""
    s = 0.0
    for j in range(n):
        tj = gt[i0 + j]
        if tj > T:
            break
        ia = 1.0 / deg_at(a_d0, a_gt, a_i0, a_n, a_tdet, a_ddet, tj)
        ib = 1.0 / deg_at(b_d0, b_gt, b_i0, b_n, b_tdet, b_ddet, tj)
        s += np.log((gr[i0 + j] + ia) / (gr[i0 + j] + ib))
    s -= inv_int(a_d0, a_gt, a_i0, a_n, a_tdet, a_ddet, T) - inv_int(b_d0, b_gt, b_i0, b_n, b_tdet, b_ddet, T)
    return s


@njit(cache=True)
def table_children(l, k, a, ctabA, mtab, ctab, aw, DR):
    """Generation 0: credit a vertex receives from its own children, from the closure's tables (as in age_pair2.py):
    (k-1) times the per-child credit by state and age on R; outside R the state credit of the table."""
    if l + k <= DR:
        b = int(a / aw)
        if b >= mtab.shape[0]:
            b = mtab.shape[0] - 1
        return (k - 1) * mtab[b, l, k]
    kk = k if k < ctab.shape[0] else ctab.shape[0] - 1
    ll = l if l < ctab.shape[1] else ctab.shape[1] - 1
    return ctab[kk, ll]


@njit(cache=True)
def tail_int(kst, ust, mu, lam):
    """Integral over s >= 0 of exp(-mu s) (1 - k(s)/d(s)): births of a vertex above DSTAR at rate l/d = 1 - k/d along
    its mean path, d(s) = (DSTAR - u*) e^s + u* + u' s - u'(e^s - 1), k(s) = k* + 0.97 s (note, 'Large vertices').
    Returns the integral and its derivative in mu."""
    up = 1.0 - lam
    I = 0.0; dI = 0.0; h = 0.02
    for q in range(1500):
        s_ = (q + 0.5) * h
        es = np.exp(s_)
        d = (DSTAR - ust) * es + ust + up * s_ - up * (es - 1.0)
        r = 1.0 - (kst + 0.97 * s_) / d
        if r < 0.0:
            r = 0.0
        e = np.exp(-mu * s_) * r * h
        I += e; dI -= s_ * e
    return I, dI


@njit(cache=True)
def tail_sample(kst, ust, mu, lam):
    """A birth time s above DSTAR with density proportional to exp(-mu s)(1 - k(s)/d(s)) (rejection from Exp(mu))."""
    up = 1.0 - lam
    while True:
        s_ = -np.log(1.0 - np.random.random()) / mu
        es = np.exp(s_)
        d = (DSTAR - ust) * es + ust + up * s_ - up * (es - 1.0)
        if np.random.random() < 1.0 - (kst + 0.97 * s_) / d:
            return s_


@njit(cache=True)
def model_path(z0, ustar, lam, grid, row):
    """Large-vertex path above DSTAR (note, P2): Z = d - u with Yule increments in normal form and drift u' = 1 - lam;
    writes d at s = DGRID, 2 DGRID, ... into grid[row, 1:], grid[row, 0] = DSTAR; returns W = d(s_end) e^{-s_end}."""
    up = 1.0 - lam
    Z = z0; e = np.exp(DGRID)
    grid[row, 0] = z0 + ustar
    for j in range(1, NGRID + 1):
        Z = Z * e - up * (e - 1.0) + np.sqrt(max(Z * e * (e - 1.0), 0.0)) * np.random.standard_normal()
        if Z < 1.0:
            Z = 1.0
        grid[row, j] = Z + ustar + up * j * DGRID
    return grid[row, NGRID] * np.exp(-NGRID * DGRID)


@njit(cache=True)
def deg_rec(d0, gt, i0, n, tstar, grid, row, W, t):
    """Degree at time t (from the vertex's birth) of a record: jumps until tstar, model grid above, W e^s beyond."""
    if tstar < 0.0 or t < tstar:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if gt[i0 + mid] <= t:
                lo = mid + 1
            else:
                hi = mid
        return float(d0 + lo)
    s = t - tstar
    j = int(s / DGRID)
    if j >= NGRID:
        return W * np.exp(s)
    f = (s - j * DGRID) / DGRID
    return grid[row, j] * (grid[row, j + 1] / grid[row, j]) ** f


@njit(cache=True)
def gen0_children(nrec, seed, cum, sl, sk, ctabA, mtab, ctab, aw, DR, H, cap):
    """Generation 0 of the child records (amended 19:58): reference parent path d0 exp(t) (d0 from the parent law with
    the tail); child gain rate l + 1/D(t) + children's credit from the tables, birth l/d; jumps until degree DSTAR,
    then the large-vertex model path. Stored: key and reference d0, gain times and rates without the parent (CSR),
    tstar, model grid, W."""
    np.random.seed(seed)
    d0r = np.zeros(nrec, np.int64); off = np.zeros(nrec + 1, np.int64)
    gt = np.zeros(cap); gr = np.zeros(cap)
    tstar = -np.ones(nrec); grid = np.zeros((nrec, NGRID + 1), np.float32); W = np.zeros(nrec); tO = -np.ones(nrec)
    pos = 0
    for r in range(nrec):
        x = np.random.random() * cum[-1]
        lo, hi = 0, cum.shape[0] - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        d0 = sl[lo] + sk[lo]
        d0r[r] = d0
        l = 1; k = 1; t = 0.0
        tout = -1.0
        while t < H:
            d = l + k
            # outside R the table's state credit already contains the parent's part (as in age_pair2.py): no 1/D
            # there, and the record does not depend on the parent path from the time tout it leaves R (fixed 20:05)
            if d > DR and tout < 0.0:
                tout = t
            inv_p = 1.0 / (d0 * np.exp(t)) if d <= DR else 0.0
            ch = table_children(l, k, t, ctabA, mtab, ctab, aw, DR)
            if d >= DSTAR:
                tstar[r] = t
                W[r] = model_path(float(d) - (k - ch - inv_p), k - ch - inv_p, LAM_BIG, grid, r)
                break
            bound = l + inv_p + ch + l / d
            dt = -np.log(1.0 - np.random.random()) / bound
            if d <= DR:
                edge = (int(t / aw) + 1) * aw
                if t + dt >= edge:
                    t = edge
                    continue
            t += dt
            if t >= H:
                break
            inv_p1 = 1.0 / (d0 * np.exp(t)) if d <= DR else 0.0
            u = np.random.random() * bound
            g = l + inv_p1 + ch
            if u < g:
                gt[pos] = t; gr[pos] = l + ch; pos += 1
                l += 1
            elif u < g + l / d:
                l -= 1; k += 1
        off[r + 1] = pos
        tO[r] = tout
    return d0r, off, gt[:pos], gr[:pos], tstar, grid, W, tO


@njit(cache=True)
def v_inv_int(dv_b, vgt, ng, b, T):
    """Integral over [0, T] of 1/degree of the output vertex v from its child's birth time b: degree dv_b at b, then +1
    at each of v's gain times (absolute, vgt[:ng]) after b."""
    s = 0.0; t = b; d = float(dv_b)
    for j in range(ng):
        tj = vgt[j]
        if tj <= b:
            continue
        if tj >= b + T:
            break
        s += (tj - t) / d
        t = tj; d += 1.0
    s += (b + T - t) / d
    return s


@njit(cache=True)
def v_deg(dv_b, vgt, ng, b, t_abs):
    d = dv_b
    for j in range(ng):
        if vgt[j] <= b:
            continue
        if vgt[j] <= t_abs:
            d += 1
        else:
            break
    return float(d)


@njit(cache=True)
def k1_output(nv, seed, mu, ctabA, mtab, ctab, aw, DR, H, dend,
              d0r, off, gt, gr, tstar, grid, W, tout, keyidx, keyoff, keymax):
    """Check K1: output vertex v from creation; parent part of its credit from the closure's tables
    (ctabA - (k-1) mtab by v's state and age); its children are generation-0 records drawn at each birth among those
    whose reference parent degree equals v's degree then (pooled from keymax up), each reweighted to v's actual path
    over the time it influences v (until v reaches degree dend or age H, and at most the child's tstar).
    Returns weighted occupation W2[l,k] and credit WC2[l,k] (weight exp(-mu a)), and the weights' sum and sum of squares."""
    np.random.seed(seed)
    Wq = np.zeros((DR + 1, DR + 1)); WCq = np.zeros((DR + 1, DR + 1))
    sw = 0.0; sw2 = 0.0
    MAXC = 64; MAXG = 64
    crec = np.zeros(MAXC, np.int64); cb = np.zeros(MAXC); cdb = np.zeros(MAXC, np.int64)
    vgt = np.zeros(MAXG)
    locW = np.zeros((DR + 1, DR + 1)); locC = np.zeros((DR + 1, DR + 1))
    for iv in range(nv):
        l = 1; k = 1; t = 0.0; nc = 0; ng = 0
        locW[:, :] = 0.0; locC[:, :] = 0.0
        while t < H:
            d = l + k
            if d >= dend:
                break
            b_ = int(t / aw)
            if b_ >= ctabA.shape[0]:
                b_ = ctabA.shape[0] - 1
            if d <= DR:
                par = ctabA[b_, l, k] - (k - 1) * mtab[b_, l, k]
                if par < 0.0:
                    par = 0.0
            else:
                par = 0.0
            chc = 0.0
            for i in range(nc):
                r = crec[i]
                chc += 1.0 / deg_rec(2, gt, off[r], off[r + 1] - off[r], tstar[r], grid, r, W[r], t - cb[i])
            bound = l + par + chc + l / d
            dt = -np.log(1.0 - np.random.random()) / bound
            edge = (b_ + 1) * aw
            tn = t + dt
            hit_edge = tn >= edge
            if hit_edge:
                tn = edge
            if tn > H:
                tn = H
            if d <= DR:   # tallies on [t, tn): the children's 1/d changes within it; sub-steps of at most 0.02
                # (fixed 20:06: a single midpoint value underestimated the mean of 1/d over the long intervals of
                # leafless states)
                nsub = int((tn - t) / 0.02) + 1
                hsub = (tn - t) / nsub
                for q in range(nsub):
                    s0 = t + q * hsub; s1 = s0 + hsub; tm = 0.5 * (s0 + s1)
                    chm = 0.0
                    for i in range(nc):
                        r = crec[i]
                        chm += 1.0 / deg_rec(2, gt, off[r], off[r + 1] - off[r], tstar[r], grid, r, W[r], tm - cb[i])
                    wo = (np.exp(-mu * s0) - np.exp(-mu * s1)) / mu
                    locW[l, k] += wo; locC[l, k] += wo * (par + chm)
            if hit_edge or tn >= H:
                t = tn
                continue
            t = tn
            ch1 = 0.0
            for i in range(nc):
                r = crec[i]
                ch1 += 1.0 / deg_rec(2, gt, off[r], off[r + 1] - off[r], tstar[r], grid, r, W[r], t - cb[i])
            u = np.random.random() * bound
            g = l + par + ch1
            if u < g:
                if ng < MAXG:
                    vgt[ng] = t; ng += 1
                l += 1
            elif u < g + l / d:
                # a birth: draw a record with reference parent degree = d (pooled from keymax)
                kk = d if d < keymax else keymax
                n_k = keyoff[kk + 1] - keyoff[kk]
                if nc < MAXC and n_k > 0:
                    crec[nc] = keyidx[keyoff[kk] + int(np.random.random() * n_k)]
                    cb[nc] = t; cdb[nc] = d; nc += 1
                l -= 1; k += 1
        tend = t
        # weights of the children: v's actual path against the record's reference path d0 e^s
        lw = 0.0
        for i in range(nc):
            r = crec[i]
            T = tend - cb[i]
            if tstar[r] >= 0.0 and tstar[r] < T:
                T = tstar[r]
            if tout[r] >= 0.0 and tout[r] < T:
                T = tout[r]
            if T <= 0.0:
                continue
            i0 = off[r]; n = off[r + 1] - off[r]
            dref = float(d0r[r])
            for j in range(n):
                tj = gt[i0 + j]
                if tj > T:
                    break
                ia = 1.0 / v_deg(cdb[i], vgt, ng, cb[i], cb[i] + tj)
                ib = 1.0 / (dref * np.exp(tj))
                lw += np.log((gr[i0 + j] + ia) / (gr[i0 + j] + ib))
            lw -= v_inv_int(cdb[i], vgt, ng, cb[i], T) - (1.0 - np.exp(-T)) / dref
        w = np.exp(lw)
        sw += w; sw2 += w * w
        Wq += w * locW; WCq += w * locC
    return Wq, WCq, sw, sw2


@njit(cache=True)
def gen_children(nrec, seed, cum, sl, sk, H, MAXC,
                 p_d0, p_off, p_gt, p_gr, p_ts, p_grid, p_W, keyidx, keyoff, keymax):
    """Generation n+1 of the child records from generation n (pool p_*): a child c from birth under the reference
    parent path d0 exp(t) (d0 from the parent law with the tail); gain rate l + 1/D(t) + sum of its children's 1/d,
    birth l/d; each child g of c is a generation-n record drawn among those whose reference d0 equals c's degree at
    the birth (pooled from keymax), reweighted to c's actual path over the time it influences c (until c reaches
    DSTAR or H, and at most g's tstar). From DSTAR on, c follows the large-vertex model. Returns the new records
    and their log-weights (sum over c's children)."""
    np.random.seed(seed)
    d0r = np.zeros(nrec, np.int64); off = np.zeros(nrec + 1, np.int64)
    cap = nrec * DSTAR
    gt = np.zeros(cap); gr = np.zeros(cap)
    tstar = -np.ones(nrec); grid = np.zeros((nrec, NGRID + 1), np.float32); W = np.zeros(nrec); logw = np.zeros(nrec)
    crec = np.zeros(MAXC, np.int64); cb = np.zeros(MAXC); cdb = np.zeros(MAXC, np.int64)
    vgt = np.zeros(DSTAR + 2)
    pos = 0
    for r in range(nrec):
        x = np.random.random() * cum[-1]
        lo, hi = 0, cum.shape[0] - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        d0 = sl[lo] + sk[lo]
        d0r[r] = d0
        l = 1; k = 1; t = 0.0; nc = 0; ng = 0
        while t < H:
            d = l + k
            inv_p = 1.0 / (d0 * np.exp(t))
            chc = 0.0
            for i in range(nc):
                q = crec[i]
                chc += 1.0 / deg_rec(2, p_gt, p_off[q], p_off[q + 1] - p_off[q], p_ts[q], p_grid, q, p_W[q], t - cb[i])
            if d >= DSTAR:
                tstar[r] = t
                u0 = k - chc - inv_p
                W[r] = model_path(float(d) - u0, u0, LAM_BIG, grid, r)
                break
            bound = l + inv_p + chc + l / d
            t += -np.log(1.0 - np.random.random()) / bound
            if t >= H:
                break
            inv_p1 = 1.0 / (d0 * np.exp(t))
            ch1 = 0.0
            for i in range(nc):
                q = crec[i]
                ch1 += 1.0 / deg_rec(2, p_gt, p_off[q], p_off[q + 1] - p_off[q], p_ts[q], p_grid, q, p_W[q], t - cb[i])
            u = np.random.random() * bound
            g = l + inv_p1 + ch1
            if u < g:
                gt[pos] = t; gr[pos] = l + ch1; pos += 1
                vgt[ng] = t; ng += 1
                l += 1
            elif u < g + l / d:
                kk = d if d < keymax else keymax
                n_k = keyoff[kk + 1] - keyoff[kk]
                if nc < MAXC and n_k > 0:
                    crec[nc] = keyidx[keyoff[kk] + int(np.random.random() * n_k)]
                    cb[nc] = t; cdb[nc] = d; nc += 1
                l -= 1; k += 1
        off[r + 1] = pos
        tend = tstar[r] if tstar[r] >= 0.0 else H
        lw = 0.0
        for i in range(nc):
            q = crec[i]
            T = tend - cb[i]
            if p_ts[q] >= 0.0 and p_ts[q] < T:
                T = p_ts[q]
            if T <= 0.0:
                continue
            i0 = p_off[q]; n = p_off[q + 1] - p_off[q]
            dref = float(p_d0[q])
            for j in range(n):
                tj = p_gt[i0 + j]
                if tj > T:
                    break
                ia = 1.0 / v_deg(cdb[i], vgt, ng, cb[i], cb[i] + tj)
                ib = 1.0 / (dref * np.exp(tj))
                lw += np.log((p_gr[i0 + j] + ia) / (p_gr[i0 + j] + ib))
            lw -= v_inv_int(cdb[i], vgt, ng, cb[i], T) - (1.0 - np.exp(-T)) / dref
        logw[r] = lw
    return d0r, off, gt[:pos], gr[:pos], tstar, grid, W, logw


def key_index(d0r, keymax):
    keys = np.minimum(d0r, keymax)
    order = np.argsort(keys, kind='stable').astype(np.int64)
    keyoff = np.zeros(keymax + 2, np.int64)
    for kk in range(keymax + 1):
        keyoff[kk + 1] = keyoff[kk] + np.sum(keys == kk)
    return order, keyoff


def resample(rng, recs, logw, keymax):
    """Resample records to equal weights within each key (the key is the reference parent degree, which the draws
    condition on). Returns the new pool and the effective sample size fraction per key."""
    d0r, off, gt, gr, tstar, grid, W = recs
    keys = np.minimum(d0r, keymax)
    pick = np.empty(d0r.size, np.int64); ess = {}
    for kk in np.unique(keys):
        idx = np.nonzero(keys == kk)[0]
        w = np.exp(logw[idx] - logw[idx].max()); w /= w.sum()
        ess[int(kk)] = 1.0 / np.sum(w * w) / idx.size
        pick[idx] = idx[rng.choice(idx.size, idx.size, p=w)]
    n = np.diff(off)[pick]
    noff = np.concatenate([[0], np.cumsum(n)])
    sel = np.concatenate([np.arange(off[p], off[p + 1]) for p in pick]) if n.sum() > 0 else np.zeros(0, np.int64)
    return (d0r[pick], noff, gt[sel], gr[sel], tstar[pick], grid[pick], W[pick]), ess


@njit(cache=True)
def rec_logratio(gt, gr, i0, n, T, xd0, xgt, xng, xb, rd0, rgt, ri0, rn, rts, rgrid, rrow, rW, rdet0):
    """log density ratio over [0, T] of a record (gains gt[i0:i0+n] relative to the start of the conditioning
    neighbour, rates without it gr) under the neighbour path X (live path: degree xd0 at absolute time xb, then +1 at
    each absolute gain time in xgt[:xng] after xb) against its reference path R (a record path: degree rd0 at 0,
    gains rgt[ri0:ri0+rn], tstar rts, model grid; or, if rdet0 > 0, rdet0 exp(t))."""
    s = 0.0
    for j in range(n):
        tj = gt[i0 + j]
        if tj > T:
            break
        ia = 1.0 / v_deg(xd0, xgt, xng, xb, xb + tj)
        if rdet0 > 0.0:
            ib = 1.0 / (rdet0 * np.exp(tj))
        else:
            ib = 1.0 / deg_rec(rd0, rgt, ri0, rn, rts, rgrid, rrow, rW, tj)
        s += np.log((gr[i0 + j] + ia) / (gr[i0 + j] + ib))
    s -= v_inv_int(xd0, xgt, xng, xb, T)
    if rdet0 > 0.0:
        s += (1.0 - np.exp(-T)) / rdet0
    else:   # integral of 1/d along the reference record path, by sub-steps of 0.01
        nsub = int(T / 0.01) + 1; h = T / nsub
        for q in range(nsub):
            s += h / deg_rec(rd0, rgt, ri0, rn, rts, rgrid, rrow, rW, (q + 0.5) * h)
    return s


@njit(cache=True)
def sim_vertex(seed_offset, mu, H, TCAP, MAXC, mode,
               # parent of the vertex: mode 0 = deterministic reference d0 exp(t) (generation 0); mode 1 = parent record
               cum, sl, sk,
               P_d0, P_off, P_gt, P_gr, P_ts, P_grid, P_W, P_sh, P_pick,
               # child pool
               c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W, keyidx, keyoff, keymax,
               # outputs for this vertex (work arrays)
               vgt, gl, gip, gch, crec, cb, cdb, lstate, kstate, tstate):
    """Simulates one vertex from its creation until degree DSTAR or age TCAP. Its parent: mode 0, the path d0 exp(t)
    with d0 from the parent law; mode 1, the parent record P_pick (degree P_d0 at the birth, gains after it, model
    path above its tstar). Children: records of the pool, by the vertex's degree at each birth. Records the vertex's
    gain times (vgt), the states it passes through with their entry times (lstate, kstate, tstate), its children.
    Returns (number of gains, number of children, number of state changes, tstar or -1, u* at tstar, d0 of mode 0)."""
    l = 1; k = 1; t = 0.0; nc = 0; ng = 0; ns = 0; ts = -1.0; ust = 0.0; d0 = 0
    if mode == 0:
        x = np.random.random() * cum[-1]
        lo, hi = 0, cum.shape[0] - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        d0 = sl[lo] + sk[lo]
    q0 = P_pick
    lstate[0] = 1; kstate[0] = 1; tstate[0] = 0.0; ns = 1
    while t < TCAP:
        d = l + k
        if mode == 0:
            ip = 1.0 / (d0 * np.exp(t))
        else:
            ip = 1.0 / deg_rec(P_d0[q0], P_gt, P_off[q0], P_off[q0 + 1] - P_off[q0], P_ts[q0], P_grid, q0, P_W[q0], t + P_sh[q0])
        chc = 0.0
        for i in range(nc):
            q = crec[i]
            chc += 1.0 / deg_rec(2, c_gt, c_off[q], c_off[q + 1] - c_off[q], c_ts[q], c_grid, q, c_W[q], t - cb[i])
        if d >= DSTAR:
            ts = t; ust = k - chc - ip
            break
        bound = l + ip + chc + l / d
        t += -np.log(1.0 - np.random.random()) / bound
        if t >= TCAP:
            break
        if mode == 0:
            ip1 = 1.0 / (d0 * np.exp(t))
        else:
            ip1 = 1.0 / deg_rec(P_d0[q0], P_gt, P_off[q0], P_off[q0 + 1] - P_off[q0], P_ts[q0], P_grid, q0, P_W[q0], t + P_sh[q0])
        ch1 = 0.0
        for i in range(nc):
            q = crec[i]
            ch1 += 1.0 / deg_rec(2, c_gt, c_off[q], c_off[q + 1] - c_off[q], c_ts[q], c_grid, q, c_W[q], t - cb[i])
        u = np.random.random() * bound
        g = l + ip1 + ch1
        if u < g:
            vgt[ng] = t; gl[ng] = l; gip[ng] = ip1; gch[ng] = ch1; ng += 1
            l += 1
        elif u < g + l / d:
            kk = d if d < keymax else keymax
            n_k = keyoff[kk + 1] - keyoff[kk]
            if nc < MAXC and n_k > 0:
                crec[nc] = keyidx[keyoff[kk] + int(np.random.random() * n_k)]
                cb[nc] = t; cdb[nc] = d; nc += 1
            l -= 1; k += 1
        else:
            continue
        if ns < lstate.shape[0]:
            lstate[ns] = l; kstate[ns] = k; tstate[ns] = t; ns += 1
    return ng, nc, ns, ts, ust, d0


@njit(cache=True)
def child_logw(i, crec, cb, cdb, vgt, ng, tend, c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W):
    q = crec[i]
    T = tend - cb[i]
    if c_ts[q] >= 0.0 and c_ts[q] < T:
        T = c_ts[q]
    if T <= 0.0:
        return 0.0
    return rec_logratio(c_gt, c_gr, c_off[q], c_off[q + 1] - c_off[q], T, cdb[i], vgt, ng, cb[i],
                        0, c_gt, 0, 0, -1.0, c_grid, 0, 0.0, float(c_d0[q]))


@njit(cache=True)
def gen_parents(nrec, seed, mu, H, TCAP, MAXC, mode, cum, sl, sk,
                G_d0, G_off, G_gt, G_gr, G_ts, G_grid, G_W, G_sh, G_rdoff, G_rdgt, G_rdts, G_rdgrid, G_rdW,
                c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W, keyidx, keyoff, keymax):
    """Parent records (upward message). A vertex P from creation (its own parent: mode 0 the path d0 exp(t); mode 1 a
    record of the previous generation, reweighted to P's path), its children from the child pool (reweighted). One of
    its births is designated with probability proportional to exp(-mu A) (births after its tstar at rate 1, by the
    analytic tail); the record weight carries the sum S of exp(-mu A). Stored: P's path from the designated birth
    (degree, gains, rates without the designated child, tstar, model path, shift for designations above tstar) and
    the designated child's path (a child record) as the reference for later reweighting."""
    np.random.seed(seed)
    cap = nrec * DSTAR
    d0o = np.zeros(nrec); off = np.zeros(nrec + 1, np.int64); gt = np.zeros(cap); gr = np.zeros(cap)
    tso = -np.ones(nrec); grid = np.zeros((nrec, NGRID + 1), np.float32); Wo = np.zeros(nrec); sh = np.zeros(nrec)
    rdoff = np.zeros(nrec + 1, np.int64); rdgt = np.zeros(cap); rdts = -np.ones(nrec); rdgrid = np.zeros((nrec, NGRID + 1), np.float32); rdW = np.zeros(nrec)
    logw = np.zeros(nrec)
    vgt = np.zeros(DSTAR + 2); gl = np.zeros(DSTAR + 2); gip = np.zeros(DSTAR + 2); gch = np.zeros(DSTAR + 2)
    crec = np.zeros(MAXC, np.int64); cb = np.zeros(MAXC); cdb = np.zeros(MAXC, np.int64)
    ls = np.zeros(4 * DSTAR + 8, np.int64); ks = np.zeros(4 * DSTAR + 8, np.int64); tsx = np.zeros(4 * DSTAR + 8)
    pos = 0; rpos = 0
    nG = G_d0.shape[0]
    dumg = np.zeros((1, NGRID + 1))
    for r in range(nrec):
        qG = int(np.random.random() * nG) if mode == 1 else 0
        ng, nc, ns, ts, ust, d0 = sim_vertex(0, mu, H, TCAP, MAXC, mode, cum, sl, sk,
                                             G_d0, G_off, G_gt, G_gr, G_ts, G_grid, G_W, G_sh, qG,
                                             c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W, keyidx, keyoff, keymax,
                                             vgt, gl, gip, gch, crec, cb, cdb, ls, ks, tsx)
        tend = ts if ts >= 0.0 else TCAP
        # designation
        S = 0.0
        for i in range(nc):
            S += np.exp(-mu * cb[i])
        tail = 0.0
        if ts >= 0.0:
            tail = np.exp(-mu * ts) * tail_int(float(ks[ns - 1]), ust, mu, LAM_BIG)[0]
        S += tail
        x = np.random.random() * S
        di = -1
        acc = 0.0
        for i in range(nc):
            acc += np.exp(-mu * cb[i])
            if x < acc:
                di = i
                break
        lw = np.log(S) if S > 0.0 else -1e300
        for i in range(nc):
            if i != di:
                lw += child_logw(i, crec, cb, cdb, vgt, ng, tend, c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W)
        if mode == 1:   # P's own parent record, from its reference designated path to P's actual path
            T = tend
            if G_ts[qG] >= 0.0 and G_ts[qG] - G_sh[qG] < T and G_sh[qG] == 0.0:
                T = G_ts[qG]
            if G_sh[qG] == 0.0:
                lw += rec_logratio(G_gt, G_gr, G_off[qG], G_off[qG + 1] - G_off[qG], T, 2, vgt, ng, 0.0,
                                   2, G_rdgt, G_rdoff[qG], G_rdoff[qG + 1] - G_rdoff[qG], G_rdts[qG], G_rdgrid, qG, G_rdW[qG], 0.0)
        logw[r] = lw
        # model path of P above its tstar
        if ts >= 0.0:
            Wo[r] = model_path(float(DSTAR) - ust, ust, LAM_BIG, grid, r)
        if di >= 0:
            A = cb[di]
            dA = 2
            for j in range(ng):
                if vgt[j] <= A:
                    dA += 1
            d0o[r] = dA; sh[r] = 0.0
            tso[r] = ts - A if ts >= 0.0 else -1.0
            qd = crec[di]
            for j in range(ng):
                if vgt[j] > A:
                    tj = vgt[j] - A
                    dd = deg_rec(2, c_gt, c_off[qd], c_off[qd + 1] - c_off[qd], c_ts[qd], c_grid, qd, c_W[qd], tj)
                    gt[pos] = tj; gr[pos] = gl[j] + gip[j] + gch[j] - 1.0 / dd; pos += 1
        else:
            # designated above tstar: A = ts + Exp(mu)
            A = ts + tail_sample(float(ks[ns - 1]), ust, mu, LAM_BIG)
            d0o[r] = DSTAR; tso[r] = 0.0; sh[r] = A - ts
            kk = keymax
            n_k = keyoff[kk + 1] - keyoff[kk]
            qd = keyidx[keyoff[kk] + int(np.random.random() * n_k)]
        off[r + 1] = pos
        # the designated child's path (reference)
        for j in range(c_off[qd], c_off[qd + 1]):
            rdgt[rpos] = c_gt[j]; rpos += 1
        rdoff[r + 1] = rpos
        rdts[r] = c_ts[qd]; rdgrid[r, :] = c_grid[qd, :]; rdW[r] = c_W[qd]
    return (d0o, off, gt[:pos], gr[:pos], tso, grid, Wo, sh, rdoff, rdgt[:rpos], rdts, rdgrid, rdW), logw


@njit(cache=True)
def output(nv, seed, mu, H, TCAP, MAXC, P_d0, P_off, P_gt, P_gr, P_ts, P_grid, P_W, P_sh,
           P_rdoff, P_rdgt, P_rdts, P_rdgrid, P_rdW,
           c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W, keyidx, keyoff, keymax):
    """Output: newborn vertices v, parent from the parent records, children from the child records, each record
    reweighted to v's path. Returns per vertex its log-weight, its birth ages (MAXC), its tstar; and weighted (by the
    vertex's weight times exp(-mu a)) occupation and credit by state (l, k) with l + k < DSTAR, unnormalised per
    vertex: arrays Wv, WCv summed with weights after normalisation by the caller."""
    np.random.seed(seed)
    D = DSTAR
    Wq = np.zeros((D + 1, D + 1)); WCq = np.zeros((D + 1, D + 1))
    lws = np.zeros(nv); births = -np.ones((nv, MAXC)); tsv = -np.ones(nv); tI = np.zeros(nv); tdI = np.zeros(nv)
    vgt = np.zeros(D + 2); gl = np.zeros(D + 2); gip = np.zeros(D + 2); gch = np.zeros(D + 2)
    crec = np.zeros(MAXC, np.int64); cb = np.zeros(MAXC); cdb = np.zeros(MAXC, np.int64)
    ls = np.zeros(4 * D + 8, np.int64); ks = np.zeros(4 * D + 8, np.int64); tsx = np.zeros(4 * D + 8)
    locW = np.zeros((D + 1, D + 1)); locC = np.zeros((D + 1, D + 1))
    nP = P_d0.shape[0]
    dum = np.zeros(1, np.int64)
    for iv in range(nv):
        qP = int(np.random.random() * nP)
        ng, nc, ns, ts, ust, d0 = sim_vertex(0, mu, H, TCAP, MAXC, 1, dum.astype(np.float64), dum, dum,
                                             P_d0, P_off, P_gt, P_gr, P_ts, P_grid, P_W, P_sh, qP,
                                             c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W, keyidx, keyoff, keymax,
                                             vgt, gl, gip, gch, crec, cb, cdb, ls, ks, tsx)
        tend = ts if ts >= 0.0 else TCAP
        lw = 0.0
        for i in range(nc):
            lw += child_logw(i, crec, cb, cdb, vgt, ng, tend, c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W)
        if P_sh[qP] == 0.0:
            T = tend
            if P_ts[qP] >= 0.0 and P_ts[qP] < T:
                T = P_ts[qP]
            lw += rec_logratio(P_gt, P_gr, P_off[qP], P_off[qP + 1] - P_off[qP], T, 2, vgt, ng, 0.0,
                               2, P_rdgt, P_rdoff[qP], P_rdoff[qP + 1] - P_rdoff[qP], P_rdts[qP], P_rdgrid, qP, P_rdW[qP], 0.0)
        lws[iv] = lw; tsv[iv] = ts
        if ts >= 0.0:
            tI[iv], tdI[iv] = tail_int(float(ks[ns - 1]), ust, mu, LAM_BIG)
        for i in range(nc):
            births[iv, i] = cb[i]
        # tallies along v's path: state intervals, credit integrated by sub-steps of at most 0.02
        locW[:, :] = 0.0; locC[:, :] = 0.0
        for j in range(ns):
            t0 = tsx[j]
            t1 = tsx[j + 1] if j + 1 < ns else tend
            if t1 > H:
                t1 = H
            if t1 <= t0:
                continue
            l = ls[j]; k = ks[j]
            if l + k >= D:
                continue
            nsub = int((t1 - t0) / 0.02) + 1; h = (t1 - t0) / nsub
            for q in range(nsub):
                s0 = t0 + q * h; tm = s0 + 0.5 * h
                c = 1.0 / deg_rec(P_d0[qP], P_gt, P_off[qP], P_off[qP + 1] - P_off[qP], P_ts[qP], P_grid, qP, P_W[qP], tm + P_sh[qP])
                for i in range(nc):
                    if cb[i] < tm:
                        qq = crec[i]
                        c += 1.0 / deg_rec(2, c_gt, c_off[qq], c_off[qq + 1] - c_off[qq], c_ts[qq], c_grid, qq, c_W[qq], tm - cb[i])
                wo = (np.exp(-mu * s0) - np.exp(-mu * (s0 + h))) / mu
                locW[l, k] += wo; locC[l, k] += wo * c
        w = np.exp(min(lw, 50.0))
        Wq += w * locW; WCq += w * locC
    return lws, births, tsv, tI, tdI, Wq, WCq


@njit(cache=True)
def output_age(nv, seed, mu, H, TCAP, MAXC, aw, P_d0, P_off, P_gt, P_gr, P_ts, P_grid, P_W, P_sh,
           P_rdoff, P_rdgt, P_rdts, P_rdgrid, P_rdW,
           c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W, keyidx, keyoff, keymax):
    """As output, and also (reduction S1) the weighted occupation Wa, parent part PIa and children part CHa of the
    credit by age bin (width aw) and state, for states of degree below DSTAR.
    Output: newborn vertices v, parent from the parent records, children from the child records, each record
    reweighted to v's path. Returns per vertex its log-weight, its birth ages (MAXC), its tstar; and weighted (by the
    vertex's weight times exp(-mu a)) occupation and credit by state (l, k) with l + k < DSTAR, unnormalised per
    vertex: arrays Wv, WCv summed with weights after normalisation by the caller."""
    np.random.seed(seed)
    D = DSTAR
    Wq = np.zeros((D + 1, D + 1)); WCq = np.zeros((D + 1, D + 1))
    lws = np.zeros(nv); births = -np.ones((nv, MAXC)); tsv = -np.ones(nv); tI = np.zeros(nv); tdI = np.zeros(nv)
    vgt = np.zeros(D + 2); gl = np.zeros(D + 2); gip = np.zeros(D + 2); gch = np.zeros(D + 2)
    crec = np.zeros(MAXC, np.int64); cb = np.zeros(MAXC); cdb = np.zeros(MAXC, np.int64)
    ls = np.zeros(4 * D + 8, np.int64); ks = np.zeros(4 * D + 8, np.int64); tsx = np.zeros(4 * D + 8)
    locW = np.zeros((D + 1, D + 1)); locC = np.zeros((D + 1, D + 1))
    nab = int(H / aw + 0.5)
    Wa = np.zeros((nab, D + 1, D + 1)); PIa = np.zeros((nab, D + 1, D + 1)); CHa = np.zeros((nab, D + 1, D + 1))

    nP = P_d0.shape[0]
    dum = np.zeros(1, np.int64)
    for iv in range(nv):
        qP = int(np.random.random() * nP)
        ng, nc, ns, ts, ust, d0 = sim_vertex(0, mu, H, TCAP, MAXC, 1, dum.astype(np.float64), dum, dum,
                                             P_d0, P_off, P_gt, P_gr, P_ts, P_grid, P_W, P_sh, qP,
                                             c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W, keyidx, keyoff, keymax,
                                             vgt, gl, gip, gch, crec, cb, cdb, ls, ks, tsx)
        tend = ts if ts >= 0.0 else TCAP
        lw = 0.0
        for i in range(nc):
            lw += child_logw(i, crec, cb, cdb, vgt, ng, tend, c_d0, c_off, c_gt, c_gr, c_ts, c_grid, c_W)
        if P_sh[qP] == 0.0:
            T = tend
            if P_ts[qP] >= 0.0 and P_ts[qP] < T:
                T = P_ts[qP]
            lw += rec_logratio(P_gt, P_gr, P_off[qP], P_off[qP + 1] - P_off[qP], T, 2, vgt, ng, 0.0,
                               2, P_rdgt, P_rdoff[qP], P_rdoff[qP + 1] - P_rdoff[qP], P_rdts[qP], P_rdgrid, qP, P_rdW[qP], 0.0)
        lws[iv] = lw; tsv[iv] = ts
        if ts >= 0.0:
            tI[iv], tdI[iv] = tail_int(float(ks[ns - 1]), ust, mu, LAM_BIG)
        for i in range(nc):
            births[iv, i] = cb[i]
        # tallies along v's path: state intervals, credit integrated by sub-steps of at most 0.02
        locW[:, :] = 0.0; locC[:, :] = 0.0
        wv = np.exp(min(lw, 50.0))   # weight known before the tallies: add directly (fixed 21:07, no per-vertex arrays)
        for j in range(ns):
            t0 = tsx[j]
            t1 = tsx[j + 1] if j + 1 < ns else tend
            if t1 > H:
                t1 = H
            if t1 <= t0:
                continue
            l = ls[j]; k = ks[j]
            if l + k >= D:
                continue
            nsub = int((t1 - t0) / 0.02) + 1; h = (t1 - t0) / nsub
            for q in range(nsub):
                s0 = t0 + q * h; tm = s0 + 0.5 * h
                cp = 1.0 / deg_rec(P_d0[qP], P_gt, P_off[qP], P_off[qP + 1] - P_off[qP], P_ts[qP], P_grid, qP, P_W[qP], tm + P_sh[qP])
                cc = 0.0
                for i in range(nc):
                    if cb[i] < tm:
                        qq = crec[i]
                        cc += 1.0 / deg_rec(2, c_gt, c_off[qq], c_off[qq + 1] - c_off[qq], c_ts[qq], c_grid, qq, c_W[qq], tm - cb[i])
                c = cp + cc
                wo = (np.exp(-mu * s0) - np.exp(-mu * (s0 + h))) / mu
                locW[l, k] += wo; locC[l, k] += wo * c
                ab = int(tm / aw)
                if ab < nab:
                    Wa[ab, l, k] += wv * h; PIa[ab, l, k] += wv * h * cp; CHa[ab, l, k] += wv * h * cc
        w = np.exp(min(lw, 50.0))
        Wq += w * locW; WCq += w * locC
    return lws, births, tsv, tI, tdI, Wq, WCq, Wa, PIa, CHa
