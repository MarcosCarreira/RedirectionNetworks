"""Test D: direct measurement of the credit distribution in complete redirection (IR) trees.

Theory and pre-registration: paper/CreditMeasurement_2026-11-23.tex.

For a tree T_N and a state s = (l, k) (l leaf neighbours, k non-leaf neighbours), M_s is the number
of non-leaves in state s and H_s the sum of their credits, the credit of v being the sum of 1/d(u)
over the non-leaf neighbours u of v. Pooled over runs, c_s = sum H_s / sum M_s is the mean credit of
a vertex in state s. With these rates the individual chain (gain rate l + c_s, birth rate l/d) gives
mu_plug from the complementary Euler-Lotka equation 1 - mu = sum_s q_s k/d (Proposition 1 of the note).

Phases (no arguments; settings below):
  0. Solver check: with c_s = k * lambda of the equal-sharing closure at LMAX = 3000, the solver must
     return the closure value 0.56884815 (closure_solver, 22 Nov), to 1e-6.
  1. Validation: VALID_RUNS trees to N = 21, mean C and S against the exact values (|z| <= 3).
  2. Main: RUNS trees to NMAX, state tallies of M and H at the checkpoints CHECKS.
     D1 identities in every tree: sum_s H_s = C - S to relative 1e-9 and sum_s M_s = C.
     D2 mu_plug at each checkpoint, sparse-state thresholds 200 (pre-registered), 50, 1000;
        jackknife over GROUPS groups; decision at N = 10^7, threshold 200.
     D3 ratios r_s = c_s / (k * lambda_hat) for the measured states; predictions P1, P2.
Output: printed and written to results/credit_sim_<date_time>.txt; pooled and per-group tallies in
results/credit_sim_<date_time>_tallies.npz; per-state table results/credit_sim_<date_time>_states.csv.
"""
import os, sys, time, datetime
import numpy as np

WORKERS = 2                      # threads used for the runs (never derived from the machine)
NMAX = 10**7                     # final tree size of the main runs
CHECKS = [10**6, 10**7]          # checkpoints with state tallies
RUNS = 200                       # number of independent main runs
GROUPS = 20                      # jackknife groups (RUNS / GROUPS runs each)
SEED0 = 20261123                 # master seed; per-run seeds from numpy SeedSequence([SEED0, 3, 0]) (stream 3: Test D)
BURN = 1000                      # random draws discarded after seeding each run
VALID_N = 21                     # validation trees grown to this size
VALID_RUNS = 400000              # number of validation trees
KCAP = 96                        # maximum non-leaf neighbours stored per non-leaf (as in ratio_sim)
LCAP = 3000                      # states with l <= LCAP are tallied one by one; l > LCAP go to an overflow bin per k
LMAX = 3000                      # chain cut-off in l (the note: l <= 3000)
KMAX = 100                       # chain cut-off in k (the note: k <= 100)
THRESHOLDS = [200, 50, 1000]     # sparse-state rule: a state is measured if its pooled count >= threshold; first = pre-registered
RHO_HAT, RHO_SE = 0.565584, 0.000114           # rho_N(10^8) for IR, ratio_sim 2026-11-23_121529
PASS_TOL, STRONG_TOL = 0.0010, 0.0003          # pre-registered tolerances for |mu_plug(10^7) - RHO_HAT|
ES_LAM_3000, ES_MU_3000 = 0.21557593, 0.56884815   # equal-sharing closure at LMAX = 3000 (closure_solver 2026-11-22_225529)
BISECT_TOL = 1e-10

try:
    import numba
    from numba import njit, prange
except ImportError:
    print("numba is required: install it in this Python environment (conda install numba).")
    sys.exit(1)

numba.set_num_threads(WORKERS)

# Exact E[C_N], E[S_N] for IR (cr_exact.py, N <= 16), as fractions.
EXACT = {8: ("4717423/2177280", "7220369/5443200"),
         12: ("2175595493439775699/782190452736000000", "40641826640790127777/24638999261184000000"),
         16: ("2259642509981131170779006369702211737/683804583677425454094090240000000000", None)}


# ---------------------------------------------------------------- simulation
@njit(cache=True)
def _fen_add(fen, i, delta):
    i += 1
    n = fen.shape[0] - 1
    while i <= n:
        fen[i] += delta
        i += i & (-i)


@njit(cache=True)
def _fen_find(fen, target, logn):
    pos = 0
    rem = target
    step = 1 << logn
    n = fen.shape[0] - 1
    while step > 0:
        nxt = pos + step
        if nxt <= n and fen[nxt] <= rem:
            pos = nxt
            rem -= fen[nxt]
        step >>= 1
    return pos


@njit(cache=True)
def grow(nmax, checkpoints, seed, cmax, kcap, lcap, tally):
    """Grow one IR tree from T_3 to nmax. At each checkpoint return C, S and, if tally, the state
    tallies M[l', k], H[l', k] with l' = min(l, lcap + 1), and the identity errors."""
    np.random.seed(seed)
    for _ in range(BURN):
        np.random.random()
    ell = np.zeros(cmax, np.int64)
    kap = np.zeros(cmax, np.int64)
    nbr = np.zeros((cmax, kcap), np.int32)
    fen = np.zeros(cmax + 1, np.int64)
    logn = 0
    while (1 << (logn + 1)) <= cmax:
        logn += 1
    nck = checkpoints.shape[0]
    outC = np.zeros(nck, np.int64)
    outS = np.zeros(nck, np.float64)
    nt = nck if tally else 1
    outM = np.zeros((nt, lcap + 2 if tally else 1, kcap + 1 if tally else 1), np.int64)
    outH = np.zeros((nt, lcap + 2 if tally else 1, kcap + 1 if tally else 1), np.float64)
    idH = np.zeros(nck, np.float64)              # |sum H - (C - S)| / C
    idM = np.zeros(nck, np.int64)                # |sum M - C|
    ell[0] = 2
    _fen_add(fen, 0, 2)
    C = 1
    L = 2
    N = 3
    ic = 0
    while ic < nck and checkpoints[ic] < 3:
        ic += 1
    err = 0
    while N < nmax or (ic < nck and checkpoints[ic] == N):
        if ic < nck and checkpoints[ic] == N:
            s_exact = 0.0
            for v in range(C):
                s_exact += ell[v] / (ell[v] + kap[v])
            outC[ic] = C
            outS[ic] = s_exact
            if tally:
                hsum = 0.0
                msum = 0
                for v in range(C):
                    rho = 0.0
                    for j in range(kap[v]):
                        u = nbr[v, j]
                        rho += 1.0 / (ell[u] + kap[u])
                    lb = ell[v] if ell[v] <= lcap else lcap + 1
                    outM[ic, lb, kap[v]] += 1
                    outH[ic, lb, kap[v]] += rho
                    hsum += rho
                    msum += 1
                idH[ic] = abs(hsum - (C - s_exact)) / C
                idM[ic] = abs(msum - C)
            ic += 1
            if N >= nmax:
                break
        x = np.random.random() * N
        if x < L:
            v = _fen_find(fen, int(x), logn)
            ell[v] += 1
            _fen_add(fen, v, 1)
            L += 1
        else:
            v = np.random.randint(C)
            d = ell[v] + kap[v]
            y = np.random.randint(d)
            if y < ell[v]:
                if C >= cmax or kap[v] >= kcap:
                    err = 1
                    break
                w = C
                C += 1
                ell[v] -= 1
                _fen_add(fen, v, -1)
                nbr[v, kap[v]] = w
                kap[v] += 1
                ell[w] = 1
                kap[w] = 1
                nbr[w, 0] = v
                _fen_add(fen, w, 1)
            else:
                u = nbr[v, np.random.randint(kap[v])]
                ell[u] += 1
                _fen_add(fen, u, 1)
                L += 1
        N += 1
    return outC, outS, outM, outH, idH, idM, err


@njit(parallel=True, cache=True)
def grow_many(nmax, checkpoints, seeds, cmax, kcap, lcap, tally):
    R = seeds.shape[0]
    nck = checkpoints.shape[0]
    nt = nck if tally else 1
    a1 = lcap + 2 if tally else 1
    a2 = kcap + 1 if tally else 1
    Cs = np.zeros((R, nck), np.int64)
    Ss = np.zeros((R, nck), np.float64)
    Ms = np.zeros((R, nt, a1, a2), np.int64)
    Hs = np.zeros((R, nt, a1, a2), np.float64)
    IH = np.zeros((R, nck), np.float64)
    IM = np.zeros((R, nck), np.int64)
    errs = np.zeros(R, np.int64)
    for r in prange(R):
        c, s, m, h, ih, im, e = grow(nmax, checkpoints, seeds[r], cmax, kcap, lcap, tally)
        Cs[r, :] = c
        Ss[r, :] = s
        Ms[r] = m
        Hs[r] = h
        IH[r, :] = ih
        IM[r, :] = im
        errs[r] = e
    return Cs, Ss, Ms, Hs, IH, IM, errs


# ---------------------------------------------------------------- measured-credit chain
def first_order(a, b):
    """Solve q_l = a_l q_{l-1} + b_l (q_{-1} = 0) exactly; a_l = 0 restarts the recursion (as closure_solver)."""
    n = len(a); q = np.empty(n)
    cuts = sorted(set([0] + [i for i in np.where(a == 0)[0].tolist() if i > 0] + [n]))
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        aa = a[s0:s1].copy(); aa[0] = 1.0
        logP = np.cumsum(np.log(aa))
        P = np.exp(logP - logP[0])
        q[s0:s1] = P * np.cumsum(b[s0:s1] / P)
    return q


def solve_q(mu, c):
    """Occupation solution q = mu R of the chain with gain rate l + c[k, l], birth rate l/d; c has shape (KMAX+1, LMAX+1)."""
    l = np.arange(0, LMAX + 1, dtype=float)
    q = np.zeros((KMAX + 1, LMAX + 1))
    prev = np.zeros(LMAX + 2)
    for k in range(1, KMAX + 1):
        d = l + k
        s = l / d
        ck = c[k]
        out = l + s + ck
        inA = np.zeros_like(l); inA[1:] = (l[1:] - 1) + ck[:-1]          # from (l-1, k)
        lp1 = l + 1
        s_prev = lp1 / (lp1 + (k - 1)) if k > 1 else np.zeros_like(l)    # birth from (l+1, k-1)
        inB = s_prev * prev[1:LMAX + 2] if k > 1 else np.zeros_like(l)
        src = np.zeros_like(l)
        if k == 1:
            src[1] = mu
        den = mu + out
        a = np.where(l >= 1, inA / den, 0.0)
        b = (inB + src) / den
        valid = (l + k >= 2)
        a = np.where(valid, a, 0.0); b = np.where(valid, b, 0.0)
        ql = first_order(a, b)
        q[k] = ql
        prev = np.append(ql, 0.0)
    return q


def mu_of(c):
    """Root of 1 - mu = sum_s q_s k/d (complementary Euler-Lotka) by bisection."""
    l = np.arange(0, LMAX + 1, dtype=float)
    K = np.arange(0, KMAX + 1, dtype=float)[:, None]
    d = l[None, :] + K
    with np.errstate(divide='ignore', invalid='ignore'):
        nb = np.where(d > 0, K / d, 0.0)
    lo, hi = 1e-6, 1.0
    while hi - lo > BISECT_TOL:
        mid = 0.5 * (lo + hi)
        g = 1.0 - float((solve_q(mid, c) * nb).sum()) - mid
        if g > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def credit_rates(M, H, thr):
    """c[k, l] for the chain from pooled tallies M, H (indexed [l', k], overflow row l' = LCAP+1 excluded).
    Measured states (count >= thr): c = H/M. Others: k * lambda_k, lambda_k = pooled credit per non-leaf
    neighbour over the non-measured states with the same k; if there are none, over all non-measured states."""
    M = M[:LCAP + 1].astype(float)          # [l, k], l <= LCAP
    H = H[:LCAP + 1]
    kk = np.arange(M.shape[1], dtype=float)[None, :]
    meas = M >= thr
    nm = ~meas
    den_all = float((kk * M)[nm].sum())
    lam_all = float(H[nm].sum()) / den_all
    c = np.zeros((KMAX + 1, LMAX + 1))
    nfall = 0
    for k in range(1, KMAX + 1):
        if k < M.shape[1]:
            mk, hk, sel = M[:LMAX + 1, k], H[:LMAX + 1, k], meas[:LMAX + 1, k]
            den = k * float(mk[~sel].sum())
            lam_k = float(hk[~sel].sum()) / den if den > 0 else lam_all
            with np.errstate(divide='ignore', invalid='ignore'):
                c[k] = np.where(sel, hk / np.where(mk > 0, mk, 1.0), k * lam_k)
            nfall += int((~sel).sum())
        else:
            c[k] = k * lam_all
            nfall += LMAX + 1
    assert (c >= 0).all() and (c <= np.arange(KMAX + 1)[:, None] / 2 + 1e-12).all(), "credit rate outside [0, k/2]"
    return c, int(meas.sum()), lam_all


# ---------------------------------------------------------------- output
def dated_path(stem, ext):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    path, i = f"{base}.{ext}", 1
    while os.path.exists(path):
        path, i = f"{base}_{i}.{ext}", i + 1
    return path


OUT = None
def log(msg):
    print(msg, flush=True)
    if OUT is not None:
        with open(OUT, "a") as f:
            f.write(msg + "\n")


def main():
    global OUT
    from fractions import Fraction
    OUT = dated_path("credit_sim", "txt")
    t0 = time.time()
    log(f"=== run started {datetime.datetime.now().isoformat(timespec='seconds')} NMAX={NMAX} CHECKS={CHECKS} RUNS={RUNS} "
        f"GROUPS={GROUPS} WORKERS={WORKERS} LCAP={LCAP} LMAX={LMAX} KMAX={KMAX} THRESHOLDS={THRESHOLDS} numba {numba.__version__} ===")
    log(f"Pre-registered (CreditMeasurement_2026-11-23.tex): PASS if |mu_plug(10^7) - {RHO_HAT}| <= {PASS_TOL}, strong if <= {STRONG_TOL}")

    # Phase 0: solver check against the equal-sharing closure
    kk = np.arange(KMAX + 1, dtype=float)[:, None]
    mu_es = mu_of(np.broadcast_to(kk * ES_LAM_3000, (KMAX + 1, LMAX + 1)).copy())
    ok0 = abs(mu_es - ES_MU_3000) <= 1e-6
    log(f"--- phase 0, solver check: c = k*{ES_LAM_3000} gives mu = {mu_es:.8f}, closure_solver {ES_MU_3000:.8f} -> "
        f"{'PASS' if ok0 else 'FAIL'} ({time.time()-t0:.1f}s)")
    if not ok0:
        log("solver check failed: stop"); return

    # Phase 1: validation
    vck = np.unique(np.array(sorted(EXACT) + [VALID_N], dtype=np.int64))
    seeds = np.random.SeedSequence([SEED0, 1, 0]).generate_state(VALID_RUNS, dtype=np.uint32).astype(np.int64)
    Cs, Ss, _, _, _, _, errs = grow_many(VALID_N, vck, seeds, 64, 32, 0, False)
    log(f"--- phase 1, validation: {VALID_RUNS} trees to N={VALID_N} ({time.time()-t0:.1f}s) ---")
    valid_ok = not errs.any()
    for j, n in enumerate(vck):
        mC, mS = Cs[:, j].mean(), Ss[:, j].mean()
        seC = Cs[:, j].std(ddof=1) / np.sqrt(VALID_RUNS)
        seS = Ss[:, j].std(ddof=1) / np.sqrt(VALID_RUNS)
        line = f"N={n:3d}: mean C={mC:.6f} (s.e. {seC:.6f})  mean S={mS:.6f} (s.e. {seS:.6f})"
        if int(n) in EXACT:
            eC, eS = EXACT[int(n)]
            zc = (mC - float(Fraction(eC))) / seC
            line += f"  exact C={float(Fraction(eC)):.6f} z={zc:+.2f}"
            valid_ok &= abs(zc) <= 3
            if eS is not None:
                zs = (mS - float(Fraction(eS))) / seS
                line += f"  exact S={float(Fraction(eS)):.6f} z={zs:+.2f}"
                valid_ok &= abs(zs) <= 3
        log(line)
    log(f"validation (all |z| <= 3): {'PASS' if valid_ok else 'FAIL'}")

    # Phase 2: main runs with state tallies
    cks = np.array(CHECKS, dtype=np.int64)
    nck = len(cks)
    cmax = int(3 * NMAX ** 0.6) + 10000
    per = RUNS // GROUPS
    gM = np.zeros((GROUPS, nck, LCAP + 2, KCAP + 1), np.int64)
    gH = np.zeros((GROUPS, nck, LCAP + 2, KCAP + 1))
    gC = np.zeros((GROUPS, nck), np.int64)
    gS = np.zeros((GROUPS, nck))
    maxIH = np.zeros(nck); maxIM = np.zeros(nck, np.int64)
    log(f"--- phase 2, main: {RUNS} trees to N={NMAX}, checkpoints {list(cks)}, capacity {cmax} non-leaves ---")
    seeds = np.random.SeedSequence([SEED0, 3, 0]).generate_state(RUNS, dtype=np.uint32).astype(np.int64)
    done = 0
    while done < RUNS:
        k = min(WORKERS, RUNS - done)
        Cs, Ss, Ms, Hs, IH, IM, errs = grow_many(NMAX, cks, seeds[done:done + k], cmax, KCAP, LCAP, True)
        if errs.any():
            log(f"capacity exceeded in runs {done}..{done+k-1} - increase KCAP or the capacity factor: stop"); return
        for i in range(k):
            g = (done + i) // per
            gM[g] += Ms[i]; gH[g] += Hs[i]; gC[g] += Cs[i]; gS[g] += Ss[i]
        maxIH = np.maximum(maxIH, IH.max(axis=0)); maxIM = np.maximum(maxIM, IM.max(axis=0))
        done += k
        if done % 20 == 0 or done == RUNS:
            log(f"  runs done {done}/{RUNS}  ({time.time()-t0:.0f}s)")
    tal = OUT.replace(".txt", "_tallies.npz")
    with open(tal, "xb") as f:
        np.savez_compressed(f, checks=cks, gM=gM, gH=gH, gC=gC, gS=gS)
    log(f"tallies written to {os.path.basename(tal)}")

    # D1: identities
    okD1 = bool((maxIH <= 1e-9).all() and (maxIM == 0).all())
    for j, n in enumerate(cks):
        log(f"D1 N={n}: max relative |sum H - (C - S)|/C = {maxIH[j]:.2e}, max |sum M - C| = {maxIM[j]}")
    log(f"D1 identities: {'PASS' if okD1 else 'FAIL'}")
    if not okD1:
        log("identity check failed: stop (analysis not run)"); return

    # D2: mu_plug with jackknife
    final = None
    for j, n in enumerate(cks):
        M, H = gM[:, j].sum(0), gH[:, j].sum(0)
        rho_n = gS[:, j].sum() / gC[:, j].sum()
        over = int(M[LCAP + 1].sum())
        log(f"--- D2, N={n}: rho_N = sum S / sum C = {rho_n:.6f}; vertices with l > {LCAP}: {over} of {int(M.sum())} ---")
        for thr in THRESHOLDS:
            c, nmeas, lam_nm = credit_rates(M, H, thr)
            mu = mu_of(c)
            jk = []
            for g in range(GROUPS):
                cg, _, _ = credit_rates(M - gM[g, j], H - gH[g, j], thr)
                jk.append(mu_of(cg))
            jk = np.array(jk)
            se = np.sqrt((GROUPS - 1) / GROUPS * ((jk - jk.mean()) ** 2).sum())
            dev = mu - RHO_HAT
            log(f"  threshold {thr:5d}: measured states {nmeas}, fallback credit per slot {lam_nm:.5f}; "
                f"mu_plug = {mu:.6f} (jackknife s.e. {se:.6f}); mu_plug - rho_hat = {dev:+.6f} "
                f"(z = {dev/np.sqrt(se**2 + RHO_SE**2):+.2f})  ({time.time()-t0:.0f}s)")
            if n == cks[-1] and thr == THRESHOLDS[0]:
                final = (mu, se, dev)
    if final is not None and cks[-1] == 10**7:
        mu, se, dev = final
        verdict = "STRONG PASS" if abs(dev) <= STRONG_TOL else ("PASS" if abs(dev) <= PASS_TOL else "FAIL")
        log(f"PRE-REGISTERED TEST D: mu_plug(10^7) = {mu:.6f} (s.e. {se:.6f}), rho_hat = {RHO_HAT:.6f}, "
            f"difference {dev:+.6f} -> {verdict} (PASS <= {PASS_TOL}, strong <= {STRONG_TOL})")

    # D3: credit relative to equal sharing, last checkpoint
    j = nck - 1
    M, H = gM[:, j].sum(0).astype(float), gH[:, j].sum(0)
    kk = np.arange(KCAP + 1, dtype=float)[None, :]
    lam_hat = H.sum() / (kk * M).sum()
    log(f"--- D3, N={cks[j]}: lambda_hat (mean credit per non-leaf neighbour) = {lam_hat:.6f} ---")
    def pooled(sel):
        n = int(M[sel].sum())
        return (H[sel].sum() / (lam_hat * (kk * M)[sel].sum()) if n > 0 else float("nan")), n
    L_ = np.arange(LCAP + 2)[:, None] * np.ones((1, KCAP + 1))
    K_ = np.ones((LCAP + 2, 1)) * np.arange(KCAP + 1)[None, :]
    meas = M >= THRESHOLDS[0]
    meas[LCAP + 1] = False
    r1, n1 = pooled(meas & (L_ == 0))
    r2, n2 = pooled(meas & (L_ >= 5) & (K_ >= 2))
    r3, n3 = pooled((L_ == 1) & (K_ == 1))
    log(f"P1  l = 0, measured states pooled:            r = {r1:.4f} (vertices {n1})  predicted r < 1: {('confirmed' if r1 < 1 else 'not confirmed') if n1 > 0 else 'no measured states'}")
    log(f"P2  l >= 5, k >= 2, measured states pooled:   r = {r2:.4f} (vertices {n2})  predicted r > 1: {('confirmed' if r2 > 1 else 'not confirmed') if n2 > 0 else 'no measured states'}")
    log(f"    fresh vertices, state (1,1):               r = {r3:.4f} (vertices {n3})  no prediction")
    log("    r for l = 0..10 (rows) and k = 1..8 (columns), measured states; '-' = not measured")
    log("      l " + "".join(f"{k:>8d}" for k in range(1, 9)))
    for l in range(0, 11):
        row = ""
        for k in range(1, 9):
            row += f"{H[l, k] / (lam_hat * k * M[l, k]):8.4f}" if meas[l, k] else f"{'-':>8s}"
        log(f"  {l:5d} " + row)
    st = OUT.replace(".txt", "_states.csv")
    with open(st, "x") as f:
        f.write(f"N,l,k,count,credit_c,ratio_r\n")
        for l in range(LCAP + 1):
            for k in range(1, KCAP + 1):
                if M[l, k] > 0:
                    c = H[l, k] / M[l, k]
                    f.write(f"{cks[j]},{l},{k},{int(M[l, k])},{c:.8g},{c / (k * lam_hat):.6f}\n")
    log(f"per-state values in {os.path.basename(st)}")
    log(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
