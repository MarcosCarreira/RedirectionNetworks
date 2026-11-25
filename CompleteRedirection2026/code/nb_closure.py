"""Closure from the neighbourhood of each vertex (complete redirection, IR).

Theory and pre-registration: paper/DistributionFormulas_2026-11-23.tex, section "A closure from the
neighbourhood of each vertex" (checks L1-L4) and its amendment of 24 November (PARENT_EXACT: the parent of v gains
leaves at rate l_p + max(c_table - m, 0) + 1/d_v, with v's exact credit in place of one mean child credit). A vertex v is followed from its creation in (1,1) with its parent
p and the children c_k created at its births, while its degree is at most D_R:
  p: starts in (l-1, k+1), (l, k) ~ pi_b(s) = q_s l/(mu d); then the individual chain with the table credit;
  v: gain at rate l_v + 1/d_p + sum_k 1/d_{c_k}, birth at rate l_v/d_v (exact given the degrees);
  c_k: created in (1,1); gain at rate l_c + 1/d_v + (k_c - 1) m, birth at rate l_c/d_c; m = sum_s q_s/d(s).
The new credit of a state s in R is the exp(-mu a)-weighted mean credit of v while in s. Outside R, c = k lambda
with lambda from the exact total sum_s c_s q_s = 1 - mu. mu and q from the ratio equations (credit_sim.py solver).
TAIL (fourth amendment, 24 Nov 12:40 BRT, note 'Tail of the state table and pre-registered Test P'): the mass
T = 1 - sum q beyond l = LMAX enters the parent law, m, lambda and the ratio equation for mu (leading order in k/l).
PCENTRED (fifth amendment, 24 Nov 13:46 BRT, note 'Parent followed from its creation and pre-registered Test S'): run4
follows a parent from its creation with its own parent and all its children; the tables come from its children in full.
OUTSIDE_P (sixth amendment, 24 Nov 14:24 BRT, note 'Credits outside the region from the followed parents and pre-registered
Test T'): outside R, credit k * lambda_B in the degree bands BANDS from the followed parents; lambda from the total above.

No arguments; settings below. Output: printed and written to results/nb_closure_<date_time>.txt.
"""
import os, sys, time, datetime
import numpy as np

D_R = 10                         # region R: l + k <= D_R
AMAX = 12.0                      # ages followed
DBIG = 2000                      # above this degree a neighbour grows as d(t0) exp(t - t0)
N_IND = 400000                   # vertices per iteration
ITERS_MAX = 24                   # closed iterations from equal sharing (L3), at most (16 until run U, 24 Nov 15:04)
CONV_TOL, CONV_RUN = 1e-4, 2     # stop when |change of mu| < CONV_TOL in CONV_RUN consecutive iterations
AVG_LAST = 12                    # reported mu: mean over the last iterations (8 until run U)
SEED0 = 2026112410               # model seed: the same seed in every iteration (common random numbers)
N_SPREAD = 10                    # extra evaluations at the fixed point with seeds SEED0 + 1.. (Monte Carlo spread; 5 until run U)
MU_REF = 0.565723
CREDIT_FILE = "credit_sim_2026-11-23_154518_tallies.npz"   # Test D (L1, L2 and L4 comparisons only)
CREDIT_MIN = 200
L1_M = 0.33
L1_C11, L1_C02 = 0.1833, 0.4076  # three-vertex model (zero2_model_2026-11-24_054227_cloud.txt)
L1_MUPLUG = 0.565808             # Test D mu_plug at threshold 200
L2_QMIN, L2_TOL, L2_FRAC = 0.005, 0.05, 0.80
L3_TOL, L3_STRONG = 0.0010, 0.0003
AGE_MODEL = True                 # second amendment of 24 Nov (note, 09:49 BRT): parent credit by state and age, and the credit a child
                                 # receives from its children by their age, both from the model's own output (run3)
AGE_W = 0.25                     # age bins of the model tables
GMAX = 12                        # grandchild birth times stored per child
DBIGV = 200                      # v beyond R: degree deterministic above this
CELL_N = 200                     # model table cell: at least this many intervals, else pooled (2, 4, 8 bins), else state credit
POOL_HW = (2, 4, 8)
CHILD_SUB_AGE = True             # third amendment (24 Nov 10:45, after diagnostic N): subtract the model's own per-child credit
                                 # by parent state and age (False: second amendment)
PARENT_EXACT = True              # amendment of 24 Nov (note): parent gain rate l_p + max(c_table - m, 0) + 1/d_v
                                 # (False: l_p + c_table, the registered run of 05:50; L1 always uses False)
TAIL = True                      # fourth amendment (24 Nov 12:40 BRT, after Test O): tail of the state table beyond LMAX
TAIL_DEC = 12                    # tail grid in the parent law: l from LMAX to LMAX * 10**TAIL_DEC
TAIL_PER_DEC = 10                # grid points per decade
TAILD_LAM = 0.1692               # diagnostic Q only: Test D credit per non-leaf neighbour in the overflow row l > 3000
PCENTRED = True                  # fifth amendment (24 Nov 13:46 BRT, after Test R): parent followed from its creation (run4)
N_PAR = 600000                   # parents per iteration (PCENTRED; 300000 until run U)
A_FULL = 1.5                     # children born at the parent's age A followed in full with probability min(1, exp(-mu (A - A_FULL)))
DBIG_PC = 200                    # PCENTRED: degree above which a vertex grows deterministically (smoke test: 200 and 2000 agree)
NN_PC = 64                       # PCENTRED: capacity in vertices followed per parent
TSTOP_PC = 40.0                  # PCENTRED: largest age of the parent followed
LAST_PC = {}                     # PCENTRED: diagnostics of the last step (EL, overflows, Test S tallies)
OUTSIDE_P = True                 # sixth amendment (24 Nov 14:24 BRT, Test T): credit outside R by degree band from the followed parents
BANDS = ((11, 20), (21, 50), (51, 200))   # degree bands with lambda_B from the parents; above: lambda_big from the total
T2_TESTD = (0.2438, 0.2412, 0.2302)       # Test D credit per non-leaf neighbour in the bands (T2)

try:
    import numba
    from numba import njit
except ImportError:
    print("numba is required: install it in this Python environment (conda install numba).")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import credit_sim as cs           # ratio-equation solver: solve_q(mu, c), mu_of(c), credit_rates; c[k, l]

LMAX, KMAX = cs.LMAX, cs.KMAX


@njit(cache=True)
def _wexp(a0, a1, mu):
    return (np.exp(-mu * a0) - np.exp(-mu * a1)) / mu


@njit(cache=True)
def _wexp2(a0, a1, mu, t0):
    r = mu + 1.0
    return np.exp(t0) * (np.exp(-r * a0) - np.exp(-r * a1)) / r


@njit(cache=True)
def run(cum, sl, sk, ctab, lmax, kmax, lam_out, m, mu, nind, amax, dr, dbig, seed, pexact):
    np.random.seed(seed)
    W = np.zeros((dr + 1, dr + 1))
    WC = np.zeros((dr + 1, dr + 1))
    ncmax = dr
    lc = np.zeros(ncmax, np.int64); kc = np.zeros(ncmax, np.int64)
    cdet = np.zeros(ncmax, np.bool_); dc0 = np.zeros(ncmax); tc0 = np.zeros(ncmax)
    for i in range(nind):
        x = np.random.random() * cum[-1]
        lo, hi = 0, cum.shape[0] - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        lp = sl[lo] - 1; kp = sk[lo] + 1
        pdet = False; dp0 = 0.0; tp0 = 0.0
        lv = 1; kv = 1; nc = 0
        a = 0.0
        while a < amax:
            dv = lv + kv
            # inverse degrees now
            if pdet:
                ip = np.exp(-(a - tp0)) / dp0
            else:
                ip = 1.0 / (lp + kp)
            sic = 0.0
            for j in range(nc):
                if cdet[j]:
                    sic += np.exp(-(a - tc0[j])) / dc0[j]
                else:
                    sic += 1.0 / (lc[j] + kc[j])
            rpg = 0.0; rpb = 0.0
            if not pdet:
                if lp <= lmax and kp <= kmax:
                    cp = ctab[kp, lp]
                else:
                    cp = kp * lam_out
                if pexact:
                    cp = cp - m
                    if cp < 0.0:
                        cp = 0.0
                    cp += 1.0 / dv
                rpg = lp + cp
                rpb = lp / (lp + kp)
            rcs = 0.0
            for j in range(nc):
                if not cdet[j]:
                    rcs += lc[j] + 1.0 / dv + (kc[j] - 1) * m + lc[j] / (lc[j] + kc[j])
            rvg = lv + ip + sic
            rvb = lv / dv
            tot = rpg + rpb + rcs + rvg + rvb
            dt = -np.log(1.0 - np.random.random()) / tot
            a1 = a + dt
            stop = False
            if a1 > amax:
                a1 = amax; stop = True
            # occupation and weighted credit of v in its state over [a, a1]
            wo = _wexp(a, a1, mu)
            wcr = 0.0
            if pdet:
                wcr += _wexp2(a, a1, mu, tp0) / dp0
            else:
                wcr += wo / (lp + kp)
            for j in range(nc):
                if cdet[j]:
                    wcr += _wexp2(a, a1, mu, tc0[j]) / dc0[j]
                else:
                    wcr += wo / (lc[j] + kc[j])
            W[lv, kv] += wo
            WC[lv, kv] += wcr
            a = a1
            if stop:
                break
            u = np.random.random() * tot
            if u < rpg:
                lp += 1
                if lp + kp >= dbig:
                    pdet = True; dp0 = lp + kp; tp0 = a
                continue
            u -= rpg
            if u < rpb:
                lp -= 1; kp += 1
                continue
            u -= rpb
            if u < rcs:
                for j in range(nc):
                    if cdet[j]:
                        continue
                    g = lc[j] + 1.0 / dv + (kc[j] - 1) * m
                    bth = lc[j] / (lc[j] + kc[j])
                    if u < g:
                        lc[j] += 1
                        if lc[j] + kc[j] >= dbig:
                            cdet[j] = True; dc0[j] = lc[j] + kc[j]; tc0[j] = a
                        break
                    u -= g
                    if u < bth:
                        lc[j] -= 1; kc[j] += 1
                        break
                    u -= bth
                continue
            u -= rcs
            if u < rvg:
                # thinning: actual gain rate of v at time a
                if pdet:
                    ipa = np.exp(-(a - tp0)) / dp0
                else:
                    ipa = 1.0 / (lp + kp)
                sa = 0.0
                for j in range(nc):
                    if cdet[j]:
                        sa += np.exp(-(a - tc0[j])) / dc0[j]
                    else:
                        sa += 1.0 / (lc[j] + kc[j])
                if np.random.random() * rvg < lv + ipa + sa:
                    lv += 1
                    if lv + kv > dr:
                        break
                continue
            # birth of v
            lv -= 1; kv += 1
            lc[nc] = 1; kc[nc] = 1; cdet[nc] = False; dc0[nc] = 0.0; tc0[nc] = 0.0
            nc += 1
    return W, WC


@njit(cache=True)
def gcred(j, kcj, a, gb, m, useB, invd, aw, gmax):
    """Credit a child receives from its own non-leaf children: (kcj - 1) m, or (B) the measured mean 1/d at each
    grandchild's age (first gmax grandchildren; any further ones at m)."""
    ng = kcj - 1
    if not useB:
        return ng * m
    s = 0.0
    nb = invd.shape[0]
    for g in range(ng):
        if g < gmax:
            b = int((a - gb[j, g]) / aw)
            if b >= nb:
                b = nb - 1
            s += invd[b]
        else:
            s += m
    return s


@njit(cache=True)
def run3(cum, sl, sk, ctab, lmax, kmax, lam_out, m, mu, nind, amax, dr, dbig, seed, pexact,
         useA, ctabA, agecum, aw, useB, invd, gmax, dbigv, useM, mtab):
    """As run (nb_closure) with: (A) the parent's credit by state and age from the model's own table ctabA[bin, l, k]
    for parent states in R, its age at v's birth drawn from agecum[l, k, :]; (B) the credit a child receives from
    each of its children from invd[age]; outputs by age bin; and v followed beyond R by its own chain (state credit)
    to collect the mean 1/d by age."""
    np.random.seed(seed)
    nbin = int(round(amax / aw))
    W = np.zeros((dr + 1, dr + 1))
    WC = np.zeros((dr + 1, dr + 1))
    WA = np.zeros((nbin, dr + 1, dr + 1))
    WCA = np.zeros((nbin, dr + 1, dr + 1))
    WCAC = np.zeros((nbin, dr + 1, dr + 1))       # child part of the credit
    NA = np.zeros((nbin, dr + 1, dr + 1))
    IV = np.zeros(nbin)
    ncmax = dr
    lc = np.zeros(ncmax, np.int64); kc = np.zeros(ncmax, np.int64)
    cdet = np.zeros(ncmax, np.bool_); dc0 = np.zeros(ncmax); tc0 = np.zeros(ncmax)
    gb = np.zeros((ncmax, gmax))
    nag = agecum.shape[2]
    for i in range(nind):
        x = np.random.random() * cum[-1]
        lo, hi = 0, cum.shape[0] - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        lp = sl[lo] - 1; kp = sk[lo] + 1
        ap0 = 0.0
        if useA and sl[lo] + sk[lo] <= dr and agecum[sl[lo], sk[lo], nag - 1] > 0:
            xa = np.random.random() * agecum[sl[lo], sk[lo], nag - 1]
            bb = 0
            while agecum[sl[lo], sk[lo], bb] <= xa:
                bb += 1
            ap0 = (bb + np.random.random()) * aw
        pdet = False; dp0 = 0.0; tp0 = 0.0
        lv = 1; kv = 1; nc = 0
        a = 0.0
        while a < amax:
            dv = lv + kv
            # inverse degrees now
            if pdet:
                ip = np.exp(-(a - tp0)) / dp0
            else:
                ip = 1.0 / (lp + kp)
            sic = 0.0
            for j in range(nc):
                if cdet[j]:
                    sic += np.exp(-(a - tc0[j])) / dc0[j]
                else:
                    sic += 1.0 / (lc[j] + kc[j])
            rpg = 0.0; rpb = 0.0
            if not pdet:
                if useA and lp + kp <= dr:
                    bp = int((ap0 + a) / aw)
                    if bp >= nag:
                        bp = nag - 1
                    cp = ctabA[bp, lp, kp]
                elif lp <= lmax and kp <= kmax:
                    cp = ctab[kp, lp]
                else:
                    cp = kp * lam_out
                if pexact:
                    msub = m
                    if useA and useM and lp + kp <= dr:
                        msub = mtab[bp, lp, kp]
                    cp = cp - msub
                    if cp < 0.0:
                        cp = 0.0
                    cp += 1.0 / dv
                rpg = lp + cp
                rpb = lp / (lp + kp)
            rcs = 0.0
            for j in range(nc):
                if not cdet[j]:
                    rcs += lc[j] + 1.0 / dv + gcred(j, kc[j], a, gb, m, useB, invd, aw, gmax) + lc[j] / (lc[j] + kc[j])
            rvg = lv + ip + sic
            rvb = lv / dv
            tot = rpg + rpb + rcs + rvg + rvb
            dt = -np.log(1.0 - np.random.random()) / tot
            a1 = a + dt
            stop = False
            if a1 > amax:
                a1 = amax; stop = True
            # occupation and weighted credit of v in its state over [a, a1]
            wo = _wexp(a, a1, mu)
            wcr = 0.0
            if pdet:
                wcr += _wexp2(a, a1, mu, tp0) / dp0
            else:
                wcr += wo / (lp + kp)
            wpar = wcr
            for j in range(nc):
                if cdet[j]:
                    wcr += _wexp2(a, a1, mu, tc0[j]) / dc0[j]
                else:
                    wcr += wo / (lc[j] + kc[j])
            W[lv, kv] += wo
            WC[lv, kv] += wcr
            # age tables: the interval [a, a1] split at the bin edges (fixed 24 Nov 11:36, E26: it was all put in the bin of a)
            s0 = a
            while s0 < a1:
                ba = int(s0 / aw)
                if ba >= nbin:
                    break
                e = (ba + 1) * aw
                s1 = a1 if a1 < e else e
                wos = _wexp(s0, s1, mu)
                if pdet:
                    wps = _wexp2(s0, s1, mu, tp0) / dp0
                else:
                    wps = wos / (lp + kp)
                wcs = wps
                for j in range(nc):
                    if cdet[j]:
                        wcs += _wexp2(s0, s1, mu, tc0[j]) / dc0[j]
                    else:
                        wcs += wos / (lc[j] + kc[j])
                WA[ba, lv, kv] += wos
                WCA[ba, lv, kv] += wcs
                WCAC[ba, lv, kv] += wcs - wps
                NA[ba, lv, kv] += 1.0
                IV[ba] += (s1 - s0) / dv
                s0 = s1
            a = a1
            if stop:
                break
            u = np.random.random() * tot
            if u < rpg:
                lp += 1
                if lp + kp >= dbig:
                    pdet = True; dp0 = lp + kp; tp0 = a
                continue
            u -= rpg
            if u < rpb:
                lp -= 1; kp += 1
                continue
            u -= rpb
            if u < rcs:
                for j in range(nc):
                    if cdet[j]:
                        continue
                    g = lc[j] + 1.0 / dv + gcred(j, kc[j], a, gb, m, useB, invd, aw, gmax)
                    bth = lc[j] / (lc[j] + kc[j])
                    if u < g:
                        lc[j] += 1
                        if lc[j] + kc[j] >= dbig:
                            cdet[j] = True; dc0[j] = lc[j] + kc[j]; tc0[j] = a
                        break
                    u -= g
                    if u < bth:
                        if kc[j] - 1 < gmax:
                            gb[j, kc[j] - 1] = a
                        lc[j] -= 1; kc[j] += 1
                        break
                    u -= bth
                continue
            u -= rcs
            if u < rvg:
                # thinning: actual gain rate of v at time a
                if pdet:
                    ipa = np.exp(-(a - tp0)) / dp0
                else:
                    ipa = 1.0 / (lp + kp)
                sa = 0.0
                for j in range(nc):
                    if cdet[j]:
                        sa += np.exp(-(a - tc0[j])) / dc0[j]
                    else:
                        sa += 1.0 / (lc[j] + kc[j])
                if np.random.random() * rvg < lv + ipa + sa:
                    lv += 1
                    if lv + kv > dr:
                        # v leaves R: follow its degree only, by its own chain with the state credit, for the mean 1/d
                        while a < amax:
                            dvv = lv + kv
                            if dvv >= dbigv:
                                # deterministic growth d(t) = dvv exp(t - a)
                                s0 = a
                                while s0 < amax:
                                    bb2 = int(s0 / aw)
                                    if bb2 >= nbin:
                                        break
                                    e2 = (bb2 + 1) * aw
                                    s1 = amax if amax < e2 else e2
                                    IV[bb2] += (np.exp(-(s0 - a)) - np.exp(-(s1 - a))) / dvv
                                    s0 = s1
                                break
                            if lv <= lmax and kv <= kmax:
                                cv = ctab[kv, lv]
                            else:
                                cv = kv * lam_out
                            gv = lv + cv
                            bv = lv / dvv
                            tv = gv + bv
                            dtt = -np.log(1.0 - np.random.random()) / tv
                            a2 = a + dtt
                            if a2 > amax:
                                a2 = amax
                            s0 = a
                            while s0 < a2:
                                bb2 = int(s0 / aw)
                                if bb2 >= nbin:
                                    break
                                e2 = (bb2 + 1) * aw
                                s1 = a2 if a2 < e2 else e2
                                IV[bb2] += (s1 - s0) / dvv
                                s0 = s1
                            a = a2
                            if a >= amax:
                                break
                            if np.random.random() * tv < gv:
                                lv += 1
                            else:
                                lv -= 1
                                kv += 1
                        break
                continue
            # birth of v
            lv -= 1; kv += 1
            lc[nc] = 1; kc[nc] = 1; cdet[nc] = False; dc0[nc] = 0.0; tc0[nc] = 0.0
            nc += 1
    for b in range(nbin):
        IV[b] /= nind * aw
    return W, WC, WA, WCA, NA, IV, WCAC


@njit(cache=True)
def _deg(i, t, ll, kk, det, d0, t0):
    if det[i]:
        return d0[i] * np.exp(t - t0[i])
    return float(ll[i] + kk[i])


@njit(cache=True)
def _next_full(t, e, mu, afull):
    """Next birth followed in full of a deterministic parent (birth rate 1, kept with probability pi(A)): the time T with
    int_t^T pi = e, or -1 if there is none."""
    if t < afull:
        if afull - t >= e:
            return t + e
        e -= afull - t
        t = afull
    x = np.exp(-mu * (t - afull)) - mu * e
    if x <= 0.0:
        return -1.0
    return afull - np.log(x) / mu


@njit(cache=True)
def run4(cum, sl, sk, ctab, lmax, kmax, lam_out, m, mu, npar, amax, dr, dbig, seed,
         useA, ctabA, agecum, aw, useB, invd, gmax, useM, mtab, afull, nn, tstop):
    """Fifth amendment (PCENTRED, note 24 Nov 13:46): a parent p followed from its creation in (1,1) with its own parent g
    (state from the parent law, table credit) and all its children; children born at p's age A followed in full with
    probability pi(A) = min(1, exp(-mu (A - afull))) and weight exp(-mu A)/pi(A), otherwise in brief. Tallies from the
    children in full, as run3: W, WC, WA, WCA, NA, IV, WCAC; and for Test S: PA (1/d_p), PR (rest credit of p),
    PT (time) by age bin and class of d_p at the birth, for children in (l,1); EL = sum over p of sum_births exp(-mu A);
    WF = sum of the weights of the births followed in full; NOV = capacity overflows."""
    np.random.seed(seed)
    nbin = int(round(amax / aw))
    W = np.zeros((dr + 1, dr + 1)); WC = np.zeros((dr + 1, dr + 1))
    WA = np.zeros((nbin, dr + 1, dr + 1)); WCA = np.zeros((nbin, dr + 1, dr + 1)); WCAC = np.zeros((nbin, dr + 1, dr + 1))
    NA = np.zeros((nbin, dr + 1, dr + 1)); IV = np.zeros(nbin)
    PA = np.zeros((nbin, 9)); PR = np.zeros((nbin, 9)); PT = np.zeros((nbin, 9))
    EL = 0.0; WF = 0.0; NOV = 0
    WP = np.zeros((dbig + 1, dbig + 1)); WCP = np.zeros((dbig + 1, dbig + 1))   # p's occupation and credit by [l, k], d <= dbig
    act = np.zeros(nn, np.bool_); kind = np.zeros(nn, np.int64); ll = np.zeros(nn, np.int64); kk = np.zeros(nn, np.int64)
    par = np.zeros(nn, np.int64); det = np.zeros(nn, np.bool_); d0 = np.zeros(nn); t0 = np.zeros(nn); tb = np.zeros(nn)
    wt = np.zeros(nn); full = np.zeros(nn, np.bool_); inr = np.zeros(nn, np.bool_); cls = np.zeros(nn, np.int64)
    gb = np.zeros((nn, gmax)); gain = np.zeros(nn); brate = np.zeros(nn); dg = np.zeros(nn)
    nag = agecum.shape[2]
    edges = np.array([3, 4, 5, 6, 8, 11, 21, 51])
    for ip_ in range(npar):
        for i in range(nn):
            act[i] = False; det[i] = False; full[i] = False; inr[i] = False
        top = 2
        # g: the parent of p, after the birth of p
        x = np.random.random() * cum[-1]
        lo, hi = 0, cum.shape[0] - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] > x:
                hi = mid
            else:
                lo = mid + 1
        act[0] = True; kind[0] = 0; ll[0] = sl[lo] - 1; kk[0] = sk[lo] + 1; par[0] = -1; tb[0] = 0.0
        ag0 = 0.0
        if useA and sl[lo] + sk[lo] <= dr and agecum[sl[lo], sk[lo], nag - 1] > 0:
            xa = np.random.random() * agecum[sl[lo], sk[lo], nag - 1]
            bb = 0
            while agecum[sl[lo], sk[lo], bb] <= xa:
                bb += 1
            ag0 = (bb + np.random.random()) * aw
        if ll[0] + kk[0] >= dbig:
            det[0] = True; d0[0] = ll[0] + kk[0]; t0[0] = 0.0
        # p
        act[1] = True; kind[1] = 1; ll[1] = 1; kk[1] = 1; par[1] = 0; tb[1] = 0.0
        t = 0.0
        pdet_births = False; tnext = -1.0
        while True:
            while top > 2 and not act[top - 1]:
                top -= 1
            # degrees now
            for i in range(top):
                if act[i]:
                    dg[i] = _deg(i, t, ll, kk, det, d0, t0)
            # credit sums from followed neighbours (children of p and of children in full; parents)
            nfull = 0
            for i in range(top):
                gain[i] = 0.0; brate[i] = 0.0
            for i in range(top):
                if not act[i] or i == 0:
                    continue
                pi = par[i]
                if pi >= 0 and act[pi]:
                    gain[i] += 1.0 / dg[pi]
                    if kind[pi] == 1 or (kind[pi] == 2 and inr[pi]):
                        gain[pi] += 1.0 / dg[i]
                if full[i]:
                    nfull += 1
            tot = 0.0
            for i in range(top):
                if not act[i] or det[i]:
                    continue
                if i == 0:
                    if useA and ll[0] + kk[0] <= dr:
                        bp = int((ag0 + t) / aw)
                        if bp >= nag:
                            bp = nag - 1
                        cg = ctabA[bp, ll[0], kk[0]]
                        msub = mtab[bp, ll[0], kk[0]] if useM else m
                    elif ll[0] <= lmax and kk[0] <= kmax:
                        cg = ctab[kk[0], ll[0]]; msub = m
                    else:
                        cg = kk[0] * lam_out; msub = m
                    cg -= msub
                    if cg < 0.0:
                        cg = 0.0
                    gain[0] = cg + (1.0 / dg[1] if act[1] else 0.0)
                elif kind[i] == 3 or kind[i] == 4:
                    gain[i] += gcred(i, kk[i], t, gb, m, useB, invd, aw, gmax)
                gain[i] += ll[i]
                brate[i] = ll[i] / dg[i]
                tot += gain[i] + brate[i]
            # stop: nothing in full, and no further births in full
            if nfull == 0 and (det[1] and tnext < 0.0):
                break
            if t >= tstop:
                break
            if tot > 0.0:
                dt = -np.log(1.0 - np.random.random()) / tot
            else:
                dt = 1e30
            t1 = t + dt
            sched = False
            if tnext >= 0.0 and tnext < t1:
                t1 = tnext; sched = True
            # sixth amendment: the parent's own occupation (weight exp(-mu t)) and credit by state, before it is deterministic
            if act[1] and not det[1] and ll[1] + kk[1] <= dbig:
                wop = _wexp(t, t1, mu)
                WP[ll[1], kk[1]] += wop
                WCP[ll[1], kk[1]] += wop * (gain[1] - ll[1])
            # tallies over [t, t1] for the children in full
            for j in range(top):
                if not act[j] or not full[j]:
                    continue
                a0 = t - tb[j]; a1 = t1 - tb[j]
                if a1 > amax:
                    a1 = amax
                if a0 >= a1:
                    continue
                s0 = a0
                while s0 < a1:
                    ba = int(s0 / aw)
                    if ba >= nbin:
                        break
                    e = (ba + 1) * aw
                    s1 = a1 if a1 < e else e
                    # 1/d of v (IV), unweighted in age
                    if det[j]:
                        IV[ba] += wt[j] * (np.exp(-(s0 + tb[j] - t0[j])) - np.exp(-(s1 + tb[j] - t0[j]))) / d0[j]
                    else:
                        IV[ba] += wt[j] * (s1 - s0) / dg[j]
                    if inr[j]:
                        lv = ll[j]; kv = kk[j]
                        wos = wt[j] * _wexp(s0, s1, mu)
                        if det[1]:
                            wps = wt[j] * _wexp2(s0, s1, mu, t0[1] - tb[j]) / d0[1]
                        else:
                            wps = wos / dg[1]
                        wcs = wps
                        for i in range(top):
                            if act[i] and par[i] == j:
                                if det[i]:
                                    wcs += wt[j] * _wexp2(s0, s1, mu, t0[i] - tb[j]) / d0[i]
                                else:
                                    wcs += wos / dg[i]
                        W[lv, kv] += wos; WC[lv, kv] += wcs
                        WA[ba, lv, kv] += wos; WCA[ba, lv, kv] += wcs; WCAC[ba, lv, kv] += wcs - wps
                        NA[ba, lv, kv] += 1.0
                        if kv == 1:
                            # Test S: parent's 1/d and rest credit (from g and the other children), plain time
                            rest = 0.0
                            if act[0]:
                                rest += 1.0 / dg[0]
                            for i in range(top):
                                if act[i] and par[i] == 1 and i != j:
                                    rest += 1.0 / dg[i]
                            PT[ba, cls[j]] += wt[j] * (s1 - s0)
                            PA[ba, cls[j]] += wt[j] * (s1 - s0) / dg[1]
                            PR[ba, cls[j]] += wt[j] * (s1 - s0) * rest
                    s0 = s1
            t = t1
            # children in full that reach amax
            for j in range(top):
                if act[j] and full[j] and t - tb[j] >= amax:
                    full[j] = False
                    if det[1]:
                        act[j] = False
                        for i in range(top):
                            if act[i] and par[i] == j:
                                act[i] = False
                    elif kind[j] == 2:
                        kind[j] = 3; inr[j] = False
                        for i in range(top):
                            if act[i] and par[i] == j:
                                act[i] = False
            if sched:
                # a birth in full of the deterministic p
                slot = -1
                for i in range(2, nn):
                    if not act[i]:
                        slot = i; break
                if slot < 0:
                    NOV += 1
                else:
                    A = t
                    pr = 1.0 if A < afull else np.exp(-mu * (A - afull))
                    if slot + 1 > top:
                        top = slot + 1
                    act[slot] = True; kind[slot] = 2; ll[slot] = 1; kk[slot] = 1; par[slot] = 1; det[slot] = False
                    tb[slot] = t; wt[slot] = np.exp(-mu * A) / pr; full[slot] = True; inr[slot] = True
                    WF += wt[slot]
                    cls[slot] = 8
                tnext = _next_full(t, -np.log(1.0 - np.random.random()), mu, afull)
                continue
            if t1 >= 1e29:
                break
            # the event
            u = np.random.random() * tot
            ev = -1; isg = True
            for i in range(top):
                if not act[i] or det[i]:
                    continue
                if u < gain[i]:
                    ev = i; isg = True; break
                u -= gain[i]
                if u < brate[i]:
                    ev = i; isg = False; break
                u -= brate[i]
            if ev < 0:
                continue
            i = ev
            if isg:
                # thinning: gain rate now (degrees of deterministic neighbours have grown)
                g = 0.0
                if i == 0:
                    g = gain[0] - (1.0 / dg[1] if act[1] else 0.0) + (1.0 / _deg(1, t, ll, kk, det, d0, t0) if act[1] else 0.0)
                else:
                    g = ll[i]
                    pi = par[i]
                    if pi >= 0 and act[pi]:
                        g += 1.0 / _deg(pi, t, ll, kk, det, d0, t0)
                    if kind[i] == 1 or (kind[i] == 2 and inr[i]):
                        for q in range(top):
                            if act[q] and par[q] == i:
                                g += 1.0 / _deg(q, t, ll, kk, det, d0, t0)
                    else:
                        g += gcred(i, kk[i], t, gb, m, useB, invd, aw, gmax)
                if np.random.random() * gain[i] >= g:
                    continue
                ll[i] += 1
                if ll[i] + kk[i] >= dbig:
                    det[i] = True; d0[i] = ll[i] + kk[i]; t0[i] = t
                    if i == 1:
                        # p deterministic: g and the children in brief no longer matter
                        act[0] = False
                        for q in range(2, nn):
                            if act[q] and kind[q] == 3 and not full[q]:
                                act[q] = False
                        EL += np.exp(-mu * t) / mu
                        tnext = _next_full(t, -np.log(1.0 - np.random.random()), mu, afull)
                if kind[i] == 2 and inr[i] and ll[i] + kk[i] > dr:
                    inr[i] = False; kind[i] = 3
                    for q in range(top):
                        if act[q] and par[q] == i:
                            act[q] = False
            else:
                ll[i] -= 1; kk[i] += 1
                if i == 0:
                    continue
                if kk[i] - 2 < gmax:
                    gb[i, kk[i] - 2] = t
                if i == 1 or (kind[i] == 2 and inr[i]):
                    slot = -1
                    for q in range(2, nn):
                        if not act[q]:
                            slot = q; break
                    if slot < 0:
                        NOV += 1
                        continue
                    if slot + 1 > top:
                        top = slot + 1
                    act[slot] = True; ll[slot] = 1; kk[slot] = 1; par[slot] = i; det[slot] = False
                    tb[slot] = t; full[slot] = False; inr[slot] = False
                    if i == 1:
                        A = t
                        EL += np.exp(-mu * A)
                        pr = 1.0 if A < afull else np.exp(-mu * (A - afull))
                        dpb = dg[1]
                        c = 0
                        for e_ in range(8):
                            if dpb >= edges[e_]:
                                c = e_ + 1
                        cls[slot] = c
                        if np.random.random() < pr:
                            kind[slot] = 2; full[slot] = True; inr[slot] = True; wt[slot] = np.exp(-mu * A) / pr
                            WF += wt[slot]
                        else:
                            kind[slot] = 3
                    else:
                        kind[slot] = 4
    for b in range(nbin):
        IV[b] /= WF * aw
    return W, WC, WA, WCA, NA, IV, WCAC, PA, PR, PT, EL / npar, WF, NOV, WP, WCP


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


def tail_stats(q, mu):
    """Tail beyond l = LMAX to leading order in k/l (note, Observation 'Tail to leading order'): mass T = 1 - sum q,
    q(l) = T mu L^mu l^-(1+mu), k = kbar_L + ln(l/L). Returns T, kbar_L, sum k q, sum q/d, sum k q/d over the tail,
    and q(L) L / mu (check P1)."""
    T = 1.0 - float(q.sum())
    col = q[:, LMAX]
    kL = float((col * np.arange(KMAX + 1)).sum() / col.sum())
    a = T * mu / ((1.0 + mu) * LMAX)
    return T, kL, T * (kL + 1.0 / mu), a, a * (kL + 1.0 / (1.0 + mu)), float(col.sum()) * LMAX / mu


def parent_law(q, mu=None):
    """States (l, k) with weights q_s l/d for the parent at a birth; q has shape (KMAX+1, LMAX+1).
    With mu given (TAIL): the tail block beyond LMAX on a geometric grid, mass T in total."""
    K, L = np.meshgrid(np.arange(KMAX + 1), np.arange(LMAX + 1), indexing="ij")
    D = K + L
    w = np.where((L >= 1) & (D > 0), q * L / np.maximum(D, 1), 0.0)
    sel = w > 0
    wl, ll, kl = w[sel], L[sel].astype(np.int64), K[sel].astype(np.int64)
    if mu is not None:
        T, kL = tail_stats(q, mu)[:2]
        n = TAIL_DEC * TAIL_PER_DEC
        edges = LMAX * 10.0 ** (np.arange(n + 1) / TAIL_PER_DEC)
        F = (LMAX / edges) ** mu                       # mass beyond each edge, as a fraction of T
        mass = T * (F[:-1] - F[1:])                    # mass of each cell
        mass[-1] += T * F[-1]                          # the last cell carries the rest
        lt = np.sqrt(edges[:-1] * edges[1:])
        kt = np.maximum(1, np.rint(kL + np.log(lt / LMAX)))
        wt = mass * lt / (lt + kt)
        wl = np.concatenate([wl, wt]); ll = np.concatenate([ll, lt.astype(np.int64)]); kl = np.concatenate([kl, kt.astype(np.int64)])
    return np.cumsum(wl), ll, kl


def mean_inv_d(q, mu=None):
    """m = sum_s q_s/d. Without mu: over the table, divided by its mass (as before TAIL). With mu: table plus tail."""
    K, L = np.meshgrid(np.arange(KMAX + 1), np.arange(LMAX + 1), indexing="ij")
    D = K + L
    s = float(np.where(D > 0, q / np.maximum(D, 1), 0.0).sum())
    if mu is None:
        return s / float(q.sum())
    return s + tail_stats(q, mu)[3]


def mu_full(c):
    """Root of 1 - mu = sum_s q_s k/d with the tail term (TAIL); as cs.mu_of otherwise."""
    if not TAIL:
        return cs.mu_of(c)
    l = np.arange(0, LMAX + 1, dtype=float)
    K = np.arange(0, KMAX + 1, dtype=float)[:, None]
    d = l[None, :] + K
    with np.errstate(divide='ignore', invalid='ignore'):
        nbk = np.where(d > 0, K / d, 0.0)
    lo, hi = 1e-6, 1.0
    while hi - lo > cs.BISECT_TOL:
        mid = 0.5 * (lo + hi)
        q = cs.solve_q(mid, c)
        g = 1.0 - float((q * nbk).sum()) - tail_stats(q, mid)[4] - mid
        if g > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def region_states():
    return [(l, k) for k in range(1, D_R + 1) for l in range(0, D_R + 1 - k) if l + k >= 2]


def step(c, mu, m, seed, nind, pexact=PARENT_EXACT, tail=TAIL):
    """One Monte Carlo step: new credits on R from the current table c, the occupation q at mu, and m."""
    q = cs.solve_q(mu, c)
    cum, sl, sk = parent_law(q, mu if tail else None)
    # lambda outside the table for parents beyond LMAX/KMAX: current mean credit per non-leaf neighbour
    K = np.arange(KMAX + 1, dtype=float)[:, None]
    lam = float((c * q).sum() / (K * q).sum())
    W, WC = run(cum, sl, sk, np.ascontiguousarray(c), LMAX, KMAX, lam, m, mu, nind, AMAX, D_R, DBIG, seed, pexact)
    return W, WC, q


def step_age(c, mu, m, seed, nind, tabs, tail=TAIL):
    """As step, with the age-structured parent and grandchild credits (run3); tabs = (ctabA, agecum, invd) or None
    (first iteration: state credit only). Returns W, WC, q and the new tables."""
    q = cs.solve_q(mu, c)
    cum, sl, sk = parent_law(q, mu if tail else None)
    K = np.arange(KMAX + 1, dtype=float)[:, None]
    lam = float(c[KMAX, LMAX] / KMAX) if tail else float((c * q).sum() / (K * q).sum())   # credit per child outside R
    nbin = int(round(AMAX / AGE_W))
    if tabs is None:
        useA = useB = False
        ctabA = np.zeros((nbin, D_R + 1, D_R + 1)); agecum = np.ones((D_R + 1, D_R + 1, nbin)); invd = np.zeros(nbin)
        mtab = np.zeros((nbin, D_R + 1, D_R + 1))
    else:
        useA = useB = True
        ctabA, agecum, invd, mtab = tabs
    if PCENTRED:
        o = run4(cum, sl, sk, np.ascontiguousarray(c), LMAX, KMAX, lam, m, mu, N_PAR, AMAX, D_R, DBIG_PC, seed,
                 useA, np.ascontiguousarray(ctabA), np.ascontiguousarray(agecum), AGE_W, useB, invd, GMAX, CHILD_SUB_AGE,
                 np.ascontiguousarray(mtab), A_FULL, NN_PC, TSTOP_PC)
        W, WC, WA, WCA, NA, IV, WCAC = o[:7]
        LAST_PC.update(PA=o[7], PR=o[8], PT=o[9], EL=o[10], WF=o[11], NOV=o[12], WP=o[13], WCP=o[14])
    else:
        W, WC, WA, WCA, NA, IV, WCAC = run3(cum, sl, sk, np.ascontiguousarray(c), LMAX, KMAX, lam, m, mu, nind, AMAX, D_R, DBIG,
                                            seed, PARENT_EXACT, useA, np.ascontiguousarray(ctabA), np.ascontiguousarray(agecum),
                                            AGE_W, useB, invd, GMAX, DBIGV, CHILD_SUB_AGE, np.ascontiguousarray(mtab))
    # new tables from this step's output
    newA = np.zeros((nbin, D_R + 1, D_R + 1))
    newM = np.full((nbin, D_R + 1, D_R + 1), m)
    for l in range(D_R + 1):
        for k in range(1, D_R + 1 - l):
            if l + k < 2:
                continue
            cs_state = WC[l, k] / W[l, k] if W[l, k] > 0 else k * lam
            for b in range(nbin):
                if NA[b, l, k] >= CELL_N:
                    newA[b, l, k] = WCA[b, l, k] / WA[b, l, k]
                    if k >= 2:
                        newM[b, l, k] = WCAC[b, l, k] / WA[b, l, k] / (k - 1)
                    continue
                got = False
                for hw in POOL_HW:
                    b0, b1 = max(0, b - hw), min(nbin, b + hw + 1)
                    if NA[b0:b1, l, k].sum() >= CELL_N:
                        newA[b, l, k] = WCA[b0:b1, l, k].sum() / WA[b0:b1, l, k].sum()
                        if k >= 2:
                            newM[b, l, k] = WCAC[b0:b1, l, k].sum() / WA[b0:b1, l, k].sum() / (k - 1)
                        got = True; break
                if not got:
                    newA[b, l, k] = cs_state
    newcum = np.cumsum(WA.transpose(1, 2, 0), axis=2)
    return W, WC, q, (newA, newcum, IV, newM)


def bands_from(WP, WCP):
    """Sixth amendment: credit per non-leaf neighbour of the followed parents in each degree band of BANDS."""
    L_, K_ = np.meshgrid(np.arange(WP.shape[0]), np.arange(WP.shape[1]), indexing="ij")
    D_ = L_ + K_
    out = []
    for lo, hi in BANDS:
        sel = (D_ >= lo) & (D_ <= hi) & (K_ >= 1)
        out.append((lo, hi, float(WCP[sel].sum() / (K_ * WP)[sel].sum())))
    return out


def table_from(cR, mu, q, tail=TAIL, lam_tail=None, bands=None):
    """Credit table: cR on R, k * lambda outside with sum_s c_s q_s = 1 - mu (with TAIL the sum includes the tail,
    credited k * lambda, or k * lam_tail if given: diagnostic Q). With bands (sixth amendment): k * lambda_B in each
    band (lo, hi, lambda_B) outside R, and lambda (returned) for the states above the bands and the tail."""
    K = np.arange(KMAX + 1, dtype=float)[:, None]
    c = np.zeros((KMAX + 1, LMAX + 1))
    inR = np.zeros_like(c, dtype=bool)
    for (l, k), v in cR.items():
        c[k, l] = v; inR[k, l] = True
    if bands is not None:
        Dm = K + np.arange(LMAX + 1, dtype=float)[None, :]
        for lo, hi, lb in bands:
            sel = (~inR) & (Dm >= lo) & (Dm <= hi) & (K >= 1)
            c = np.where(sel, K * lb, c); inR = inR | sel
    rest = (1.0 - mu) - float((c * q)[inR].sum())
    den = float((K * q)[~inR].sum())
    if tail:
        tKq = tail_stats(q, mu)[2]
        if lam_tail is None:
            den += tKq
        else:
            rest -= lam_tail * tKq
    lam = rest / den
    c = np.where(inR, c, K * lam)
    c = np.minimum(c, K / 2)
    return c, lam


def main():
    global OUT
    OUT = dated_path("nb_closure", "txt")
    t0 = time.time()
    log(f"=== run started {datetime.datetime.now().isoformat(timespec='seconds')} D_R={D_R} AMAX={AMAX} DBIG={DBIG} N_IND={N_IND} "
        f"ITERS_MAX={ITERS_MAX} CONV_TOL={CONV_TOL} PARENT_EXACT={PARENT_EXACT} AGE_MODEL={AGE_MODEL} CHILD_SUB_AGE={CHILD_SUB_AGE} TAIL={TAIL} "
        f"PCENTRED={PCENTRED} N_PAR={N_PAR} A_FULL={A_FULL} DBIG_PC={DBIG_PC} OUTSIDE_P={OUTSIDE_P} BANDS={BANDS} numba {numba.__version__} ===")
    log(f"Pre-registered (DistributionFormulas_2026-11-23.tex, section 'A closure from the neighbourhood of each vertex'): "
        f"L2 credit of R-states with q >= {L2_QMIN} within {100*L2_TOL:.0f}% of Test D in >= {100*L2_FRAC:.0f}%; "
        f"L3 closed mu within {L3_TOL} of {MU_REF} (strong {L3_STRONG})")
    rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    z = np.load(os.path.join(rd, CREDIT_FILE))
    M = z["gM"][:, 1].sum(0); H = z["gH"][:, 1].sum(0)
    cD, _, _ = cs.credit_rates(M, H, CREDIT_MIN)            # Test D table, shape (KMAX+1, LMAX+1)
    states = region_states()

    # L1
    mu_plug = cs.mu_of(cD)
    W, WC, qD = step(cD, mu_plug, L1_M, SEED0 - 1, N_IND, False, False)
    c11, c02 = WC[1, 1] / W[1, 1], WC[0, 2] / W[0, 2]
    okL1a = abs(mu_plug - L1_MUPLUG) <= 1e-5
    log(f"L1: solver with the Test D credit: mu = {mu_plug:.6f} (Test D {L1_MUPLUG}) -> {'PASS' if okL1a else 'FAIL'}")
    log(f"L1: with m = {L1_M}: c(1,1) = {c11:.4f} (three-vertex model {L1_C11}), c(0,2) = {c02:.4f} ({L1_C02}) "
        f"[Monte Carlo s.e. of order 0.001 at N_IND = {N_IND}] ({time.time()-t0:.0f}s)")

    # L2: one step from Test D inputs
    mD = mean_inv_d(qD)
    W, WC, _ = step(cD, mu_plug, mD, SEED0 - 2, N_IND, PARENT_EXACT, False)
    log(f"--- L2: one step from the Test D inputs (m = sum q/d = {mD:.4f}) ---")
    log("   state      q(TestD)  c model   c TestD   diff%")
    qn = qD / qD.sum()
    used = good = 0
    for (l, k) in states:
        if W[l, k] <= 0:
            continue
        cm = WC[l, k] / W[l, k]; ct = cD[k, l]
        dd = cm / ct - 1 if ct > 0 else np.nan
        if qn[k, l] >= L2_QMIN:
            used += 1; good += abs(dd) <= L2_TOL
            log(f"  ({l:2d},{k:2d})   {qn[k,l]:.4f}   {cm:.4f}   {ct:.4f}   {100*dd:+6.1f}")
    fr = good / used if used else 0
    log(f"PRE-REGISTERED L2: {good} of {used} states within {100*L2_TOL:.0f}% ({100*fr:.0f}%) -> {'PASS' if fr >= L2_FRAC else 'FAIL'}")

    # L3: closed iteration from equal sharing
    K = np.arange(KMAX + 1, dtype=float)[:, None]
    c = np.broadcast_to(K * cs.ES_LAM_3000, (KMAX + 1, LMAX + 1)).copy()
    mu = mu_full(c)
    log(f"--- L3: closed iteration from equal sharing (mu = {mu:.6f}) ---")
    hist = []
    cRs = []
    bhist = []; lhist = []
    small = 0
    tabs = None
    for it in range(ITERS_MAX):
        q = cs.solve_q(mu, c)
        m = mean_inv_d(q, mu if TAIL else None)
        if TAIL:
            T, kL, tKq, _, _, Tpow = tail_stats(q, mu)
            log(f"  P1 tail: T = {T:.6f}, q(L) L/mu = {Tpow:.6f} ({100*(T/Tpow-1):+.2f}%), tail mean k = {tKq/T:.3f} "
                f"(Test D overflow row 9.27: {100*(tKq/T/9.27-1):+.1f}%)")
        if AGE_MODEL:
            W, WC, _, tabs = step_age(c, mu, m, SEED0, N_IND, tabs)
        else:
            W, WC, _ = step(c, mu, m, SEED0, N_IND)
        cR = {(l, k): WC[l, k] / W[l, k] for (l, k) in states if W[l, k] > 0}
        if PCENTRED and AGE_MODEL:
            PT_ = LAST_PC["PT"]
            log(f"  S1: overflows {LAST_PC['NOV']}, Euler-Lotka sum {LAST_PC['EL']:.4f}, share of parents d >= 51 at ages 0-1 "
                f"{100*PT_[:4, 8].sum()/PT_[:4].sum():.2f}% (measured 14.28%)")
        bands = None
        if PCENTRED and OUTSIDE_P:
            bands = bands_from(LAST_PC["WP"], LAST_PC["WCP"]); bhist.append([b[2] for b in bands])
            WP_, WCP_ = LAST_PC["WP"], LAST_PC["WCP"]
            log("  T: lambda_B " + ", ".join(f"{lo}-{hi}: {lb:.4f}" for lo, hi, lb in bands) +
                f"; T1 parent's own credit (1,1) {WCP_[1,1]/WP_[1,1]:.4f}, (0,2) {WCP_[0,2]/WP_[0,2]:.4f}")
        c, lam = table_from(cR, mu, q, bands=bands)
        mu = mu_full(c)
        hist.append(mu); cRs.append(cR); lhist.append(lam)
        lam_lab = "lambda above the bands and in the tail" if (PCENTRED and OUTSIDE_P) else "lambda outside R"   # label fixed 24 Nov 14:55
        log(f"  iteration {it+1}: m = {m:.4f}, {lam_lab} = {lam:.5f}, c(1,1) = {cR[(1,1)]:.4f}, c(0,2) = {cR[(0,2)]:.4f}, "
            f"mu = {mu:.6f} ({time.time()-t0:.0f}s)")
        if len(hist) >= 2 and abs(hist[-1] - hist[-2]) < CONV_TOL:
            small += 1
        else:
            small = 0
        nfirst = 1 if AGE_MODEL else 0        # with AGE_MODEL the first iteration uses the state credit only
        if small >= CONV_RUN and len(hist) - nfirst >= AVG_LAST:
            log(f"  converged: |change| < {CONV_TOL} in {CONV_RUN} consecutive iterations")
            break
    else:
        log(f"  not converged after {ITERS_MAX} iterations")
    if AGE_MODEL and tabs is not None:
        tfile = OUT.replace(".txt", "_tables.npz")
        with open(tfile, "xb") as f:
            np.savez_compressed(f, ctabA=tabs[0], agecum=tabs[1], invd=tabs[2], mtab=tabs[3], c=c, mu=mu,
                                q=cs.solve_q(mu, c), m=mean_inv_d(cs.solve_q(mu, c), mu if TAIL else None), AGE_W=AGE_W, D_R=D_R,
                                TAIL=TAIL, PCENTRED=PCENTRED,
                                **({k_: v_ for k_, v_ in LAST_PC.items()} if PCENTRED else {}),
                                CHILD_SUB_AGE=CHILD_SUB_AGE, hist=np.array(hist))
        log(f"  tables of the last iteration written to {os.path.basename(tfile)} (added 24 Nov 11:04)")
    last = np.array(hist[(1 if AGE_MODEL else 0):][-AVG_LAST:])   # never the first iteration with AGE_MODEL (fixed 24 Nov 10:28)
    mu_cl = float(last.mean()); sd = float(last.std(ddof=1))
    spread = []
    q = cs.solve_q(mu, c); m = mean_inv_d(q, mu if TAIL else None)
    for j in range(1, N_SPREAD + 1):
        if AGE_MODEL:
            W, WC, _, _ = step_age(c, mu, m, SEED0 + j, N_IND, tabs)
        else:
            W, WC, _ = step(c, mu, m, SEED0 + j, N_IND)
        cRj = {(l, k): WC[l, k] / W[l, k] for (l, k) in states if W[l, k] > 0}
        bj = bands_from(LAST_PC["WP"], LAST_PC["WCP"]) if (PCENTRED and OUTSIDE_P) else None
        cj, _ = table_from(cRj, mu, q, bands=bj)
        spread.append(mu_full(cj))
    log(f"  one more step from the fixed point with {N_SPREAD} other seeds: mu = " + ", ".join(f"{v:.6f}" for v in spread) +
        f" (s.d. {np.std(spread, ddof=1):.6f}; Monte Carlo error of the closure value)")
    d3 = mu_cl - MU_REF
    v3 = "STRONG PASS" if abs(d3) <= L3_STRONG else ("PASS" if abs(d3) <= L3_TOL else "FAIL")
    log(f"PRE-REGISTERED L3: mu = {mu_cl:.6f} (mean of the last {AVG_LAST} iterations, s.d. {sd:.6f}); mu - {MU_REF} = {d3:+.6f} -> {v3}")
    tol5 = max(3 * float(np.std(spread, ddof=1)) / np.sqrt(len(spread)), 1e-4)
    log(f"PRE-REGISTERED L5: |mu - {MU_REF}| = {abs(d3):.6f} against max(3 s.d./sqrt({len(spread)}), 1e-4) = {tol5:.6f} -> "
        f"{'PASS' if abs(d3) <= tol5 else 'FAIL'}")
    # run U (note, 24 Nov 15:04): the closure value against the reference with the Monte Carlo error alone
    se_u = float(np.std(spread, ddof=1)) / np.sqrt(len(spread)); ms = float(np.mean(spread))
    for nm, val in (("L3 mean", mu_cl), ("mean of the other seeds", ms)):
        zu = (val - MU_REF) / se_u
        log(f"PRE-REGISTERED U: {nm} {val:.6f}, minus {MU_REF} = {val-MU_REF:+.6f}, s.e. {se_u:.6f}, z = {zu:+.2f} -> "
            f"{'gap remains (|z| > 3)' if abs(zu) > 3 else 'no gap resolved (|z| <= 3)'}")

    if PCENTRED and OUTSIDE_P:
        bm = np.array(bhist[(1 if AGE_MODEL else 0):][-AVG_LAST:]).mean(0); lm = float(np.mean(lhist[(1 if AGE_MODEL else 0):][-AVG_LAST:]))
        okT1 = min(lhist) > 0
        dev = [bm[i] / T2_TESTD[i] - 1 for i in range(len(BANDS))]
        log(f"PRE-REGISTERED T1: lambda_big > 0 at every iteration (min {min(lhist):.5f}) -> {'PASS' if okT1 else 'FAIL'}")
        log("PRE-REGISTERED T2: lambda_B (mean of the last iterations) " +
            ", ".join(f"{BANDS[i][0]}-{BANDS[i][1]}: {bm[i]:.4f} (Test D {T2_TESTD[i]}, {100*dev[i]:+.1f}%)" for i in range(len(BANDS))) +
            f" -> {'PASS' if max(abs(x) for x in dev) <= 0.03 else 'FAIL'}; lambda_big {lm:.4f} (Test D 0.2118 for 201-1000, 0.1930 for 1001-3000)")
        bands_last = [(BANDS[i][0], BANDS[i][1], float(bm[i])) for i in range(len(BANDS))]
    else:
        bands_last = None
    if TAIL:
        cQ, lamQ = table_from({(l, k): float(np.mean([cr[(l, k)] for cr in cRs[(1 if AGE_MODEL else 0):][-AVG_LAST:] if (l, k) in cr]))
                               for (l, k) in states if any((l, k) in cr for cr in cRs)}, mu, q, True, TAILD_LAM, bands_last)
        cA, lamA = table_from({(l, k): float(np.mean([cr[(l, k)] for cr in cRs[(1 if AGE_MODEL else 0):][-AVG_LAST:] if (l, k) in cr]))
                               for (l, k) in states if any((l, k) in cr for cr in cRs)}, mu, q, True, None, bands_last)
        log(f"Diagnostic Q (no decision): mean credits on R; tail at k*lambda: lambda = {lamA:.5f}, mu = {mu_full(cA):.6f}; "
            f"tail at k*{TAILD_LAM}: lambda = {lamQ:.5f}, mu = {mu_full(cQ):.6f}")

    # L4
    log("--- L4: closed credit on R (mean of the last iterations) against Test D, states with q >= 0.002 ---")
    log("   state      q(TestD)  c closed  c TestD   diff%")
    for (l, k) in states:
        vals = [cr[(l, k)] for cr in cRs[(1 if AGE_MODEL else 0):][-AVG_LAST:] if (l, k) in cr]
        if not vals or qn[k, l] < 0.002:
            continue
        cm = float(np.mean(vals)); ct = cD[k, l]
        log(f"  ({l:2d},{k:2d})   {qn[k,l]:.4f}   {cm:.4f}   {ct:.4f}   {100*(cm/ct-1):+6.1f}")
    log(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
