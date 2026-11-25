"""Reduction S1 and S2 (24 Nov 21:0x BRT, cloud). A cavity solution at moderate size (records and output vertices
NREC per generation, NGEN generations, one replica); then C(l,k,a) by state and age bin, split into the parent's and
the children's parts (output_age, NBATCH batches), saved to results/red_s1_<date_time>.npz. S2: the single-vertex
forward equation with this C (states of degree >= 50: k * lambda of the solution) gives mu by Euler-Lotka, against
the cavity's (E) and (R) of the same solution."""
import os, time, datetime, numpy as np
import cavity as C, nb_closure as nb, credit_sim as cs
import builtins, datetime, os as _os
_RD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results"); _os.makedirs(_RD, exist_ok=True)
_OUT = open(_os.path.join(_RD, "red_s1_log_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"), "w")
def print(*a, **k):
    """Prints and writes the same line to the dated result file (added 25 Nov 2026, 12:16 BRT)."""
    k.pop("flush", None); builtins.print(*a, **k, flush=True); builtins.print(*a, **k, file=_OUT, flush=True)
NREC, NGEN, NBATCH, KEYMAX, MAXC, TCAP, AW = 40000, 4, 4, 11, 64, 24.0, 0.25
DMAX, KM, DA = 400, 60, 0.002
HERE = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(HERE, "results"); os.makedirs(RD, exist_ok=True)
stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
z = np.load(C.TABLES if os.path.exists(C.TABLES) else os.path.join(RD, C.TABLES)); aw = float(z['AGE_W']); DR = int(z['D_R'])
rng = np.random.default_rng(7); sd = 777
exec(open(os.path.join(HERE, 'red_common.py')).read())
mu = float(z['mu']); c_prev = z['c']
cum, sl, sk = nb.parent_law(z['q'], mu)
t0 = time.time()
chp = C.gen0_children(NREC, sd + 1, cum, sl, sk, z['ctabA'], z['mtab'], z['c'], aw, DR, C.H, NREC * C.DSTAR)[:7]
order, keyoff = C.key_index(chp[0], KEYMAX)
pp, lwp = C.gen_parents(NREC, sd + 2, mu, C.H, TCAP, MAXC, 0, cum, sl, sk, *G0(), *chp, order, keyoff, KEYMAX)
pp, _ = resample_par(rng, pp, lwp)
for gen in range(1, NGEN + 1):
    cum, sl, sk = nb.parent_law(cs.solve_q(mu, c_prev), mu)
    res = C.gen_children(NREC, sd + 10 * gen + 3, cum, sl, sk, C.H, MAXC, *chp, order, keyoff, KEYMAX)
    chp, _ = resample_ch(rng, res[:7], res[7]); order, keyoff = C.key_index(chp[0], KEYMAX)
    pp2, lwp = C.gen_parents(NREC, sd + 10 * gen + 4, mu, C.H, TCAP, MAXC, 1, cum, sl, sk, *pp, *chp, order, keyoff, KEYMAX)
    pp, _ = resample_par(rng, pp2, lwp)
    Wq = WCq = Wa = PIa = CHa = 0.0; acc = np.zeros(5)
    for bt in range(NBATCH):
        lws, births, tsv, tI, tdI, Wb, WCb, Wab, PIb, CHb = C.output_age(NREC, sd + 10 * gen + 100 * bt + 5, mu, C.H, TCAP, MAXC, AW,
                                                                         *pp, *chp, order, keyoff, KEYMAX)
        Wq = Wq + Wb; WCq = WCq + WCb; Wa = Wa + Wab; PIa = PIa + PIb; CHa = CHa + CHb
        acc += el_sums(lws, births, tsv, tI, tdI, mu)
    mE = el_root(acc, mu)
    mR, cR, lam, c_new = ratio(Wq, WCq, mu, c_prev)
    print(f"generation {gen} ({time.time()-t0:.0f}s): mu (E) {mE:.6f}, (R) {mR:.6f}; lambda {lam:.4f}", flush=True)
    mu_used = mu; mu = mR; c_prev = c_new
np.savez(os.path.join(RD, f"red_s1_{stamp}.npz"), Wa=Wa, PIa=PIa, CHa=CHa, lam=lam, muE=mE, muR=mR, mu_used=mu_used, AW=AW, DSTAR=C.DSTAR)
# S2: forward equation with C(l,k,a) = (PIa + CHa)/Wa, fallback by state over ages, k*lambda from degree DSTAR
D = C.DSTAR; nab = Wa.shape[0]
Ctab = np.where(Wa > 0, (PIa + CHa) / np.maximum(Wa, 1e-300), np.nan)
Cst = (PIa + CHa).sum(0) / np.maximum(Wa.sum(0), 1e-300)
L = np.arange(DMAX + 1)[:, None] + 0 * np.arange(KM + 1)[None, :]; K = 0 * L + np.arange(KM + 1)[None, :]; Dg = L + K
ok = (Dg <= DMAX) & (Dg >= 2) & (K >= 1)
br = np.where(ok, L / np.maximum(Dg, 1), 0.0)
Cbig = np.where(ok, K * lam, 0.0)
def fwd_mu(Cfun):
    P = np.zeros(Dg.shape); P[1, 1] = 1.0; a = 0.0; beta = []; esc = 0.0
    for st in range(int(round(C.H / DA))):
        Cm = Cfun(a)
        g = np.where(ok, L + Cm, 0.0); gf = g * P; bf = br * P
        lost = gf[Dg == DMAX].sum()
        beta.append((bf.sum() + esc) )
        gin = np.zeros_like(P); gin[1:, :] = gf[:-1, :]; bin_ = np.zeros_like(P); bin_[:-1, 1:] = bf[1:, :-1]
        P = P + DA * (np.where(ok, gin, 0.0) + bin_ - gf - bf); esc += lost * DA; a += DA
    beta = np.array(beta); ages = (np.arange(beta.size) + 0.5) * DA
    def f(m): return (np.exp(-m * ages) * beta).sum() * DA + np.exp(-m * C.H) * beta[-1] / m - 1.0
    lo, hi = 0.4, 0.8
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
def C_cav(a):
    b = min(int(a / AW), nab - 1)
    Cm = Cbig.copy()
    blk = Ctab[b]; blk = np.where(np.isnan(blk), Cst, blk)
    m = ok[:D, :D] & (Dg[:D, :D] < D)
    Cm[:D, :D] = np.where(m, np.nan_to_num(blk[:D, :D]), Cm[:D, :D])
    return Cm
t1 = time.time(); mF = fwd_mu(C_cav)
print(f"S2: forward equation with the cavity's C(l,k,a): mu {mF:.6f}  (cavity (E) {mE:.6f}, (R) {mR:.6f}) ({time.time()-t1:.0f}s)")
