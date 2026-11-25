"""Ratio equations for redirection trees in the (leaf degree, non-leaf degree) state.

Each non-leaf node is described by (l, k): l leaf neighbours, k non-leaf neighbours,
degree d = l + k >= 2, k >= 1. q(l,k) = (number of such nodes)/C, C = number of non-leaves,
and C ~ N^mu. All rates below are per unit of 1/N.

  leaf hit (a leaf of the node is picked, redirect to the node):      rate l       -> (l+1, k)
  node picked (rate 1):
     IR : to one of its leaves w.p. l/d  -> node (l-1, k+1), new non-leaf (1,1)
          to a non-leaf neighbour w.p. k/d -> that neighbour gets a new leaf
     DAN: l>=1 -> (l-1, k+1), new non-leaf (1,1);   l=0 -> (1, k)
     PAN: l>=1 -> (l-1, k+1), new non-leaf (1,1);   l=0 -> nothing
  core hit (IR only): rate k*lam -> (l+1, k).
     CLOSURE (IR): the total rate of non-leaf -> non-leaf redirections, sum q k/d, is shared
     equally among the non-leaf neighbour slots, lam = sum(q k/d) / sum(q k).
     This is the only approximation; DAN and PAN need none.

Stationary ratios: mu*q = (in) - (out)*q + mu*[state (1,1)], with mu = sum q*s, s = rate at
which a node creates a new non-leaf (IR: l/d; DAN/PAN: 1[l>=1]).
Since transitions only raise k or raise l within a layer, each k-layer is a first-order
recursion in l, solved exactly in vector form. mu is found by bisection on
g(mu) = sum q(mu)*s - mu; for IR, lam is iterated to a fixed point outside.

mu is determined from the complementary Euler-Lotka form (ClosureTheory.tex, Proposition
"Euler-Lotka equation"): 1 - mu = sum_s q_s * (non-birth share of s), non-birth share = k/d (IR) or
1[l=0] (DAN/PAN). Its truncation error is O(1/LMAX) (Proposition "Truncation bound"), against
O(LMAX^-mu) for mu = sum q*s. For IR, the tail of sum k*q beyond LMAX is added from the discounted
flux into l = LMAX+1 times (k/mu + 1/mu^2) (Remark "Tail of the lambda condition").

No arguments; settings below. Progressive output, also written to a new dated file results/closure_solver_<date_time>.txt.
"""
import numpy as np, time, builtins, os, datetime


def dated_path(stem, ext):
    """New file results/<stem>_<YYYY-MM-DD_HHMMSS>.<ext> next to this script; never an existing file."""
    import os, datetime
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    path, i = f"{base}.{ext}", 1
    while os.path.exists(path):
        path, i = f"{base}_{i}.{ext}", i + 1
    return path

OUTFILE = None   # set in main(): each run writes a new dated file results/closure_solver_<date_time>.txt
_print = builtins.print
def print(*args, **kw):
    """Print to the console and append the same line to OUTFILE."""
    _print(*args, **kw)
    if OUTFILE is not None:
        with open(OUTFILE, "a") as f:
            _print(*args, file=f)

MODELS = ["DAN", "PAN", "IR"]
LMAX_LIST = [300, 1000, 3000, 10000]  # leaf-degree truncations; with the complementary form the error is O(1/LMAX)
KMAX = 100                           # non-leaf-degree truncation (60 vs 120 gave the same mu to 6 digits)
BISECT_TOL = 1e-9
LAM_ITERS = 30                       # IR: outer iterations for lam

def first_order(a, b):
    """Solve q_l = a_l q_{l-1} + b_l (q_{-1} = 0) exactly; a_l = 0 restarts the recursion."""
    n = len(a); q = np.empty(n)
    cuts = [0] + [i for i in np.where(a == 0)[0].tolist() if i > 0] + [n]
    cuts = sorted(set(cuts))
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        aa = a[s0:s1].copy(); aa[0] = 1.0       # q_{s0} = b_{s0} (a_{s0} is 0 or start)
        logP = np.cumsum(np.log(aa))
        P = np.exp(logP - logP[0])
        q[s0:s1] = P * np.cumsum(b[s0:s1] / P)
    return q

def solve_q(model, mu, lam, LMAX):
    l = np.arange(0, LMAX + 1, dtype=float)
    q = np.zeros((KMAX + 1, LMAX + 1))
    prev = np.zeros(LMAX + 2)                    # layer k-1, index l
    for k in range(1, KMAX + 1):
        d = l + k
        if model == "IR":
            s = l / d
            out = l + s + k * lam
            inA = (l - 1) + k * lam               # from (l-1, k)
            lp1 = l + 1
            s_prev = lp1 / (lp1 + (k - 1)) if k > 1 else np.zeros_like(l)
        else:
            s = (l >= 1).astype(float)
            direct = 1.0 if model == "DAN" else 0.0
            out = l + s + direct * (l == 0)
            inA = (l - 1) + direct * (l == 1)     # from (l-1, k); DAN: (0,k)->(1,k)
            s_prev = np.ones_like(l)              # (l+1,k-1) with l+1>=1 always creates
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

def stats(model, q, LMAX, mu, lam):
    """Return (J, Kq, mass): J = sum q * non-birth share; Kq = sum k q with the IR tail beyond LMAX."""
    l = np.arange(0, LMAX + 1, dtype=float)
    K = np.arange(0, KMAX + 1, dtype=float)[:, None]
    d = l[None, :] + K
    with np.errstate(divide='ignore', invalid='ignore'):
        if model == "IR":
            nb = np.where(d > 0, K / d, 0.0)
        else:
            nb = (l[None, :] == 0).astype(float) * (K >= 1)
    J = float((q * nb).sum())
    Kq = float((q * K).sum())
    if model == "IR":
        kk = np.arange(0, KMAX + 1, dtype=float)
        flux = (LMAX + lam * kk) * q[:, LMAX] / mu            # discounted flux into l = LMAX+1 (R = q/mu)
        Kq += mu * float((flux * (kk / mu + 1.0 / mu**2)).sum())
    return J, Kq, float(q.sum())

def mu_for(model, lam, LMAX):
    lo, hi = 1e-6, 1.0
    while hi - lo > BISECT_TOL:
        mid = 0.5 * (lo + hi)
        q = solve_q(model, mid, lam, LMAX)
        g = 1.0 - stats(model, q, LMAX, mid, lam)[0] - mid
        if g > 0: lo = mid
        else: hi = mid
    mu = 0.5 * (lo + hi)
    return mu, solve_q(model, mu, lam, LMAX)

def main():
    global OUTFILE
    if OUTFILE is None:
        OUTFILE = dated_path("closure_solver", "txt")
    t0 = time.time()
    print(f"=== run started {datetime.datetime.now().isoformat(timespec='seconds')} "
          f"LMAX_LIST={LMAX_LIST} KMAX={KMAX} ===")
    print("Reference (Hartle-Krapivsky-Redner-Zhang 2026): DAN 0.771192, PAN 0.549735; "
          "IR proven bracket (Uzay) [0.5650, 0.5665]", flush=True)
    for LMAX in LMAX_LIST:
        for model in MODELS:
            if model == "IR":
                lam = 0.2156                  # start near the fixed point found in tests
                for it in range(LAM_ITERS):
                    mu, q = mu_for(model, lam, LMAX)
                    J, Kq, mass = stats(model, q, LMAX, mu, lam)
                    new = (1.0 - mu) / Kq
                    print(f"  IR LMAX={LMAX} it={it} lam={lam:.8f} mu={mu:.8f} ({time.time()-t0:.0f}s)", flush=True)
                    if abs(new - lam) < 1e-9: break
                    lam = new
                print(f"IR  LMAX={LMAX} KMAX={KMAX}: mu={mu:.8f} lam={lam:.8f} captured mass={mass:.6f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            else:
                mu, q = mu_for(model, 0.0, LMAX)
                print(f"{model} LMAX={LMAX} KMAX={KMAX}: mu={mu:.8f}  ({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
