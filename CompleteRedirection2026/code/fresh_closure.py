"""Fresh-child closure for complete redirection (test of the conjectured sign of the error).

Individual chain on (l, k, j): l leaf neighbours, k non-leaf neighbours, j of which are FRESH,
i.e. in state (1,1) (degree 2, one leaf, single non-leaf neighbour = this vertex).
Exact facts used (Lemma 4): a fresh neighbour u sends this vertex credit 1/2 (it is picked at
rate 1 per N and redirects here with probability 1/2); u stops being fresh at rate
1 (its leaf is picked) + 1/2 (it buds) + 1/d (this vertex redirects into it).
Approximation: credit from each NON-fresh non-leaf neighbour is the average lam' of 1/d over
non-fresh non-leaves, weighted by their non-leaf degree (equal sharing among non-fresh slots):
   lam' = (sum q k/d - q_F/2) / (sum k q - q_F),   q_F = fraction of non-leaves that are fresh.
Rates (time ln N):
   (l,k,j) -> (l+1,k,j)       at  l + j/2 + (k-j) lam'
   (l,k,j) -> (l-1,k+1,j+1)   at  l/(l+k)              [birth of a fresh vertex (1,1,0)]
   (l,k,j) -> (l,k,j-1)       at  j (3/2 + 1/(l+k))
mu from the complementary Euler-Lotka form 1 - mu = sum_s q_s k/d (ClosureTheory.tex).
With FRESH = False the fresh credit is replaced by lam (equal sharing over all slots), which must
reproduce the equal-sharing value 0.568842 (regression check).
No arguments; settings below; output also written to results/fresh_closure_<date_time>.txt.
"""
import numpy as np, time, os, datetime, builtins

LMAX_LIST = [300, 1000, 3000, 10000]   # same truncations as the equal-sharing run of closure_solver.py
KMAX = 100                              # same non-leaf-degree cut-off as closure_solver.py
JMAX = 10
BISECT_TOL = 1e-10
LAM_ITERS = 40

def dated_path(stem, ext):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{stem}_{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    path, i = f"{base}.{ext}", 1
    while os.path.exists(path):
        path, i = f"{base}_{i}.{ext}", i + 1
    return path

OUTFILE = None
_print = builtins.print
def print(*args, **kw):
    _print(*args, **kw)
    if OUTFILE is not None:
        with open(OUTFILE, "a") as f:
            _print(*args, file=f)

def first_order(a, b):
    n = len(a); q = np.empty(n)
    cuts = sorted(set([0] + [i for i in np.where(a == 0)[0].tolist() if i > 0] + [n]))
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        aa = a[s0:s1].copy(); aa[0] = 1.0
        P = np.exp(np.cumsum(np.log(aa)) - np.log(aa[0]))
        q[s0:s1] = P * np.cumsum(b[s0:s1] / P)
    return q

def solve_q(mu, lamp, L, fresh):
    """q[k, j, l] = mu * discounted occupation of the individual chain."""
    l = np.arange(0, L + 1, dtype=float)
    q = np.zeros((KMAX + 1, JMAX + 2, L + 2))
    for k in range(1, KMAX + 1):
        d = l + k
        valid = (d >= 2)
        for j in range(min(k - 1, JMAX), -1, -1):
            cred = (j * 0.5 if fresh else j * lamp) + (k - j) * lamp
            out = l + cred + np.where(d > 0, l / np.maximum(d, 1), 0) + j * (1.5 + 1.0 / np.maximum(d, 1))
            inA = (l - 1) + cred                                       # from (l-1, k, j)
            b = np.zeros_like(l)
            if k >= 2 and j >= 1:                                      # from (l+1, k-1, j-1) by budding
                b += (l + 1) / (l + k) * q[k - 1, j - 1, 1:L + 2]
            b += (j + 1) * (1.5 + 1.0 / np.maximum(d, 1)) * q[k, j + 1, 0:L + 1]   # from (l, k, j+1)
            if k == 1 and j == 0:
                b[1] += mu
            den = mu + out
            a = np.where((l >= 1) & valid, inA / den, 0.0)
            bb = np.where(valid, b / den, 0.0)
            q[k, j, 0:L + 1] = first_order(a, bb)
    return q

def stats(q, mu, lamp, L):
    l = np.arange(0, L + 1, dtype=float)
    K = np.arange(0, KMAX + 1, dtype=float)[:, None, None]
    d = l[None, None, :] + K
    with np.errstate(divide='ignore', invalid='ignore'):
        kd = np.where(d > 0, K / d, 0.0)
    qq = q[:, :, 0:L + 1]
    J = float((qq * kd).sum())
    Kq = float((qq * K).sum())
    # tail of sum k q beyond L: discounted flux into l = L+1 times (k/mu + 1/mu^2)
    kk = np.arange(0, KMAX + 1, dtype=float)
    jj = np.arange(0, JMAX + 2, dtype=float)[None, :]
    rate = L + 0.5 * jj + (kk[:, None] - jj) * lamp
    flux = (rate * q[:, :, L] / mu).sum(axis=1)
    Kq += mu * float((flux * (kk / mu + 1.0 / mu**2)).sum())
    qF = float(q[1, 0, 1])                                             # state (1,1,0)
    sumj = float((qq * np.arange(0, JMAX + 2)[None, :, None]).sum())   # should equal qF
    return J, Kq, qF, sumj, float(qq.sum())

def mu_for(lamp, L, fresh):
    lo, hi = 1e-6, 1.0
    while hi - lo > BISECT_TOL:
        mid = 0.5 * (lo + hi)
        J = stats(solve_q(mid, lamp, L, fresh), mid, lamp, L)[0]
        if 1.0 - J - mid > 0: lo = mid
        else: hi = mid
    mu = 0.5 * (lo + hi)
    return mu, solve_q(mu, lamp, L, fresh)

def main():
    global OUTFILE
    if OUTFILE is None:
        OUTFILE = dated_path("fresh_closure", "txt")
    t0 = time.time()
    print(f"=== run started {datetime.datetime.now().isoformat(timespec='seconds')} LMAX_LIST={LMAX_LIST} KMAX={KMAX} JMAX={JMAX} ===")
    print("Reference: equal sharing 0.568842; proven bracket (Uzay) [0.5650, 0.5665]; simulation ~0.565723")
    for fresh in (False, True):
        for L in LMAX_LIST:
            lamp = 0.2156
            for it in range(LAM_ITERS):
                mu, q = mu_for(lamp, L, fresh)
                J, Kq, qF, sumj, mass = stats(q, mu, lamp, L)
                new = (1.0 - mu) / Kq if not fresh else ((1.0 - mu) - 0.5 * qF) / (Kq - qF)
                if abs(new - lamp) < 1e-9: break
                lamp = new
            print(f"{'fresh' if fresh else 'equal'} L={L}: mu={mu:.7f} lam'={lamp:.7f} q_fresh={qF:.6f} "
                  f"sum j q={sumj:.6f} mass={mass:.6f} ({time.time()-t0:.0f}s)", flush=True)
    print("done")

if __name__ == "__main__":
    main()
