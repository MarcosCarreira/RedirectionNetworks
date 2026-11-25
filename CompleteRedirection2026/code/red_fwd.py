"""Reduction: single-vertex forward equation in age with a given conditional credit C(l, k, a) (Crank-Nicolson
step), and mu by Euler-Lotka; S3 sensitivities of mu to the parent's and the children's parts of C by block of
states and ages (24 Nov 21:1x BRT, cloud). Input: the tallies of red_s1.py (RS1 below)."""
import numpy as np, os
import cavity as C
RS1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'red_s1_2026-11-24_210740_cloud.npz')   # tallies of red_s1.py (fixed 25 Nov 12:16 BRT; was the newest file by pattern)
DMAX, KM = 400, 60
r = np.load(RS1)
Wa, PIa, CHa, lam = r['Wa'], r['PIa'], r['CHa'], float(r['lam'])
D = int(r['DSTAR']); nab = Wa.shape[0]; AW = float(r['AW'])
L = np.arange(DMAX + 1)[:, None] + 0 * np.arange(KM + 1)[None, :]; K = 0 * L + np.arange(KM + 1)[None, :]; Dg = L + K
ok = (Dg <= DMAX) & (Dg >= 2) & (K >= 1); br = np.where(ok, L / np.maximum(Dg, 1), 0.0)
m50 = ok & (Dg < D)
def tables(Wa, PIa, CHa):
    """parent and children parts by age bin on the full grid (states of degree < D; empty cells: the state's value
    over all ages)."""
    Wst = Wa.sum(0); pst = PIa.sum(0) / np.maximum(Wst, 1e-300); cst = CHa.sum(0) / np.maximum(Wst, 1e-300)
    Pi = np.zeros((nab,) + Dg.shape); Ch = np.zeros((nab,) + Dg.shape)
    for b in range(nab):
        p = np.where(Wa[b] > 0, PIa[b] / np.maximum(Wa[b], 1e-300), pst)
        c = np.where(Wa[b] > 0, CHa[b] / np.maximum(Wa[b], 1e-300), cst)
        Pi[b][:D + 1, :D + 1] = np.where(m50[:D + 1, :D + 1], p, 0.0)
        Ch[b][:D + 1, :D + 1] = np.where(m50[:D + 1, :D + 1], c, 0.0)
    return Pi, Ch
PI0, CH0 = tables(Wa, PIa, CHa)
CBIG = np.where(ok & ~m50, K * lam, 0.0)
from numba import njit


@njit(cache=True)
def _cn(Pi, Ch, cbig, AW, DA, n, DMAX, KM):
    """Crank-Nicolson in age for the forward equation of the state (l, k), solved by a sweep in increasing degree and,
    within a degree, decreasing l (gains raise the degree; births keep it and lower l). Returns beta(a) on the grid:
    the birth rate sum P l/d plus the mass that has left the grid at degree DMAX (birth rate 1 there)."""
    nab = Pi.shape[0]
    P = np.zeros((DMAX + 1, KM + 1)); P[1, 1] = 1.0
    beta = np.zeros(n + 1); esc = 0.0
    def_out = np.zeros((DMAX + 1, KM + 1))
    for st in range(n + 1):
        a = st * DA
        b0 = min(int(a / AW), nab - 1)
        # birth rate now
        s = 0.0
        for l in range(1, DMAX + 1):
            for k in range(1, KM + 1):
                if l + k <= DMAX:
                    s += P[l, k] * l / (l + k)
        beta[st] = s + esc
        if st == n:
            break
        b1 = min(int((a + 0.999999 * DA) / AW), nab - 1)
        Pn = np.zeros((DMAX + 1, KM + 1))
        lost_old = 0.0; lost_new = 0.0
        for d in range(2, DMAX + 1):
            for l in range(d - 1, -1, -1):
                k = d - l
                if k < 1 or k > KM:
                    continue
                g0 = l + Pi[b0, l, k] + Ch[b0, l, k] + cbig[l, k]
                g1 = l + Pi[b1, l, k] + Ch[b1, l, k] + cbig[l, k]
                br = l / d
                # explicit half: old outflow and old inflow
                fin_old = 0.0; fin_new = 0.0
                if l >= 1 and d - 1 >= 2:
                    gp0 = (l - 1) + Pi[b0, l - 1, k] + Ch[b0, l - 1, k] + cbig[l - 1, k]
                    gp1 = (l - 1) + Pi[b1, l - 1, k] + Ch[b1, l - 1, k] + cbig[l - 1, k]
                    if l - 1 + k >= 2 and (l - 1 >= 0):
                        fin_old += gp0 * P[l - 1, k]; fin_new += gp1 * Pn[l - 1, k]
                if l + 1 <= DMAX and k - 1 >= 1:
                    fin_old += (l + 1) / d * P[l + 1, k - 1]; fin_new += (l + 1) / d * Pn[l + 1, k - 1]
                rhs = P[l, k] + 0.5 * DA * (fin_old - (g0 + br) * P[l, k]) + 0.5 * DA * fin_new
                Pn[l, k] = rhs / (1.0 + 0.5 * DA * (g1 + br))
                if d == DMAX:
                    lost_old += g0 * P[l, k]; lost_new += g1 * Pn[l, k]
        esc += 0.5 * DA * (lost_old + lost_new)
        P = Pn
    return beta


def fwd_mu(Pi, Ch, cbig=CBIG, DA=0.002, HH=12.0):
    n = int(round(HH / DA))
    beta = _cn(Pi, Ch, cbig, AW, DA, n, DMAX, KM)
    ages = np.arange(n + 1) * DA
    def f(m):
        y = np.exp(-m * ages) * beta
        return DA * (y.sum() - 0.5 * (y[0] + y[-1])) + np.exp(-m * HH) * beta[-1] / m - 1.0
    lo, hi = 0.4, 0.8
    for _ in range(50):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    import time
    import builtins, datetime, os as _os
    _RD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results"); _os.makedirs(_RD, exist_ok=True)
    _OUT = open(_os.path.join(_RD, "red_fwd_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"), "w")
    def print(*a, **k):
        """Prints and writes the same line to the dated result file (added 25 Nov 2026, 12:16 BRT)."""
        k.pop("flush", None); builtins.print(*a, **k, flush=True); builtins.print(*a, **k, file=_OUT, flush=True)
    print(f"input {os.path.basename(RS1)}")
    t = time.time()
    for DA in (0.008, 0.004, 0.002, 0.001):
        print(f"Crank-Nicolson step {DA}: mu {fwd_mu(PI0, CH0, DA=DA):.7f}  ({time.time()-t:.0f}s)")
    print(f"ratio equations of the same solution {float(r['muR']):.6f}")
