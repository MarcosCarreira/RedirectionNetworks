"""Reduction S4 (24 Nov 21:2x BRT, cloud): simple descriptions of the parent's part of C on (l,1) and the children's
part on the leafless states, each tested on mu through the forward equation (Crank-Nicolson, step 0.004)."""
import numpy as np
import red_fwd as F
import builtins, datetime, os as _os
_RD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results"); _os.makedirs(_RD, exist_ok=True)
_OUT = open(_os.path.join(_RD, "red_s4_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"), "w")
def print(*a, **k):
    """Prints and writes the same line to the dated result file (added 25 Nov 2026, 12:16 BRT)."""
    k.pop("flush", None); builtins.print(*a, **k, flush=True); builtins.print(*a, **k, file=_OUT, flush=True)
print(f"input {_os.path.basename(F.RS1)}")
DA = 0.004
base = F.fwd_mu(F.PI0, F.CH0, DA=DA)
W = F.Wa; nab = F.nab; aw = F.AW
def occ(b, l, k): return W[b, l, k]
print("parent part pi(l,1,a) by age bin (rows) and l = 1..6:")
for b in (0, 1, 2, 3, 4, 6, 8, 12):
    print(f"  a {b*aw:4.2f}: " + " ".join(f"{F.PI0[b, l, 1]:.4f}" for l in range(1, 7)))
print("children part chi(0,k,a), k = 2, 3, 4:")
for b in (0, 1, 2, 3, 4, 6, 8, 12):
    print(f"  a {b*aw:4.2f}: " + " ".join(f"{F.CH0[b, 0, k]:.4f}" for k in (2, 3, 4)))
def test(name, Pi, Ch):
    m = F.fwd_mu(Pi, Ch, DA=DA)
    print(f"{name}: d mu = {1e5*(m-base):+.2f} e-5", flush=True)
# D1: parent part on (l,1) replaced by its occupation-weighted mean over l at each age
Pi = F.PI0.copy()
for b in range(nab):
    ls = np.arange(1, F.D - 1); w = np.array([occ(b, l, 1) for l in ls]); v = np.array([F.PI0[b, l, 1] for l in ls])
    if w.sum() > 0:
        Pi[b, 1:F.D - 1, 1] = (w * v).sum() / w.sum()
test("D1 parent on (l,1) independent of l", Pi, F.CH0)
# D2: separable pi(l,1,a) = p(a) h(l), h from the age-pooled ratio
Pi = F.PI0.copy(); ls = np.arange(1, F.D - 1)
Wl = W[:, ls, 1]; Pl = F.PI0[:, ls, 1]
h = (Wl * Pl).sum(0) / np.maximum(Wl.sum(0), 1e-300)
pa = (Wl * Pl).sum(1) / np.maximum((Wl * h[None]).sum(1), 1e-300)
Pi[:, 1:F.D - 1, 1] = pa[:, None] * h[None]
test("D2 parent on (l,1) = p(a) h(l)", Pi, F.CH0)
# D3: children part of leafless states = (k-1) g(a), g pooled per child over k
Ch = F.CH0.copy(); ks = np.arange(2, F.D)
Wk = W[:, 0, ks]; Ck = F.CH0[:, 0, ks] / (ks[None] - 1)
g = (Wk * Ck).sum(1) / np.maximum(Wk.sum(1), 1e-300)
Ch[:, 0, ks] = g[:, None] * (ks[None] - 1)
test("D3 children of (0,k) = (k-1) g(a)", F.PI0, Ch)
# D4: children part of (0,k) by state only (age averaged with the occupation)
Ch = F.CH0.copy()
for k in ks:
    w = W[:, 0, k]; 
    if w.sum() > 0: Ch[:, 0, k] = (w * F.CH0[:, 0, k]).sum() / w.sum()
test("D4 children of (0,k) without age", F.PI0, Ch)
# D5: parent part of (l,1) without age
Pi = F.PI0.copy()
for l in ls:
    w = W[:, l, 1]
    if w.sum() > 0: Pi[:, l, 1] = (w * F.PI0[:, l, 1]).sum() / w.sum()
test("D5 parent of (l,1) without age", Pi, F.CH0)
