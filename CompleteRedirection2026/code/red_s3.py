"""Reduction S3 (24 Nov 21:1x BRT, cloud): change of mu (forward equation, Crank-Nicolson step 0.004) when the
parent's part or the children's part of C(l,k,a) is raised by 1% in one block of states and ages."""
import numpy as np, time
import red_fwd as F
import builtins, datetime, os as _os
_RD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results"); _os.makedirs(_RD, exist_ok=True)
_OUT = open(_os.path.join(_RD, "red_s3_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"), "w")
def print(*a, **k):
    """Prints and writes the same line to the dated result file (added 25 Nov 2026, 12:16 BRT)."""
    k.pop("flush", None); builtins.print(*a, **k, flush=True); builtins.print(*a, **k, file=_OUT, flush=True)
print(f"input {_os.path.basename(F.RS1)}")
DA = 0.004
base = F.fwd_mu(F.PI0, F.CH0, DA=DA)
print(f"base mu {base:.7f}")
L, K = F.L, F.K
states = {"(l,1)": (K == 1), "(0,k>=2)": (L == 0) & (K >= 2), "(l>=1,k>=2)": (L >= 1) & (K >= 2)}
ages = {"0-1": (0, 4), "1-3": (4, 12), "3-12": (12, F.nab)}
rows = []
for part in ("parent", "children"):
    for sn, sm in states.items():
        if part == "children" and sn == "(l,1)":
            continue
        for an, (b0, b1) in ages.items():
            Pi = F.PI0.copy(); Ch = F.CH0.copy()
            tgt = Pi if part == "parent" else Ch
            tgt[b0:b1] = np.where(sm[None], tgt[b0:b1] * 1.01, tgt[b0:b1])
            m = F.fwd_mu(Pi, Ch, DA=DA)
            rows.append((part, sn, an, (m - base)))
            print(f"  {part:8s} {sn:12s} ages {an:5s}: d mu = {1e5*(m-base):+7.2f} e-5", flush=True)
m = F.fwd_mu(F.PI0, F.CH0, cbig=F.CBIG * 1.01, DA=DA)
print(f"  credit k*lambda above degree 49 +1%: d mu = {1e5*(m-base):+7.2f} e-5")
