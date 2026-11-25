"""Exact distribution of the complete-redirection (random friend) tree, small N.

Model: from a tree on n vertices, pick a vertex v uniformly, pick a neighbour w of v
uniformly, attach a new leaf to w. Start from the single edge (n = 2).

The script carries the exact probability (fractions.Fraction) of every unlabeled tree up
to NMAX vertices and checks, at every n:
  (A) drift identity  E[L_{n+1}] = E[L_n] + 1 - E[S_n]/n,
      S_n = sum over vertices v of (leaf neighbours of v)/deg(v)   (corrected Formulas §8.1.6);
  (B) star probability P(star on n vertices) = 2/(n-1) for n >= 3;
  (C) the non-leaf count never decreases along any transition;
  (D) for every tree with at least two non-leaves (n <= NCHECK), the one-step conditional
      drift of every state count M[(l,k)] (Proposition 5 of ClosureTheory.tex) and the total
      core-hit intensity (Lemma 7); and the one-step drift of the ordered non-leaf pair
      counts by the states of both ends, with the pair identities.
No arguments. Settings are the constants below.
"""
from fractions import Fraction
from collections import defaultdict
import sys, time

NMAX = 16          # largest tree size enumerated (13 takes under 1 s; 16 should take well under a minute)
NCHECK = 12        # per-tree exact check of Proposition 5 / Lemma 7 up to this n
# exact values, one row per n: each run writes a new dated file results/cr_exact_<date_time>.csv

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

def canon(adj):
    """Canonical string of an unlabeled tree (AHU encoding rooted at the centre)."""
    n = len(adj)
    deg = [len(a) for a in adj]
    layer = [v for v in range(n) if deg[v] <= 1]
    remaining = n
    d = deg[:]
    while remaining > 2:
        remaining -= len(layer)
        nxt = []
        for v in layer:
            for w in adj[v]:
                d[w] -= 1
                if d[w] == 1:
                    nxt.append(w)
        layer = nxt
    centres = layer
    def enc(v, parent):
        return "(" + "".join(sorted(enc(w, v) for w in adj[v] if w != parent)) + ")"
    return min(enc(c, -1) for c in centres)

def leaf_share_sum(adj):
    s = Fraction(0)
    for v in range(len(adj)):
        dv = len(adj[v])
        lv = sum(1 for w in adj[v] if len(adj[w]) == 1)
        if lv:
            s += Fraction(lv, dv)
    return s

def states_of(adj):
    """Counts M[(l,k)], core-hit intensities H[(l,k)] and S for a tree (non-leaves only)."""
    M = defaultdict(int); H = defaultdict(Fraction); S = Fraction(0)
    deg = [len(a) for a in adj]
    for v in range(len(adj)):
        if deg[v] < 2:
            continue
        l = sum(1 for w in adj[v] if deg[w] == 1); k = deg[v] - l
        M[(l, k)] += 1
        S += Fraction(l, deg[v])
        H[(l, k)] += sum(Fraction(1, deg[u]) for u in adj[v] if deg[u] >= 2)
    return M, H, S

def check_prop5(adj):
    """Exact check, conditional on this tree, of the one-step drift of every M[(l,k)]
    (Proposition 5 of ClosureTheory.tex) and of the total core-hit intensity (Lemma 7)."""
    n = len(adj)
    M, H, S = states_of(adj)
    C = sum(M.values())
    if n < 3 or C < 2:
        return True
    # exact conditional expectation of M(n+1) by enumerating all moves
    EM1 = defaultdict(Fraction)
    for v in range(n):
        for w in adj[v]:
            a2 = [list(a) for a in adj] + [[w]]
            a2[w].append(n)
            M2, _, _ = states_of(a2)
            pr = Fraction(1, n * len(adj[v]))
            for key, c in M2.items():
                EM1[key] += pr * c
    keys = set(EM1) | set(M) | {(l + 1, k) for (l, k) in M} | {(l - 1, k + 1) for (l, k) in M} | {(1, 1)}
    for (l, k) in keys:
        lhs = n * (EM1[(l, k)] - M.get((l, k), 0))
        rhs = ((l - 1) * M.get((l - 1, k), 0) - l * M.get((l, k), 0)
               + Fraction(l + 1, l + k) * M.get((l + 1, k - 1), 0)
               - Fraction(l, l + k) * M.get((l, k), 0)
               + H.get((l - 1, k), 0) - H.get((l, k), 0)
               + (S if (l, k) == (1, 1) else 0))
        if lhs != rhs:
            print("Prop 5 FAILS at state", (l, k), "lhs", lhs, "rhs", rhs, "tree", adj)
            return False
        if k < 1 or l + k < 2:
            if EM1[(l, k)] != 0:
                print("state outside D reached", (l, k)); return False
    if sum(H.values()) != C - S:
        print("Lemma 7 FAILS", adj); return False
    return True

def pairs_of(adj):
    """Ordered non-leaf pairs: P[(s,s2)] counts (v,u) adjacent non-leaves with state(v)=s,
    state(u)=s2; G[(s,s2)] sums, over those pairs, 1/d(x) over non-leaf neighbours x of v, x != u."""
    deg = [len(a) for a in adj]
    st = {}
    for v in range(len(adj)):
        if deg[v] >= 2:
            l = sum(1 for w in adj[v] if deg[w] == 1); st[v] = (l, deg[v] - l)
    P = defaultdict(int); G = defaultdict(Fraction)
    for v in st:
        for u in adj[v]:
            if u in st:
                P[(st[v], st[u])] += 1
                G[(st[v], st[u])] += sum(Fraction(1, deg[x]) for x in adj[v] if x in st and x != u)
    return P, G

def check_pairs(adj):
    """Exact check, conditional on this tree, of the one-step drift of the ordered-pair counts
    (pair-drift proposition of ClosureTheory.tex) and of sum_{s2} G[(s,s2)] = (k-1) H[s]."""
    n = len(adj)
    M, H, S = states_of(adj)
    if n < 3 or sum(M.values()) < 2:
        return True
    P, G = pairs_of(adj)
    EP1 = defaultdict(Fraction)
    for v in range(n):
        for w in adj[v]:
            a2 = [list(a) for a in adj] + [[w]]
            a2[w].append(n)
            P2, _ = pairs_of(a2)
            pr = Fraction(1, n * len(adj[v]))
            for key, c in P2.items():
                EP1[key] += pr * c
    def F(s, s2):          # first-slot flux into minus out of (s, s2)
        l, k = s; d2 = s2[0] + s2[1]
        sm = (l - 1, k); spl = (l + 1, k - 1)
        val = ((l - 1) * P.get((sm, s2), 0) + Fraction(P.get((sm, s2), 0), d2) + G.get((sm, s2), 0)
               - l * P.get((s, s2), 0) - Fraction(P.get((s, s2), 0), d2) - G.get((s, s2), 0))
        if k - 1 >= 1 and l + k >= 2:
            val += Fraction(l + 1, l + k) * P.get((spl, s2), 0)
        val -= Fraction(l, l + k) * P.get((s, s2), 0)
        return val
    def B(s, s2):          # new pairs from case (b)
        val = Fraction(0)
        for a, b in ((s, s2), (s2, s)):
            if b == (1, 1):
                l, k = a
                val += Fraction(l + 1, l + k) * M.get((l + 1, k - 1), 0)
        return val
    keys = set(EP1) | set(P)
    for (s, s2) in keys:
        lhs = n * (EP1[(s, s2)] - P.get((s, s2), 0))
        rhs = F(s, s2) + F(s2, s) + B(s, s2)
        if lhs != rhs:
            print("pair drift FAILS at", (s, s2), "lhs", lhs, "rhs", rhs, "tree", adj); return False
    for s in M:
        if sum(g for (a, b), g in G.items() if a == s) != (s[1] - 1) * H[s]:
            print("pair total FAILS", s, adj); return False
        if sum(c for (a, b), c in P.items() if a == s) != s[1] * M[s]:
            print("pair marginal FAILS", s, adj); return False
    return True

def main():
    t0 = time.time()
    states = {canon([[1], [0]]): ([[1], [0]], Fraction(1))}
    OUTCSV = dated_path("cr_exact", "csv")
    print(f"writing {OUTCSV}")
    out = open(OUTCSV, "x")
    out.write("n,trees,E_L,E_C,E_L_over_n,P_star,E_S\n")
    print("n  trees  E[L_n]  E[C_n]  E[L_n]/n(float)  P(star)  checks")
    for n in range(2, NMAX + 1):
        EL = sum(p * sum(1 for a in adj if len(a) == 1) for adj, p in states.values())
        ES = sum(p * leaf_share_sum(adj) for adj, p in states.values())
        star = sum(p for adj, p in states.values() if max(len(a) for a in adj) == n - 1)
        okB = (n < 3) or (star == Fraction(2, n - 1))
        print(f"{n:2d} {len(states):6d}  {EL}  {n-EL}  {float(EL/n):.6f}  {star}  "
              f"star_ok={okB}  ({time.time()-t0:.1f}s)", flush=True)
        out.write(f"{n},{len(states)},{EL},{n-EL},{float(EL/n):.12f},{star},{ES}\n"); out.flush()
        if n == NMAX:
            break
        okD = all(check_prop5(adj) and check_pairs(adj) for adj, _ in states.values()) if n <= NCHECK else True
        print(f"   Proposition 5, Lemma 7 and pair drift, every tree with C>=2 at n={n}: {okD if n <= NCHECK else 'skipped'}", flush=True)
        if not okD:
            print("CHECK FAILED"); sys.exit(1)
        new = defaultdict(lambda: [None, Fraction(0)])
        okC = True
        for adj, p in states.values():
            C0 = sum(1 for a in adj if len(a) > 1)
            for v in range(n):
                for w in adj[v]:
                    a2 = [list(a) for a in adj] + [[w]]
                    a2[w].append(n)
                    if sum(1 for a in a2 if len(a) > 1) < C0:
                        okC = False
                    key = canon(a2)
                    slot = new[key]
                    if slot[0] is None:
                        slot[0] = a2
                    slot[1] += p * Fraction(1, n * len(adj[v]))
        states = {k: (v[0], v[1]) for k, v in new.items()}
        assert sum(p for _, p in states.values()) == 1
        EL1 = sum(p * sum(1 for a in adj if len(a) == 1) for adj, p in states.values())
        okA = (EL1 == EL + 1 - ES / n)
        print(f"   drift identity n={n}->{n+1}: {okA}; non-leaf count monotone: {okC}", flush=True)
        if not (okA and okC and okB):
            print("CHECK FAILED"); sys.exit(1)
    print(f"done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
