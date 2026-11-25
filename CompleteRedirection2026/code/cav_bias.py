"""Run P (DistributionFormulas, 'Offset of the ratio-equation estimate: run P, pre-registered', 25 Nov 00:21 BRT).
No arguments; settings below. Variants N30 (3e4 records), N100 (1e5), N100L (1e5, large-vertex drift with
lambda = 0.20); 4 replicas each, generations 0-5, four output batches per generation; the point at 3e5 is the base
of the K5 run (KFILE). Uses the child process of cav_k5.py with these settings. Output: results/cav_bias_<date>.txt."""
import os, sys, time, datetime, json, subprocess
import numpy as np
import cav_k5 as K

VARIANTS = {"N30": (50, 12.0, 30000), "N100": (50, 12.0, 100000), "N100L": (50, 12.0, 100000)}
LAMS = {"N30": 0.230, "N100": 0.230, "N100L": 0.20}
REPLICAS = 4
KFILE = "cav_k5_2026-11-24_233354_B.json"     # K5 base, 3e5 records
MU_REF = 0.565723
HERE = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(HERE, "results")


def setup():
    K.VARIANTS = VARIANTS; K.REPLICAS = REPLICAS


if __name__ == "__main__":
    setup()
    if len(sys.argv) == 4:
        K.child(sys.argv[1], sys.argv[2], sys.argv[3]); sys.exit(0)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    LOG = os.path.join(RD, f"cav_bias_{stamp}.txt")
    def log(s):
        print(s, flush=True)
        with open(LOG, "a") as f:
            f.write(s + "\n")
    log(f"=== cav_bias started {datetime.datetime.now().isoformat(timespec='seconds')} VARIANTS={VARIANTS} LAMS={LAMS} "
        f"REPLICAS={REPLICAS} NBATCH={K.NBATCH} NGEN={K.NGEN} VALGEN={K.VALGEN} base point {KFILE} ===")
    log("Pre-registered (DistributionFormulas, 'Offset of the ratio-equation estimate: run P'): P1, P2 below.")
    T0 = time.time(); summ = {}
    for vname, (dstar, H, nrec) in VARIANTS.items():
        env = dict(os.environ, CAV_DSTAR=str(dstar), CAV_H=str(H), CAV_LAM=str(LAMS[vname]),
                   NUMBA_CACHE_DIR=os.path.join(HERE, f"numba_cache_bias_{vname}"))
        sumfile = os.path.join(RD, f"cav_bias_{stamp}_{vname}.json")
        subprocess.run([sys.executable, os.path.abspath(__file__), vname, LOG, sumfile], env=env, check=True)
        with open(sumfile) as f:
            summ[vname] = json.load(f)
    with open(os.path.join(RD, KFILE)) as f:
        summ["B300"] = json.load(f)
    res = {}
    for vname in ("N30", "N100", "N100L", "B300"):
        for col, est in ((1, "E"), (2, "R")):
            v, se, sl_, ses = K.value(summ[vname], col)
            res[(vname, est)] = (v, se)
            log(f"[{vname}] ({est}) value {v:.6f}, s.e. {se:.6f}; slope over generations {sl_:+.2e} (s.e. {ses:.2e})")
    essmin = min(r[3] for k_, s in summ.items() if k_ != "B300" for rows in s.values() for r in rows)
    log(f"minimum effective sample size in this run {essmin:.2f}")
    for est in ("R", "E"):
        Ns = np.array([3e4, 1e5, 3e5]); v = np.array([res[(n, est)][0] for n in ("N30", "N100", "B300")])
        se = np.array([res[(n, est)][1] for n in ("N30", "N100", "B300")])
        A = np.vstack([np.ones(3), 1.0 / Ns]).T; Wt = np.diag(1 / se ** 2)
        cov = np.linalg.inv(A.T @ Wt @ A); coef = cov @ A.T @ Wt @ v
        muinf, b = coef; se_muinf, se_b = np.sqrt(np.diag(cov))
        bias300, se_bias300 = b / 3e5, se_b / 3e5
        z = (muinf - MU_REF) / se_muinf
        verdict = "finite-population bias found" if abs(bias300) >= 2 * se_bias300 else "no finite-population bias resolved"
        log(f"PRE-REGISTERED P1 ({est}): mu_inf {muinf:.6f} (s.e. {se_muinf:.6f}), bias at 3e5 {bias300:+.6f} (s.e. {se_bias300:.6f}) -> {verdict}; "
            f"mu_inf - {MU_REF} = {muinf-MU_REF:+.6f}, z = {z:+.2f} -> {'agreement' if abs(z) <= 2 else 'no agreement'}")
    for est in ("R", "E"):
        (vl, sl), (v0, s0) = res[("N100L", est)], res[("N100", est)]
        d = vl - v0; lim = 2 * np.hypot(sl, s0)
        log(f"PRE-REGISTERED P2 ({est}): mu(lambda 0.20) - mu(lambda 0.230) = {d:+.6f} against {lim:.6f} -> "
            f"{'drift matters' if abs(d) > lim else 'no effect resolved'}; per 0.01 of lambda {d/(-3):+.6f}")
    log(f"done ({time.time()-T0:.0f}s)")
