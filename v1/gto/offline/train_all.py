"""Train every model in parallel processes, then merge.

    python -m gto.offline.train_all
"""
import os
import subprocess
import sys

from gto.offline.train import ARCHETYPES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    logs = os.path.join(HERE, "logs")
    os.makedirs(logs, exist_ok=True)
    jobs = []
    running = []
    limit = int(os.environ.get("PARALLEL", "4"))
    for name in (sys.argv[1:] or ["eq"] + ARCHETYPES):
        while len([p for p in running if p.poll() is None]) >= limit:
            running[0].wait()
            running = [p for p in running if p.poll() is None]
        env = dict(os.environ)
        env["ITERS"] = os.environ.get("ITERS_EQ" if name == "eq" else "ITERS_EX",
                                      "6000000" if name == "eq" else "4000000")
        env["NUMBA_CACHE_DIR"] = os.path.join(HERE, "numba_cache", name)  # no shared-cache races
        log = open(os.path.join(logs, f"{name}.log"), "w")
        jobs.append((name, subprocess.Popen([sys.executable, "-m", "gto.offline.train", name],
                                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
        running.append(jobs[-1][1])
    failed = [n for n, p in jobs if p.wait() != 0]
    if failed:
        sys.exit(f"failed: {failed}")
    if os.path.exists(os.path.join(HERE, "parts", "eq.npz")):
        subprocess.check_call([sys.executable, "-m", "gto.offline.train", "merge"], cwd=ROOT)


if __name__ == "__main__":
    main()
