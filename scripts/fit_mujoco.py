"""Fit a MuJoCo feature map's weight matrix with Augmented Random Search (ARS).

This is the fitting step of the continuous-control pipeline, as it ran for the paper. It starts
from W = 0, runs 90 iterations of ARS on returns from real episodes, keeps the best weights it
probed, and scores them on the ten test episodes (seeds 4000-4009, median). The language model
is not involved.

    python scripts/fit_mujoco.py reacher --seed 1 --check   # re-create a released controller
    python scripts/fit_mujoco.py halfcheetah --refits 3     # the paper's protocol: median of 3 fits
    python scripts/fit_mujoco.py hopper --program terms.py --seed 0 --out fitted.json

The paper's number for each task is the median of three fits of its feature map, with fit
seeds 0, 1 and 2, and champions/mujoco/ holds the median one. `--check` refits with the
released controller's seed and compares every weight. The fits are seeded, but their exact
weights depend on the numerical libraries, so run the check in an environment built from
requirements.txt.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_mujoco as R  # noqa: E402

ROOT = R.ROOT
TASKS = json.load(open(os.path.join(ROOT, "scripts", "mujoco_tasks.json")))

# The search settings used for every task in the paper.
BUDGET, DIRS, TOP, STEP, NOISE = 1440, 8, 4, 0.02, 0.03   # 1440 episodes = 90 iterations
TUNE_SEEDS = (100, 101)
PROBE_EVERY = 10
REFIT_SEED_BASE = 12345      # fit seed s uses np.random.default_rng([12345, s])


class Norm:
    """Running mean and variance of the feature vector.

    The features are not on a common scale, so ARS's fixed perturbation size would mean
    something different for each column. A feature whose value never changed (such as a
    constant bias) is passed through as it is, since centring it would zero it.
    """

    def __init__(self, n):
        self.n = 0
        self.mu = np.zeros(n)
        self.m2 = np.ones(n)
        self.lo = np.full(n, np.inf)
        self.hi = np.full(n, -np.inf)

    def push(self, x):
        self.n += 1
        d = x - self.mu
        self.mu += d / self.n
        self.m2 += d * (x - self.mu)
        np.minimum(self.lo, x, out=self.lo)
        np.maximum(self.hi, x, out=self.hi)

    def sigma(self):
        return np.sqrt(np.maximum(self.m2 / max(self.n, 1), 1e-8))

    def apply(self, x):
        sg = self.sigma()
        const = (self.hi - self.lo) < 1e-9
        out = (x - self.mu) / np.where(const, 1.0, sg)
        return np.where(const, x, out)


class Fitter:
    def __init__(self, task, mod, n_terms):
        self.task = task
        self.spec = TASKS[task]
        self.mod = mod
        probe = R.Episode(self.spec["env_id"], self.spec["labels"], 0)
        self.low, self.high = probe.low, probe.high
        self.lo = [float(x) for x in self.low]
        self.hi = [float(x) for x in self.high]
        self.ad = len(self.lo)
        self.n_terms = n_terms
        self.steps = 0

    def rollout(self, W, seed, norm, collect):
        """One episode's return under weights W, or None if the feature map misbehaves.
        With collect=True the episode's features also update the normaliser."""
        ep = R.Episode(self.spec["env_id"], self.spec["labels"], seed)
        if hasattr(self.mod, "reset_terms"):
            try:
                self.mod.reset_terms()
            except Exception:
                pass
        n = 0
        while n < self.spec["cap"]:
            o = ep.state()
            if o["lost"]:
                break
            try:
                f = np.array([float(x) for x in self.mod.terms(o, n)])
                if f.size != self.n_terms or not np.all(np.isfinite(f)):
                    return None
            except Exception:
                return None
            if collect:
                norm.push(f)
            a = W @ norm.apply(f)
            ep.step([max(self.lo[j], min(self.hi[j], float(a[j]))) for j in range(self.ad)])
            n += 1
            self.steps += 1
        return float(ep.state()["score"][0])

    def fit(self, rng, log=None):
        """ARS-V2 on the features. Returns (W, normaliser, best probed return)."""
        W = np.zeros((self.ad, self.n_terms))
        norm = Norm(self.n_terms)
        best = -1e18
        bestW, bestscore = None, -1e18
        iters = max(1, BUDGET // (2 * DIRS))
        for it in range(iters):
            D = [rng.normal(0, 1, size=(self.ad, self.n_terms)) for _ in range(DIRS)]
            seed = TUNE_SEEDS[it % len(TUNE_SEEDS)]
            Rs = []
            for d in D:
                rp = self.rollout(W + NOISE * d, seed, norm, True)
                rm = self.rollout(W - NOISE * d, seed, norm, True)
                Rs.append((-1e18 if rp is None else rp, -1e18 if rm is None else rm))
            Rs = np.array(Rs)
            order = np.argsort(-np.max(Rs, axis=1))[:TOP]
            sd = Rs[order].std()
            if sd < 1e-8:
                sd = 1.0
            upd = sum((Rs[i, 0] - Rs[i, 1]) * D[i] for i in order)
            W = W + (STEP / (TOP * sd)) * upd
            best = max(best, float(np.max(Rs)))
            # Keep the best iterate, probed on one tuning episode, not the last one.
            if it % PROBE_EVERY == 0 or it == iters - 1:
                sc = self.rollout(W, TUNE_SEEDS[it % len(TUNE_SEEDS)], norm, False)
                if sc is not None and sc > bestscore:
                    bestscore, bestW = float(sc), W.copy()
            if log and (it % 30 == 0 or it == iters - 1):
                log(f"    iteration {it:2d}/{iters}: best perturbed {best:10.2f}, "
                    f"best kept {bestscore:10.2f}")
        if bestW is not None:
            return bestW, norm, bestscore
        return W, norm, best

    def controller(self, W, norm, extra=None):
        """A champions/mujoco-style controller record for these weights."""
        rec = {
            "task": self.task, "env_id": self.spec["env_id"], "cap": self.spec["cap"],
            "test_seeds": list(range(4000, 4010)), "n_terms": self.n_terms,
            "labels": self.spec["labels"], "weights": np.asarray(W).tolist(),
            "norm_mu": norm.mu.tolist(), "norm_sigma": norm.sigma().tolist(),
            "norm_lo": norm.lo.tolist(), "norm_hi": norm.hi.tolist(),
        }
        rec.update(extra or {})
        return rec

    def test(self, rec):
        """Median return, forward displacement and episode length over the test seeds."""
        ctrl = R.Controller(rec, self.mod)
        runs = [R.play(rec, ctrl, s) for s in rec["test_seeds"]]
        return (statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs),
                statistics.median(r[2] for r in runs))


def count_terms(task, program):
    """How many terms the feature map returns, asked of a separate copy of the module so the
    copy that is fitted starts untouched, as in the original runs."""
    spec = TASKS[task]
    probe = R.load_module(program, f"count_{task}")
    return len(probe.terms(R.Episode(spec["env_id"], spec["labels"], 0).state(), 0))


def fit_once(task, program, seed_spec, log=None, mod=None):
    """Fit a feature map once. Returns its controller record, with test results."""
    n_terms = count_terms(task, program)
    if mod is None:
        mod = R.load_module(program, f"fit_{task}_{'_'.join(map(str, seed_spec))}")
    fitter = Fitter(task, mod, n_terms)
    t0 = time.time()
    W, norm, tune = fitter.fit(np.random.default_rng(list(seed_spec)), log)
    fit_steps = fitter.steps
    rec = fitter.controller(W, norm)
    ret, disp, steps = fitter.test(rec)
    rec.update({"reported_return": ret, "displacement_m": disp, "episode_steps": steps,
                "fit": {"method": "ARS", "budget": BUDGET, "directions": DIRS, "top": TOP,
                        "step": STEP, "noise": NOISE, "rng": list(seed_spec),
                        "tune_seeds": list(TUNE_SEEDS), "env_steps": fit_steps},
                "tune_return": tune, "seconds": round(time.time() - t0, 1)})
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--program", help="feature-map file (default: the released one)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", type=int, help="one fit with this fit seed")
    g.add_argument("--refits", type=int, help="fits with seeds 0..N-1, then their median")
    ap.add_argument("--check", action="store_true",
                    help="refit with the released controller's seed and compare its weights")
    ap.add_argument("--out", help="write the fitted controller here (JSON)")
    a = ap.parse_args()

    program = a.program or os.path.join(R.CHAMP, f"{a.task}.py")
    say = lambda m: print(m, flush=True)  # noqa: E731
    if a.check:
        released = json.load(open(os.path.join(R.CHAMP, f"{a.task}.json")))
        seed = released["fit"]["rng"][1]
        say(f"{a.task}: refitting the released feature map with fit seed {seed}")
        rec = fit_once(a.task, program, (REFIT_SEED_BASE, seed), say)
        same = (rec["weights"] == released["weights"] and rec["norm_mu"] == released["norm_mu"]
                and rec["norm_sigma"] == released["norm_sigma"])
        say(f"  return {rec['reported_return']:.2f} (released {released['reported_return']:.2f}), "
            f"environment steps {rec['fit']['env_steps']:,} "
            f"(released {released['fit']['env_steps']:,}), {rec['seconds']:.0f} s")
        say("  weights identical to the released controller" if same else
            "  WEIGHTS DIFFER from the released controller")
        sys.exit(0 if same else 1)

    seeds = [a.seed if a.seed is not None else 0] if not a.refits else list(range(a.refits))
    recs = []
    for s in seeds:
        say(f"{a.task}: fit seed {s}")
        rec = fit_once(a.task, program, (REFIT_SEED_BASE, s), say)
        recs.append(rec)
        say(f"  test return {rec['reported_return']:.2f}, displacement "
            f"{rec['displacement_m']:.2f} m, {rec['episode_steps']:g} steps, "
            f"{rec['fit']['env_steps']:,} environment steps, {rec['seconds']:.0f} s")
    if len(recs) > 1:
        med = statistics.median(r["reported_return"] for r in recs)
        say(f"median test return over {len(recs)} fits: {med:.2f}")
    if a.out:
        chosen = sorted(recs, key=lambda r: r["reported_return"])[(len(recs) - 1) // 2]
        json.dump(chosen, open(a.out, "w"), indent=1)
        say(f"wrote {a.out}" + (" (the median fit)" if len(recs) > 1 else ""))


if __name__ == "__main__":
    main()
