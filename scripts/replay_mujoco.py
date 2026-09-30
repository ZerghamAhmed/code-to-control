"""Replay the MuJoCo controllers in champions/mujoco/ and compare their scores with the paper.

A continuous controller has two parts. The language model wrote a feature map,
champions/mujoco/<task>.py, which turns the observation into 14 numbers. Random search then
fitted a matrix that maps those numbers to torques; it is stored with the observation names and
the feature normaliser in champions/mujoco/<task>.json.

This script rebuilds each controller, plays the paper's ten test episodes (seeds 4000-4009),
and reports the median return, forward displacement and episode length. It makes no
language-model calls.

    python scripts/replay_mujoco.py                 # every task
    python scripts/replay_mujoco.py halfcheetah     # one task
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAMP = os.path.join(ROOT, "champions", "mujoco")

# Tasks whose first position coordinate is the body's forward position. Elsewhere it is a cart
# position or a joint angle, so no displacement is printed.
LOCOMOTION = {"halfcheetah", "hopper", "walker2d", "swimmer", "ant", "humanoid", "humanoidstandup"}


class Episode:
    """One test episode, with the observation and step semantics of the original runs.

    The controller sees each observation value rounded to 4 decimals, under the names in the
    task's JSON, plus the running return as `score`.
    """

    def __init__(self, env_id, labels, seed):
        import gymnasium as gym
        gym.logger.min_level = gym.logger.ERROR     # silence the "-v4 is out of date" notice
        self.env = gym.make(env_id, max_episode_steps=10 ** 9)
        self.labels = labels
        self.obs, _ = self.env.reset(seed=seed)
        if len(np.asarray(self.obs).ravel()) != len(labels):
            raise SystemExit(f"{env_id}: {len(np.asarray(self.obs).ravel())} observation "
                             f"values but {len(labels)} names")
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        self.ret = 0.0
        self.lost = False

    def state(self):
        s = {k: [round(float(v), 4)] for k, v in zip(self.labels, np.asarray(self.obs).ravel())}
        s["score"] = [round(float(self.ret), 4)]
        s["won"] = False
        s["lost"] = self.lost
        return s

    def step(self, action):
        if self.lost:          # as in the original wrapper: an ended episode no longer moves
            return
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1),
                    self.low, self.high).astype(np.float32)
        self.obs, r, terminated, _truncated, _info = self.env.step(a)
        self.ret += float(r)
        if terminated:
            self.lost = True

    def x(self):
        return float(self.env.unwrapped.data.qpos[0])


def load_module(path, name):
    """Import a feature-map file as a fresh module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(task):
    meta = json.load(open(os.path.join(CHAMP, f"{task}.json")))
    return meta, load_module(os.path.join(CHAMP, f"{task}.py"), f"feature_map_{task}")


class Controller:
    """features -> normalise -> matrix -> clip, exactly as in the fitted runs."""

    def __init__(self, meta, mod):
        self.mod = mod
        self.W = np.array(meta["weights"])
        self.mu = np.array(meta["norm_mu"])
        self.sigma = np.array(meta["norm_sigma"])
        # A feature that never varied during fitting (such as a constant bias) is passed
        # through unnormalised, since centring it would zero it.
        self.const = (np.array(meta["norm_hi"]) - np.array(meta["norm_lo"])) < 1e-9
        self.n_terms = meta["n_terms"]

    def act(self, state, t, low, high):
        f = [float(x) for x in self.mod.terms(state, t)]
        if len(f) != self.n_terms or not all(math.isfinite(x) for x in f):
            raise RuntimeError(f"feature map returned {len(f)} values or a non-finite one")
        x = np.array(f)
        v = np.where(self.const, x, (x - self.mu) / np.where(self.const, 1.0, self.sigma))
        lo = [float(b) for b in low]
        hi = [float(b) for b in high]
        return [max(lo[j], min(hi[j], float(np.dot(self.W[j], v)))) for j in range(len(lo))]


def play(meta, ctrl, seed, on_step=None):
    """Play one episode. Returns (return, forward displacement in m, steps)."""
    ep = Episode(meta["env_id"], meta["labels"], seed)
    if hasattr(ctrl.mod, "reset_terms"):
        ctrl.mod.reset_terms()
    x0 = ep.x()
    n = 0
    if on_step:
        on_step(ep)
    while n < meta["cap"]:
        s = ep.state()
        if s["lost"]:
            break
        ep.step(ctrl.act(s, n, ep.low, ep.high))
        n += 1
        if on_step:
            on_step(ep)
    return round(float(ep.ret), 4), ep.x() - x0, n


def evaluate(task):
    meta, mod = load(task)
    ctrl = Controller(meta, mod)
    runs = [play(meta, ctrl, s) for s in meta["test_seeds"]]
    ret = statistics.median(r[0] for r in runs)
    disp = statistics.median(r[1] for r in runs)
    steps = statistics.median(r[2] for r in runs)
    return meta, runs, ret, disp, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tasks", nargs="*")
    a = ap.parse_args()
    tasks = a.tasks or sorted(f[:-5] for f in os.listdir(CHAMP) if f.endswith(".json"))
    print(f"{'task':<24}{'paper':>12}{'replayed':>12}{'displacement':>14}{'steps':>8}  result")
    bad = 0
    for task in tasks:
        meta, _runs, ret, disp, steps = evaluate(task)
        ok = round(ret, 2) == round(meta["reported_return"], 2)
        bad += not ok
        moved = f"{disp:>12.2f} m" if task in LOCOMOTION else " " * 14
        print(f"{task:<24}{meta['reported_return']:>12.2f}{ret:>12.2f}{moved}"
              f"{steps:>8g}  {'match' if ok else 'MISMATCH'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
