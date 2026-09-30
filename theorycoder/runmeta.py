#!/usr/bin/env python
"""Provenance and per-episode instrumentation shared by every runner.

WHY THIS MODULE EXISTS
----------------------
Every bug that cost us a result this project was a MISSING FIELD, not a wrong one. A missing
field renders as a legitimate value: no token count reads as a blank column, no `ended` reads as
"not a rail", no timing reads as "instant", and no action fingerprint reads as "the action set
was fine". The fixes were all retrofits, and none of them can be backfilled.

So the additions below are deliberately cheap and deliberately central: one module, called from
each runner's record setup, so a new runner cannot quietly omit them.

WHAT EACH ONE IS FOR, WITH THE FAILURE THAT MOTIVATED IT
-------------------------------------------------------
`code_version()`   -- `cmd.json` records argv, NOT which code ran. Two arms with byte-identical
                      argv can execute different code if a file was edited between launches. This
                      is exactly why WorldCoder's boxing 2-episode stop could not be ruled out as
                      drift, and why the 2026-08-17 action-map fix silently splits the corpus into
                      before/after with nothing in the records to say so.

`action_fingerprint()` -- MsPacman's action names were shifted by one for the entire project and
                      no metric surfaced it; a second agent had to read the source. Recording the
                      resolved `actions_set` makes that a one-line diff between arms instead of an
                      archaeology exercise.

`episode_meta()`   -- we had per-RUN totals only, so a learning curve could not be plotted against
                      time, and "which episode got slow" was unanswerable. Also carries the action
                      HISTOGRAM: "does this policy ever fire?" was answered by grepping generated
                      source, which only works when the source is kept.

`llm_health()`     -- we lost whole arms to HTTP 429 and have no record of which, how many, or how
                      much backoff was spent. Three seaquest/boxing/mspacman arms died this way
                      and the records show only a short run.

`Watchdog`         -- a seaquest planner arm ran 62 minutes at 99.9% CPU with zero log output and
                      zero episodes, wedged in an unbounded search. Nothing in the system noticed.
                      A heartbeat makes the difference between "slow" and "dead" observable.
"""
import os
import subprocess
import threading
import time
from collections import Counter

_CACHE = {}


def code_version(root=None):
    """git HEAD, branch, and whether the tree was dirty AT THE MOMENT THE RUN STARTED.

    Cached per process: a run must record one version, and shelling out to git per episode would
    both cost time and, worse, let the recorded version CHANGE mid-run if the tree is edited
    while the arm is alive -- which is precisely the situation this field exists to detect.

    `dirty_files` is the list, not just a boolean, because "dirty" is useless on a research tree
    that is nearly always dirty. Knowing that `ocatari_generic_env.py` was uncommitted when the
    arm ran is the actionable part.
    """
    if "cv" in _CACHE:
        return _CACHE["cv"]
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args):
        try:
            return subprocess.run(("git",) + args, cwd=root, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    dirty = [ln[3:] for ln in git("status", "--porcelain").splitlines() if ln[3:]]
    out = {
        "head": git("rev-parse", "HEAD") or None,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty_count": len(dirty),
        # capped: a tree with 400 untracked artifact files should not bloat every record
        "dirty_files": sorted(dirty)[:40],
        "dirty_truncated": len(dirty) > 40,
    }
    _CACHE["cv"] = out
    return out


def action_fingerprint(env=None, actions=None, game=None):
    """The action set as the RUN resolved it, plus how it was resolved.

    `source` distinguishes a curated per-game map from the fallback default -- the fallback is
    what produced the MsPacman mislabelling, so an arm that used it should be visibly suspect
    rather than indistinguishable.
    """
    acts = list(actions) if actions is not None else list(getattr(env, "actions_set", []) or [])
    src = "unknown"
    # accept a bare game NAME too: the controller builds its env per episode via a factory, so
    # it has the resolved action list but no env object to read `game` off.
    game = getattr(env, "game", None) or game
    if game is not None:
        try:
            from ocatari_generic_env import ACTIONS
            game = {"breakoutoc": "Breakout", "mspacman": "MsPacman", "montezuma":
                    "MontezumaRevenge", "spaceinvaders": "SpaceInvaders"}.get(game, game)
            if game not in ACTIONS and str(game).capitalize() in ACTIONS:
                game = str(game).capitalize()
            src = "ACTIONS[%s]" % game if game in ACTIONS else "_DEFAULT_ACTIONS(fallback)"
        except Exception:
            pass
    return {"actions": acts, "n": len(acts), "source": src,
            "game": str(game) if game is not None else None}


def episode_meta(ep, score, transitions=None, steps=None, t0=None, ended=None,
                 top_actions=6):
    """Per-episode record fields: wall-clock, steps, termination, and an action histogram.

    The histogram is capped at `top_actions` and reports a total, so the record stays small while
    still answering the question that mattered on seaquest -- whether the policy ever chose to
    fire. `fire_frac` is broken out because it is the one we actually asked, repeatedly.
    """
    out = {"episode": ep, "score": score}
    if t0 is not None:
        out["wall_sec"] = round(time.time() - t0, 3)
    if steps is not None:
        out["env_steps"] = int(steps)
    elif transitions is not None:
        out["env_steps"] = len(transitions)
    if ended is not None:
        out["ended"] = ended
    if transitions:
        names = [str(t.get("action") if isinstance(t, dict) else t) for t in transitions]
        c = Counter(names)
        tot = max(1, len(names))
        out["action_hist"] = dict(c.most_common(top_actions))
        out["action_distinct"] = len(c)
        out["fire_frac"] = round(
            sum(v for k, v in c.items() if "fire" in k.lower()) / float(tot), 4)
    return out


def llm_health(client):
    """Retries, errors and backoff spent, read off the client's own timing events.

    Counted from what LLMClient already records rather than by instrumenting call sites, so it
    cannot disagree with the token and timing numbers taken from the same place.
    """
    ev = ((getattr(client, "timing", None) or {}).get("events") or [])
    errs = [e for e in ev if e.get("error")]
    kinds = Counter()
    for e in errs:
        msg = str(e.get("error"))
        if "429" in msg or "rate_limit" in msg:
            kinds["rate_limit"] += 1
        elif "timeout" in msg.lower():
            kinds["timeout"] += 1
        else:
            kinds["other"] += 1
    return {
        "calls": sum(1 for e in ev if e.get("category") == "llm"),
        "errors": len(errs),
        "error_kinds": dict(kinds),
        # wall-clock inside failed calls: retry backoff is real time and was invisible before.
        "error_sec": round(sum(e.get("duration_s") or 0.0 for e in errs), 3),
    }


class Watchdog:
    """Print a heartbeat when a run goes quiet, so wedged is distinguishable from slow.

    Motivated by a real arm: 62 minutes, 99.9% CPU, zero episodes, no output after line 12. Its
    two siblings progressed normally, so nothing about the launch was wrong and nothing in the
    logs said it was stuck.

    It does NOT kill anything. Killing on a timer would end legitimately slow searches, and the
    action-map fix just raised two games from 6 to 14 actions, which makes long searches expected.
    Observability first; the decision stays with a human reading the log.
    """

    def __init__(self, interval=300.0, label="run", stream=None):
        self.interval = float(interval)
        self.label = label
        self._last = time.time()
        self._stop = threading.Event()
        self._t = None
        self._stream = stream

    def beat(self, note=""):
        """Call after each episode / synthesis so the watchdog knows work happened."""
        self._last = time.time()
        self._note = note

    def start(self):
        def loop():
            while not self._stop.wait(self.interval):
                quiet = time.time() - self._last
                if quiet >= self.interval:
                    print("[watchdog] %s: no progress for %.0fs (last: %s)"
                          % (self.label, quiet, getattr(self, "_note", "n/a")), flush=True)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()


def stamp(rec, env=None, actions=None, game=None, extra=None):
    """Attach every provenance field to a record's `meta`. Call once at record creation."""
    meta = rec.setdefault("meta", {})
    meta["code_version"] = code_version()
    meta["action_fingerprint"] = action_fingerprint(env=env, actions=actions, game=game)
    meta["runmeta_version"] = 1
    if extra:
        meta.update(extra)
    return rec
