#!/usr/bin/env python
"""CONTROLLER ARMS: the LLM writes a reactive policy — no world model, no planner.

    python scripts/run_controller_flappy.py --mode llm  --rep 0
    python scripts/run_controller_flappy.py --mode ars  --rep 0

TWO ARMS, differing ONLY in who supplies the numeric constants:

    --mode llm   prompt: controller_only.txt
                 LLM writes `policy(state)` with its constants baked in.
                 Nothing is fitted. Improvement comes only from revision.

    --mode ars   prompt: controller_ars.txt
                 LLM writes `PARAMS` + `policy(state, params)`; Augmented Random
                 Search fits PARAMS against REAL returns between revisions. The
                 fitted values are fed back as evidence about the FORM.

Against the existing planning arm (unified_clean_h.txt: abstractions + world model
+ plan, sysid on the model's constants), that gives a three-way over WHO PICKS THE
NUMBERS — the same question the paper already asks about world models, asked again
for policies:

    LLM-only policy    structure LLM   constants LLM     search none
    LLM + ARS          structure LLM   constants fitted  search none
    TheoryCoder today  structure LLM   constants sysid   search planning

WHY A SEPARATE SCRIPT. The harness is built around cycling grounded operators;
a reactive policy has no plan to cycle and no horizon to exhaust, so wiring this
into run_autonomous_survive.py would mean a second execution path through code
every live arm depends on. Nothing here imports from that file's main loop, and
abstraction_prompts/general/unified_clean_h.txt is never read or written.

DEFAULTS match the archived flappy cell exactly (descriptive names, obs=objects,
DISCRETE actions, seed 42, haiku, 30 LLM calls, 20 episodes) so results sit
directly beside them.
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from theorycoder import runmeta as RM
from theorycoder.abstractions.objapi import NAMESPACE as OBJ_NAMESPACE, API_DOC
from theorycoder.abstractions.trajectory import format_trajectory_table
from theorycoder.llm_client import LLMClient

PROMPTS = {"llm": "abstraction_prompts/general/controller_only.txt",
           "ars": "abstraction_prompts/general/controller_ars.txt"}
HYPOTHESIS_PROMPT = "abstraction_prompts/general/controller_hyp.txt"
MISSIONS = {
    "flappy": ("Survive as long as possible and pass as many obstacles as you "
               "can. Score increases by 1 for each obstacle passed."),
    "pong":   ("Win the game. Return the ball past the opponent to score, and "
               "stop it getting past you. Score is your points minus theirs."),
    "freeway": ("Cross to the far side as many times as you can without being "
                "hit. Score increases by 1 for each crossing."),
    "breakout": ("Keep the ball in play and destroy as many bricks as you can. "
                 "Score increases for each brick destroyed."),
    "spaceinvaders": ("Destroy the aliens and avoid being hit. Score increases "
                      "for each alien destroyed."),
    "boxing": ("Land hits on the opponent and avoid being hit. Score is your "
               "hits minus theirs."),
    "skiing": ("Get down the course as fast as you can, passing between the "
               "gates. Score improves the better you do."),
    "tennis": ("Win points against the opponent. Score is your points minus "
               "theirs."),
    "seaquest": ("Score points and stay alive. Your score rises when you achieve "
                 "the game's objective."),
    "kangaroo": ("Score points and avoid losing. Your score rises when you make "
                 "progress toward the game's objective."),
    "mspacman": ("Score points and avoid being caught. Your score rises when you "
                 "collect things."),
}


# --------------------------------------------------------------------------- #
def _state_for_prompt(state, budget=1500, compact=False):
    """Serialise the state for the prompt WITHOUT losing the fields that matter.

    TWO MEASURED BUGS THIS FIXES.

    1. `json.dumps(state)[:1500]` truncates the TAIL, and `_project()` appends
       `score` / `won` / `lost` LAST. Measured: MsPacman's dump is 6,397 chars, so
       the agent optimising `score` never saw the `score` field at all. Kangaroo
       (857) and Seaquest (133) were unaffected, which is why this went unnoticed
       -- it only bites object-rich games, and those are exactly the ones we just
       started running.

    2. Truncating whole object lists silently hides ENTIRE CATEGORIES. Dropping
       half of `ghost` still tells the agent ghosts exist; dropping the last key
       tells it nothing.

    So: emit the scalar/terminal fields FIRST, then as many object categories as
    fit, then an explicit note naming any category that was abridged. The agent is
    told what was cut rather than silently shown a partial world.
    """
    prio = [k for k in ("score", "opponent_score", "won", "lost") if k in state]
    rest = [k for k in state if k not in prio]
    out, cut = {}, []
    for k in prio:
        out[k] = state[k]
    # BUG 3, MEASURED ON MSPACMAN. Truncating each long list to its FIRST 6 entries spends
    # the budget on whichever objects the tracker happens to list first, and spends it even
    # on lists that carry no information at all. MsPacman's reset state:
    #
    #   pill           150 entries, 6 shown ( 4%)   <- the entire point of the game
    #   pill_size      150 entries, every one [4.0, 2.0]        (identical)
    #   pill_velocity  150 entries, every one [0.0, 0.0]        1,800 chars of zeros
    #
    # So two of three long lists were uninformative while the one that decides the score
    # was 96% hidden. A constant list is fully described by ONE line -- "150 entries, all
    # [0.0, 0.0]" tells the agent strictly more than six samples of it -- and the freed
    # budget goes to the list that varies. When a `player` position exists, the entries
    # kept are the ones NEAREST the player, because a reactive policy needs the closest
    # food and the tracker's ordering is arbitrary with respect to that.
    #
    # GATED: `compact` defaults False so every archived cell stays comparable, exactly as
    # --settled-state is gated. Enable with --compact-state and A/B it.
    _player = None
    if compact:
        pv = state.get("player")
        if isinstance(pv, list) and pv and isinstance(pv[0], (int, float)):
            _player = pv
        elif isinstance(pv, list) and pv and isinstance(pv[0], (list, tuple)):
            _player = list(pv[0])
    for k in rest:
        v = state[k]
        if isinstance(v, list) and len(v) > 6:
            if compact and len(set(json.dumps(e, default=str) for e in v)) == 1:
                # constant list: describe it, do not sample it
                out[k] = "ALL %d ENTRIES IDENTICAL: %s" % (len(v), json.dumps(v[0], default=str))
                continue
            keep = 6
            if compact:
                keep = 12                      # budget freed by collapsing constant lists
                if _player is not None and v and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2:
                    try:
                        v = sorted(v, key=lambda e: (float(e[0]) - _player[0]) ** 2
                                                    + (float(e[1]) - _player[1]) ** 2)
                        cut.append("%s (%d of %d shown, NEAREST the player first)"
                                   % (k, min(keep, len(v)), len(state[k])))
                        out[k] = v[:keep]
                        if len(json.dumps(out, default=str)) > budget:
                            out.pop(k); cut[-1] = "%s (OMITTED -- too large)" % k
                            break
                        continue
                    except (TypeError, ValueError, IndexError):
                        pass
            out[k] = v[:keep]
            cut.append("%s (%d of %d shown)" % (k, keep, len(v)))
        else:
            out[k] = v
        if len(json.dumps(out, default=str)) > budget:
            out.pop(k)
            cut.append("%s (OMITTED -- too large)" % k)
            break
    txt = json.dumps(out, default=str)
    if cut:
        txt += "\n(abridged for length: %s. These keys DO exist in the real state.)" % "; ".join(cut)
    return txt


def extract_block(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no fenced python block in the response")
    return max(blocks, key=len)


def _normalize_change(text):
    """Stable key for counting an exact re-proposal after a refutation."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:240]


def parse_hypothesis_fields(text):
    """Parse the three required fields emitted by controller_hyp.txt.

    The model is asked to put these lines before its fenced Python block.  Only
    that prefix is eligible: fields copied into a code comment are malformed
    output, not a reason to reject the whole run.  Every malformed input is
    represented as ``UNPARSED`` and returned to the caller without raising.
    """
    if not isinstance(text, str):
        return {
            "change": None, "predict": None, "falsifier": None,
            "predicted_score": None, "change_key": "", "parse_ok": False,
            "parse_missing": ["CHANGE", "PREDICT", "FALSIFIER", "PREDICT.score"],
            "status": "UNPARSED",
        }
    # A valid response has the metadata before the first fenced block.  Taking
    # the prefix also prevents a policy comment such as ``# CHANGE: ...`` from
    # being mistaken for a required top-level field.
    candidate = text.split("```", 1)[0]

    def field(name):
        m = re.search(r"(?im)^\s*%s:\s*(.*?)\s*$" % name, candidate)
        return m.group(1).strip() if m else None

    change = field("CHANGE")
    predict = field("PREDICT")
    falsifier = field("FALSIFIER")
    predicted_score = None
    if predict:
        number = r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?"
        m = re.search(r"(?i)score\s*=\s*(%s)" % number, predict)
        if m:
            predicted_score = float(m.group(1))
    missing = [name for name, value in (
        ("CHANGE", change), ("PREDICT", predict), ("FALSIFIER", falsifier)
    ) if not value]
    if predicted_score is None:
        missing.append("PREDICT.score")
    return {
        "change": change,
        "predict": predict,
        "falsifier": falsifier,
        "predicted_score": predicted_score,
        "change_key": _normalize_change(change),
        "parse_ok": not missing,
        "parse_missing": missing,
        "status": "PENDING" if not missing else "UNPARSED",
    }


def stamp_hypothesis(hypothesis, measured_score):
    """Compare a numeric prediction with the measured episode score."""
    if not hypothesis or not hypothesis.get("parse_ok"):
        return
    predicted = float(hypothesis["predicted_score"])
    tolerance = max(1.0, 0.25 * abs(predicted))
    error = float(measured_score) - predicted
    hypothesis["measured_score"] = float(measured_score)
    hypothesis["absolute_error"] = abs(error)
    hypothesis["tolerance"] = tolerance
    hypothesis["status"] = "CONFIRMED" if abs(error) <= tolerance else "REFUTED"


def hypothesis_history_text(record):
    rows = []
    for revision in record.get("revisions", []):
        h = revision.get("hypothesis")
        if not h or not h.get("parse_ok"):
            continue
        rows.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                revision.get("ep", "?"),
                h.get("change", "")[:180].replace("|", "/"),
                h.get("predicted_score"),
                h.get("measured_score", "pending"),
                h.get("status", "PENDING"),
                h.get("falsifier", "")[:180].replace("|", "/"),
            )
        )
    if not rows:
        return "(no scored hypothesis history yet)"
    return ("| revision | CHANGE | predicted score | measured score | status | FALSIFIER |\n"
            "|---:|---|---:|---:|---|---|\n" + "\n".join(rows))


def add_repeat_metadata(record, hypothesis):
    """Mark whether this exact normalized CHANGE follows a refutation."""
    key = hypothesis.get("change_key", "")
    prior_refuted = []
    if key:
        for revision in record.get("revisions", []):
            h = revision.get("hypothesis") or {}
            if h.get("parse_ok") and h.get("status") == "REFUTED" \
                    and h.get("change_key") == key:
                prior_refuted.append(revision.get("ep"))
    hypothesis["prior_refuted_revisions"] = prior_refuted
    hypothesis["repeat_after_refuted"] = bool(prior_refuted)


def hypothesis_summary(record):
    scored = [
        r.get("hypothesis") for r in record.get("revisions", [])
        if r.get("hypothesis", {}).get("parse_ok")
        and r.get("hypothesis", {}).get("status") in {"CONFIRMED", "REFUTED"}
    ]
    confirmed = sum(h.get("status") == "CONFIRMED" for h in scored)
    repeats = [h for h in scored if h.get("repeat_after_refuted")]
    return {
        "scored_predictions": len(scored),
        "confirmed": confirmed,
        "refuted": len(scored) - confirmed,
        "calibration": (confirmed / len(scored)) if scored else None,
        "repeats_after_refuted": len(repeats),
        "repeat_change_keys": sorted({h.get("change_key") for h in repeats}),
        "unparsed": sum(
            not r.get("hypothesis", {}).get("parse_ok")
            for r in record.get("revisions", []) if r.get("hypothesis")
        ),
    }


def load_policy(src, mode):
    """Exec the returned code with the object API injected, as the runner does."""
    ns = dict(OBJ_NAMESPACE)
    ns["math"] = math
    exec(compile(src, "<controller>", "exec"), ns)
    fn = ns.get("policy")
    if not callable(fn):
        raise ValueError("no callable `policy` defined")
    params = ns.get("PARAMS")
    # CALLABLE IS NOT ENOUGH. A policy whose body does `from objects import
    # objects` execs fine and is callable, then raises ModuleNotFoundError on
    # EVERY tick -- observed on disc_llm_rep0, where the bird free-fell into the
    # ground 25 ticks at a time for four episodes. The object API is INJECTED,
    # not importable, so a function-level import of it always fails at call time.
    # Smoke-test on a real state here so the failure becomes a synthesis error
    # the revision loop is told about, instead of silent per-tick fallbacks.
    ns["__smoke__"] = None
    if mode == "ars":
        if not isinstance(params, (list, tuple)) or not params:
            raise ValueError("ars mode requires a non-empty PARAMS list")
        params = [float(x) for x in params]
    return fn, params


def smoke_test(fn, params, probe_state, actions):
    """Call the policy once on a real state; a raise here is a SYNTHESIS failure."""
    try:
        a = fn(probe_state) if params is None else fn(probe_state, params)
    except Exception as exc:
        raise ValueError("policy raised on a real state: %s: %s"
                         % (type(exc).__name__, exc))
    _, invalid = _coerce(a, actions)
    if invalid:
        raise ValueError("policy returned an invalid action %r on a real state" % (a,))


def save_episode_gif(env_factory, transitions, path):
    """Replay an episode's logged actions on a fresh seeded env and save a GIF.

    The planner and the WorldCoder baseline have emitted per-episode GIFs for a while;
    this runner never did, so controller episodes -- including every arm that solves
    flappy and the +21 pong arms -- had no visual artifact at all. `play(collect=True)`
    already returns transitions carrying `action`, which is all a replay needs.

    Best-effort by design: a missing `_render_pil` returns None rather than failing the
    run. A GIF is an artifact, never a result -- it must not be able to kill an arm.
    """
    try:
        env = env_factory()
        env.reset()
        render = getattr(env, "_render_pil", None)
        if render is None:
            return None
        frames = [render()]
        for tr in transitions:
            env.step(tr["action"])
            frames.append(render())
            if getattr(env, "lost", False) or getattr(env, "won", False):
                break
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sub = frames[::2] or frames
        sub[0].save(path, save_all=True, append_images=sub[1:], duration=50, loop=0)
        return path
    except Exception as e:
        print("[gif] skipped: %s: %s" % (type(e).__name__, e))
        return None


def _coerce(a, actions):
    """Validate an action. DISCRETE: membership in the action list. CONTINUOUS:
    `actions` is an (lo, hi) tuple and the action is one float, so validity is an
    interval check -- `a not in actions` would reject every legal thrust."""
    if isinstance(actions, tuple):
        lo, hi = actions
        try:
            v = float(a[0] if isinstance(a, (list, tuple)) else a)
        except Exception:
            return lo, True
        if v != v or v < lo or v > hi:
            return lo, True
        return [v], False
    return (a, False) if a in actions else (actions[-1], True)


def play(env_factory, fn, params, actions, max_ticks, collect=False,
         stall_ticks=0):
    """One real episode. Returns (score, ticks, transitions, n_bad).

    STALL_TICKS ends a STALEMATE. Measured on the +21 pong arm: episodes 1, 3, 5
    and 8 each ran the full 50,000-tick cap and scored -1 -- the paddle returned
    everything and neither side ever scored, so the game could not end and the
    runner burned the cap. Those four episodes alone were 200,000 of the 246,628
    env steps that arm spent reaching its best score. They taught the loop
    nothing: a hung game is not experience. Ending an episode once the score has
    not moved for `stall_ticks` cuts that ~5x with no effect on any episode that
    is actually progressing. Default 0 = OFF, so archived runs stay reproducible.
    """
    env = env_factory()
    trans, bad, t = [], 0, 0
    # FULL ACTION SEQUENCE, collected unconditionally. `trans` only holds actions when
    # collect=True (it exists for gif rendering), so before this the ONLY action record on a
    # controller episode was `action_hist` -- a histogram, which cannot be replayed. These envs
    # are deterministic given the seed, so (seed, acts) reproduces the episode exactly at ~6
    # bytes/step, which makes every frame/gif/diagnostic regenerable after the fact.
    acts = []
    errs = {}                          # exception text -> count
    last_change, last_score = 0, float(env.score)
    # MATCHED-HORIZON SNAPSHOT. Freeway scores by cumulative crossings, so the score at tick
    # H is what a run truncated at H would have reported. Capturing it mid-episode yields the
    # full-game number AND the truncated-comparable number from ONE run. 834 is where the
    # un-overridden FreewayEnv max_ticks=2500 (ALE frames, /3 at frame_skip 3) has been
    # stopping every freeway episode in this corpus.
    _HORIZONS = (834,)
    horizon_scores = {}
    while t < max_ticks and not env.lost and not env.won:
        if t in _HORIZONS:
            horizon_scores[t] = float(env.score)
        if stall_ticks and (t - last_change) >= stall_ticks:
            errs["__stalled__"] = t
            break
        s = env.get_obs()
        try:
            a = fn(s) if params is None else fn(s, params)
        except Exception as exc:
            # KEEP THE MESSAGE. The first version reported only a COUNT of failed
            # ticks, so the model was told "your policy raised 50 times" and never
            # what the error was -- four arms rewrote the same
            # `from objects import objects` for all 20 revisions without ever
            # learning it was a ModuleNotFoundError. A failure report that does
            # not name the failure cannot be acted on.
            k = "%s: %s" % (type(exc).__name__, exc)
            errs[k] = errs.get(k, 0) + 1
            a = None                       # _coerce counts it once, below
        a, invalid = _coerce(a, actions)
        if invalid:
            bad += 1
        env.step(a)
        acts.append(a)
        if float(env.score) != last_score:
            last_score, last_change = float(env.score), t
        if collect:
            trans.append({"state": s, "action": a, "actual_next": env.get_obs()})
        t += 1
    # A horizon NOT reached means the episode ended first, and the score at that horizon is
    # therefore the final score -- "score at H" is score at min(H, episode_end). Without this
    # the common case is silently empty: `t` increments at the END of the loop body, so an
    # episode that stops exactly at 834 never sees t == 834 inside the loop.
    for _h in _HORIZONS:
        horizon_scores.setdefault(_h, float(env.score))
    # Everything main() needs FROM THE ENV travels back here. `env` is local to play(); the
    # 2026-08-31 rail-provenance change read getattr(env, ...) in main() and raised NameError
    # on the first controller launch since (2026-09-07, all three kangaroo flat reps).
    extras = {"horizon_scores": horizon_scores,
              "env_max_ticks": getattr(env, "max_ticks", None),
              "env_turn_number": getattr(env, "turn_number", None)}
    return float(env.score), t, trans, (bad, errs), acts, extras


def ars_fit(env_factory, fn, params, actions, max_ticks, budget,
            n_dirs=8, step=0.15, noise=0.12, seed=0):
    """ARS-V1 on REAL returns. Only the numbers move; the code is never edited."""
    rng = random.Random(seed)
    W = list(params)
    used = 0
    def fitness(w):
        # ONE episode per evaluation. The first version called play() twice --
        # once for score, once for ticks -- which silently DOUBLED the sample
        # cost, the very quantity this arm exists to measure.
        sc, tk, _, _, _, _ = play(env_factory, fn, w, actions, max_ticks)
        return sc * 1000.0 + tk
    while used < budget:
        deltas = [[rng.gauss(0, 1) for _ in W] for _ in range(n_dirs)]
        rows = []
        for d in deltas:
            rp = fitness([w + noise * x for w, x in zip(W, d)])
            rm = fitness([w - noise * x for w, x in zip(W, d)])
            used += 2                      # one episode per fitness call, x2 (+/-)
            rows.append((rp, rm, d))
            if used >= budget:
                break
        sigma = 1e-6 + (sum((r[0] - r[1]) ** 2 for r in rows) / len(rows)) ** 0.5
        for rp, rm, d in rows:
            for i in range(len(W)):
                W[i] += step / (len(rows) * sigma) * (rp - rm) * d[i]
    return W, used


# --------------------------------------------------------------------------- #
# Single source of truth -- see game_registry.py for why four copies of
# this dict had already drifted apart.
try:
    from game_registry import OCATARI_GENERIC as _GENERIC
except Exception:                      # never break a live runner
    _GENERIC = {"spaceinvaders": "SpaceInvaders", "boxing": "Boxing",
        "skiing": "Skiing", "tennis": "Tennis", "pooyan": "Pooyan",
        "seaquest": "Seaquest", "kangaroo": "Kangaroo",
        "mspacman": "MsPacman", "breakoutoc": "Breakout"}

def _mission_for(game, env=None):
    """The mission string, WITHOUT inventing one for a game that lacks an entry.

    `MISSIONS` covers 11 games and omits breakoutoc, pooyan and montezuma, so
    `MISSIONS[a.game]` raised `KeyError: 'breakoutoc'` -- the reason breakoutoc has never
    had a controller arm and sits as an orphan (planner-only) row in the results table.

    The fallback is the ENVIRONMENT's own mission, not a new hand-written line, and that
    choice matters for disclosure: the hand-written entries name domain objects ("destroy
    as many bricks", "Destroy the aliens") while the generic adapter's mission is
    domain-free ("Increase your score and avoid losing"). A game reaching the fallback
    therefore receives STRICTLY LESS information than one with an entry -- a conservative
    asymmetry, not a new give-away -- and that must be stated wherever such a cell is
    quoted. See docs/WHAT_WE_GIVE_THE_AGENT.md section 5.
    """
    if game in MISSIONS:
        return MISSIONS[game]
    return getattr(env, "mission", None) or "Increase your score and avoid losing."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["llm", "ars"])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--query-mode", default="anthropic",
                    help="how to reach the language model: anthropic (the default; "
                         "reads ANTHROPIC_API_KEY), openai_direct, groq, or custom "
                         "(an OpenAI-compatible endpoint set by CUSTOM_BASE_URL and "
                         "CUSTOM_API_KEY)")
    ap.add_argument("--max-llm-calls", type=int, default=30)
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--max-ticks", type=int, default=50000)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--ars-budget", type=int, default=200,
                    help="real episodes of ARS per revision (ars mode only)")
    ap.add_argument("--game", default="flappy", choices=sorted(set(list(_GENERIC) + ["flappy", "pong", "freeway",
                                            "breakout", "montezuma"])),
                    help="pong uses the FULL-GAME protocol (play to 21, score = "
                         "point differential) so a controller number sits beside "
                         "our planning cells and PoE-World's published -12.33.")
    ap.add_argument("--champion-banking", default="off", choices=["off", "on"],
                    help="BEHAVIOURAL. off (default) = the architecture that "
                         "produced every completed cell, frozen at tag "
                         "arch-pre-champion-banking: every episode plays the "
                         "NEWEST revision, so a winning policy is replaced the "
                         "moment the next revision lands. on = when a challenger "
                         "regresses hard against the banked champion, REPLAY the "
                         "champion instead of the challenger, and tell the reviser "
                         "it is improving the champion. Motivated by five recorded "
                         "instances of the loop destroying its own winner: the "
                         "planning +10 rewritten next episode; the haiku no-fit "
                         "controller collapsing to -21 after +14/+13; gpt-4o "
                         "scoring +9 at ep6 then -21 for fourteen straight; the "
                         "flappy champion frozen on a crashing policy; the moon "
                         "refit destroyed by the next rewrite.")
    ap.add_argument("--bank-regress-frac", type=float, default=0.5,
                    help="a challenger counts as REGRESSED when its score falls "
                         "below this fraction of the champion's (for negative "
                         "champions, when it is worse by the same margin). 0.5 is "
                         "deliberately loose: banking on every dip would freeze "
                         "exploration, and the failures we measured were total "
                         "collapses (+14 -> -21), not marginal dips.")
    ap.add_argument("--target-score", type=int, default=10**6,
                    help="env win threshold. flappy_env defaults to 5, which ends "
                         "the episode on a WIN and looks exactly like a score "
                         "ceiling. Effectively disabled here so episodes end by "
                         "DYING, which is what we want to measure.")
    ap.add_argument("--continuous", action="store_true",
                    help="continuous action space: Box(0,1) thrust instead of "
                         "flap/noop. The env applies a CONVEX BLEND "
                         "vy=(1-t)*vy+t*FLAP, so t=0 is exactly noop and t=1 is "
                         "exactly flap -- the discrete set is a strict SUBSET, and "
                         "any drop is attributable to searching a real interval "
                         "rather than to lost capability.")
    ap.add_argument("--stall-ticks", type=int, default=0,
                    help="End an episode when the score has not moved for this "
                         "many ticks. 0 = off. On the +21 pong arm, episodes 1/3/5/8 "
                         "each ran the full 50,000-tick cap at score -1 -- hung "
                         "games, 200,000 of the 246,628 steps that arm spent "
                         "reaching its best. All four happened BEFORE any champion "
                         "existed, so no stop-revising rule can recover them; only "
                         "this can.")
    ap.add_argument("--stop-after-stale", type=int, default=0,
                    help="After this many consecutive revisions fail to beat the "
                         "champion, stop synthesizing and replay the champion. "
                         "0 = off. Complements --stall-ticks rather than replacing "
                         "it: this saves LLM calls and episodes AFTER a good policy "
                         "exists, that one saves env steps inside bad episodes.")
    ap.add_argument("--settled-state", action="store_true",
                    help="Show the prompt a SETTLED state instead of the reset "
                         "state: advance until the object inventory stops "
                         "changing. Reset is unrepresentative on every game "
                         "checked -- pong hides the ball ~20 decisions, seaquest "
                         "dumps 133 chars because only `player` exists, mspacman "
                         "reports all 4 ghosts at [0,0]. OFF by default: it "
                         "changes what the agent sees, so archived cells stay "
                         "comparable and this gets its own A/B.")
    ap.add_argument("--env-config", default=None,
                    help="VARIANT (flappy only): JSON file or inline JSON with "
                         "'physics' overrides and/or a 'shift_schedule', e.g. "
                         "configs/flappy_variants/heavy.json. The PLANNER runner has "
                         "had this since the variant work; the controller runner "
                         "never did, which is why no controller has ever been "
                         "evaluated on moon/heavy gravity.")
    ap.add_argument("--replay-src", default=None,
                    help="ZERO-LLM-CALL replay: path to a base_scores record whose "
                         "champion_src is loaded and played unchanged. Forces "
                         "--max-llm-calls 0 so the loop never synthesizes or "
                         "revises. This is how a controller is tested on a variant "
                         "without letting it re-learn the new physics.")
    ap.add_argument("--seed-traj", default=None,
                    help="probe json whose transitions fill the __TRAJECTORY__ slot on "
                         "EPISODE 1, so controller #1 can see velocity. Use a RANDOM "
                         "probe; a controller-derived one is circular.")
    ap.add_argument("--seed-src", default=None,
                    help="WARM START: path to a saved controller record whose "
                         "champion_src is played once as revision 0, then the "
                         "normal LLM revision loop begins. Unlike --replay-src, "
                         "this does not disable synthesis.")
    ap.add_argument("--hypothesis-history", action="store_true",
                    help="LLM mode: use controller_hyp.txt, parse CHANGE/PREDICT/"
                         "FALSIFIER, score predictions, and feed the accumulated "
                         "history table into the next revision.")
    ap.add_argument("--mission-source", default="generic",
                    choices=["controller", "env", "generic", "minimal"],
                    help="WHICH mission string this arm gets. `controller` (default, and what "
                         "every archived cell ran with) reads the hand-written MISSIONS dict, "
                         "falling back to the env's generic line. `env` always uses the env's "
                         "generic line -- which is what the planner and BOTH WorldCoder columns "
                         "get, since they read `env.mission` directly.\n\n"
                         "WHY THIS EXISTS. The dict is ours, not OCAtari's (OCAtari ships no "
                         "mission text), and it covers 11 games while the other three methods "
                         "never read it. On two games its wording names domain objects: "
                         "spaceinvaders 'Destroy the aliens ... for each alien destroyed' and "
                         "pong 'Return the ball past the opponent'. Those are two of the four "
                         "games where the controller beats the random floor, so the table "
                         "currently cannot separate 'better method' from 'told more' there. "
                         "Setting `env` matches the controller to the other three columns and "
                         "prices that sentence directly. Default is unchanged so archived "
                         "cells stay reproducible.",)
    ap.add_argument("--win-truth", default="off", choices=["on", "off"],
                    help="Append the env's real win condition (from --target-score, OUR "
                         "config) to the mission. Counters invented victory thresholds "
                         "like seaquest's synthesized `if score >= 20: freeze`.")
    ap.add_argument("--stagnation-feedback", default="off", choices=["on", "off"],
                    help="Report in revision feedback when the score stopped moving long "
                         "before the episode ended, with the dominant action in that "
                         "stretch. Feedback on the agent's own trace; no game knowledge.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--artifact-dir", default=None,
                    help="Write per-episode GIFs, the champion source, and every "
                         "revision's source into this directory. Off unless given, so "
                         "archived runs keep their current on-disk shape.")
    a = ap.parse_args()

    if a.replay_src and a.seed_src:
        raise SystemExit("--replay-src and --seed-src are mutually exclusive")
    if a.hypothesis_history and a.mode != "llm":
        raise SystemExit("--hypothesis-history is only supported with --mode llm")

    # ---- variant physics (flappy only) -------------------------------------
    _envkw, _variant = {}, None
    if a.env_config:
        raw = (open(a.env_config).read() if os.path.exists(a.env_config)
               else a.env_config)
        cfg = json.loads(raw)
        _variant = cfg.get("name") or a.env_config
        if cfg.get("physics"):
            _envkw["physics"] = cfg["physics"]
        if cfg.get("shift_schedule"):
            _envkw["shift_schedule"] = cfg["shift_schedule"]
        if a.game != "flappy":
            raise SystemExit("--env-config is flappy-only; %r has no physics "
                             "override hook." % a.game)
        print("[variant] %s -> %s" % (_variant, _envkw))

    if a.game in _GENERIC:
        # one adapter for every OCAtari game; fixes the prev_xy==(0,0) velocity
        # sentinel and multi-instance categories once, for all of them
        from ocatari_generic_env import OCAtariGenericEnv
        _name = _GENERIC[a.game]

        def env_factory():
            return OCAtariGenericEnv(_name, seed=a.seed, frame_skip=3,
                                     velocity=True, target_score=a.target_score)
    elif a.game == "freeway":
        # OCAtari Freeway -- EXTERNAL benchmark, same parser PoE-World reads.
        from ocatari_freeway_env import FreewayEnv

        def env_factory():
            # BUG FIX 2026-09-10: this ignored BOTH runner flags. FreewayEnv defaults to
            # target_score=10 and max_ticks=2500 (= 834 decisions at frame_skip 3, i.e. 31 % of a
            # real 2,731-decision game), so every freeway cell ever run here was cut short twice
            # over while --target-score 1000000 and --max-ticks 6000 sat unused. Episodes were
            # labelled 'env_rail'/'death_or_win' rather than truncated, which is why it went
            # unnoticed. Pass both through so freeway obeys the same rails as every other game.
            return FreewayEnv(seed=a.seed, frame_skip=3,
                              target_score=a.target_score, max_ticks=a.max_ticks * 3)
    elif a.game == "breakout":
        # OUR breakout (reflection physics), not OCAtari's -- internal domain.
        from breakout_env import BreakoutEnv

        def env_factory():
            return BreakoutEnv(seed=a.seed)
    elif a.game == "pong":
        from ocatari_pong_env import PongEnv

        def env_factory():
            # full_game + frame_skip 3 + velocity ON: identical observation and
            # scoring protocol to the pong_fs3 planning cells, so the ONLY thing
            # that differs is planner-with-world-model vs reactive controller.
            return PongEnv(seed=a.seed, frame_skip=3, full_game=True,
                           velocity=True, target_score=a.target_score)
    else:
        from flappy_env import FlappyEnv

        def env_factory():
            # DISCRETE actions (no continuous_actions), CONTINUOUS pixel state via
            # obs=objects, descriptive names -- identical to the archived flappy cell.
            # target_score DEFAULTS TO 5 in flappy_env, and play() stops on env.won,
            # so leaving it unset silently ends every good episode at exactly 5 --
            # which is precisely what happened: 15 runs across two action spaces and
            # four fitting budgets all "capped" at 5, every one of them at 358 ticks,
            # and it was a WIN rather than a plateau. The archived planning cell
            # passes --max-score 1000000 for the same reason. Third time a rail has
            # been read as a result in this project (--max-score 15 on helicopter,
            # --max-ticks 20000 on flappy).
            return FlappyEnv(entity_names="descriptive", obs_mode="objects",
                             seed=a.seed, continuous_actions=a.continuous,
                             target_score=a.target_score, **_envkw)

    probe = env_factory()
    if a.continuous:
        # actions_set is None for a Box space (loud failure if iterated); the
        # policy returns a float and validity is an interval check, not membership.
        lo = float(probe.action_space.low[0]); hi = float(probe.action_space.high[0])
        actions = (lo, hi)
        action_desc = ("a single FLOAT thrust in [%g, %g]. %g behaves exactly like "
                       "no thrust and %g exactly like full thrust; intermediate "
                       "values blend between them." % (lo, hi, lo, hi))
        fallback = lo
    else:
        actions = list(probe.actions_set)
        action_desc = repr(actions)
        fallback = actions[-1]
    # THE RESET STATE IS NOT A REPRESENTATIVE STATE, and it is what we paste into
    # the prompt. Measured on three separate games:
    #
    #   pong      -- `ball` and `enemy` do not exist for ~20 decisions (the serve),
    #                which also broke the in-model fit and PoE-World's planner.
    #   seaquest  -- the whole dump is 133 chars because only `player` exists;
    #                `oxygenbar`, `diver` and `shark` all appear later, so the agent
    #                wrote a controller for a game whose objects it never saw, and
    #                never once referenced the oxygen bar.
    #   mspacman  -- all four ghosts report [0,0] and all four powerpills share one
    #                coordinate at tick 0; by tick 1 the pills separate properly.
    #
    # Three games, one cause. So advance a few decisions until the CATEGORY SET
    # stops growing and positions have separated, and show the agent that instead.
    # Domain-free: the criterion is "the object inventory stopped changing", not
    # any knowledge of what the objects are.
    def _settled_state(env, window=30):
        """Return the RICHEST state seen in a short window, not the first stable one.

        First attempt used "object inventory unchanged for 3 decisions" and it
        FAILED on pong -- the very game that motivated the fix. Pong's inventory is
        stable for ~20 decisions while the ball is still absent, so the criterion
        fired at decision 3 and returned the same impoverished state as reset.

        Taking the max-category state over a fixed window cannot be fooled that
        way: whatever appears late is still captured, and no assumption is made
        about WHEN things settle. Ties go to the later state, which also has
        positions separated (mspacman's ghosts read [0,0] at reset).
        """
        best, best_n = env.get_obs(), -1
        for _ in range(window):
            s_ = env.get_obs()
            n = len({k for k, v in s_.items()
                     if isinstance(v, list) and v and isinstance(v[0], list)})
            if n >= best_n:
                best, best_n = s_, n
            if getattr(env, "lost", False) or getattr(env, "won", False):
                break
            try:
                env.step(fallback)
            except Exception:
                break
        return best

    raw_state = _settled_state(env_factory()) if a.settled_state else probe.get_obs()

    # PER-MODEL COMPLETION CAP. gpt-4o rejects max_tokens>16384 with a 400, and
    # our default is 32000 -- so every call failed and a full 20-episode arm
    # produced ZERO episodes before anyone noticed. Clamp rather than let the
    # whole run die on a limit the provider advertises in its error message.
    _CAPS = {"gpt-4o": 16384, "gpt-4o-2024-08-06": 16384, "gpt-4o-mini": 16384}
    for _m, _cap in _CAPS.items():
        if a.model.startswith(_m) and a.max_tokens > _cap:
            print("[cap] %s supports at most %d completion tokens; lowering "
                  "--max-tokens from %d." % (a.model, _cap, a.max_tokens))
            a.max_tokens = _cap
            break

    prompt_path = HYPOTHESIS_PROMPT if a.hypothesis_history else PROMPTS[a.mode]
    prompt_tpl = open(prompt_path).read()
    client = LLMClient(query_mode=a.query_mode, language_model=a.model,
                       max_tokens=a.max_tokens)

    _tag = "cont" if a.continuous else "disc"
    out = a.out or ("run_logs/base_scores/%s_ctrl_%s_%s_rep%d.json"
                    % (a.game, _tag, a.mode, a.rep))
    # ---------------------------------------------------------------------- #
    # DO NOT LET A SHORT RUN DESTROY A LONGER ONE AT THE SAME PATH.
    #
    # The path above is a pure function of (game, tag, mode, rep) -- it carries no
    # run identity -- so ANY second run of the same config writes over the first.
    # Measured cost on 2026-08-17: three `--max-llm-calls 0` smoke tests wrote
    # 0-episode records over completed 20-episode arms, including flappy's best
    # controller (780.0). The data was recoverable from the logs only because the
    # logs happened to survive; nothing in the code noticed.
    #
    # The guard reads the incumbent's episode count ONCE, here, before the run can
    # write anything. It cannot be a per-flush comparison: flush() runs after every
    # episode, so a legitimate rerun's first flush would always look "shorter" than
    # the finished record and would divert forever. Deciding up front means a probe
    # (which never reaches the incumbent's count) always diverts, while a real rerun
    # takes the canonical name the moment it has matched what it is replacing.
    _incumbent = 0
    if not a.out and os.path.exists(out):
        try:
            from record_io import episodes_done as _eps_done
            _incumbent = _eps_done(json.load(open(out)))
        except Exception:
            _incumbent = 0            # an unreadable incumbent protects nothing
        if _incumbent:
            print("[record] incumbent at %s has %d scored episode(s); this run must "
                  "match that before it may overwrite" % (out, _incumbent), flush=True)
    _out_canonical, _diverted = out, None
    # PROVENANCE. The planner runner records git sha + per-source hashes via
    # RunRecorder; this script writes its own record and had none -- so our BEST
    # results (the pong controllers) could not be traced to the code that made
    # them. Hash the three files that actually determine behaviour: this runner,
    # the prompt, and the env.
    def _hash(path):
        import hashlib
        try:
            return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        except Exception:
            return None
    _envfile = {"flappy": "flappy_env.py", "pong": "ocatari_pong_env.py",
                "freeway": "ocatari_freeway_env.py",
                "breakout": "breakout_env.py"}.get(a.game,
                                                   "ocatari_generic_env.py")
    _prov = {
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip() or None,
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                                         capture_output=True, text=True).stdout.strip()),
        "source_hashes": {"runner": _hash(__file__),
                          "prompt": _hash(prompt_path),
                          "env": _hash(_envfile)},
        "argv": ["scripts/run_controller_flappy.py"] + sys.argv[1:],
    }
    rec = {"meta": {"mode": a.mode, "rep": a.rep, "model": a.model,
                    "max_ticks": a.max_ticks,
                    "query_mode": a.query_mode, **_prov, "seed": a.seed,
                    # Was hardcoded "flappy" from when this script only ran flappy,
                    # so EVERY controller record written before 2026-08-16 mislabels
                    # its game -- boxing_ctrl_*.json and breakout_ctrl_*.json both
                    # claim meta.game=='flappy'. Any table that groups by meta.game
                    # silently piles all 37 controller arms onto flappy. Group by
                    # the filename or by this field only for runs after the fix.
                    "prompt": prompt_path, "game": a.game,
                    "variant": _variant, "env_kwargs": _envkw,
                    "hypothesis_history": bool(a.hypothesis_history),
                    "actions": actions, "started": time.time()},
           "episodes": [], "calls": [], "revisions": []}

    # TIMING. The planner and the WorldCoder baseline both build a
    # `theorycoder.runrecord.RunRecord`, whose `totals` are filled by `rec.timer(...)` context
    # managers; this runner hand-rolls a plain dict and so recorded NOTHING -- 0 of 42 arms had
    # `wall_sec`, `llm_sec`, `plan_sec`, `env_sec` or `sysid_sec`. That is the same shape as the
    # token bug: the field was structurally ABSENT, not zero, so any timing table would have
    # read "the controller is instant" rather than "unmeasured".
    #
    # Retrofitting RunRecord mid-experiment would change the record schema under 42 completed
    # arms, so the accumulators are kept here and written into the same `totals` KEYS the other
    # runners use, which is what makes the column comparable across methods.
    #
    # `plan_sec` is deliberately 0.0 and not omitted: this method does no search, and an
    # explicit zero is a measurement ("no planning time") while a missing key is not. Same for
    # `sysid_sec`. `synth_sec` is the controller's analogue of plan_sec -- the time spent turning
    # an LLM response into a runnable policy (parse + smoke test) -- and is reported separately
    # so `wall_sec` does not silently absorb it.
    _T = {"llm": 0.0, "env": 0.0, "synth": 0.0, "gif": 0.0}
    _T0 = time.time()

    # PROVENANCE. This runner does not use `RunRecord`, so it gets none of that class's
    # code-identity fields for free. Stamped here with the same content so a controller arm and a
    # planner arm can be compared for "did these run the same code and the same action map".
    # `actions` rather than an env object: the env is constructed per episode by `env_factory`,
    # while `actions` is the resolved set this arm will actually use.
    RM.stamp(rec, actions=actions, game=a.game)

    # Resolve the mission ONCE. `MISSIONS` omits breakoutoc/pooyan/montezuma, which is why
    # `MISSIONS[a.game]` raised KeyError and those games could never get a controller arm.
    # The short-circuit means an env is only constructed for a game that needs the fallback.
    if getattr(a, "mission_source", "controller") in ("generic", "minimal"):
        from theorycoder.missions import resolve as _mresolve
        _MISSION, _MSRC = _mresolve(a.game, None, a.mission_source)
    elif getattr(a, "mission_source", "controller") == "env":
        _MISSION = (getattr(env_factory(), "mission", None)
                    or "Increase your score and avoid losing.")
        _MSRC = "env.mission (--mission-source env)"
    else:
        _MISSION = _mission_for(a.game, env_factory() if a.game not in MISSIONS else None)
        _MSRC = ("MISSIONS[%s]" % a.game) if a.game in MISSIONS else "env.mission (no dict entry)"
    # RECORDED. No controller or planner record in this corpus stores the mission it ran
    # with, so "what was this cell told?" is answerable only by reading the code at the
    # commit it ran on. That is exactly the class of undocumented per-game input that
    # already produced three misreadings in this project (pong's protocol, freeway's
    # frame_skip, the 3-episode floors).
    rec["meta"]["mission_source"] = _MSRC
    rec["meta"]["mission"] = _MISSION
    print("[mission] %s -> %r" % (_MSRC, _MISSION[:90]), flush=True)
    # WIN-CONDITION TRUTH (--win-truth on). Measured failure this addresses: on seaquest the
    # synthesized policy contained `if score >= 20: return "noop"` with the comment "If we've
    # achieved the objective, freeze completely" -- every episode scored exactly 20 and then
    # idled until death or the cap. The model INVENTED a victory threshold because the mission
    # says "achieve the game's objective" without saying what winning is, and nothing in the
    # feedback ever contradicted the theory ("RETURN: score 20" reads as success).
    #
    # This is not game knowledge: `a.target_score` is OUR config, the same class of harness
    # fact as the PRIMITIVE ACTIONS list. When the threshold is effectively unreachable the
    # truth is "there is no winning score", and saying so kills the invented-victory failure
    # without naming a single game concept. Gated off so archived cells stay comparable.
    if getattr(a, "win_truth", "off") == "on":
        if a.target_score >= 10 ** 5:
            _MISSION += (" There is NO winning score in this environment: the score "
                         "accumulates until the episode ends, so never stop trying to "
                         "score. Any score threshold you invent as a stopping point is "
                         "wrong.")
        else:
            _MISSION += (" The episode is WON when the score reaches %d." % a.target_score)

    def flush():
        # See the _incumbent note above: a run writes the canonical path only once it
        # has produced at least as many scored episodes as the record it would
        # replace. Until then it writes a `.short` sibling, so the shorter run's data
        # still lands somewhere and the longer run survives untouched.
        nonlocal _diverted
        dest = _out_canonical
        n = len([e for e in (rec.get("episodes") or [])
                 if isinstance(e.get("score"), (int, float))])
        if _incumbent and n < _incumbent:
            dest = _out_canonical + ".short"
            if _diverted != dest:
                print("[record] %d/%d episodes -- writing %s instead, NOT "
                      "overwriting the longer incumbent"
                      % (n, _incumbent, os.path.basename(dest)), flush=True)
                _diverted = dest
        elif _diverted:
            # Graduated: this run has matched the incumbent and may take the real
            # name. Drop the sibling so it cannot be mistaken for a separate arm.
            try:
                os.remove(_diverted)
            except OSError:
                pass
            print("[record] %d episodes >= incumbent %d -- taking %s"
                  % (n, _incumbent, os.path.basename(dest)), flush=True)
            _diverted = None
        # Written on EVERY flush, not only at the end, so a killed arm still carries the time it
        # actually spent. Six of the arms in this project were terminated mid-run; an end-only
        # write would have left all of them with no timing at all.
        _t = rec.setdefault("totals", {})
        _t["wall_sec"] = round(time.time() - _T0, 3)
        _t["llm_sec"] = round(_T["llm"], 3)
        _t["env_sec"] = round(_T["env"], 3)
        _t["synth_sec"] = round(_T["synth"], 3)
        _t["gif_sec"] = round(_T["gif"], 3)
        _t["plan_sec"] = 0.0      # explicit: this method searches nothing
        _t["sysid_sec"] = 0.0     # explicit: no constant fitting in the controller
        _t["other_sec"] = round(max(0.0, _t["wall_sec"] - _T["llm"] - _T["env"]
                                    - _T["synth"] - _T["gif"]), 3)
        # Retry/error accounting. Three arms in this project died on HTTP 429 and the records
        # showed only a short run -- nothing said the cause was the API rather than the method.
        # Read off the client's own timing events so it cannot disagree with the token numbers,
        # which come from the same place.
        try:
            rec["llm_health"] = RM.llm_health(client)
        except Exception:
            pass
        with open(dest, "w") as f:
            json.dump(rec, f, indent=2, default=str)

    cur_src, cur_fn, cur_params = None, None, None
    if a.replay_src:
        # ZERO-LLM REPLAY. Load a finished champion and play it unchanged, so the
        # score measures TRANSFER of the learned controller rather than its
        # ability to re-learn the new physics from scratch.
        _r = json.load(open(a.replay_src))
        cur_src = _r.get("champion_src") or next(
            (x["src"] for x in reversed(_r.get("revisions") or []) if x.get("src")), None)
        if not cur_src:
            raise SystemExit("no champion_src in %s" % a.replay_src)
        cur_fn, cur_params = load_policy(cur_src, a.mode)
        _cp = _r.get("champion_params")
        if a.mode == "ars" and _cp:
            cur_params = list(_cp)
        smoke_test(cur_fn, cur_params, raw_state, actions)
        a.max_llm_calls = 0                     # the loop can never synthesize
        rec["meta"]["replay_src"] = a.replay_src
        print("[replay] %d chars from %s, params=%s -- ZERO LLM calls"
              % (len(cur_src), os.path.basename(a.replay_src), cur_params))
    if a.seed_src:
        # WARM START. Unlike replay, keep the loaded champion in the revision
        # loop. It is played once as ep=0's revision, then ep=1 is the first
        # normal LLM revision. Do not smoke-test against target actions here:
        # the intended experiment seeds Breakout from Pong, so invalid source
        # action names are part of the measured transfer evidence and are fed
        # back as bad actions on the next revision.
        _r = json.load(open(a.seed_src))
        cur_src = _r.get("champion_src") or next(
            (x["src"] for x in reversed(_r.get("revisions") or []) if x.get("src")), None)
        if not cur_src:
            raise SystemExit("no champion_src in %s" % a.seed_src)
        cur_fn, cur_params = load_policy(cur_src, a.mode)
        _cp = _r.get("champion_params")
        if a.mode == "ars" and _cp:
            cur_params = list(_cp)
        rec["meta"]["seed_src"] = a.seed_src
        rec["revisions"].append({
            "ep": 0, "ok": True, "seed": True, "params": cur_params,
            "chars": len(cur_src), "src": cur_src,
        })
        print("[seed] %d chars from %s, params=%s -- revision 0"
              % (len(cur_src), os.path.basename(a.seed_src), cur_params))
    _last_score = None
    stale_revisions = 0      # consecutive revisions that failed to beat the champion
    _stop_announced = False
    champ = None                      # (score, src, params)
    # EPISODE-1 TRAJECTORY SEED (flag-gated, inert when off).
    #
    # WHY. Controller #1 is synthesized from `__RAW_STATE__` -- a SINGLE state -- with this
    # string in the `__TRAJECTORY__` slot. A single frame carries positions but no velocity,
    # so the first controller is structurally blind to dx/dy and cannot express "move toward
    # where the ball WILL be". Measured on pong: the first win lands at episode 4-5, i.e.
    # only after `format_trajectory_table(tail)` starts supplying real motion from episode 2.
    #
    # The seed MUST be a policy-free trajectory (random probe). Seeding from a winning
    # controller's trajectory would be circular -- you cannot have a good controller before
    # you have one -- and would make any episode-1 result meaningless.
    feedback = "(no play yet — this is your first controller)"
    if getattr(a, "seed_traj", None):
        _st = json.load(open(a.seed_traj))
        _tr = _st.get("transitions", _st) if isinstance(_st, dict) else _st
        feedback = format_trajectory_table(_tr)
        print("[seed-traj] episode 1 gets %d transitions from %s (velocity is now observable)"
              % (len(_tr), os.path.basename(a.seed_traj)))
    calls = 0

    for epi in range(1, a.max_episodes + 1):
        # ---- synthesize / revise -------------------------------------------
        revision_entry = None
        # STOP REVISING WHEN REVISION HAS STOPPED HELPING. Measured on the +21
        # pong arm: it reached the CEILING at ep10 (+21 is 21-0, the maximum the
        # game can return) and then kept synthesizing for ten more episodes,
        # scoring +2, -12, -17, -6, +3 -- every fresh attempt worse, every other
        # episode saved only by banking replaying the champion. The loop has no
        # notion of "done", because it revises unconditionally. This is
        # game-agnostic on purpose: "beat the champion" works where there is no
        # ceiling to detect, which is every game except pong.
        if (a.stop_after_stale and champ is not None
                and stale_revisions >= a.stop_after_stale):
            if not _stop_announced:
                print("[stop] %d consecutive revisions failed to beat the champion "
                      "(%g); replaying it and skipping synthesis. %d LLM calls left "
                      "unspent." % (stale_revisions, champ[0][0],
                                    a.max_llm_calls - calls))
                _stop_announced = True
            cur_fn, cur_params = load_policy(champ[1], a.mode)
            cur_src = champ[1]
            if champ[2] is not None:
                cur_params = list(champ[2])
        elif calls < a.max_llm_calls and (cur_fn is None or epi > 1):
            parsed_hypothesis = None
            p = (prompt_tpl
                 .replace("__OBJAPI__", API_DOC)
                 .replace("__ABSTRACTIONS__", cur_src or "(nothing yet)")
                 .replace("__TRAJECTORY__", feedback)
                 .replace("__HYPOTHESIS_HISTORY__",
                          hypothesis_history_text(rec) if a.hypothesis_history
                          else "(hypothesis history disabled)")
                 .replace("__ACTIONS__", action_desc)
                 .replace("__RAW_STATE__", _state_for_prompt(raw_state))
                 .replace("__MISSION__", _MISSION))
            t0 = time.time()
            # Accumulated in a `finally` below rather than after a successful query, because a
            # FAILED call still costs wall-clock and we had 429 storms whose retry-backoff time
            # (5s + 10s + 20s + 40s) would otherwise vanish from llm_sec entirely -- the arms
            # that lost the most time to the API would have looked like the cheapest.
            _llm_charged = False
            try:
                text, _ = client.query(p)
                calls += 1
                # TOKENS. The planner records these via RunRecorder.log_call; this runner
                # never did, so `calls[]` held only {ep, seconds} and EVERY controller arm
                # in the corpus has no token data -- the results table's controller token
                # column is structurally empty and cannot be backfilled. LLMClient already
                # accumulates usage into `timing["events"]`, so this reads what is already
                # there rather than counting anything itself.
                _ev = ((getattr(client, "timing", None) or {}).get("events") or [])
                _last = _ev[-1] if _ev else {}
                _dt = time.time() - t0
                _T["llm"] += _dt
                _llm_charged = True
                # everything after this point in the try -- extract, load_policy,
                # smoke_test -- is SYNTHESIS time, not LLM time. Split here so a slow
                # smoke test is never reported as a slow model.
                _t_synth0 = time.time()
                rec["calls"].append({
                    "ep": epi, "seconds": round(_dt, 2),
                    "tokens_in": _last.get("prompt_tokens"),
                    "tokens_out": _last.get("completion_tokens"),
                })
                # Keep a running total so a reader does not have to sum the list, and so a
                # provider that omits per-call usage still yields a usable arm-level number.
                _ti = sum(c.get("tokens_in") or 0 for c in rec["calls"])
                _to = sum(c.get("tokens_out") or 0 for c in rec["calls"])
                rec.setdefault("totals", {})
                rec["totals"]["tokens_in"] = _ti
                rec["totals"]["tokens_out"] = _to
                if a.hypothesis_history:
                    parsed_hypothesis = parse_hypothesis_fields(text)
                    add_repeat_metadata(rec, parsed_hypothesis)
                src = extract_block(text)
                fn, params = load_policy(src, a.mode)
                # SMOKE TEST BEFORE ADOPTING IT. The first version assigned first
                # and tested after, so a policy that raises on every tick was
                # already installed by the time the test failed -- the run then
                # played 25 ticks of fallback and died, exactly the failure the
                # test exists to prevent.
                smoke_test(fn, params, raw_state, actions)
                cur_src, cur_fn, cur_params = src, fn, params
                revision_entry = {"ep": epi, "ok": True, "params": params,
                                  "chars": len(src), "src": src}
                if parsed_hypothesis is not None:
                    revision_entry["hypothesis"] = parsed_hypothesis
                rec["revisions"].append(revision_entry)
            except Exception as exc:
                revision_entry = {"ep": epi, "ok": False,
                                  "error": "%s: %s" % (type(exc).__name__, exc)}
                if parsed_hypothesis is not None:
                    revision_entry["hypothesis"] = parsed_hypothesis
                rec["revisions"].append(revision_entry)
                print("[ep %d] synthesis failed: %s" % (epi, exc))
                flush()
                if cur_fn is None:
                    continue
            finally:
                # Charge the time even when the block exits by exception or `continue`.
                # A failed query is the case that matters: its retry backoff (5+10+20+40s
                # on a 429) is real wall-clock, and leaving it uncharged would make the
                # arms that lost the most time to the API look like the cheapest.
                if not _llm_charged:
                    _T["llm"] += time.time() - t0
                else:
                    _T["synth"] += time.time() - _t_synth0
        if cur_fn is None:
            break

        # ---- fit constants (ars mode only) ---------------------------------
        fit_eps = 0
        if a.mode == "ars":
            try:
                cur_params, fit_eps = ars_fit(env_factory, cur_fn, cur_params, actions,
                                              a.max_ticks, a.ars_budget, seed=a.seed + epi)
            except Exception as exc:
                print("[ep %d] ARS failed: %s" % (epi, exc))

        # ---- CHAMPION BANKING: play the champion, not a collapsed challenger --
        played_champion = False
        if a.champion_banking == "on" and champ is not None and _last_score is not None:
            champ_score = champ[0][0]
            margin = abs(champ_score) * (1.0 - a.bank_regress_frac) \
                if champ_score > 0 else abs(champ_score) * a.bank_regress_frac
            if _last_score < champ_score - margin:
                # The previous revision collapsed against the banked champion.
                # Restore the champion's code AND params and play THAT this
                # episode, instead of letting a broken challenger keep the seat.
                try:
                    cur_fn, cur_params = load_policy(champ[1], a.mode)
                    cur_src = champ[1]
                    if champ[2] is not None:
                        cur_params = list(champ[2])
                    played_champion = True
                    print("[bank] last revision scored %g vs champion %g -- "
                          "REPLAYING the champion this episode."
                          % (_last_score, champ_score))
                except Exception as exc:
                    print("[bank] could not restore champion: %s" % exc)

        # ---- play the real game --------------------------------------------
        _t_env0 = time.time()
        score, ticks, trans, (bad, errs), acts, extras = play(
            env_factory, cur_fn, cur_params, actions, a.max_ticks,
            collect=True, stall_ticks=a.stall_ticks)
        # env_sec includes evaluating the POLICY on every tick, because for this method the
        # policy IS the per-step compute -- there is no separate search to attribute it to.
        # Named `env_sec` anyway so the column lines up with the other three runners.
        _T["env"] += time.time() - _t_env0
        stalled = errs.pop("__stalled__", None) is not None
        _last_score = score
        # ---- ARTIFACTS: one GIF per episode, plus the source that produced it ----
        # Written here rather than at the end so a crashed or killed arm still leaves the
        # episodes it did finish. The score goes in the filename because that is what makes
        # a directory of GIFs readable without opening them.
        if a.artifact_dir:
            _ad = a.artifact_dir
            # Timed and reported separately: GIF writing replays the whole episode through a
            # fresh env, so it is a real and sometimes large cost that has nothing to do with
            # the method. Folding it into env_sec would make artifact-producing arms look
            # slower at playing the game than arms run without --artifact-dir.
            _t_gif0 = time.time()
            save_episode_gif(env_factory, trans,
                             os.path.join(_ad, "gifs", "ep%d_score%g.gif" % (epi, score)))
            _T["gif"] += time.time() - _t_gif0
            if cur_src:
                _cd = os.path.join(_ad, "code")
                os.makedirs(_cd, exist_ok=True)
                with open(os.path.join(_cd, "ep%d_score%g.py" % (epi, score)), "w") as _f:
                    _f.write(cur_src)
        # RAIL PROVENANCE. Without `max_ticks` on the record, nobody downstream can tell a
        # score that is real gameplay from a score that is a TRUNCATION. Measured
        # 2026-08-17: 47 arms in the corpus have no recorded cap, so rail-vs-death is
        # simply undecidable for them -- and flappy's headline "780" turned out to be
        # exactly `--max-ticks 50000` reached alive, i.e. a solve, not a score. The runner
        # already computes this at print time (see the `rails` block at the end of main);
        # it just never stored it. `ended` names the reason so no reader has to infer it
        # from tick arithmetic -- inferring caps from repeated tick counts manufactured a
        # false "freeway is 90% rail-bound" the first time I tried it.
        # ENV-INTERNAL CAP. `a.max_ticks` is the RUNNER's cap on decision ticks; several envs
        # carry their OWN cap on a DIFFERENT counter and reach it first. Freeway is the case
        # that exposed this: FreewayEnv defaults to max_ticks=2500 counted in ALE frames, and
        # the builder at ~line 676 never overrides it, so at frame_skip 3 every episode ends
        # at decision tick 834 (834*3 = 2502 >= 2500) with env.lost set. The rail test below
        # compared 834 >= 6000, saw False, and labelled that truncation "death_or_win" -- so
        # every freeway record in the corpus claims a natural ending for a game that was cut
        # off at ~31% of its real ~8190-frame length. Record the env's own cap and let it
        # decide the label too. Verified 2026-08-31 at seed 42.
        horizon_scores = extras["horizon_scores"]
        _env_cap = extras["env_max_ticks"]
        _env_ctr = extras["env_turn_number"]
        _env_railed = (_env_cap is not None and _env_ctr is not None
                       and _env_ctr >= _env_cap)
        rec["episodes"].append({"ep": epi, "score": score, "ticks": ticks,
                                "max_ticks": a.max_ticks,
                                "env_max_ticks": _env_cap,
                                "horizon_scores": horizon_scores,
                                "env_turn_number": _env_ctr,
                                "ended": ("rail" if ticks >= a.max_ticks else
                                          "env_rail" if _env_railed else
                                          "stalled" if stalled else "death_or_win"),
                                "bad_actions": bad, "fit_episodes": fit_eps,
                                "params": cur_params, "errors": errs,
                                "played_champion": played_champion,
                                # per-episode wall-clock, so a learning curve can be plotted
                                # against TIME and "which episode got slow" is answerable --
                                # previously only run-level totals existed.
                                "wall_sec": round(time.time() - _t_env0, 3),
                                # ACTION HISTOGRAM. "Does this policy ever fire?" was the central
                                # seaquest question and it was answered by grepping generated
                                # source, which only works while the source is kept. `fire_frac`
                                # is broken out because it is the one we kept asking.
                                # see the note in play(): the histogram cannot be replayed,
                                # this can. (seed, actions) reproduces the episode exactly.
                                "actions": list(acts),
                                **{k: v for k, v in
                                   RM.episode_meta(epi, score, transitions=trans).items()
                                   if k in ("action_hist", "action_distinct", "fire_frac")}})
        if revision_entry is not None and revision_entry.get("ok") \
                and revision_entry.get("hypothesis"):
            stamp_hypothesis(revision_entry["hypothesis"], score)
        print("[ep %2d] score=%-6g ticks=%-6d bad=%-3d fit_eps=%d params=%s"
              % (epi, score, ticks, bad, fit_eps,
                 [round(x, 3) for x in (cur_params or [])]))

        # Tie-break on ticks survived, then on validity. With every score 0 the
        # original kept the FIRST policy forever -- including one that raised on
        # every tick -- so champion_src showed a broken program while the live
        # policy had long since been repaired.
        key = (score, ticks, -bad)
        # A revision is STALE if it did not improve on the champion. Banking
        # replays do not count -- replaying the champion is not an attempt.
        if champ is not None and not played_champion:
            stale_revisions = 0 if key > champ[0] else stale_revisions + 1
        if champ is None or key > champ[0]:
            champ = (key, cur_src, cur_params)
        flush()

        # ---- build the evidence for the next revision ----------------------
        # NOT prediction error -- nothing predicted anything. What happened, what
        # it scored, and where it failed. Plus, in ars mode, the fitted constants,
        # which say as much about the FORM as a fitted coefficient says about a
        # physical model.
        tail = trans[-40:]
        fb = format_trajectory_table(tail) if tail else "(no ticks)"
        parts = ["RETURN: score %g over %d ticks." % (score, ticks)]
        if bad:
            detail = "; ".join("%s (x%d)" % (k, v) for k, v in
                               sorted(errs.items(), key=lambda kv: -kv[1])[:3])
            parts.append("Your policy FAILED on %d of %d ticks and the runner "
                         "substituted a safe default each time. The exact "
                         "error(s): %s. Fix this FIRST — a policy that raises "
                         "cannot steer, and the agent simply falls."
                         % (bad, ticks, detail or "invalid action returned"))
        if champ and score < champ[0][0]:
            parts.append("This REGRESSED against your best controller (score %g). "
                         "The code above is your LAST revision, not the best one."
                         % champ[0][0])
        if a.mode == "ars" and cur_params:
            parts.append("FITTED PARAMS after %d real episodes of search: %s. A value "
                         "driven to ~0 means that term earned nothing; a value pinned "
                         "far from your guess means the form is too weak there."
                 % (fit_eps, [round(x, 4) for x in cur_params]))
        if revision_entry is not None and revision_entry.get("ok") \
                and revision_entry.get("hypothesis"):
            h = revision_entry["hypothesis"]
            if h.get("parse_ok"):
                parts.append("HYPOTHESIS RESULT: predicted score %s, measured %s, "
                             "%s (tolerance %s)."
                             % (h.get("predicted_score"), score,
                                h.get("status"), h.get("tolerance")))
        # STAGNATION (--stagnation-feedback on). The default feedback shows the final
        # score and the last 40 ticks -- so a policy that froze at tick 280 and idled for
        # 5,720 ticks produces "RETURN: score 20 over 6000 ticks" plus 40 ticks of noop,
        # which reads like a finished success. The killer fact the model never saw is WHEN
        # the score last moved. Computed from the agent's own trace; no game knowledge.
        if getattr(a, "stagnation_feedback", "off") == "on" and trans:
            _sc = []
            for _t in trans:
                _st = _t.get("state") or {}
                _v = _st.get("score", [0])
                _sc.append(float(_v[0] if isinstance(_v, (list, tuple)) and _v else _v or 0))
            _last = 0
            for _i in range(1, len(_sc)):
                if _sc[_i] != _sc[_i - 1]:
                    _last = _i
            _idle = len(_sc) - _last
            if _last > 0 and _idle >= max(150, int(0.3 * len(_sc))):
                from collections import Counter as _Ctr
                # TOP THREE, NOT TOP ONE. A stall has two distinguishable shapes and top-1
                # cannot tell them apart: FROZEN (one action at ~100%) and THRASHING (two
                # actions near 50/50, i.e. an oscillation loop). Reporting only the most
                # common turns "left 50%, right 49%" into "most common was 'left', 50%",
                # which reads as indecision rather than a loop -- so the model cannot see
                # the defect it needs to fix. The 98%-noop case is unchanged, since the
                # trailing entries are negligible.
                _cnt = _Ctr(str(_t.get("action")) for _t in trans[_last:]).most_common(3)
                _mix = ", ".join("%r %d%%" % (_a, round(100.0 * _n / max(1, _idle)))
                                 for _a, _n in _cnt)
                parts.append("STAGNATION: your score last changed at tick %d, then you "
                             "scored NOTHING for the final %d ticks while the game was "
                             "still running (your actions in that stretch: %s). The "
                             "episode did not end because you won."
                             % (_last, _idle, _mix))
        parts.append("The last %d ticks before the episode ended:\n%s" % (len(tail), fb))
        feedback = "\n\n".join(parts)

    rec["meta"]["finished"] = time.time()
    rec["best"] = champ[0][0] if champ else None
    rec["champion_src"] = champ[1] if champ else None
    rec["champion_params"] = champ[2] if champ else None
    if a.hypothesis_history:
        rec["hypothesis_summary"] = hypothesis_summary(rec)
    # RAIL AUDIT -- a rail only confounds a result if it BINDS, and that is a
    # question for the data. The main runner has one; this script did not, which
    # is how target_score=5 was read as a ceiling.
    rails = []
    n_win = sum(1 for e in rec["episodes"] if e["score"] >= a.target_score)
    if n_win:
        rails.append("--target-score %d (WON %d episode(s) -- score is a rail, "
                     "not a measurement)" % (a.target_score, n_win))
    n_tick = sum(1 for e in rec["episodes"] if e["ticks"] >= a.max_ticks)
    if n_tick:
        rails.append("--max-ticks %d (ended %d episode(s) STILL ALIVE)"
                     % (a.max_ticks, n_tick))
    rec["rails_bound"] = rails or None
    if rails:
        print("\n!! RAILS BOUND: " + "; ".join(rails))
    flush()          # AFTER the audit -- the first version computed
                     # rails_bound and then never wrote it, so every record
                     # said rails=None while episodes sat at exactly 50,000
                     # ticks. An audit that is not persisted is not an audit.
    scores = [e["score"] for e in rec["episodes"]]
    print("\nMODE=%s rep=%d  best=%s  scores=%s" % (a.mode, a.rep, rec.get("best"), scores))
    print("record: %s" % out)


if __name__ == "__main__":
    main()
