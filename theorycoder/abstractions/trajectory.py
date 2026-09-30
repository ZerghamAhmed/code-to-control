"""Trajectory capture — the fuel for world-model / abstraction revision.

Two producers:
  * ``collect_random_trajectory`` — roll RANDOM actions on the real engine. Used
    before any plan exists to gather initial experience for the first synthesis.
  * ``rollout`` — step a given action sequence (e.g. a plan's primitive actions) on
    the real engine WHILE rolling the world model in lockstep, so every step records
    the world model's prediction next to reality and flags mismatches.

A trajectory is a list of transition dicts:
    {
      "step": int,
      "action": str,
      "state": {...},            # state BEFORE the action (real)
      "predicted_next": {...}|None,   # transition_model(state, action), if provided
      "actual_next": {...},      # real engine state after the action
      "mismatch": bool|None,     # predicted_next != actual_next (None if no model)
      "done": bool,
    }

``format_trajectory`` renders a compact, mismatch-prioritized text block suitable for
dropping into a revision prompt.
"""
from __future__ import annotations

import json
import random
from copy import deepcopy

_RESERVED_KEYS = {"_grounding_trace"}


def _step_engine(engine, action):
    """Call engine.step and normalize to (state, done). BabyAI returns a 4-tuple
    (state, reward, done, info); tolerate other shapes."""
    result = engine.step(action)
    if isinstance(result, tuple):
        state = result[0]
        done = bool(result[2]) if len(result) > 2 else bool(getattr(engine, "won", False))
    else:
        state = result
        done = bool(getattr(engine, "won", False)) or bool(getattr(engine, "lost", False))
    # Prefer the engine's own accessor when available (authoritative).
    if hasattr(engine, "get_obs"):
        state = engine.get_obs()
    return state, done


def _canon(value):
    """Canonicalize a state (or fragment) for order-insensitive comparison.

    Lists of coordinates ([[x,y], ...]) are sorted; flat coords ([dx,dy]) and other
    scalars are left as-is; reserved bookkeeping keys are dropped.
    """
    if isinstance(value, dict):
        # Underscore-prefixed keys are MODEL-INTERNAL latent state (e.g. an
        # inventory the world model tracks that perception cannot observe) —
        # excluded from predicted-vs-actual comparison so they never produce
        # spurious mismatches against parsed observations.
        return {k: _canon(v) for k, v in value.items()
                if k not in _RESERVED_KEYS and not str(k).startswith("_")}
    if isinstance(value, list):
        if value and all(isinstance(e, (list, tuple)) for e in value):
            # list of coordinate pairs -> order-insensitive
            return sorted((_canon(e) for e in value), key=lambda x: json.dumps(x, sort_keys=True))
        return [_canon(e) for e in value]
    if isinstance(value, tuple):
        return [_canon(e) for e in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canon(e) for e in value), key=repr)
    return value


def _approx_eq(a, b, tol):
    """Structural equality where NUMBERS compare within an absolute tolerance.

    Continuous domains need this: the env wrapper rounds observations while the
    world model computes full precision, so exact float equality flags a ghost
    mismatch on every step even for a PERFECT model."""
    if isinstance(a, bool) or isinstance(b, bool):  # bool is an int subclass
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_approx_eq(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_approx_eq(x, y, tol) for x, y in zip(a, b))
    return a == b


def states_differ(a, b, float_round=None):
    """True if two states differ semantically (order-insensitive, reserved keys
    ignored). If either is None / carries an error marker, treat as differing.

    float_round: if set (continuous domains), numbers count as EQUAL when they
    agree to that many decimals — i.e. within an absolute tolerance of
    0.5 * 10**-float_round. Default None keeps exact comparison (discrete
    behavior unchanged)."""
    if a is None or b is None:
        return True
    if isinstance(a, dict) and a.get("_error"):
        return True
    ca, cb = _canon(a), _canon(b)
    if ca == cb:
        return False
    if float_round is None:
        return True
    return not _approx_eq(ca, cb, 0.5 * 10.0 ** (-float_round))


def rollout(engine, actions, transition_model=None, start_state=None, float_round=None):
    """Execute ``actions`` on the real engine, rolling ``transition_model`` in
    lockstep. Returns the trajectory (list of transition dicts). Stops early if the
    engine signals done.

    float_round: optional float tolerance (decimals) for mismatch detection on
    continuous states — see ``states_differ``. None = exact (discrete default).
    """
    state = deepcopy(start_state) if start_state is not None else engine.get_obs()
    traj = []
    for i, a in enumerate(actions):
        predicted = None
        if transition_model is not None:
            try:
                predicted = transition_model(deepcopy(state), a)
            except Exception as e:  # a crashing world model is itself a revision signal
                predicted = {"_error": f"{type(e).__name__}: {e}"}

        actual, done = _step_engine(engine, a)

        traj.append({
            "step": i,
            "action": a,
            "state": deepcopy(state),
            "predicted_next": deepcopy(predicted) if predicted is not None else None,
            "actual_next": deepcopy(actual),
            "mismatch": (states_differ(predicted, actual, float_round=float_round)
                         if transition_model is not None else None),
            "done": bool(done),
        })
        state = deepcopy(actual)
        if done:
            break
    return traj


def collect_random_trajectory(engine, n_steps, transition_model=None, seed=None,
                              float_round=None):
    """Reset the engine and roll ``n_steps`` RANDOM actions. Used to gather initial
    experience before any plan exists. If ``transition_model`` is given, predictions
    and mismatches are recorded too.
    """
    engine.reset()
    rng = random.Random(seed) if seed is not None else random
    actions = [rng.choice(list(engine.actions_set)) for _ in range(n_steps)]
    return rollout(engine, actions, transition_model=transition_model,
                   float_round=float_round)


def mismatch_count(traj):
    return sum(1 for t in traj if t.get("mismatch"))


def format_trajectory(traj, max_transitions=30, errors_only=False):
    """Render a compact text block for a revision prompt.

    When ``errors_only`` is True, only mismatching transitions are shown. Transitions
    are truncated to ``max_transitions``, mismatches first so the signal survives the
    cap.
    """
    def j(d):
        return json.dumps(d, sort_keys=True, default=str)

    n, nm = len(traj), mismatch_count(traj)

    if errors_only:
        # Sparse: show each mismatching transition WITH its before-state for context.
        items = [t for t in traj if t.get("mismatch")][:max_transitions]
        header = f"{n} transitions total, {nm} mismatches (showing only mismatches):\n"
        blocks = []
        for t in items:
            b = [f"step {t['step']}: action = {t['action']}  <-- MISMATCH (world model wrong here)",
                 f"  before:    {j(t['state'])}"]
            if t["predicted_next"] is not None:
                b.append(f"  predicted: {j(t['predicted_next'])}")
            b.append(f"  actual:    {j(t['actual_next'])}")
            blocks.append("\n".join(b))
        return header + "\n".join(blocks)

    # Full trajectory, compact: the start state is the current state (shown separately),
    # and each step's "before" equals the previous step's result — so we show only
    # `action -> resulting state`, and the model's prediction ONLY where it mismatched.
    items = traj[:max_transitions]
    header = (f"{n} transitions from the current state, {nm} mismatches "
              f"(each line: action -> resulting state):\n")
    lines = []
    for t in items:
        tag = "   <-- MISMATCH" if t.get("mismatch") else ""
        lines.append(f"  {t['action']:>8} -> {j(t['actual_next'])}{tag}")
        if t.get("mismatch") and t["predicted_next"] is not None:
            lines.append(f"           (world model predicted: {j(t['predicted_next'])})")
    return header + "\n".join(lines)


def format_trajectory_table(traj, max_rows=60, max_cols=14):
    """GENERIC columnar rendering: flatten every numeric leaf of the canonical
    schema into `entity[i].x/y` columns, keep the most-CHANGING columns (what
    moves is what teaches dynamics), align rows so constants are readable by
    eye, and annotate CONTACT events / deaths (PoE-World-style). Falls back to
    the classic renderer when states don't conform."""
    import json as _json
    if not traj:
        return "(no transitions)"

    def leaves(state):
        out = {}
        for k, v in state.items():
            ks = str(k)
            if ks.startswith("_") or ks in ("won", "lost"):
                continue
            if ks == "score":
                out["score"] = v[0] if isinstance(v, list) else v
                continue
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                for i, pos in enumerate(v):
                    for j, name in enumerate(("x", "y")[:len(pos)]):
                        out[f"{ks}[{i}].{name}"] = pos[j]
            elif (isinstance(v, list) and len(v) == 1
                  and isinstance(v[0], (int, float))
                  and not isinstance(v[0], bool)):
                # VECTOR-STATE FIELDS, e.g. `cart_x: [0.0274]`. Without this the
                # only extractable key for a vector-state game was `score`, so the
                # revision feedback the LLM received was a single incrementing
                # counter and NOTHING about the dynamics it was asked to model.
                # It silently affected every CartPole run: the agent revised its
                # world model blind, which is why the zero-feedback first synthesis
                # was the best program and all later revisions were worse.
                out[ks] = v[0]
        return out

    rows = []
    for tr in traj:
        # Annotate the WIN edge as well as the death edge. An achievement task's
        # win transition is the single most informative tick in an episode, and it
        # was previously invisible in this table.
        rows.append((tr["action"], leaves(tr["actual_next"]),
                     bool(tr.get("mismatch")),
                     (bool(tr["actual_next"].get("lost")) and
                      not bool(tr["state"].get("lost"))),
                     (bool(tr["actual_next"].get("won")) and
                      not bool(tr["state"].get("won")))))
    # `score` alone is not a conforming row: it carries no dynamics. Require at
    # least one non-score column before trusting the table, otherwise fall back to
    # the classic renderer, which prints whole states and handles any schema.
    if not rows or len([c for c in rows[0][1] if c != "score"]) == 0:
        return format_trajectory(traj)          # non-conforming: fall back

    # choose columns by change count (plus score always)
    cols = {}
    prev = None
    for _a, lv, _m, _d, _w in rows:
        if prev is not None:
            for k in lv:
                if k in prev and lv[k] != prev[k]:
                    cols[k] = cols.get(k, 0) + 1
        prev = lv
    keep = sorted(cols, key=lambda k: -cols[k])[:max_cols]
    if "score" not in keep and any("score" in r[1] for r in rows):
        keep.append("score")
    omitted = len({k for _a, lv, _m, _d, _w in rows for k in lv}) - len(keep)

    # contact events via the object API
    try:
        from .objapi import objects
        def contacts(state):
            objs = list(objects(state))
            return {(a.category + str(a.index), b.category + str(b.index))
                    for i, a in enumerate(objs) for b in objs[i + 1:]
                    if a.category != b.category and a.touches(b)}
    except Exception:
        contacts = lambda s: set()

    # row selection: changes in rare columns, contact changes, deaths, stride
    sel = set()
    prev_c = contacts(traj[0]["state"])
    events = {}
    for i, tr in enumerate(traj):
        c = contacts(tr["actual_next"])
        if c != prev_c:
            sel.add(i)
            started = c - prev_c
            if started:
                events[i] = "CONTACT: " + ", ".join(f"{a}~{b}" for a, b in started)
        prev_c = c
        if rows[i][2] or rows[i][3] or rows[i][4]:
            sel.add(i)
    stride = max(1, len(rows) // max(1, max_rows - len(sel)))
    sel.update(range(0, len(rows), stride))
    idxs = sorted(sel)[:max_rows]

    hdr = ["t", "action"] + keep + ["note"]
    widths = [max(6, len(h) + 1) for h in hdr]
    lines = ["".join(h.ljust(w) for h, w in zip(hdr, widths))]
    for i in idxs:
        a, lv, mism, death, win = rows[i]
        note = []
        if death:
            note.append("DIED")
        if win:
            note.append("WON")
        if mism:
            note.append("MISMATCH")
        if i in events:
            note.append(events[i])
        vals = [str(i), str(a)] + [
            # SIGNIFICANT digits, not decimal places. `.1f` renders CartPole's
            # entire state as 0.0 / -0.0 / 0.1 (|angle| < 0.21, |x| < 2.4), so the
            # table showed the right COLUMNS at a resolution that destroyed them --
            # the fourth instance in this codebase of an absolute-precision constant
            # breaking on small-magnitude state. It also contradicted the prompt,
            # which promises "values ... reported to 4 decimal places".
            # `.4g` keeps 4 significant digits at any magnitude: 0.0274 stays
            # 0.0274, and Flappy's 507.0 stays 507.
            (f"{lv[k]:.4g}" if isinstance(lv.get(k), float) else str(lv.get(k, "")))
            for k in keep] + ["; ".join(note)]
        lines.append("".join(v.ljust(w) for v, w in zip(vals, widths)))
    if omitted > 0:
        lines.append(f"(+{omitted} mostly-static columns omitted)")
    return "\n".join(lines)
