"""Flappy Bird under changed physics: zero-shot transfer, then refitting constants (Figure 3B).

The three Flappy Bird champions in champions/flappy_transfer/ each play eight physics
changes, unchanged, for up to 20,000 steps. Where a champion dies early, its program is held
fixed and only its numeric constants are refit, by coordinate search over real rollouts.
No language model is called. The script prints survival before and after refitting and
checks every cell against Figure 3B of the paper.

    python scripts/flappy_transfer.py              # zero-shot and refit, a few minutes
    python scripts/flappy_transfer.py --zero-shot  # zero-shot only, under a minute

The refit is the procedure behind the figure, unchanged: the program's inline numbers are
lifted to named constants, then each constant in turn is tried at a ladder of multiples,
including negatives, keeping any change that survives longer. Two passes. The winner is
then replayed for the full 20,000 steps.
"""
import argparse
import ast
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flappy_env import FlappyEnv  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_controller_flappy", os.path.join(ROOT, "scripts", "run_controller_flappy.py"))
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)

CAP = 20000
ACTIONS = ["flap", "noop"]

# (name, label in Figure 3B, physics overrides, mid-episode schedule), top row first.
# The base game has GRAV 1.15, DRAG 0, GAP 168, FLAP -11.5 and PIPE_SPEED 4.2.
VARIANTS = [
    ("heavy",    "gravity x1.5",            {"GRAV": 1.725}, None),
    ("drag",     "velocity-prop. drag",     {"DRAG": 0.2}, None),
    ("shift",    "gravity x1.5 at t=500",   None, [{"at_tick": 500, "set": {"GRAV": 1.725}}]),
    ("narrow",   "gap 168->120",            {"GAP": 120}, None),
    ("floaty",   "flap -11.5->-7.0",        {"FLAP": -7.0}, None),
    ("moon",     "gravity x0.5, flap x0.6", {"GRAV": 0.575, "FLAP": -6.9}, None),
    ("fast",     "pipe speed x1.5",         {"PIPE_SPEED": 6.3}, None),
    ("inverted", "gravity sign flipped",    {"GRAV": -1.15, "FLAP": 11.5}, None),
]
RUNS = (1, 2, 3)

# Figure 3B, as steps survived. Cells not listed survive all 20,000 steps zero-shot and were
# not refit. Each entry is (zero-shot steps, steps after refitting).
EXPECTED = {
    (1, "inverted"): (22, 27),
    (2, "floaty"): (3605, 20000),
    (2, "moon"): (18133, 20000),
    (2, "fast"): (18117, 20000),
    (2, "inverted"): (22, 27),
    (3, "moon"): (8853, 17109),
    (3, "fast"): (530, 1691),
    (3, "inverted"): (22, 27),
}

# Refit search: multiples of each constant, and their negatives. A constant that is zero
# is tried at these values instead.
LADDER = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 4.0]
LADDER = LADDER + [-m for m in LADDER]
ZERO = [0.0, 0.1, -0.1, 0.5, 1.0, -1.0]
PASSES = 2


# ---- lifting inline numbers to named constants ---------------------------------------------
# Literals in structural positions (subscripts, slices, range() arguments) stay inline: an
# index is not a control constant, and refitting one could make the program raise.

_SKIP_PARENTS = (ast.Subscript, ast.Slice)


def _mark(tree):
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            ch._parent = node
    return tree


def _structural(node):
    p = getattr(node, "_parent", None)
    while p is not None:
        if isinstance(p, _SKIP_PARENTS):
            return True
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "range":
            return True
        p = getattr(p, "_parent", None)
    return False


class _Hoister(ast.NodeTransformer):
    def __init__(self):
        self.consts = {}          # name -> value
        self._seen = {}           # value -> name (one name per distinct value)
        self.sites = []           # (literal node, name), for in_place()

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            return node
        if _structural(node):
            return node
        v = float(node.value)
        if v in self._seen:
            name = self._seen[v]
        else:
            name = "C%d" % (len(self.consts) + 1)
            self._seen[v] = name
            self.consts[name] = node.value
        self.sites.append((node, name))
        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)


def hoist(src):
    """-> (source with inline numbers lifted to C1, C2, ..., {name: value})."""
    tree = _mark(ast.parse(src))
    h = _Hoister()
    tree = h.visit(tree)
    if not h.consts:
        return src, {}
    ast.fix_missing_locations(tree)
    lines = ast.unparse(tree).split("\n")
    header = "\n".join("%s = %r" % (k, v) for k, v in h.consts.items())
    cut = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            cut = i + 1
    return "\n".join(lines[:cut] + ["", header, ""] + lines[cut:]), h.consts


def set_constants(code, values):
    """Rewrite the `NAME = <number>` line for every NAME in `values`."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Z][A-Z0-9_]*)\s*=", line)
        if m and m.group(2) in values:
            v = values[m.group(2)]
            v = int(v) if float(v).is_integer() and abs(v) < 1e9 else v
            lines[i] = f"{m.group(1)}{m.group(2)} = {v}"
    return "\n".join(lines)


def in_place(src, values):
    """`src` with every literal that hoist() lifts to a constant in `values` replaced where it
    stands, so the refit program keeps the model's comments and layout. Values are written as
    set_constants() writes them."""
    h = _Hoister()
    h.visit(_mark(ast.parse(src)))
    lines = [line.encode() for line in src.split("\n")]      # AST offsets count bytes
    for node, name in sorted(h.sites, key=lambda s: (s[0].lineno, s[0].col_offset), reverse=True):
        if name not in values:
            continue
        v = values[name]
        v = int(v) if float(v).is_integer() and abs(v) < 1e9 else v
        text = repr(v)
        if v < 0 and isinstance(node._parent, (ast.UnaryOp, ast.BinOp, ast.Compare)):
            text = f"({text})"
        line = lines[node.lineno - 1]
        lines[node.lineno - 1] = line[:node.col_offset] + text.encode() + line[node.end_col_offset:]
    return "\n".join(line.decode() for line in lines)


# ---- playing and refitting -------------------------------------------------------------------

def survive(src, physics, schedule, cap):
    """Steps survived by the program `src` under the given physics, up to `cap`."""
    def make():
        return FlappyEnv(entity_names="descriptive", obs_mode="objects", seed=42,
                         target_score=10 ** 9, physics=physics, shift_schedule=schedule)
    fn, _ = rcf.load_policy(src, "llm")
    _, ticks, _, _, _, _ = rcf.play(make, fn, None, ACTIONS, cap)
    return ticks


def refit(src, physics, schedule, start):
    """Coordinate search over the program's constants. Returns (steps survived after the
    full 20,000-step replay, environment steps the search used, constants that moved).

    A candidate only has to outlast the incumbent, so each rollout stops one step after
    the incumbent's survival. Survival only increases, so this finds the same winner as
    running every candidate to 20,000."""
    hsrc, consts = hoist(src)
    cur, cur_t, used = dict(consts), start, 0
    for _ in range(PASSES):
        improved = False
        for name in sorted(cur):
            x = cur[name]
            for v in (ZERO if x == 0 else [x * m for m in LADDER]):
                if v == cur[name]:
                    continue
                trial = dict(cur)
                trial[name] = v
                t = survive(set_constants(hsrc, trial), physics, schedule, min(CAP, cur_t + 1))
                used += t
                if t > cur_t:
                    cur, cur_t, improved = trial, t, True
        if not improved:
            break
    final = survive(set_constants(hsrc, cur), physics, schedule, CAP)
    moved = {k: (consts[k], cur[k]) for k in cur if cur[k] != consts[k]}
    return final, used, moved


def pct(steps):
    return 100.0 * steps / CAP


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zero-shot", action="store_true", help="skip the refits")
    ap.add_argument("--save", metavar="PATH",
                    help="write every refit, with the constants it moved and the refit "
                         "program, as JSON")
    a = ap.parse_args()

    progs = {r: json.load(open(os.path.join(ROOT, "champions", "flappy_transfer", f"run{r}.json")))
             ["champion_src"] for r in RUNS}
    print(f"{'physics change':<25}{'run':>4}{'zero-shot':>11}{'refit':>9}"
          f"{'search steps':>14}  result")
    bad, saved = 0, []
    for name, label, physics, schedule in VARIANTS:
        for r in RUNS:
            z = survive(progs[r], physics, schedule, CAP)
            want_z, want_f = EXPECTED.get((r, name), (CAP, None))
            f = used = None
            if z < CAP and not a.zero_shot:
                f, used, moved = refit(progs[r], physics, schedule, z)
                if a.save:
                    prog = in_place(progs[r], {k: new for k, (_, new) in moved.items()})
                    saved.append({"run": r, "change": name, "label": label, "zero_shot": z,
                                  "refit": f, "search_steps": used,
                                  "constants": {k: list(v) for k, v in moved.items()},
                                  "program": prog})
                    # the program as saved must be the one the refit scored
                    f = f if survive(prog, physics, schedule, CAP) == f else None
            ok = z == want_z and (a.zero_shot or f == want_f)
            bad += not ok
            print(f"{label:<25}{r:>4}{pct(z):>10.1f}%"
                  f"{'' if f is None else f'{pct(f):.1f}%':>9}"
                  f"{'' if used is None else format(used, ','):>14}  "
                  f"{'match' if ok else 'MISMATCH'}", flush=True)
    if a.save:
        json.dump({"cap": CAP, "refits": saved}, open(a.save, "w"), indent=1)
        print(f"\nwrote {a.save}: {len(saved)} refits")
    print("\nSurvival is the percentage of the 20,000-step episode. Refit runs only where the "
          "zero-shot controller dies early.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
