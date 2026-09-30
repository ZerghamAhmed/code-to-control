"""ONE source of truth for which games the OCAtari adapter serves.

    from game_registry import OCATARI_GENERIC, is_ocatari

WHY THIS FILE EXISTS
--------------------
The same fact was stored independently in four places and they had already drifted
apart (measured 2026-08-17):

    scripts/build_games_table.py      OCATARI            12 games
    scripts/run_controller_flappy.py  _GENERIC            4 games  <- measured 2026-08-17
                                                                     from HEAD: only
                                                                     spaceinvaders, boxing,
                                                                     skiing, tennis
    scripts/run_autonomous_survive.py _OCATARI_GENERIC    8 games
    scripts/run_worldcoder_baseline.py _OCATARI_GENERIC   9 games  <- has breakoutoc

Two concrete failures came out of that drift:

1. **Three OCAtari games were reported as "self-made"** in the results table,
   because `kangaroo`/`mspacman`/`seaquest` were added to the runners and the
   table's own set was never updated. That understates external-benchmark coverage,
   which is exactly the number a reviewer checks.

2. **The "never leave an orphan row" rule was unenforceable.** The controller runner
   cannot run `pooyan` while the planner can, so "run both arms on every game" is
   silently impossible for part of the set. A rule that the code cannot satisfy is
   not a rule.

Adding a game here now makes it available to every runner and correctly classified
in every table, in one edit.

NAME MAPPING. Keys are our lowercase CLI names (`--game mspacman`); values are the
OCAtari environment names (`MsPacman`), which are case-sensitive and not derivable
by `.title()` -- `MsPacman` and `SpaceInvaders` both break that guess.
"""

# our CLI name -> OCAtari env name
OCATARI_GENERIC = {
    "spaceinvaders": "SpaceInvaders",
    "boxing": "Boxing",
    "skiing": "Skiing",
    "tennis": "Tennis",
    "pooyan": "Pooyan",
    "seaquest": "Seaquest",
    "kangaroo": "Kangaroo",
    "mspacman": "MsPacman",
    # OCAtari's Breakout, distinct from our internal 16-brick `breakout_env.py`.
    # Kept under a separate key on purpose: the two are DIFFERENT GAMES with
    # different ceilings (ours wins at 16 bricks, OCAtari's scores past 400), and
    # conflating them once produced a PPO baseline row with no comparable
    # controller row.
    "breakoutoc": "Breakout",
    # ---------------------------------------------------- 2026-08-17: TRANSFER PAIRS
    # Chosen to COMPLETE MECHANIC PAIRS rather than to add breadth, because the claim we
    # want is generalisation, and a transfer claim needs a matched partner game -- a
    # seventh unrelated game supports "works on N games" and says nothing about transfer.
    # Each of these shares its core mechanic with a game we already have, so learn-on-A /
    # evaluate-on-B is a controlled comparison, and a cross-PAIR run is the negative control.
    #
    # Entity exposure verified live (400 warm-up steps, because a snapshot taken at reset
    # shows only `player` on every one of these -- the mistake that made me briefly report
    # seaquest as "blind"):
    #   alien       -> alien, egg, player, pulsar
    #   riverraid   -> fueldepot, helicopter, house, player, playermissile, tanker
    #   demonattack -> enemy, enemymissile, enemypart, player, playermissile
    "alien": "Alien",              # pairs with mspacman: maze pursuit + collect
    # pairs with seaquest: side-scroll, shoot, AND a consumable resource. Deliberate:
    # `oxygenbar` appeared in only 5 of 120 seaquest policies, so `fueldepot` tests whether
    # ignoring a resource meter is systematic or a seaquest quirk.
    "riverraid": "Riverraid",
    "demonattack": "DemonAttack",  # pairs with spaceinvaders: fixed shooter
    # ------------------------------------------- 2026-08-19: landscape-screen candidates
    # Added for the controller-first paper benchmark screen. Selected from a mechanical
    # probe of all 47 untested OCAtari games (scratchpad landscape_probe.json; 21 fail to
    # instantiate, several more expose empty object state). These six run, expose rich
    # object state, and cover behavior regimes the existing set lacks: competitive
    # luring/timing (fishingderby), rotational thrust control (asteroids), moving-platform
    # crossing (frostbite), fire-less collection/dodging (asterix), defensive
    # interception (gopher), and defense+oxygen survival (namethisgame).
    "fishingderby": "FishingDerby",
    "asteroids": "Asteroids",
    "frostbite": "Frostbite",
    "asterix": "Asterix",
    "gopher": "Gopher",
    "namethisgame": "NameThisGame",
    # planning/boundary candidate (2026-08-19): exploration + hazard sequencing +
    # delayed credit; screened as an expected-negative for the boundary seat.
    "pitfall": "Pitfall",
}

# Games sharing a core mechanic. Used for transfer experiments: WITHIN a pair is the
# treatment, ACROSS pairs is the control. Stated here rather than inferred so the
# pairing cannot be chosen after seeing the results.
TRANSFER_PAIRS = {
    "maze_pursuit": ["mspacman", "alien"],
    "scroll_shoot_resource": ["seaquest", "riverraid"],
    "fixed_shooter": ["spaceinvaders", "demonattack"],
    "differential_racket": ["pong", "tennis"],
    "deflect_upward": ["breakoutoc", "pooyan"],
}

# Games served by a dedicated adapter rather than the generic one. Listed here so
# `is_ocatari` can answer correctly for classification in results tables.
OCATARI_DEDICATED = {
    "pong": "ocatari_pong_env.py",
    "freeway": "ocatari_freeway_env.py",
    "montezuma": "ocatari_montezuma_env.py",
}

# Everything OCAtari-backed, for "is this an external benchmark?" questions.
OCATARI_ALL = set(OCATARI_GENERIC) | set(OCATARI_DEDICATED)


def is_ocatari(game):
    """True when the game comes from OCAtari (an external benchmark).

    `breakout` is deliberately NOT here: bare `breakout` is our internal
    reflection-physics env. OCAtari's is `breakoutoc`.
    """
    return game.lower() in OCATARI_ALL


def ocatari_name(game):
    """OCAtari env name for a generic-adapter game, or None."""
    return OCATARI_GENERIC.get(game.lower())


# ---------------------------------------------------------------------------
# SHARED PROBE FIXTURE
#
# Written after making the same mistake twice in consecutive experiments: probing
# Breakout from a reset state, which has NO BALL because the game spawns one only
# after `fire`. Both probes silently measured a model that returns None, and one of
# them would have reported "no difference" between two broken arms.
#
# The lesson had already been written down when the repeat happened, so the fix is a
# FIXTURE, not another note. Anything that probes or replays a game should call
# `start_state()` instead of hand-rolling `for _ in range(12): env.step(noop)`.
#
# Two things it does, both measured rather than assumed:
#   1. applies the game's SPAWN ACTION if it has one (Breakout needs `fire`; the
#      control-axis probe measured this, and it is recorded here rather than
#      rediscovered);
#   2. returns the RICHEST state in a short window, not the first stable one -- the
#      criterion `--settled-state` uses, because "inventory unchanged for 3 steps"
#      fires at step 3 on Pong while the ball is still absent for ~20.
# ---------------------------------------------------------------------------

# Measured by scripts/control_axis_probe.py (7/7 against ground truth). An action
# here is one that makes a category absent at reset appear.
SPAWN_ACTION = {
    "breakout": "fire",
    "breakoutoc": "fire",
}


def start_state(env, window=30, spawn=None):
    """A state worth probing: past any serve/launch, with the inventory settled.

    `env` is any adapter following the engine contract (`get_obs`, `step`,
    `actions_set`, `lost`, `won`). Returns the observation with the most object
    categories seen in `window` decisions, which also has positions separated --
    MsPacman reports all four ghosts at [0,0] at reset.
    """
    def cats(s):
        return {k for k, v in s.items()
                if isinstance(v, list) and v and isinstance(v[0], list)}

    if spawn is None:
        spawn = SPAWN_ACTION.get(getattr(env, "game", "").lower())
    if spawn and spawn in getattr(env, "actions_set", []):
        env.step(spawn)

    idle = (env.actions_set[0] if getattr(env, "actions_set", None) else None)
    best, best_n = env.get_obs(), -1
    for _ in range(window):
        s = env.get_obs()
        n = len(cats(s))
        if n >= best_n:
            best, best_n = s, n
        if getattr(env, "lost", False) or getattr(env, "won", False) or idle is None:
            break
        try:
            env.step(idle)
        except Exception:
            break
    return best


# ---------------------------------------------------------------------------
# WHICH ENV PRODUCED THIS CHAMPION?
#
# Written after retracting three findings that came from probing internal-env
# champions with OCAtari states. `run_logs/autonomous/*/tc_game/breakout/` is
# written by BOTH `--game breakout` (our internal reflection-physics env) and
# `--game breakoutoc` (OCAtari), and the directory name distinguishes neither. Four
# ticks of careful measurement went into a champion/env pairing that never existed.
#
# The artifact's own SCHEMA is the reliable signal, so read that instead of the path.
# ---------------------------------------------------------------------------

# Keys that appear in one env's state and not the other's.
_SCHEMA_MARKERS = {
    "breakout":   ("brick", "hazard", "balls_left"),   # internal breakout_env.py
    "breakoutoc": ("block",),                          # OCAtari Breakout
}


def env_of_champion(champion, game_hint=None):
    """Which env a champion's theory was written for: a key of OCATARI_GENERIC,
    a bare game name, or None when the schema is ambiguous.

    `champion` is the parsed champion.json dict (or its abstractions+worldmodel
    text). Ambiguity is returned as None rather than guessed -- a wrong pairing
    produces confident nonsense, as it did for breakout.
    """
    import re
    text = champion if isinstance(champion, str) else (
        (champion.get("abstractions") or "") + (champion.get("worldmodel_fitted") or ""))
    hits = {}
    for name, markers in _SCHEMA_MARKERS.items():
        hits[name] = sum(1 for m in markers
                         if re.search(r"['\"]%s['\"]" % m, text))
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] == 0:
        return None                      # no marker at all -> unknown, not a guess
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None                      # tie -> ambiguous, say so
    return ranked[0][0]


def actions_for_champion(champion, fallback=None):
    """The action set a champion's theory could legally have used.

    Guards the specific mistake that voided the breakout thread: scanning theories
    for `fire` when `fire` is not legal in the env they were written for. The
    internal breakout env exposes only ['noop','left','right'].
    """
    env = env_of_champion(champion)
    if env == "breakout":
        return ["noop", "left", "right"]
    return fallback


# ---------------------------------------------------------------------------
# HOW LONG UNTIL THE STATE IS COMPLETE?
#
# Atari withholds objects at the start, so a world model synthesized from the reset
# frame can be missing the objects that define the game. `run_autonomous_survive.py`
# has always had `--start-at` for this, with a hand-set default of 150 GAME FRAMES for
# pong (PoE-World's 50 at frameskip 3) and **0 for every other game**. Measured
# 2026-08-17, four of eleven games hide keys at reset:
#
#   breakoutoc  gains ball, ball_size, ball_velocity          <- nothing compensates
#   pong        gains ball, enemy (+sizes/velocities)         <- covered by its 150
#   seaquest    gains oxygenbar (+size/velocity)              <- nothing compensates
#   tennis      gains ballshadow (+size/velocity)             <- nothing compensates
#
# and breakoutoc was verified on the sent prompt itself: the word `ball` appeared
# **0 times**, so the agent was asked to model a ball it had never been shown and then
# scored on exactly that channel.
#
# This measures the number rather than adding another table of constants: step until the
# key set stops growing. It needs no per-game knowledge beyond the spawn action the
# control-axis probe already measured.
# ---------------------------------------------------------------------------
def frames_to_full_state(env, cap_decisions=120):
    """GAME FRAMES until no new observable key has appeared for `quiet` decisions.

    Returns 0 when the reset state is already complete, which keeps every game whose
    state is whole at reset byte-identical to the no-skip behaviour. The return value is
    in FRAMES (decisions x frame_skip) because that is the unit `--start-at` uses.

    NO EARLY EXIT ON A QUIET WINDOW. The first version stopped after 8 decisions without a
    new key and reported **0 frames for pong** -- whose ball first appears at decision 20,
    eleven decisions after that window closed. A "nothing new lately" heuristic cannot
    bound a delay it has not reached yet, so the whole cap is walked; 120 decisions is ~1s.
    """
    def keys(s):
        return {k for k in (s or {}) if not str(k).startswith("_")}

    state = env.reset()
    seen = keys(state)
    idle = (env.actions_set[0] if getattr(env, "actions_set", None) else None)
    spawn = SPAWN_ACTION.get(getattr(env, "game", "").lower())
    last_new, decisions = 0, 0
    for i in range(1, cap_decisions + 1):
        # the spawn action first, then idle: a category that only exists after `fire`
        # will never appear under idle alone, which is the breakoutoc case exactly
        act = spawn if (i == 1 and spawn and spawn in (env.actions_set or [])) else idle
        if act is None:
            break
        try:
            state, _r, done, _i = env.step(act)
        except Exception:
            break
        decisions = i
        new = keys(state) - seen
        if new:
            seen |= new
            last_new = i
        if done:
            break
    if last_new == 0:
        return 0
    return int(last_new * max(1, getattr(env, "frame_skip", 1)))
