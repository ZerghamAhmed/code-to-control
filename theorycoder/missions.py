"""ONE place that decides what mission string a run receives.

WHY THIS EXISTS. Audited 2026-08-18: only 4 of 9 games gave every method the same
mission, and the differences ran in both directions.

  * `MISSIONS` (hand-written, in the controller runner) covers 11 games and is read ONLY
    by the controller and ReAct. On spaceinvaders it names the target and the reward
    source ("Destroy the aliens ... for each alien destroyed") while the planner and both
    WorldCoder columns got the generic line.
  * `env.mission` is read by the planner, our WorldCoder reimplementation and vendored
    WorldCoder -- but it is NOT uniform, because different games use different adapters.
    `ocatari_generic_env` is genuinely generic; `ocatari_pong_env` and
    `ocatari_freeway_env` ship game-specific text. Freeway's states the policy outright
    ("move the chicken UP") and tells the agent the hazard is harmless ("no death").
    Pong's is game-specific AND factually wrong under `full_game=True`, the protocol we
    actually run: it claims "score = your points" (the code returns a differential) and
    "the episode ends if the enemy scores" (it does not). See FINDINGS 1.7.

So neither source was a standard. This module makes the choice explicit and auditable
instead of implicit in which adapter a game happens to use.

THE PROTOCOL. `generic` is the intended lead: identical text for every game and every
method, naming no game object and no mechanic. A win under `generic` is a claim about the
METHOD. `specified` is the documented arm, kept to PRICE supplied information rather than
to replace the protocol -- the same reasoning as `--env-doc` in the planner, which exists
because GIF-MCTS hands its model a 1303-word doc page while our agent normally sees 26
words, so every published comparison there confounds method with information supplied.

DEFAULT IS `env`, i.e. current behaviour, so nothing moves until a runner opts in. Flipping
the protocol is then one default, not nine edited files -- and editing the adapters' own
strings was rejected precisely because it would silently change what every archived run
would reproduce.
"""

# Deliberately says nothing about any game: no object, no mechanic, no direction, no
# reward decomposition. "achieve the game's objective" is a pointer, not a description.
GENERIC = ("Increase your score and avoid losing. Score rises when you achieve "
           "the game's objective.")

# MINIMAL states EXACTLY the metric we report (best-of-20 score) and nothing else.
#
# WHY IT IS SEPARATE FROM GENERIC. "Increase your score and avoid losing" reads as a
# pointer to two observable fields (`score`, `lost`), but "avoid losing" is a BEHAVIOURAL
# instruction, and it points the wrong way in any game whose objective requires risk.
# Kangaroo is the clearest case: the objective is to climb to the child, and "avoid losing"
# argues against climbing. We evaluate on score alone, so instructing on survival as well
# is a MISALIGNMENT between what we ask for and what we grade -- and we have measured that
# pathology twice already (the turtle rate, FINDINGS 3.2; and the seaquest policy that
# wrote `if score >= 20: return "noop"` and froze).
#
# NOT the default, and deliberately so. GENERIC's whole cost advantage is that it is
# already the text 6 of 9 games ran on, so every completed vendored cell stays valid.
# Promoting MINIMAL re-opens all of them at 7-37 h each. So it is tested as a variant on
# the cheap column first and only adopted if it wins.
MINIMAL = "Maximize your score."

SOURCES = ("generic", "minimal", "env", "dict")


def resolve(game, env=None, source="env", dict_missions=None):
    """Return (mission_text, provenance_label).

    The label is written into the record. No controller or planner record in this corpus
    stores the mission it ran with, so "what was this cell told?" is answerable only by
    reading the code at the commit it ran on -- the same class of undocumented per-game
    input that already produced three misreadings here (pong's scoring protocol,
    freeway's frame_skip, the 3-episode floors).
    """
    if source not in SOURCES:
        raise ValueError("mission source %r not in %r" % (source, SOURCES))

    if source == "generic":
        return GENERIC, "missions.GENERIC (uniform protocol)"

    if source == "minimal":
        return MINIMAL, "missions.MINIMAL (score-only, no survival clause)"

    if source == "dict" and dict_missions and game in dict_missions:
        return dict_missions[game], "MISSIONS[%s]" % game

    env_m = getattr(env, "mission", None)
    if env_m:
        # Name the adapter so a game-specific env mission cannot pass as "the generic one".
        kind = "generic" if env_m.strip() == GENERIC else "GAME-SPECIFIC"
        return env_m, "env.mission (%s, %s)" % (type(env).__name__, kind)

    if source == "dict" and dict_missions and game in dict_missions:
        return dict_missions[game], "MISSIONS[%s]" % game
    return GENERIC, "missions.GENERIC (env had no mission)"
