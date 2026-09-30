"""Decision latency of the Atari controllers: the time one call to policy(state) takes.

For each game, the champion plays 300 steps in the environment the runner uses. Each state it
saw is then timed five times, and the median of the five is that state's latency, which
removes timer noise. The script prints each game's median and spread across states, and the
median across the six games.

    python scripts/measure_latency.py

Timings depend on the machine. The paper reports 11.4 us, the median across games, measured
with this protocol on a champion from each game's runs.
"""
import importlib.util
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location(
    "run_controller_flappy", os.path.join(ROOT, "scripts", "run_controller_flappy.py"))
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)

GAMES = [("pong", "Pong"), ("spaceinvaders", "Space Invaders"), ("asterix", "Asterix"),
         ("breakout", "Breakout"), ("fishingderby", "Fishing Derby"), ("freeway", "Freeway")]
STATES, REPS, SEED, TARGET = 300, 5, 42, 10 ** 6


def make_env(game, max_ticks):
    """The same environment settings the runner builds for each game."""
    from game_registry import OCATARI_GENERIC
    if game in OCATARI_GENERIC:
        from ocatari_generic_env import OCAtariGenericEnv
        return OCAtariGenericEnv(OCATARI_GENERIC[game], seed=SEED, frame_skip=3,
                                 velocity=True, target_score=TARGET)
    if game == "freeway":
        from ocatari_freeway_env import FreewayEnv
        return FreewayEnv(seed=SEED, frame_skip=3, target_score=TARGET, max_ticks=max_ticks * 3)
    if game == "pong":
        from ocatari_pong_env import PongEnv
        return PongEnv(seed=SEED, frame_skip=3, full_game=True, velocity=True,
                       target_score=TARGET)
    raise ValueError(game)


def states_seen(fn, env):
    """The first STATES states the controller sees, restarting if the episode ends."""
    out = []
    while len(out) < STATES:
        s = env.get_obs()
        out.append(s)
        env.step(fn(s))
        if env.lost or env.won:
            env.reset()
    return out


def time_policy(fn, states):
    per_state = []
    for s in states:
        reps = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn(s)
            reps.append((time.perf_counter() - t0) * 1e6)
        per_state.append(st.median(reps))
    return sorted(per_state)


def main():
    print(f"{'game':<16}{'median us':>10}{'p5 us':>9}{'p95 us':>9}")
    medians = []
    for stem, name in GAMES:
        meta = json.load(open(os.path.join(ROOT, "champions", f"{stem}.json")))
        fn, _ = rcf.load_policy(meta["champion_src"], "llm")
        env = make_env(meta["game"], int(meta["rails"]["--max-ticks"]))
        env.reset()
        per_state = time_policy(fn, states_seen(fn, env))
        med = st.median(per_state)
        medians.append(med)
        print(f"{name:<16}{med:>10.1f}{per_state[int(0.05 * STATES)]:>9.1f}"
              f"{per_state[int(0.95 * STATES)]:>9.1f}", flush=True)
    print(f"\nmedian across games: {st.median(medians):.1f} us")


if __name__ == "__main__":
    main()
