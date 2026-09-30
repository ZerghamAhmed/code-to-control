"""Render the Flappy Bird clips of the physics changes (Figure 3B) for the project page and the
README grid.

Each clip replays a released champion under one change to the physics. The game's own renderer
draws every frame, so the bird and the pipes look exactly as in the base game; the change is to
the environment, not the bird. Only the sky differs: it is tinted per change, so a clip is
recognisable at a glance. The controller sees none of this. It reads object positions and
velocities, never pixels.

  flappy_heavy          heavier gravity, run 1 unchanged: the first 24 s of 20,000 steps survived
  flappy_moon           light gravity, run 1 unchanged: the same
  flappy_narrow         a narrower gap, run 1 unchanged: the same
  flappy_inverted       gravity reversed, run 1 unchanged: the whole episode, at half speed

and a before-and-after pair for every refit that reaches the end of the episode:

  flappy_<change>_frozen  the program unchanged: the last ten seconds before it crashes
  flappy_<change>_refit   the same program with its constants refit: the same ten seconds,
                          and it flies on

The refit programs are the ones scripts/flappy_transfer.py saves, so run that first:

    python scripts/flappy_transfer.py --save champions/flappy_transfer/refits.json
    python tools/render_flappy_variant.py      # needs ffmpeg; about 20 s

Frames follow the runner's own clips: every second step, at 20 frames a second.
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import flappy_transfer as T  # noqa: E402
import flappy_env as FE  # noqa: E402

VIDEOS = os.path.join(ROOT, "docs", "static", "videos")
POSTERS = os.path.join(ROOT, "docs", "static", "images", "posters")
REFITS = os.path.join(ROOT, "champions", "flappy_transfer", "refits.json")
BASE_SKY = (78, 192, 226)            # the sky colour flappy_env draws
SCORE_XY, SCORE_INK = (8, 8), (20, 24, 10)
SKIES = {"heavy": (238, 132, 92), "moon": (206, 210, 220), "narrow": (238, 166, 196),
         "floaty": (176, 160, 226), "fast": (150, 206, 176), "inverted": (176, 98, 112)}
SURVIVED_STEPS = 480                 # 24 s of an episode that runs to the cap
WINDOW, HOLD = 400, 40               # steps shown before a crash; frames the crash is held
PHYSICS = {name: (physics, schedule) for name, _, physics, schedule in T.VARIANTS}


def program(run):
    path = os.path.join(ROOT, "champions", "flappy_transfer", f"run{run}.json")
    return json.load(open(path))["champion_src"]


def full_refits():
    """The saved refits that reach the end of the episode: (run, change, refit program)."""
    if not os.path.exists(REFITS):
        raise SystemExit("run scripts/flappy_transfer.py --save "
                         "champions/flappy_transfer/refits.json first")
    return [(c["run"], c["change"], c["program"]) for c in json.load(open(REFITS))["refits"]
            if c["refit"] == T.CAP]


def make_env(variant):
    physics, schedule = PHYSICS[variant]
    return FE.FlappyEnv(entity_names="descriptive", obs_mode="objects", seed=42,
                        target_score=10 ** 9, physics=physics, shift_schedule=schedule)


def draw(env, variant, crashed=False):
    """The game's own frame on the variant's sky. The score is held back while the game draws,
    since its antialiased edge would keep the old sky, and redrawn on top."""
    text = ImageDraw.ImageDraw.text
    ImageDraw.ImageDraw.text = lambda *a, **k: None
    try:
        game = np.asarray(env._render_pil())
    finally:
        ImageDraw.ImageDraw.text = text
    sky = np.all(game == BASE_SKY, axis=-1)[..., None]
    img = Image.fromarray(np.where(sky, np.array(SKIES[variant], np.uint8), game))
    d = ImageDraw.Draw(img)
    d.text(SCORE_XY, f"score {env._score}", fill=SCORE_INK)
    if crashed:
        r = FE.BIRD_R + 10
        d.ellipse([FE.BIRD_X - r, env._y - r, FE.BIRD_X + r, env._y + r],
                  outline=(214, 40, 40), width=3)
    return img


def play(src, variant, start, steps, crash_at=None, every=2):
    """Frames of every `every`-th step from `start`, for `steps` steps. With crash_at, the
    episode must end on exactly that step, as scripts/flappy_transfer.py measures it, and its
    last frame is held for HOLD frames. Otherwise it must not end at all."""
    fn, _ = T.rcf.load_policy(src, "llm")
    env = make_env(variant)
    env.reset()
    frames = [draw(env, variant)] if start == 0 else []
    for t in range(1, start + steps + 1):
        env.step(fn(env.get_obs()))
        if env.lost:
            if t != crash_at:
                raise SystemExit(f"{variant}: the controller crashed at step {t}, "
                                 f"expected {crash_at or 'no crash'}")
            return frames + [draw(env, variant, crashed=True)] * HOLD
        if t >= start and (t - start) % every == 0:
            frames.append(draw(env, variant))
    if crash_at:
        raise SystemExit(f"{variant}: the controller did not crash by step {start + steps}")
    return frames


def write(frames, name):
    out = os.path.join(VIDEOS, f"{name}.mp4")
    W, H = frames[0].size
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", "20", "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "23",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out],
                          stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    frames[0].save(os.path.join(POSTERS, f"{name}.png"), optimize=True)
    print(f"wrote {os.path.relpath(out, ROOT)} ({len(frames)} frames, "
          f"{os.path.getsize(out) // 1024} KB)", flush=True)


def crash_step(src, variant):
    physics, schedule = PHYSICS[variant]
    return T.survive(src, physics, schedule, T.CAP)


def main():
    run1 = program(1)
    for variant in ("heavy", "moon", "narrow"):
        write(play(run1, variant, 0, SURVIVED_STEPS), f"flappy_{variant}")
    # Reversed gravity ends within a second, so every step is drawn: half speed.
    write(play(run1, "inverted", 0, T.CAP, crash_at=crash_step(run1, "inverted"), every=1),
          "flappy_inverted")

    for run, variant, refit in full_refits():
        src = program(run)
        crash = crash_step(src, variant)
        start = crash - WINDOW
        write(play(src, variant, start, WINDOW, crash_at=crash), f"flappy_{variant}_frozen")
        write(play(refit, variant, start, WINDOW + 2 * HOLD)[:WINDOW // 2 + HOLD],
              f"flappy_{variant}_refit")


if __name__ == "__main__":
    main()
