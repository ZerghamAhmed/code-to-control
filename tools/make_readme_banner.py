"""Build docs/static/images/readme_banner.gif, the gameplay grid at the top of the README.

Sixteen clips in a 4 x 4 grid, each cropped to a square: the six Atari games, Flappy Bird on
the base game and with gravity x1.5, and eight MuJoCo tasks. GitHub plays GIFs in a README but
not videos.

    python tools/make_readme_banner.py      # needs ffmpeg
"""
import os
import subprocess
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS = os.path.join(ROOT, "docs", "static", "videos")
OUT = os.path.join(ROOT, "docs", "static", "images", "readme_banner.gif")
FPS, SECONDS, TILE, COLS = 12, 6, 120, 4

TILES = [  # (clip, start in seconds, crop[, length in seconds, speed]), row by row
    ("pong_best.mp4", 0.0, "top"),          # the top square keeps the score in view
    # Space Invaders starts where its hits cluster: 10 of the episode's 23 fall in these six
    # seconds (steps 462-702), against 4 in the first six.
    ("spaceinvaders.mp4", 11.55, "centre"),
    ("asterix.mp4", 0.0, "centre"),
    ("breakout.mp4", 0.0, "centre"),
    ("fishingderby.mp4", 0.0, "centre"),
    ("freeway.mp4", 0.0, "centre"),
    ("flappy.mp4", 0.0, "centre"),
    ("flappy_heavy.mp4", 0.0, "centre"),    # gravity x1.5, zero-shot (Figure 3B)
    ("mujoco/halfcheetah.mp4", 0.0, "centre"),
    ("mujoco/swimmer.mp4", 0.0, "centre"),
    ("mujoco/ant.mp4", 0.0, "centre"),
    ("mujoco/walker2d.mp4", 0.0, "centre"),
    ("mujoco/humanoid.mp4", 0.0, "centre"),
    ("mujoco/inverteddoublependulum.mp4", 0.0, "centre"),
    # an earlier, higher-scoring fit of the same Humanoid Standup map (example gameplay only;
    # the released controller and the project page use the paper's median fit). Only the
    # sit-up, the first 0.9 s, looped at half speed.
    ("mujoco/humanoidstandup_champion.mp4", 0.0, "centre", 0.9, 0.5),
    ("mujoco/reacher.mp4", 0.0, "centre"),
]


def frames(path, start, crop, tmp, tag, length=None, speed=1.0):
    """SECONDS of tile frames from `start`, cropped to a square TILE. With `length`, only that
    much of the clip is used, played at `speed` and looped to fill the tile's SECONDS."""
    if not os.path.exists(path):
        raise SystemExit(f"missing clip {os.path.relpath(path, ROOT)}")
    pattern = os.path.join(tmp, f"{tag}_%03d.png")
    y = "0" if crop == "top" else "(ih-min(iw,ih))/2"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start),
                    "-t", str(length or SECONDS), "-i", path, "-vf",
                    f"setpts=PTS/{speed},fps={FPS},"
                    f"crop='min(iw,ih)':'min(iw,ih)':'(iw-min(iw,ih))/2':'{y}',"
                    f"scale={TILE}:{TILE}:flags=lanczos", pattern], check=True)
    out = [Image.open(os.path.join(tmp, f)).convert("RGB")
           for f in sorted(os.listdir(tmp)) if f.startswith(tag + "_")]
    need = FPS * SECONDS
    return (out * (need // len(out) + 1))[:need] if length else out


def main(out=OUT):
    rows = (len(TILES) + COLS - 1) // COLS
    with tempfile.TemporaryDirectory() as tmp:
        clips = [frames(os.path.join(VIDEOS, t[0]), t[1], t[2], tmp, f"t{i}", *t[3:])
                 for i, t in enumerate(TILES)]
        n = min(len(c) for c in clips)
        for k in range(n):
            grid = Image.new("RGB", (COLS * TILE, rows * TILE))
            for i, clip in enumerate(clips):
                grid.paste(clip[k], ((i % COLS) * TILE, (i // COLS) * TILE))
            grid.save(os.path.join(tmp, f"g_{k:03d}.png"))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS),
                        "-i", os.path.join(tmp, "g_%03d.png"), "-vf",
                        "split[a][b];[a]palettegen=stats_mode=diff[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                        "-loop", "0", out], check=True)
    print(f"wrote {out}  ({os.path.getsize(out) // 1024} KB, "
          f"{COLS * TILE}x{rows * TILE}, {n} frames)")


if __name__ == "__main__":
    main()
