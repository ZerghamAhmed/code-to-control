"""Render the MuJoCo clips and posters on the project page from the released controllers.

    python tools/render_mujoco.py                # every task in champions/mujoco/
    python tools/render_mujoco.py halfcheetah

Each clip plays the controller's test episodes in order (seeds 4000, 4001, ...) at real-time
speed. It holds whole episodes only, up to 20 seconds; a first episode longer than that is cut
at 20 seconds. MuJoCo draws the floor only a finite distance out (40 m for HalfCheetah), so a
clip also ends once the followed body comes within 10 m of the floor's edge. Every rendered
episode is checked against an unrendered replay, so drawing the frames cannot have changed
what the controller did. Needs ffmpeg.
"""
import fractions
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import replay_mujoco as R  # noqa: E402

VIDEOS = os.path.join(ROOT, "docs", "static", "videos", "mujoco")
POSTERS = os.path.join(ROOT, "docs", "static", "images", "posters", "mujoco")
WIDTH, HEIGHT, FPS, LIMIT = 480, 360, 30, 20.0
EDGE_MARGIN = 10.0     # metres; stop before the edge of the drawn floor comes into view

# Camera per task: follow a body, or look at a fixed point.
CAMERAS = {
    "halfcheetah":            dict(body="torso", distance=3.5, elevation=-12, azimuth=90),
    "hopper":                 dict(body="torso", distance=3.5, elevation=-8, azimuth=90),
    "walker2d":               dict(body="torso", distance=4.0, elevation=-8, azimuth=90),
    "swimmer":                dict(body="torso", distance=5.0, elevation=-90, azimuth=90),
    "ant":                    dict(body="torso", distance=5.0, elevation=-30, azimuth=90),
    "humanoid":               dict(body="torso", distance=4.0, elevation=-8, azimuth=90),
    "humanoidstandup":        dict(body="torso", distance=4.0, elevation=-8, azimuth=90),
    "invertedpendulum":       dict(lookat=(0.0, 0.0, 0.35), distance=2.4, elevation=-5, azimuth=90),
    "inverteddoublependulum": dict(lookat=(0.0, 0.0, 0.55), distance=2.6, elevation=-5, azimuth=90),
    "reacher":                dict(lookat=(0.0, 0.0, 0.0), distance=0.8, elevation=-90, azimuth=90),
}


def camera(model, cfg):
    import mujoco
    cam = mujoco.MjvCamera()
    if "body" in cfg:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["body"])
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = cfg["lookat"]
    cam.distance, cam.elevation, cam.azimuth = cfg["distance"], cfg["elevation"], cfg["azimuth"]
    return cam


def floor_x_range(model):
    """x extent of the drawn floor, or None if there is no finite floor plane."""
    import mujoco
    spans = [(model.geom_pos[i][0] - model.geom_size[i][0], model.geom_pos[i][0] + model.geom_size[i][0])
             for i in range(model.ngeom)
             if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE and model.geom_size[i][0] > 0]
    return (max(a for a, _ in spans), min(b for _, b in spans)) if spans else None


def render(task):
    import mujoco
    meta, mod = R.load(task)
    ctrl = R.Controller(meta, mod)
    plain = {s: R.play(meta, ctrl, s) for s in meta["test_seeds"]}

    dt = R.Episode(meta["env_id"], meta["labels"], meta["test_seeds"][0]).env.unwrapped.dt
    seeds, total = [], 0.0
    for s in meta["test_seeds"]:
        length = plain[s][2] * dt
        if seeds and total + length > LIMIT:
            break
        seeds.append(s)
        total += min(length, LIMIT)
    rate = fractions.Fraction(1 / dt).limit_denominator(1000)

    os.makedirs(VIDEOS, exist_ok=True)
    os.makedirs(POSTERS, exist_ok=True)
    out = os.path.join(VIDEOS, f"{task}.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-framerate", str(rate), "-i", "-", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "slow", "-crf", "31", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", out],
        stdin=subprocess.PIPE)
    state = {"renderer": None, "cam": None, "n": 0, "first": None, "stop": False, "frames": 0}

    def draw(ep):
        if state["stop"] or state["n"] * dt > LIMIT + 1e-9:
            return
        u = ep.env.unwrapped
        if state["renderer"] is None:
            state["renderer"] = mujoco.Renderer(u.model, height=HEIGHT, width=WIDTH)
            state["cam"] = camera(u.model, CAMERAS[task])
            state["floor"] = floor_x_range(u.model)
        if state["floor"] and state["cam"].type == mujoco.mjtCamera.mjCAMERA_TRACKING:
            x = u.data.xpos[state["cam"].trackbodyid][0]
            lo, hi = state["floor"]
            if not lo + EDGE_MARGIN < x < hi - EDGE_MARGIN:
                state["stop"] = True
                return
        state["renderer"].update_scene(u.data, camera=state["cam"])
        frame = state["renderer"].render()
        if state["first"] is None:
            state["first"] = frame.copy()
        ff.stdin.write(np.ascontiguousarray(frame).tobytes())
        state["frames"] += 1

    for s in seeds:
        state["n"] = 0

        def on_step(ep):
            draw(ep)
            state["n"] += 1

        drawn = R.play(meta, ctrl, s, on_step=on_step)
        if drawn != plain[s]:
            raise SystemExit(f"{task} seed {s}: rendering changed the episode {drawn} != {plain[s]}")
    ff.stdin.close()
    if ff.wait():
        raise SystemExit(f"ffmpeg failed for {task}")
    from PIL import Image
    Image.fromarray(state["first"]).save(os.path.join(POSTERS, f"{task}.png"), optimize=True)
    cut = " (cut before the floor's edge)" if state["stop"] else ""
    print(f"{task:<24} seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} episode{'s' * (len(seeds) > 1)}), "
          f"{state['frames'] * dt:5.1f} s{cut}  ->  {os.path.relpath(out, ROOT)} "
          f"({os.path.getsize(out) // 1024} KB)")


def main():
    tasks = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(R.CHAMP) if f.endswith(".json"))
    for task in tasks:
        render(task)


if __name__ == "__main__":
    main()
