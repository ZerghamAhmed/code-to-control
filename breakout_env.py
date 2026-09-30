"""BREAKOUT / ARKANOID — continuous state, DISCRETE actions, sparse-ish reward.

WHY THIS GAME EXISTS IN THIS REPO
---------------------------------
It adds REFLECTION physics, which nothing else in the suite has. The existing games cover
side-scroll avoidance (helicopter, flappy), ballistic aiming (projectile), grid movement
with no physics (snake), and thrust landing (lunarlander). None of them require predicting
a bounce.

That makes it the mechanical BRIDGE between two games we already run:

    projectile   predict where a moving object LANDS      (one arc, no bounce)
    breakout     predict where it lands AFTER reflections (arcs + walls + paddle)
    pong         intercept a reflecting object            (reflection + an opponent)

If the continual-learning line works at all, projectile -> breakout -> pong should be a
transfer chain with real shared structure: the same "integrate a velocity, detect a
collision, reflect" core. It is a stronger test than helicopter (which shares Flappy's
structure almost exactly) because the shared piece must survive a genuine change of task.

Breakout is also RECOGNISABLE and not ours: reviewers discount self-designed benchmarks,
and a faithful reimplementation of a 1976 Atari title with published rules is much harder
to accuse of being tuned to our method than an invented game.

SCHEMA. `player` (the paddle) / `ball` / `hazard` (walls) / `brick` + the usual
`*_size`, `*_velocity`, `score`, `won`, `lost`. Continuous float positions, y DOWN,
1-decimal rounding, matching flappy/helicopter conventions.

REWARD: +1 per brick destroyed, nothing else. `do nothing` lets the ball fall past a
stationary paddle and scores 0 (measured, see __main__) -- no per-tick survival term, so
there is no stalling exploit. `lost` when all balls are gone; `won` at `target_score`.

PHYSICS (v1). Deliberately simple and EXACTLY reflective -- no spin, no brick-drop
powerups, no ball acceleration -- so that a wrong prediction is diagnosable rather than
lost in incidental complexity:

    x += vx ;  y += vy
    wall hit  -> negate the corresponding velocity component
    brick hit -> negate vy, destroy the brick, +1
    paddle    -> negate vy AND add a horizontal kick proportional to the offset from
                 the paddle centre (the one non-obvious rule, and the one that makes
                 aiming possible at all)
"""
import random
import sys
from copy import deepcopy

W, H = 480.0, 480.0
PADDLE_W, PADDLE_H = 64.0, 10.0
PADDLE_Y = H - 30.0
PADDLE_DX = 8.0                  # px per tick of paddle movement
BALL_R = 5.0
BALL_SPEED = 5.0
KICK = 4.0                       # horizontal kick per unit of normalised paddle offset
WALL_T = 10.0                    # wall thickness, reported as hazard boxes

BRICK_W, BRICK_H = 48.0, 18.0
BRICK_TOP = 60.0
BRICK_GAP = 2.0

ACTIONS = ("noop", "left", "right")
ACTIONS_BLIND = ("act_a", "act_b", "act_c")

LEVELS = {
    0: dict(rows=2, cols=8, balls=3, target=16),
    1: dict(rows=3, cols=8, balls=3, target=24),
    2: dict(rows=4, cols=8, balls=2, target=32),
}

MAX_TICKS = 6000


def _r1(v):
    return round(float(v), 1)


def _mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = (t ^ (t >> 15)) * (t | 1) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rnd


class BreakoutEnv:
    """Engine contract: get_obs / step / reset / actions_set / won / lost / score."""

    mission = ("Maximize score. The episode ends when the environment reports "
               "won or lost.")

    episode_horizon = 3000

    def __init__(self, level_set="breakout", level_id=0, seed=42,
                 target_score=None, frame_skip=1, obs_mode="objects",
                 blind=False, entity_names="functional", physics=None, **kwargs):
        self.level_set = level_set
        self.level_id = int(level_id)
        if self.level_id not in LEVELS:
            raise ValueError(f"unknown level_id {level_id} — have {sorted(LEVELS)}")
        self.spec = dict(LEVELS[self.level_id])
        self.seed = int(seed) if seed is not None else 42
        self.target_score = (float(self.spec["target"]) if target_score is None
                             else float(target_score))
        self.frame_skip = max(1, int(frame_skip or 1))
        self.obs_mode = obs_mode
        self.blind = bool(blind)
        self.entity_names = entity_names
        self.actions_set = list(ACTIONS_BLIND if self.blind else ACTIONS)
        self._phys_defaults = dict(BALL_SPEED=BALL_SPEED, KICK=KICK, PADDLE_DX=PADDLE_DX)
        self._phys_overrides = dict(physics or {})
        self.phys = dict(self._phys_defaults)
        self.phys.update(self._phys_overrides)
        self.won = False
        self.lost = False
        self.turn_number = 0
        self.episode_index = -1
        self.reset()

    @property
    def score(self):
        return int(self._score)

    @property
    def obs(self):
        return self.get_obs()

    # ---- core simulation -----------------------------------------------------
    def reset(self, episode=None):
        self._rand = _mulberry32(self.seed)
        self._px = W / 2.0
        self._score = 0
        self.won = False
        self.lost = False
        self.turn_number = 0
        self._ticks = 0
        self.phys = dict(self._phys_defaults)
        self.phys.update(self._phys_overrides)
        self._balls_left = int(self.spec["balls"])
        self._bricks = []
        rows, cols = int(self.spec["rows"]), int(self.spec["cols"])
        total_w = cols * BRICK_W + (cols - 1) * BRICK_GAP
        x0 = (W - total_w) / 2.0
        for r in range(rows):
            for c in range(cols):
                self._bricks.append({
                    "x": x0 + c * (BRICK_W + BRICK_GAP) + BRICK_W / 2.0,
                    "y": BRICK_TOP + r * (BRICK_H + BRICK_GAP) + BRICK_H / 2.0,
                    "alive": True})
        self._launch()
        self.state = self._project()
        return deepcopy(self.state)

    def _launch(self):
        """Serve from the paddle with a seeded horizontal component, so the opening
        is reproducible but not identical across seeds."""
        self._bx = self._px
        self._by = PADDLE_Y - BALL_R - 2.0
        ang = (self._rand() * 0.6 - 0.3)          # radians off vertical
        sp = float(self.phys["BALL_SPEED"])
        self._vx = sp * ang * 2.0
        self._vy = -sp

    def _decode(self, action):
        if isinstance(action, str):
            if action in ACTIONS:
                return ACTIONS.index(action)
            if action in ACTIONS_BLIND:
                return ACTIONS_BLIND.index(action)
            raise ValueError(f"Invalid action: {action}. Available actions: "
                             f"{self.actions_set}")
        a = int(action)
        if not 0 <= a < len(ACTIONS):
            raise ValueError(f"Invalid action: {action}. Available actions: "
                             f"{self.actions_set}")
        return a

    def _tick(self, a):
        if self.won or self.lost:
            return
        self._ticks += 1
        self.turn_number += 1
        # paddle
        dx = float(self.phys["PADDLE_DX"])
        if a == 1:
            self._px -= dx
        elif a == 2:
            self._px += dx
        self._px = max(WALL_T + PADDLE_W / 2.0, min(W - WALL_T - PADDLE_W / 2.0, self._px))
        # ball
        self._bx += self._vx
        self._by += self._vy
        # walls
        if self._bx - BALL_R <= WALL_T:
            self._bx = WALL_T + BALL_R
            self._vx = -self._vx
        if self._bx + BALL_R >= W - WALL_T:
            self._bx = W - WALL_T - BALL_R
            self._vx = -self._vx
        if self._by - BALL_R <= WALL_T:
            self._by = WALL_T + BALL_R
            self._vy = -self._vy
        # bricks — nearest-first so one tick destroys at most one brick
        for b in self._bricks:
            if not b["alive"]:
                continue
            if (abs(self._bx - b["x"]) <= BRICK_W / 2.0 + BALL_R
                    and abs(self._by - b["y"]) <= BRICK_H / 2.0 + BALL_R):
                b["alive"] = False
                self._vy = -self._vy
                self._score += 1
                break
        # paddle bounce, with the offset kick
        if (self._vy > 0
                and PADDLE_Y - PADDLE_H / 2.0 - BALL_R <= self._by <= PADDLE_Y + PADDLE_H
                and abs(self._bx - self._px) <= PADDLE_W / 2.0 + BALL_R):
            self._vy = -abs(self._vy)
            off = (self._bx - self._px) / (PADDLE_W / 2.0)
            self._vx += float(self.phys["KICK"]) * off
        # lost ball
        if self._by - BALL_R > H:
            self._balls_left -= 1
            if self._balls_left <= 0:
                self.lost = True
            else:
                self._launch()
        if self._score >= self.target_score and not self.lost:
            self.won = True

    def step(self, action):
        a = self._decode(action)
        if self.won or self.lost:
            return deepcopy(self.state), 0.0, True, {"score": self.score}
        prev = self._score
        for i in range(self.frame_skip):
            self._tick(a if i == 0 else 0)
            if self.won or self.lost:
                break
        self.state = self._project()
        reward = ((self._score - prev) + (10.0 if self.won else 0.0)
                  - (1.0 if self.lost else 0.0))
        done = self.won or self.lost
        return deepcopy(self.state), reward, done, {"score": self.score}

    # ---- projection ----------------------------------------------------------
    def _project(self):
        walls = [[W / 2.0, WALL_T / 2.0], [WALL_T / 2.0, H / 2.0],
                 [W - WALL_T / 2.0, H / 2.0]]
        wall_sz = [[W, WALL_T], [WALL_T, H], [WALL_T, H]]
        bricks = [[_r1(b["x"]), _r1(b["y"])] for b in self._bricks if b["alive"]]
        brick_sz = [[BRICK_W, BRICK_H] for _ in bricks]
        tail = {"score": [self.score], "won": bool(self.won), "lost": bool(self.lost)}
        paddle = [[_r1(self._px), PADDLE_Y]]
        ball = [[_r1(self._bx), _r1(self._by)]]
        ballv = [[_r1(self._vx), _r1(self._vy)]]

        if self.entity_names == "descriptive":
            out = {
                "paddle": paddle, "paddle_size": [[PADDLE_W, PADDLE_H]],
                "paddle_velocity": [[0.0, 0.0]],
                "ball": ball, "ball_size": [[2 * BALL_R, 2 * BALL_R]],
                "ball_velocity": ballv,
                "brick": bricks, "brick_size": brick_sz,
                "wall": walls, "wall_size": wall_sz,
                "balls_left": [self._balls_left],
                **tail,
            }
        else:
            out = {
                "player": paddle, "player_size": [[PADDLE_W, PADDLE_H]],
                "player_velocity": [[0.0, 0.0]],
                "ball": ball, "ball_size": [[2 * BALL_R, 2 * BALL_R]],
                "ball_velocity": ballv,
                "brick": bricks, "brick_size": brick_sz,
                "hazard": walls, "hazard_size": wall_sz,
                "balls_left": [self._balls_left],
                **tail,
            }
        if self.blind:
            keys = [k for k in out if k not in ("score", "won", "lost")]
            blind = {f"key_{chr(97 + i)}": out[k] for i, k in enumerate(keys)}
            blind.update(tail)
            return blind
        return out

    def get_obs(self):
        return deepcopy(self.state)

    def _render_pil(self):
        """PIL frame — rendering ONLY. The harness's GIF writer looks for this name."""
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None
        img = Image.new("RGB", (int(W), int(H)), (14, 16, 22))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, WALL_T], fill=(60, 70, 90))
        d.rectangle([0, 0, WALL_T, H], fill=(60, 70, 90))
        d.rectangle([W - WALL_T, 0, W, H], fill=(60, 70, 90))
        for b in self._bricks:
            if not b["alive"]:
                continue
            d.rectangle([b["x"] - BRICK_W / 2, b["y"] - BRICK_H / 2,
                         b["x"] + BRICK_W / 2, b["y"] + BRICK_H / 2], fill=(200, 120, 70))
        d.rectangle([self._px - PADDLE_W / 2, PADDLE_Y - PADDLE_H / 2,
                     self._px + PADDLE_W / 2, PADDLE_Y + PADDLE_H / 2], fill=(230, 220, 120))
        d.ellipse([self._bx - BALL_R, self._by - BALL_R,
                   self._bx + BALL_R, self._by + BALL_R], fill=(245, 245, 245))
        d.text((14, 14), f"score {self._score}  balls {self._balls_left}",
               fill=(225, 225, 225))
        return img

    def get_frame(self):
        img = self._render_pil()
        if img is None:
            return None
        import numpy as np
        return np.asarray(img)


# ---- reference policies -------------------------------------------------------
if __name__ == "__main__":
    import statistics as st

    class TrackBall:
        """Hand-written baseline: move the paddle toward the ball's current x.

        Information-equivalent to the observation (it reads ball x and paddle x, both
        visible). It does NOT predict the bounce or aim at a particular brick, so the
        gap to a planning agent is exactly what PREDICTING REFLECTION is worth --
        which is the whole reason this game is in the suite.
        """

        def __init__(self, predict=False):
            self.predict = predict

        def __call__(self, env):
            target = env._bx
            if self.predict and env._vy > 0:
                dt = (PADDLE_Y - env._by) / max(1e-6, env._vy)
                target = env._bx + env._vx * dt
                span = W - 2 * WALL_T
                target = (target - WALL_T) % (2 * span)
                if target > span:
                    target = 2 * span - target
                target += WALL_T
            if target < env._px - 3:
                return 1
            if target > env._px + 3:
                return 2
            return 0

    policies = [
        ("do nothing (stationary paddle)", lambda: (lambda env: 0)),
        ("uniform random actions", lambda: (lambda env: random.randrange(3))),
        ("track ball x", TrackBall),
        ("predict bounce point", lambda: TrackBall(predict=True)),
    ]
    seeds = list(range(16))
    print("obs keys :", list(BreakoutEnv().get_obs().keys()))
    print("actions  :", BreakoutEnv().actions_set)
    print("blind keys:", list(BreakoutEnv(blind=True).get_obs().keys()))
    print("descriptive keys:",
          list(BreakoutEnv(entity_names="descriptive").get_obs().keys()))

    for lid in sorted(LEVELS):
        sp = LEVELS[lid]
        print(f"\n### level {lid}: {sp['rows']}x{sp['cols']} bricks, {sp['balls']} balls, "
              f"target {sp['target']} | {len(seeds)} seeds, max {MAX_TICKS} ticks")
        print(f"{'policy':<32} {'score':>9} {'ticks':>9} {'won':>6}   per-seed scores")
        for name, mk in policies:
            scores, ticks, wins = [], [], 0
            for s in seeds:
                random.seed(1000 + s)
                env = BreakoutEnv(level_id=lid, seed=s)
                pol = mk()
                n = 0
                while not (env.won or env.lost) and n < MAX_TICKS:
                    env.step(pol(env))
                    n += 1
                scores.append(env.score)
                ticks.append(n)
                wins += 1 if env.won else 0
            print(f"{name:<32} {st.median(scores):>9.2f} {st.median(ticks):>9.1f} "
                  f"{wins:>4}/{len(seeds)}   {scores}")
