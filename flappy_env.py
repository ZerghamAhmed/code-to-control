"""
Flappy Bird environment for TheoryCoder.

Python port of flappybird.html — identical physics + grid projection so the
symbolic state matches the visual reference exactly. Follows the same engine
contract as the other games (BabyAI / maze / sokoban): get_obs() returns an
entity -> [[x, y]] dict, step(action) advances one discrete tick, and
actions_set / won / lost / reset / save_screen / close are provided.

Symbolic state (grid GW x GH, BOTTOM-LEFT origin, y up):
    {
      'bird':          [[x, y]],            # bird cell (x fixed, y varies)
      'pipe_gap_low':  [[x, y], ...],       # bottom edge of each gap (top of lower pipe)
      'pipe_gap_high': [[x, y], ...],       # top edge of each gap (bottom of upper pipe)
      'pipe_x':        [[x], ...],          # pipe columns, left -> right
      'ground_y':      [y],                 # first solid row from the bottom
      'ceiling_y':     [y],                 # top row
      'bird_velocity': [v],                 # discretized latent, + = rising
      'score':         [n],                 # pipes passed
      'won':           bool,                # score >= target_score
      'lost':          bool,                # collided with pipe / ground / ceiling
    }

Actions: "flap" (upward impulse) or "noop".

Determinism: pipe gaps come from a seeded PRNG (mulberry32, same as the HTML),
so reset(seed) + a fixed action sequence is fully reproducible — the property
the planner's transition model relies on.
"""

from copy import deepcopy


# --- physics / geometry constants (must match flappybird.html) ---------------
W, H, GROUND = 420, 600, 64          # pixels
GW, GH = 15, 20                      # symbolic grid (cols x rows)
CW, CH = W / GW, H / GH
BIRD_X, BIRD_R = 110, 13             # bird fixed x
GRAV, FLAP, VMAX, VMIN = 1.15, -11.5, 15, -15


def _thrust_scalar(a):
    """Coerce a continuous action to a thrust scalar in [0, 1].

    Accepts a float, a 1-element sequence, or a numpy array, because the planner's
    CEM backend hands back arrays while a hand-written policy passes a plain float.
    Clamped rather than rejected: an out-of-range proposal is a search artifact, and
    raising here would turn it into a crash mid-episode.
    """
    try:
        v = float(a[0]) if hasattr(a, "__len__") and not isinstance(a, str) else float(a)
    except (TypeError, ValueError):
        v = 0.0
    return max(0.0, min(1.0, v))
PIPE_W, GAP, PIPE_SPEED, PIPE_SPACING = 58, 168, 4.2, 210


def _imul(x, y):
    """32-bit integer multiply (bit-pattern equivalent of JS Math.imul)."""
    return (x * y) & 0xFFFFFFFF


def _mulberry32(seed):
    """Deterministic PRNG matching the JS reference, yields floats in [0, 1)."""
    a = seed & 0xFFFFFFFF

    def rand():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = _imul(t ^ (t >> 15), 1 | t)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rand


class FlappyEnv:
    def __init__(self, level_set="flappy", level_id=0, seed=42,
                 target_score=5, frame_skip=1, obs_mode="grid",
                 physics=None, shift_schedule=None, entity_names="functional",
                 continuous_actions=False, **kwargs):
        # continuous_actions: the ACTION-SPACE ablation. Named `continuous_actions`
        # and NOT `continuous` on purpose — `obs_mode="continuous"` already exists and
        # means something entirely different (exact pixel OBSERVATIONS), so a bare
        # `continuous` flag here would be a genuine footgun.
        #
        # False (DEFAULT) -> actions_set ['flap','noop'], byte-identical to every
        #                    archived result.
        # True            -> actions_set is None and action_space is Box(0,1,(1,)): a
        #                    real-valued thrust. Physics, state and reward are UNCHANGED;
        #                    only how an action is chosen differs. See _tick for why the
        #                    blend makes the discrete set a strict subset of this one.
        self.continuous_actions = bool(continuous_actions)
        # entity_names: "functional" (DEFAULT — player/hazard, with pipes+ground+
        # ceiling merged into one `hazard` list) reproduces the schema every
        # archived result used. "descriptive" (bird/pipe/ground/ceiling, separate
        # entities) removes two give-aways: the label asserting these boxes are
        # lethal, and the grouping asserting the three kinds are one category.
        self.entity_names = entity_names
        self.level_set = level_set
        self.level_id = level_id
        self.seed = int(seed) if seed is not None else 42
        self.target_score = target_score
        # obs_mode: "grid" = coarse 15x20 cell coordinates (BabyAI-style);
        # "continuous" = exact pixel coordinates (y-up), same keys/shapes —
        # the symbolic dict is unchanged, only the numbers get finer. Pipes
        # appear in the state only once they are actually on screen (no
        # boundary clamping — clamping poisoned rate estimation).
        self.obs_mode = obs_mode
        # Variant support (opt-in; defaults reproduce the original constants
        # exactly). `physics` overrides any of the keys below; `shift_schedule`
        # is a list of {"at_tick": N, "set": {KEY: value}} events applied
        # mid-episode (tick-indexed, fires once per reset).
        # DRAG defaults to 0.0 => the term vanishes and behaviour is byte-identical
        # to the original game. A non-zero DRAG adds a velocity-PROPORTIONAL force,
        # i.e. a new structural term that no adjustment of GRAV/FLAP can reproduce.
        self._phys_defaults = dict(GRAV=GRAV, FLAP=FLAP, VMAX=VMAX, VMIN=VMIN,
                                   PIPE_SPEED=PIPE_SPEED, GAP=GAP,
                                   PIPE_SPACING=PIPE_SPACING, DRAG=0.0)
        self._phys_overrides = dict(physics or {})
        self.shift_schedule = list(shift_schedule or [])
        self.phys = dict(self._phys_defaults)
        self.phys.update(self._phys_overrides)
        # frame_skip > 1: the chosen action is applied on the first sub-tick,
        # then the bird coasts (noop) for the remaining ticks. Gives one decision
        # per `frame_skip` ticks — makes turn-based play (ReAct) tractable.
        self.frame_skip = max(1, int(frame_skip))
        if self.continuous_actions:
            # None, not [] — a loud failure if something iterates it, matching
            # lunarlander_env's convention. `action_space` is what routes the planner
            # to CEM instead of tree search.
            self.actions_set = None
            try:
                import gymnasium as _gym
            except Exception:
                import gym as _gym
            import numpy as _np
            self.action_space = _gym.spaces.Box(low=_np.float32(0.0),
                                                high=_np.float32(1.0),
                                                shape=(1,), dtype=_np.float32)
        else:
            self.actions_set = ["flap", "noop"]
            self.action_space = None
        self.won = False
        self.lost = False
        self.turn_number = 0
        self.reset()

    @property
    def score(self):
        return self._score

    @property
    def obs(self):
        """The symbolic state dict (same as get_obs) — exposed so the harness /
        baselines can observe the game symbolically instead of from pixels."""
        return self.get_obs()

    # ---- core simulation -----------------------------------------------------
    def reset(self):
        self._rand = _mulberry32(self.seed)
        self._y = H * 0.42
        self._vy = 0.0
        self._pipes = []            # each: {x, gapTop, gapBot, scored}
        self._score = 0
        self.won = False
        self.lost = False
        self.turn_number = 0
        self._ticks = 0
        # restore variant physics (a mid-episode shift must not leak across resets)
        self.phys = dict(self._phys_defaults)
        self.phys.update(self._phys_overrides)
        self._spawn_if_needed(force=True)
        self.state = self._project()
        return deepcopy(self.state)

    def _spawn_if_needed(self, force=False):
        last = self._pipes[-1] if self._pipes else None
        GAP_, SPACING_ = self.phys["GAP"], self.phys["PIPE_SPACING"]
        if force or last is None or (W - last["x"]) >= SPACING_:
            margin = 70
            gap_center = margin + GAP_ / 2 + self._rand() * (H - GROUND - 2 * margin - GAP_)
            self._pipes.append({
                "x": W + PIPE_W,
                "gapTop": gap_center - GAP_ / 2,
                "gapBot": gap_center + GAP_ / 2,
                "scored": False,
            })

    def _tick(self, action):
        """Advance one tick. Discrete action in {'flap','noop'}, or — when
        `continuous=True` — a float thrust in [0, 1]."""
        if self.won or self.lost:
            return
        if self.shift_schedule:
            for ev in self.shift_schedule:
                if ev.get("at_tick") == self._ticks:
                    self.phys.update(ev.get("set") or {})
        self._ticks += 1
        ph = self.phys
        if self.continuous_actions:
            # CONTINUOUS THRUST. A convex blend between the two discrete outcomes,
            # chosen so the discrete action set is a STRICT SUBSET of this one:
            #     t = 0  ->  vy unchanged      == 'noop'   exactly
            #     t = 1  ->  vy = FLAP         == 'flap'   exactly
            # so anything the discrete agent can do, the continuous agent can also do.
            # That is the property an action-space ablation needs: a drop cannot be
            # blamed on the continuous agent having lost some capability, only on the
            # search over a real interval being harder than a two-way comparison.
            # Gravity/drag/clamping below are untouched, so the PHYSICS is identical.
            t = _thrust_scalar(action)
            self._vy = (1.0 - t) * self._vy + t * ph["FLAP"]
        elif action == "flap":
            self._vy = ph["FLAP"]
        self._vy = self._vy + ph["GRAV"] - ph["DRAG"] * self._vy
        self._vy = max(ph["VMIN"], min(ph["VMAX"], self._vy))
        self._y += self._vy

        for p in self._pipes:
            p["x"] -= ph["PIPE_SPEED"]
        if self._pipes and self._pipes[0]["x"] < -PIPE_W:
            self._pipes.pop(0)
        self._spawn_if_needed()

        # scoring: pipe fully passes the bird's x
        for p in self._pipes:
            if not p["scored"] and p["x"] + PIPE_W < BIRD_X:
                p["scored"] = True
                self._score += 1

        # collisions
        floor = H - GROUND
        if self._y - BIRD_R <= 0:
            self._y = BIRD_R
            self.lost = True
        if self._y + BIRD_R >= floor:
            self._y = floor - BIRD_R
            self.lost = True
        for p in self._pipes:
            if BIRD_X + BIRD_R > p["x"] and BIRD_X - BIRD_R < p["x"] + PIPE_W:
                if self._y - BIRD_R < p["gapTop"] or self._y + BIRD_R > p["gapBot"]:
                    self.lost = True

        if self._score >= self.target_score and not self.lost:
            self.won = True

    # ---- grid projection (px -> grid, bottom-left origin, y up) --------------
    @staticmethod
    def _gx(px):
        return max(0, min(GW - 1, int(px // CW)))

    @staticmethod
    def _gy(py):
        return max(0, min(GH - 1, int((H - py) // CH)))

    def _project(self):
        if self.obs_mode == "objects":
            return self._project_objects()
        if self.obs_mode == "continuous":
            return self._project_continuous()
        bird = [[self._gx(BIRD_X), self._gy(self._y)]]
        gap_low, gap_high, pipe_x = [], [], []
        for p in sorted(self._pipes, key=lambda q: q["x"]):
            cx = self._gx(p["x"] + PIPE_W / 2)
            pipe_x.append([cx])
            gap_low.append([cx, self._gy(p["gapBot"])])   # top of LOWER pipe
            gap_high.append([cx, self._gy(p["gapTop"])])  # bottom of UPPER pipe
        return {
            "bird": bird,
            "pipe_gap_low": gap_low,
            "pipe_gap_high": gap_high,
            "pipe_x": pipe_x,
            "ground_y": [self._gy(H - GROUND)],
            "ceiling_y": [self._gy(0)],
            "bird_velocity": [int(round(-self._vy))],   # + = rising
            "score": [self._score],
            "won": bool(self.won),
            "lost": bool(self.lost),
        }

    def _project_objects(self):
        """CANONICAL SUITE SCHEMA (motion_games-compatible): raw pixels, y DOWN.
        player = the bird (box). hazard = every deadly box: each on-screen pipe
        contributes a TOP box and a BOTTOM box (the gap is just the space
        between), plus a ground box and a ceiling box. Passing a pipe scores,
        so the generic reach_goal/score-progress operator applies unchanged."""
        r1 = lambda v: round(float(v), 1)
        prev_y = getattr(self, "_prev_y_obj", self._y)
        self._prev_y_obj = self._y
        haz, hsz, hvel = [], [], []
        floor = H - GROUND
        for p in sorted(self._pipes, key=lambda q: q["x"]):
            # RAM-honest: report every existing pipe, on-screen or not (OCAtari's
            # RAM mode does the same). Hiding off-screen objects forced the model
            # to invent SPAWNING — an unlearnable-in-budget requirement.
            cx = r1(p["x"] + PIPE_W / 2)
            # top pipe box: from ceiling down to gapTop
            haz.append([cx, r1(p["gapTop"] / 2)])
            hsz.append([float(PIPE_W), r1(p["gapTop"])])
            hvel.append([-self.phys["PIPE_SPEED"], 0.0])
            # bottom pipe box: from gapBot down to the floor
            haz.append([cx, r1((p["gapBot"] + floor) / 2)])
            hsz.append([float(PIPE_W), r1(floor - p["gapBot"])])
            hvel.append([-self.phys["PIPE_SPEED"], 0.0])
        agent = {
            "pos": [[float(BIRD_X), r1(self._y)]],
            "size": [[2.0 * BIRD_R, 2.0 * BIRD_R]],
            "vel": [[0.0, r1(self._y - prev_y)]],
        }
        ground = ([[W / 2.0, r1(floor + 10)]], [[float(W), 20.0]], [[0.0, 0.0]])
        ceiling = ([[W / 2.0, -10.0]], [[float(W), 20.0]], [[0.0, 0.0]])
        tail = {"score": [self._score], "won": bool(self.won), "lost": bool(self.lost)}

        if self.entity_names == "descriptive":
            # Names describe WHAT each object is, not what it does to you, and the
            # three kinds are separate entities. "hazard" both labels these boxes as
            # lethal and asserts that pipes, ground and ceiling are one category —
            # two things the agent should have to learn from death transitions.
            return {
                "bird": agent["pos"], "bird_size": agent["size"],
                "bird_velocity": agent["vel"],
                "pipe": haz, "pipe_size": hsz, "pipe_velocity": hvel,
                "ground": ground[0], "ground_size": ground[1],
                "ground_velocity": ground[2],
                "ceiling": ceiling[0], "ceiling_size": ceiling[1],
                "ceiling_velocity": ceiling[2],
                **tail,
            }

        # DEFAULT ("functional") — byte-identical to the schema every archived
        # result used. Order is preserved exactly: pipe boxes, then ground, then
        # ceiling. Do not reorder; champions ground their operators on this.
        haz = haz + ground[0] + ceiling[0]
        hsz = hsz + ground[1] + ceiling[1]
        hvel = hvel + ground[2] + ceiling[2]
        return {
            "player": agent["pos"],
            "player_size": agent["size"],
            "player_velocity": agent["vel"],
            "hazard": haz,
            "hazard_size": hsz,
            "hazard_velocity": hvel,
            **tail,
        }

    def _project_continuous(self):
        """Exact pixel coordinates, y-UP (larger y = higher), same keys/shapes
        as grid mode. Only on-screen pipes are exposed (visible in the frame);
        no boundary clamping."""
        r1 = lambda v: round(float(v), 1)
        bird = [[float(BIRD_X), r1(H - self._y)]]
        gap_low, gap_high, pipe_x = [], [], []
        for p in sorted(self._pipes, key=lambda q: q["x"]):
            if p["x"] >= W:                      # off-screen: not observable yet
                continue
            cx = r1(p["x"] + PIPE_W / 2)
            pipe_x.append([cx])
            gap_low.append([cx, r1(H - p["gapBot"])])   # top of LOWER pipe
            gap_high.append([cx, r1(H - p["gapTop"])])  # bottom of UPPER pipe
        return {
            "bird": bird,
            "pipe_gap_low": gap_low,
            "pipe_gap_high": gap_high,
            "pipe_x": pipe_x,
            "ground_y": [float(GROUND)],
            "ceiling_y": [float(H)],
            "bird_velocity": [r1(-self._vy)],   # + = rising (px/tick)
            "score": [self._score],
            "won": bool(self.won),
            "lost": bool(self.lost),
        }

    # ---- engine contract -----------------------------------------------------
    def step(self, action):
        if not self.continuous_actions and action not in self.actions_set:
            raise ValueError(f"Invalid action: {action}. Available actions: {self.actions_set}")
        prev_score = self._score
        self._tick(action)
        for _ in range(self.frame_skip - 1):
            if self.won or self.lost:
                break
            self._tick(0.0 if self.continuous_actions else "noop")
        self.state = self._project()
        self.turn_number += 1
        reward = (self._score - prev_score) + (10.0 if self.won else 0.0) - (1.0 if self.lost else 0.0)
        done = self.won or self.lost
        return deepcopy(self.state), reward, done, {"score": self._score}

    def get_obs(self):
        return deepcopy(self.state)

    def _render_pil(self):
        """Draw the current frame to a PIL image (None if PIL unavailable)."""
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None
        img = Image.new("RGB", (W, H), (78, 192, 226))
        d = ImageDraw.Draw(img)
        for p in self._pipes:
            d.rectangle([p["x"], 0, p["x"] + PIPE_W, p["gapTop"]], fill=(87, 176, 74),
                        outline=(47, 111, 40))
            d.rectangle([p["x"], p["gapBot"], p["x"] + PIPE_W, H - GROUND], fill=(87, 176, 74),
                        outline=(47, 111, 40))
        d.rectangle([0, H - GROUND, W, H], fill=(222, 210, 154))
        # --- bird sprite (COSMETIC ONLY) --------------------------------------
        # Drawn inside the 26x26 collision box centred on (BIRD_X, self._y);
        # BIRD_R remains the physics half-extent and is untouched. The wing
        # flaps with vertical velocity purely for readability.
        cx, cy = BIRD_X, self._y
        body, outline = (255, 210, 63), (120, 82, 12)
        d.ellipse([cx - BIRD_R, cy - BIRD_R * 0.85, cx + BIRD_R * 0.85, cy + BIRD_R * 0.85],
                  fill=body, outline=outline)
        # tail
        d.polygon([(cx - BIRD_R, cy), (cx - BIRD_R - 6, cy - 5),
                   (cx - BIRD_R - 6, cy + 5)], fill=(238, 180, 40), outline=outline)
        # wing: up when rising, down when falling
        wing_dy = -4 if self._vy < 0 else 4
        d.polygon([(cx - 4, cy), (cx + 5, cy - 1), (cx - 1, cy + wing_dy + 4)],
                  fill=(240, 240, 245), outline=outline)
        # beak
        d.polygon([(cx + BIRD_R * 0.8, cy - 2), (cx + BIRD_R + 6, cy + 1),
                   (cx + BIRD_R * 0.8, cy + 4)], fill=(240, 120, 40), outline=outline)
        # eye
        d.ellipse([cx + 2, cy - 7, cx + 9, cy], fill=(255, 255, 255), outline=outline)
        d.ellipse([cx + 5, cy - 5, cx + 8, cy - 2], fill=(20, 20, 20))
        d.text((8, 8), f"score {self._score}", fill=(20, 24, 10))
        return img

    def get_frame(self):
        """RGB uint8 (H, W, 3) array of the current frame — for game_cli / ReAct."""
        import numpy as np
        img = self._render_pil()
        if img is None:
            return np.zeros((H, W, 3), dtype=np.uint8)
        return np.asarray(img, dtype=np.uint8)

    def save_screen(self, filename="screenshot.png"):
        """Best-effort PNG of the current frame; silently skips if PIL absent."""
        img = self._render_pil()
        if img is None:
            return
        try:
            img.save(filename)
        except Exception:
            pass

    def render(self):
        self.save_screen()

    def close(self):
        pass
