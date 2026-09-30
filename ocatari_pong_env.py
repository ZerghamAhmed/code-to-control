"""
OCAtari Pong adapter for TheoryCoder's engine contract.

This is the per-game shim (the analogue of flappy_env): it maps OCAtari's
object-centric Pong state onto the standard symbolic dict + engine interface.
It contains NO game rules, dynamics, or strategy — only interface plumbing:

  get_obs() -> {"player": [[x, y]], "player_size": [[w, h]],
                "ball": [[x, y]], "ball_size": [[w, h]],       # absent until served
                "enemy": [[x, y]], "enemy_size": [[w, h]],
                "score": [player_points], "enemy_score": [pts],
                "won": bool, "lost": bool}
  step("noop"|"up"|"down"), reset(), actions_set, won/lost, score

Coordinates are raw OCAtari pixels (top-left origin, y DOWN) — the agent's
theory has to discover the conventions, same as any game.

Survive-and-report framing: an episode ends ("lost") when the ENEMY scores a
point; `score` = player points won before conceding. This maps Pong onto the
same maximize-progress-before-failure loop as any survival game.
"""

from copy import deepcopy


class PongEnv:
    def __init__(self, level_set="pong", level_id=0, seed=42,
                 target_score=15, frame_skip=1, full_game=False,
                 velocity=False, **kwargs):
        # full_game=True -> PoE-World-comparable protocol: play the whole Atari
        # game to 21 and report POINT DIFFERENTIAL (player - enemy). Default
        # (False) is our survive-and-report framing: episode ends on the first
        # conceded point.
        self.full_game = bool(full_game)
        # velocity=True -> also project each object's per-frame displacement
        # (OCAtari's `dx`/`dy`).
        #
        # WHY THIS IS A FLAG AND NOT THE DEFAULT. Withholding velocity was
        # described in our own notes as a deliberate latent-state test, but the
        # evidence says it was an omission: OCAtari computes `dx`/`dy` on every
        # GameObject for free, our Flappy wrapper DOES project velocity, and
        # PoE-World -- which reads the same parser -- states plainly that it uses
        # "a list of objects, each with an object category, a bounding box, and
        # velocities" (arXiv:2505.10819 §4). So we were the only system in the
        # comparison discarding it.
        #
        # It is measurable, not cosmetic. With position only, sys-ID cannot fit
        # the ball's x-direction: a bounding ball reverses sign every rally, and
        # no single signed constant represents that. Observed directly in
        # run 20260811_063220, where consecutive fits gave
        # BALL_SPEED_X = -2.82 then +2.82 then -2.82.
        #
        # Kept OFF by default so the no-velocity cell stays comparable to what we
        # have already reported, and the two cells differ in exactly one thing.
        self.velocity = bool(velocity)
        from ocatari.core import OCAtari
        self.level_set = level_set
        self.level_id = level_id
        self.seed = int(seed) if seed is not None else 42
        self.target_score = target_score
        self.frame_skip = max(1, int(frame_skip))
        # MISSION: the canonical generic line, identical for every game and every method.
        # Decided 2026-08-18 after auditing that only 4 of 9 games gave all methods the same
        # mission. THIS adapter previously shipped game-specific text, quoted below so the
        # change is recoverable and auditable rather than silent:
        #
        #     "Win rally points in Pong: move your paddle to return the ball past the enemy\n        #      paddle. score = your points. The episode ends if the enemy scores."\n        #\n        #   NOTE: that text was also FACTUALLY WRONG under full_game=True, the protocol we\n        #   run: score is a differential (player_pts - enemy_pts), and conceding a point\n        #   does NOT end the episode. See FINDINGS 1.7.
        #
        # It was replaced because (a) it is read by the planner, our WorldCoder
        # reimplementation AND vendored WorldCoder, so game-specific text here leaks to every
        # column at once, and (b) a 3-way controller ablation (dict / generic / minimal, 20
        # episodes x 3 reps on kangaroo, seaquest, spaceinvaders) found generic best or tied
        # everywhere: seaquest 20 -> 40, spaceinvaders 510 -> 665, kangaroo 200 -> 200.
        # `missions.GENERIC` names no game, no object, no direction and no reward rule; every
        # noun in it (`score`, `lost`) is already a field in the observation.
        from theorycoder.missions import GENERIC as _GENERIC_MISSION
        self.mission = _GENERIC_MISSION
        # NOOP / RIGHT / LEFT joystick == stay / up / down in Pong
        self._action_ids = {"noop": 0, "up": 2, "down": 3}
        self.actions_set = ["noop", "up", "down"]
        self._env = OCAtari("PongNoFrameskip-v4", mode="ram", obs_mode="obj",
                            render_mode="rgb_array")
        self.reset()

    # ---- engine contract ----------------------------------------------------
    def reset(self):
        self._env.reset(seed=self.seed)
        self._player_pts = 0
        self._enemy_pts = 0
        self.won = False
        self.lost = False
        self.turn_number = 0
        self.state = self._project()
        return deepcopy(self.state)

    def step(self, action):
        if action not in self.actions_set:
            raise ValueError(f"Invalid action: {action}. Available: {self.actions_set}")
        aid = self._action_ids[action]
        reward = 0.0
        for _ in range(self.frame_skip):
            _, r, term, trunc, _ = self._env.step(aid)
            reward += r
            if r > 0:
                self._player_pts += int(r)
            elif r < 0:
                self._enemy_pts += int(-r)
                if not self.full_game:
                    self.lost = True      # conceded a point -> episode over
                elif self._enemy_pts >= 21:
                    self.lost = True      # full game over (they reached 21)
            if self._player_pts >= (21 if self.full_game else self.target_score):
                self.won = True
            if term or trunc or self.lost or self.won:
                break
        self.turn_number += 1
        self.state = self._project()
        done = self.won or self.lost
        return deepcopy(self.state), reward, done, {"score": self._player_pts}

    @property
    def score(self):
        # full-game mode reports the PoE-World metric: point differential
        return (self._player_pts - self._enemy_pts) if self.full_game \
            else self._player_pts

    @property
    def obs(self):
        return self.get_obs()

    def get_obs(self):
        obs = deepcopy(self.state)
        # ORACLE HOOK (flag-gated, inert when off). Under TC_ORACLE_MODEL=1 the
        # emulator's own state travels with the projected dict so that
        # theorycoder/oracle/pong_oracle_worldmodel.py can serve as an EXACT
        # transition model. `_ale` is invisible to the agent: underscore keys are
        # filtered by objapi.objects() and by every harness site that renders state.
        import os as _os
        if _os.environ.get("TC_ORACLE_MODEL") == "1":
            _u = self._env
            while not hasattr(_u, "ale"):
                _u = getattr(_u, "env", None) or getattr(_u, "_env", None)
            obs["_ale"] = _u.ale.cloneState()
            obs["_player_pts"] = self._player_pts
            obs["_enemy_pts"] = self._enemy_pts
            obs["_turn"] = self.turn_number
        return obs

    # ---- projection ----------------------------------------------------------
    def _project(self):
        out = {}
        for o in self._env.objects:
            cat = getattr(o, "category", "NoObject")
            if cat == "NoObject":
                continue
            key = cat.lower()          # Player / Ball / Enemy
            x, y = o.xy
            w, h = o.wh
            out[key] = [[float(x), float(y)]]
            out[key + "_size"] = [[float(w), float(h)]]
            if self.velocity:
                # OCAtari's own per-frame displacement. `getattr` with a default
                # rather than direct access: `dx`/`dy` are None on the first frame
                # after a reset (no previous position to difference against), and
                # a None reaching the agent's transition_model raises inside the
                # planner rather than at the boundary, which is far harder to
                # trace. Naming matches flappy_env's `*_velocity` so the two
                # object-centric games present the same shape.
                out[key + "_velocity"] = [[float(getattr(o, "dx", 0) or 0),
                                           float(getattr(o, "dy", 0) or 0)]]
        out["score"] = [self._player_pts]
        out["enemy_score"] = [self._enemy_pts]
        out["won"] = bool(self.won)
        out["lost"] = bool(self.lost)
        return out

    # ---- rendering / misc ----------------------------------------------------
    def _render_pil(self):
        try:
            from PIL import Image
            import numpy as np
            frame = self._env.render()
            return Image.fromarray(np.asarray(frame, dtype="uint8"))
        except Exception:
            return None

    def get_frame(self):
        """RGB uint8 array of the current frame (for game_cli traces)."""
        import numpy as np
        img = self._render_pil()
        return np.asarray(img, dtype="uint8") if img is not None \
            else np.zeros((210, 160, 3), dtype="uint8")

    def save_screen(self, filename="screenshot.png"):
        img = self._render_pil()
        if img is not None:
            try:
                img.save(filename)
            except Exception:
                pass

    def render(self):
        self.save_screen()

    def close(self):
        self._env.close()
