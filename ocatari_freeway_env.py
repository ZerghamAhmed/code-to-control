"""
OCAtari Freeway adapter (canonical schema). Interface plumbing only.

  get_obs() -> {"chicken": [[x,y]], "chicken_size": [[w,h]], "chicken_velocity": ...,
                "car": [[x,y]*10], "car_size": ..., "car_velocity": ...,
                "score": [crossings], "won": bool, "lost": bool}

Raw OCAtari pixels, top-left origin, y DOWN. Velocities are per-tick position
deltas computed by the adapter (same thing OCAtari's dx/dy are).
Survive-and-report: score = completed crossings; being hit only knocks the
chicken back (no death); the episode ends on the tick budget ("lost") or at
target_score ("won").
"""
from copy import deepcopy


class FreewayEnv:
    def __init__(self, level_set="freeway", level_id=0, seed=42,
                 target_score=10, frame_skip=1, max_ticks=2500, **kwargs):
        from ocatari.core import OCAtari
        self.level_set, self.level_id = level_set, level_id
        self.seed = int(seed) if seed is not None else 42
        self.target_score = target_score
        self.frame_skip = max(1, int(frame_skip))
        self.max_ticks = max_ticks
        # MISSION: the canonical generic line, identical for every game and every method.
        # Decided 2026-08-18 after auditing that only 4 of 9 games gave all methods the same
        # mission. THIS adapter previously shipped game-specific text, quoted below so the
        # change is recoverable and auditable rather than silent:
        #
        #     "Cross the freeway: move the chicken UP across all lanes of traffic to the top\n        #      to score a crossing, then repeat. Cars knock the chicken back down (no\n        #      death). score = crossings."\n        #\n        #   That text STATED THE POLICY ("move the chicken UP") and told the agent the\n        #   hazard was harmless ("no death") -- the worst leak in the table, and it went\n        #   to every method.
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
        self._env = OCAtari("FreewayNoFrameskip-v4", mode="ram", obs_mode="obj",
                            render_mode="rgb_array")
        meanings = self._env.env.unwrapped.get_action_meanings()
        self._aid = {"noop": meanings.index("NOOP"), "up": meanings.index("UP"),
                     "down": meanings.index("DOWN")}
        self.actions_set = ["noop", "up", "down"]
        self.reset()

    # ---- engine contract ------------------------------------------------------
    def reset(self):
        self._env.reset(seed=self.seed)
        self._score = 0
        self.won = False
        self.lost = False
        self.turn_number = 0
        self._prev = {}
        self.state = self._project()
        return deepcopy(self.state)

    def step(self, action):
        if action not in self.actions_set:
            raise ValueError(f"Invalid action: {action}. Available: {self.actions_set}")
        r_sum = 0.0
        for _ in range(self.frame_skip):
            _, r, term, trunc, _ = self._env.step(self._aid[action])
            r_sum += r
            if r > 0:
                self._score += int(r)
            self.turn_number += 1
            if self._score >= self.target_score:
                self.won = True
            if self.turn_number >= self.max_ticks:
                self.lost = True                      # time budget spent
            if term or trunc or self.won or self.lost:
                break
        self.state = self._project()
        done = self.won or self.lost
        return deepcopy(self.state), r_sum, done, {"score": self._score}

    @property
    def score(self):
        return self._score

    @property
    def obs(self):
        return self.get_obs()

    def get_obs(self):
        return deepcopy(self.state)

    # ---- projection -------------------------------------------------------------
    def _project(self):
        out = {}
        buckets = {}
        for o in self._env.objects:
            cat = getattr(o, "category", "NoObject")
            if cat == "NoObject":
                continue
            buckets.setdefault(cat.lower(), []).append(o)
        for key, objs in buckets.items():
            objs = sorted(objs, key=lambda o: (o.xy[1], o.xy[0]))  # stable order
            out[key] = [[float(o.xy[0]), float(o.xy[1])] for o in objs]
            out[key + "_size"] = [[float(o.wh[0]), float(o.wh[1])] for o in objs]
            vel = []
            for i, o in enumerate(objs):
                px, py = self._prev.get((key, i), (o.xy[0], o.xy[1]))
                vel.append([round(float(o.xy[0] - px), 1),
                            round(float(o.xy[1] - py), 1)])
                self._prev[(key, i)] = (o.xy[0], o.xy[1])
            out[key + "_velocity"] = vel
        out["score"] = [self._score]
        out["won"] = bool(self.won)
        out["lost"] = bool(self.lost)
        return out

    # ---- rendering ----------------------------------------------------------------
    def _render_pil(self):
        try:
            from PIL import Image
            import numpy as np
            return Image.fromarray(np.asarray(self._env.render(), dtype="uint8"))
        except Exception:
            return None

    def get_frame(self):
        import numpy as np
        img = self._render_pil()
        return (np.asarray(img, dtype="uint8") if img is not None
                else np.zeros((210, 160, 3), dtype="uint8"))

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
