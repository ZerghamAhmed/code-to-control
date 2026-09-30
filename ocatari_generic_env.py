"""ONE adapter for every OCAtari game, in the engine contract.

    from ocatari_generic_env import OCAtariGenericEnv
    env = OCAtariGenericEnv("SpaceInvaders")

WHY THIS REPLACES PER-GAME FILES
--------------------------------
ocatari_pong_env.py (178 lines) and ocatari_montezuma_env.py (~190) are almost
entirely the same code: iterate `env.objects`, lowercase the category, emit
`<entity>` / `<entity>_size` / `<entity>_velocity` lists, add score/won/lost, wrap
step with frame-skip. Only three things genuinely differ per game -- the action
map, the death signal, and how score is read -- and all three have sane defaults.

Writing montezuma by hand cost an hour and produced TWO silent-corruption bugs
that this file now fixes once, for every game:

  1. OCAtari's `prev_xy == (0,0)` sentinel. On the first frame after a reset an
     object has no previous position, so `dx`/`dy` come back as x-0 and y-0 --
     the POSITION masquerading as a velocity. Measured on montezuma:
     player_velocity read [[76.0, 73.0]] in a game where nothing moves faster
     than ~2 px/tick. Zeroed on frame 0 here.
  2. MULTIPLE INSTANCES per category. Pong has one ball and one player, so its
     adapter writes `out[key] = [...]`. Montezuma has three ladders and several
     platforms; SpaceInvaders has dozens of aliens. A copied pong projection
     keeps only the LAST of each. This one appends.

DEATH SIGNAL. Life loss, not `terminated` -- Atari games give several lives and
`terminated` only fires after the last one, so an agent would be told it survived
a fall it did not. Games with no lives counter fall back to `terminated`.

NOTHING HERE IS GAME KNOWLEDGE. Object names are OCAtari's own categories,
lowercased. No lethality labels, no goal hints, no strategy -- the same
`descriptive` convention the rest of the suite uses.
"""
from copy import deepcopy

from ocatari.core import OCAtari

# Full Atari joystick. Per-game subsets keep the branching factor down; a game
# absent from this table gets the minimal safe set below.
_FULL = {"noop": 0, "fire": 1, "up": 2, "right": 3, "left": 4, "down": 5,
         "upright": 6, "upleft": 7, "downright": 8, "downleft": 9,
         "upfire": 10, "rightfire": 11, "leftfire": 12, "downfire": 13}

ACTIONS = {
    # vertical paddle games
    "Pong":          {"noop": 0, "up": 2, "down": 3},
    "Tennis":        _FULL,
    # horizontal movers
    "Breakout":      {"noop": 0, "fire": 1, "right": 2, "left": 3},
    "SpaceInvaders": {"noop": 0, "fire": 1, "right": 2, "left": 3,
                      "rightfire": 4, "leftfire": 5},
    "Pooyan":        {"noop": 0, "fire": 1, "up": 2, "down": 3,
                      "upfire": 4, "downfire": 5},
    # up/down only
    "Freeway":       {"noop": 0, "up": 1, "down": 2},
    # ------------------------------------------------------------------ 2026-08-17 fix
    # These three had NO entry and fell through to `_DEFAULT_ACTIONS`, which was wrong in
    # two different ways.
    #
    # MsPacman was MISLABELLED, not merely restricted. Its ALE set is
    # ['noop','up','right','left','down','upright','upleft','downright','downleft'] -- it has
    # **no fire at all** -- so `_DEFAULT_ACTIONS`' name->index map was shifted by one and every
    # direction sent a different action. Measured by probing the player's position:
    #     asked "up"    -> dx=+64  (ALE index 2 = `right`)
    #     asked "left"  -> dy=+60  (ALE index 4 = `down`)
    #     asked "down"  -> dx=+26, dy=-48  (ALE index 5 = `upright`)
    #     asked "fire"  -> ALE index 1 = `up`
    # A policy reasoning "move up toward the pellet" was moving right. This invalidates the
    # MsPacman numbers recorded before this date for EVERY method.
    #
    # Seaquest and Kangaroo were correctly LABELLED (their 18-action sets begin with our six in
    # the same order) but truncated to 6, which removes every diagonal and every *fire combo.
    # In Seaquest that is disabling rather than merely limiting: you shoot horizontally while
    # moving vertically to dodge, so without `upfire`/`rightfire`/`leftfire`/`downfire` the
    # agent must CHOOSE between moving and shooting on every tick. That is a plausible part of
    # the evasion-only policies we observed -- with six actions, firing means standing still.
    #
    # COST, stated so it is not discovered later: this raises the branching factor from 6 to 14
    # for two games, which makes the planner's BFS meaningfully more expensive. Correctness
    # first; if the branching proves too costly the answer is a curated subset that is still
    # correctly NAMED, never a shifted map.
    "MsPacman":      {"noop": 0, "up": 1, "right": 2, "left": 3, "down": 4,
                      "upright": 5, "upleft": 6, "downright": 7, "downleft": 8},
    "Seaquest":      _FULL,
    "Kangaroo":      _FULL,
    # full joystick
    "Boxing":        _FULL,
    "Skiing":        {"noop": 0, "right": 1, "left": 2},
    "MontezumaRevenge": {"noop": 0, "fire": 1, "up": 2, "right": 3, "left": 4,
                         "down": 5, "upfire": 10, "rightfire": 11, "leftfire": 12},
    # ------------------------------------------------------------------ 2026-08-19
    # Landscape-screen additions, maps MEASURED from get_action_meanings() (never
    # assumed -- the MsPacman lesson above). Asterix has NO FIRE (9 actions, MsPacman
    # layout). Gopher's 8-action set puts UPFIRE at index 5, where _DEFAULT_ACTIONS
    # would have sent "down". AirRaid/NameThisGame are 6-action horizontal shooters
    # with no up/down. FishingDerby, Asteroids, Frostbite were verified: their
    # meanings[:6] match _DEFAULT_ACTIONS exactly, so they need no entry.
    "Asterix":      {"noop": 0, "up": 1, "right": 2, "left": 3, "down": 4,
                     "upright": 5, "upleft": 6, "downright": 7, "downleft": 8},
    "Gopher":       {"noop": 0, "fire": 1, "up": 2, "right": 3, "left": 4,
                     "upfire": 5, "rightfire": 6, "leftfire": 7},
    "NameThisGame": {"noop": 0, "fire": 1, "right": 2, "left": 3,
                     "rightfire": 4, "leftfire": 5},
    "AirRaid":      {"noop": 0, "fire": 1, "right": 2, "left": 3,
                     "rightfire": 4, "leftfire": 5},
}
_DEFAULT_ACTIONS = {"noop": 0, "fire": 1, "up": 2, "right": 3, "left": 4, "down": 5}


def _resolve_actions(game, ocenv):
    """Curated map if we have one; otherwise DERIVE the map from ALE's own meanings.

    The fixed `_DEFAULT_ACTIONS` fallback is what mislabelled MsPacman for this entire project:
    it assumes the game's action list begins `noop, fire, up, right, left, down`, and MsPacman has
    no `fire` at all, so every index was shifted by one and each direction sent a different
    action. Nothing detects that from the outside -- the env accepts the action, the episode runs,
    and the only symptom is a policy that reasons correctly and behaves wrongly.

    Deriving the map from `get_action_meanings()` makes it **correct by construction**, so the bug
    class cannot recur on a game we add later. Checked against the games we might add next: today
    Bowling and Enduro would BOTH be mislabelled by the fixed fallback, and Riverraid, Frostbite,
    Alien, BankHeist, FishingDerby, Krull and Amidar would each be silently cut 18 -> 6.

    A curated entry still WINS, for two reasons that both matter:
      * Branching. A derived map exposes every ALE action (18 on many games), and the planner's
        BFS pays for that. A curated entry is how we deliberately trade coverage for search cost
        -- but it must stay correctly NAMED, never shifted.
      * Axis-rotated games. ALE's strings are wrong for vertical-paddle games: Pong's indices 2
        and 3 are named `right`/`left` and move the paddle UP/DOWN (verified: `up` -> dy=-62,
        `down` -> dy=+156). A derived map would faithfully reproduce ALE's misleading names, so
        those games need a curated entry and are listed in `scripts/check_action_maps.py`'s
        AXIS_ROTATED set. Any NEW vertical-paddle game must be curated for the same reason.
    """
    if game in ACTIONS:
        return dict(ACTIONS[game]), "ACTIONS[%s]" % game
    try:
        meanings = [m.lower() for m in ocenv._env.unwrapped.get_action_meanings()]
        if meanings:
            return {n: i for i, n in enumerate(meanings)}, "derived_from_ALE"
    except Exception:
        pass
    # Last resort only, and it is recorded as such so an arm that used it is visibly suspect.
    return dict(_DEFAULT_ACTIONS), "_DEFAULT_ACTIONS(last-resort)"

# Games where "score" should be a DIFFERENTIAL (you vs opponent), matching how
# the opposing points are also tracked. Everything else uses raw cumulative score.
DIFFERENTIAL = {"Pong", "Tennis", "Boxing", "IceHockey", "DoubleDunk"}


class OCAtariGenericEnv:
    def __init__(self, game="Pong", seed=42, frame_skip=3, velocity=True,
                 target_score=10 ** 6, full_game=True, game_mode=None,
                 difficulty=None, repeat_action_probability=None, **kwargs):
        self.game = game
        self.seed = int(seed)
        # 3 is the Atari convention PoE-World uses (their OCAtari is built with
        # frameskip=3); matching it keeps decision rates comparable.
        self.frame_skip = max(1, int(frame_skip))
        self.velocity = bool(velocity)
        # ENORMOUS by default on purpose: flappy_env defaults target_score to 5,
        # which silently ended every good episode on a "win" and was read as a
        # score ceiling for a full day. An episode should end by dying or by
        # exhausting ticks, never by a threshold nobody chose.
        self.target_score = target_score
        self.full_game = bool(full_game)
        self.action_space = None
        self.mission = ("Increase your score and avoid losing. Score rises when "
                        "you achieve the game's objective.")
        self._env = OCAtari("%sNoFrameskip-v4" % game, mode="ram",
                            obs_mode="obj", render_mode="rgb_array")
        # HELD-OUT INSTANCE AXES. All three must be applied HERE, not by calling
        # ale.setMode/setDifficulty later: ale_py applies them inside `load_game()`
        # immediately after `loadROM` (ale_py/env.py:219-226), and a ROM load resets
        # the console, so a setter called afterwards is silently discarded. That is
        # the bug N157 recorded for `repeat_action_probability` and mis-attributed to
        # the OCAtari kwarg chain; the kwargs are fine, the ordering was not.
        #
        # Note `game_mode`, NOT `mode` -- OCAtari's own `mode=` is its object-extraction
        # backend ("ram"/"revised") and collides with ALE's game mode.
        _u = self._env._env.unwrapped
        if game_mode is not None or difficulty is not None:
            if game_mode is not None:
                _u._game_mode = int(game_mode)
            if difficulty is not None:
                _u._game_difficulty = int(difficulty)
            _u.load_game()
        if repeat_action_probability is not None:
            _u.ale.setFloat("repeat_action_probability",
                            float(repeat_action_probability))
        self.game_mode = game_mode
        self.difficulty = difficulty
        self.repeat_action_probability = repeat_action_probability
        # AFTER the env exists, because deriving the map needs ALE's own action meanings.
        # `action_source` is recorded so a run can be audited for which path it took.
        self._action_ids, self.action_source = _resolve_actions(game, self._env)
        self.actions_set = list(self._action_ids)
        self._score = 0
        self._opp = 0
        self._lives = None
        # M1: append-only record of every raw ALE termination/truncation seen by
        # this env instance. Deliberately NOT cleared by reset() so a whole unit's
        # terminations remain auditable. Purely additive telemetry.
        self.termination_log = []
        # V2 additive telemetry: exact count of underlying ALE frames stepped by
        # this instance. The frame-skip loop below breaks early on term/trunc/
        # lost/won, so ale_frames is NOT decision_ticks * frame_skip and cannot
        # be derived post hoc. Purely a counter; zero behavioural effect.
        self.ale_frames = 0
        self.reset()

    # ---- engine contract ----------------------------------------------------
    def reset(self):
        self._env.reset(seed=self.seed)
        self._score = 0
        self._opp = 0
        self._lives = None
        self.won = False
        self.lost = False
        # M1: environment completion is tracked SEPARATELY from won/lost.
        # `env_over` means "the environment says this episode is over" and is
        # never conflated with `lost`, which keeps its lives/loss meaning so
        # generated predicates that inspect state["lost"] are unaffected.
        self.env_over = False
        self.last_term = False
        self.last_trunc = False
        self.turn_number = 0
        self.ale_frames_episode = 0
        self.state = self._project()
        return deepcopy(self.state)

    def step(self, action):
        if action not in self.actions_set:
            raise ValueError("Invalid action: %r. Available: %s"
                             % (action, self.actions_set))
        aid = self._action_ids[action]
        reward = 0.0
        for _ in range(self.frame_skip):
            _, r, term, trunc, info = self._env.step(aid)
            self.ale_frames += 1
            self.ale_frames_episode = getattr(self, "ale_frames_episode", 0) + 1
            reward += r
            if self.game in DIFFERENTIAL:
                if r > 0:
                    self._score += int(r)
                elif r < 0:
                    self._opp += int(-r)
                    if not self.full_game:
                        self.lost = True
                    elif self._opp >= 21:
                        self.lost = True
                # 21 IS PONG'S WIN CONDITION AND NOTHING ELSE'S. `full_game` means "play the
                # whole match" and was written for pong, where 21 points wins; the generic
                # adapter then applied it to every game. Measured cost 2026-08-17: WorldCoder
                # passes full_game=True for all generic OCAtari games, so mspacman stopped
                # after ONE episode at score 170, seaquest after two at 140, spaceinvaders
                # after one -- its `if won: break` fired on a win that does not exist in those
                # games. Those cells then looked like 3-rep medians while containing a single
                # terminal episode, and were compared against our best-of-20.
                #
                # Worse, it truncates precisely the runs that are SUCCEEDING: freeway (0) and
                # breakoutoc (2) played all 20 episodes only because they never crossed 21.
                # Our planner passes full_game=True too, so the same cap was latent for us --
                # it just had not scored high enough to trigger it.
                #
                # MsPacman, Seaquest and SpaceInvaders have no win threshold: you play until
                # you are out of lives. Tennis is won at 6 games, boxing on points after a
                # timed round. So gate this on the game actually being pong.
                if (self.full_game and self._score >= 21
                        and str(getattr(self, "game", "")).lower().startswith("pong")):
                    self.won = True
            else:
                self._score += int(r)
                if self._score >= self.target_score:
                    self.won = True
            lives = info.get("lives") if isinstance(info, dict) else None
            if lives is not None:
                if self._lives is not None and lives < self._lives:
                    self.lost = True
                self._lives = lives
            elif term:
                self.lost = True
            # M1 REPAIR. Previously `term`/`trunc` only broke this frame-skip
            # loop and were then discarded, because the returned done flag was
            # derived solely from won/lost. `info["lives"]` is always present for
            # generic OCAtari games, so the `elif term` branch above is
            # unreachable and the only live path to `lost` was a lives DECREASE.
            # Timed games (FishingDerby, Freeway) report lives constant at 0, so
            # their genuine terminations were thrown away and the episode ran on
            # to the external safety cap. Record the raw flags and expose
            # completion separately; do NOT convert termination into a loss.
            if term or trunc:
                self.last_term = bool(term)
                self.last_trunc = bool(trunc)
                self.env_over = True
                self.termination_log.append({
                    "turn_number": self.turn_number,
                    "term": bool(term),
                    "trunc": bool(trunc),
                    "lives_at_termination": lives,
                    "lost": bool(self.lost),
                    "won": bool(self.won),
                    "score": self.score,
                })
            if term or trunc or self.lost or self.won:
                break
        self.turn_number += 1
        self.state = self._project()
        return deepcopy(self.state), reward, \
            (self.won or self.lost or self.env_over), \
            {"score": self.score, "term": self.last_term,
             "trunc": self.last_trunc, "env_over": self.env_over,
             "lost": self.lost, "won": self.won}

    @property
    def score(self):
        return (self._score - self._opp) if self.game in DIFFERENTIAL else self._score

    @property
    def obs(self):
        return self.get_obs()

    def get_obs(self):
        return deepcopy(self.state)

    # ---- projection ----------------------------------------------------------
    def _project(self):
        out = {}
        for o in self._env.objects:
            cat = getattr(o, "category", "NoObject")
            if cat == "NoObject":
                continue
            key = cat.lower()
            x, y = o.xy
            w, h = o.wh
            out.setdefault(key, []).append([float(x), float(y)])
            out.setdefault(key + "_size", []).append([float(w), float(h)])
            if self.velocity:
                dx = float(getattr(o, "dx", 0) or 0)
                dy = float(getattr(o, "dy", 0) or 0)
                if self.turn_number == 0:
                    dx = dy = 0.0          # prev_xy==(0,0) sentinel; see docstring
                out.setdefault(key + "_velocity", []).append([dx, dy])
            # ------------------------------------------------------------ 2026-08-19
            # ORIENTATION PASS-THROUGH. OCAtari tracks a discrete heading for some
            # objects (Asteroids' ship: 0-15; `left` increments, `right` decrements)
            # but this projection dropped it, making the ship's facing UNOBSERVABLE:
            # 12 pure-rotation steps produced zero change in position/size/velocity.
            # A controller therefore could not aim, which is a state-representation
            # failure, not a controller-learning one (interface audit, PAPER_CAMPAIGN).
            # Additive and generic: emitted only where OCAtari reports a non-None
            # orientation, so games without it see no new keys.
            # Coercion must be defensive: Asteroids reports a plain int, but e.g.
            # Kangaroo reports an OCAtari Orientation ENUM — float(enum) raised and
            # killed every kangaroo episode at tick 0 (caught 2026-08-19, P4 launch).
            ori = getattr(o, "orientation", None)
            if ori is not None:
                try:
                    ori_v = float(ori)
                except (TypeError, ValueError):
                    ori_v = getattr(ori, "value", None)
                    try:
                        ori_v = float(ori_v) if ori_v is not None else None
                    except (TypeError, ValueError):
                        ori_v = None
                if ori_v is not None:
                    out.setdefault(key + "_orientation", []).append(ori_v)
        out["score"] = [self.score]
        if self.game in DIFFERENTIAL:
            out["opponent_score"] = [self._opp]
        out["won"] = bool(self.won)
        out["lost"] = bool(self.lost)
        return out

    # ---- rendering -----------------------------------------------------------
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


if __name__ == "__main__":
    import random, sys
    games = sys.argv[1:] or ["Pong", "Freeway", "Breakout", "SpaceInvaders",
                             "Boxing", "Skiing"]
    for g in games:
        try:
            e = OCAtariGenericEnv(g)
            s = e.get_obs()
            cats = sorted(k for k in s if not k.endswith(("_size", "_velocity"))
                          and k not in ("score", "won", "lost", "opponent_score"))
            rng = random.Random(0); t = 0
            while not e.lost and not e.won and t < 300:
                e.step(rng.choice(e.actions_set)); t += 1
            print("  %-15s objects=%-46s random: score=%s ticks=%d"
                  % (g, ",".join(cats)[:46], e.score, t))
            e.close()
        except Exception as exc:
            print("  %-15s FAILED: %s: %s" % (g, type(exc).__name__, str(exc)[:70]))
