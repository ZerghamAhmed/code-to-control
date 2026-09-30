"""ONE way to read a results record. Import this instead of reaching into the dict.

    from record_io import scores_of, best_of, interaction_of, episodes_done, is_probe

WHY THIS FILE EXISTS
--------------------
`run_logs/base_scores/*.json` has been written by at least three generations of tooling and
the schema drifted every time. Audited 2026-08-17 over all 311 records:

  * **173 records (56%) have `best: null`** even though they contain scores. Older planner
    tooling never set the field. Both current consumers survive only because they
    independently recompute `max(scores)` -- a NEW consumer that trusted `best` would read
    None for more than half the corpus and silently drop those cells.
  * **114 records contain episodes with `score: null`** (episodes that crashed or never
    scored). Anything that does `max(e["score"] for e in episodes)` raises on these.
  * **Per-episode interaction is `ticks` in controller records and `env_steps` in planner
    records.** Reading only `ticks` scored every older planner record as ZERO interaction --
    which in one pass made the planner look infinitely sample-efficient, and in another
    produced seven bogus `ratio 0.00` rows in a per-game comparison.

I hit the third trap three separate times in one session before writing it down twice, and
the first two are still latent for any new reader. Notes did not work; a shared accessor
cannot be forgotten, because using the dict directly is now the thing that looks wrong.

The functions here are deliberately total: they never raise on a malformed record, and they
never invent a value. An empty record yields an empty score list, and a caller that cannot
proceed without scores must say so itself -- "no scored episode" must never render as a zero.
"""

__all__ = ["scores_of", "best_of", "median_of", "interaction_of", "episodes_done",
           "episodes_configured", "is_probe", "is_partial", "cap_of", "rails_of",
           "config_of", "aggregate_configs", "strongest_config", "method_of", "model_of",
           "episode_steps",
           "MIN_ARM_EPISODES", "MIN_REPS"]

MIN_ARM_EPISODES = 3      # fewer scored episodes than this is a probe, not an arm
MIN_REPS = 3              # fewer reps than this is a run, not a reproducible result


def _episodes(rec):
    eps = (rec or {}).get("episodes") or []
    return [e for e in eps if isinstance(e, dict)]


def scores_of(rec):
    """Every numeric episode score, in order. Ignores `score: null` episodes."""
    return [e["score"] for e in _episodes(rec)
            if isinstance(e.get("score"), (int, float))]


def best_of(rec):
    """Highest score, RECOMPUTED — never the stored `best`, which is null in 56% of records."""
    sc = scores_of(rec)
    return max(sc) if sc else None


def median_of(rec):
    sc = sorted(scores_of(rec))
    if not sc:
        return None
    n = len(sc)
    return sc[n // 2] if n % 2 else (sc[n // 2 - 1] + sc[n // 2]) / 2.0


def interaction_of(rec):
    """Total env steps: per-episode `ticks` OR `env_steps`, else `totals.env_steps`."""
    tot = sum((e.get("ticks") or e.get("env_steps") or 0) for e in _episodes(rec))
    if tot:
        return tot
    return ((rec or {}).get("totals") or {}).get("env_steps") or 0


def episodes_done(rec):
    """Episodes that actually produced a score — not len(episodes)."""
    return len(scores_of(rec))


def episodes_configured(rec):
    """`--max-episodes` from the run's own argv, or None.

    Configured and completed are different columns. Conflating them once made a planner
    cell look budget-starved when its argv asked for MORE than the controller's; the arms
    simply could not spend it.
    """
    import re
    argv = " ".join(((rec or {}).get("meta") or {}).get("argv") or [])
    m = re.search(r"--max-episodes (\d+)", argv)
    if m:
        return int(m.group(1))
    hp = ((rec or {}).get("meta") or {}).get("hyperparams") or {}
    v = hp.get("max_episodes")
    return int(v) if isinstance(v, (int, float)) else None


def is_probe(rec, min_eps=MIN_ARM_EPISODES):
    """True when this run cannot show learning and must not be counted as an arm."""
    return episodes_done(rec) < min_eps


def is_partial(rec):
    """True when the run was truncated (crashed after producing valid episodes)."""
    return bool(((rec or {}).get("meta") or {}).get("partial"))


def cap_of(rec):
    """`(max_ticks, where_it_came_from)` or `(None, None)`. NEVER a default.

    A rail-bound score is a TRUNCATION, not gameplay, so whether an episode hit the cap
    decides whether its number may be read as performance at all -- flappy's headline
    "780" is exactly `--max-ticks 50000` reached alive. Two measured ways to get this
    wrong, both of which produced confident false tables on 2026-08-17:

      * **Inferring the cap** from repeated tick counts ("the max, if >=2 episodes share
        it") announced "freeway is 90% rail-bound". Freeway is 0%: in a deterministic env
        two identical tick counts are a repeatable DEATH.
      * **Defaulting the cap** -- `build_games_table` used `meta.max_ticks or 50000`, so
        every one of ~42 capless arms was measured against a fabricated 50,000, and any
        game actually capped at 20,000 was silently mis-scored.

    So this returns None rather than a guess, and the caller must render "unknown".
    Checked in order of directness, INCLUDING argv: 13 arms carry the cap only there,
    which is why the first audit called them undecidable.
    """
    import re
    meta = (rec or {}).get("meta") or {}
    v = meta.get("max_ticks")
    if isinstance(v, (int, float)) and v:
        return int(v), "meta"
    v = (meta.get("hyperparams") or {}).get("max_ticks")
    if isinstance(v, (int, float)) and v:
        return int(v), "hyperparams"
    m = re.search(r"--max-ticks\s+(\d+)", " ".join(meta.get("argv") or []))
    if m:
        return int(m.group(1)), "argv"
    per = [e.get("max_ticks") for e in _episodes(rec)
           if isinstance(e.get("max_ticks"), (int, float)) and e.get("max_ticks")]
    if per:
        return int(max(per)), "episodes"
    return None, None


def rails_of(rec):
    """`(n_rail, n_scored, cap_source)`. `cap_source` is None when the cap is UNKNOWN.

    When it is None the caller must print "unknown" -- not 0. "No rails detected" and
    "cannot tell" are different claims, and collapsing them is what made 42 undecidable
    arms look clean.
    """
    cap, src = cap_of(rec)
    eps = [e for e in _episodes(rec) if isinstance(e.get("score"), (int, float))]
    if cap is None:
        return 0, len(eps), None
    # `ended` is authoritative where the runner recorded it (added 2026-08-17); the tick
    # comparison is the fallback for every record written before that.
    n = sum(1 for e in eps if (e.get("ended") == "rail"
                               or (e.get("ended") is None and (e.get("ticks") or 0) >= cap)))
    return n, len(eps), src


def config_of(arm_name):
    """Strip the rep suffix: `pong_ctrl_disc_ars_rep2` -> `pong_ctrl_disc_ars`.

    Reps of one config are repetitions of the SAME experiment and must be aggregated
    together; different configs are different experiments and must not be pooled.
    """
    import re
    return re.sub(r"_rep\d+$", "", arm_name)


def aggregate_configs(arms):
    """`[(arm_name, record)]` -> per-config `{config: {...}}` under ONE convention.

    THE CONVENTION, stated once so three tools cannot disagree about it:

        a rep's number is its BEST episode;
        a config's number is the MEDIAN across its reps of those bests;
        a config with fewer than MIN_REPS reps reports a run, not a result.

    This is not the same as pooling every episode of every arm and taking the median,
    which is what an earlier version of the results board did. Pooling dilutes a strong
    config against weak ones that merely share a game: pong's controller read a pooled
    median of -8.5 while its strongest config (`pong_ctrl_disc_ars`, 3 reps) is 18.0.
    Both numbers were computed correctly; they answer different questions, and only the
    per-config one answers "how well does this method do on this game when it works".

    `median_ep` is kept alongside as the pooled-episode median, because "what does a
    typical episode score" is still worth seeing -- but it is a SEPARATE, labelled field,
    never the headline.
    """
    import collections, statistics
    out = {}
    groups = collections.defaultdict(list)
    for name, rec in arms:
        groups[config_of(name)].append(rec)
    for cfg, recs in groups.items():
        bests = [b for b in (best_of(r) for r in recs) if b is not None]
        if not bests:
            continue
        eps = [s for r in recs for s in scores_of(r)]
        out[cfg] = {
            "reps": len(bests),
            "best": max(bests),
            "median": statistics.median(bests),        # THE headline number
            "median_ep": statistics.median(eps) if eps else None,
            "episodes": len(eps),
            "quotable": len(bests) >= MIN_REPS,
        }
    return out


def strongest_config(arms):
    """The config with the highest `median`; ties broken by reps, then best. Or None.

    Reported rather than the pooled aggregate because a method's capability on a game is
    what its BEST configuration achieves -- pooling in deliberately-weakened ablation arms
    (no-sysid, fs3, a different backend) understates the method by construction.
    """
    agg = aggregate_configs(arms)
    if not agg:
        return None
    cfg = max(agg, key=lambda k: (agg[k]["median"], agg[k]["reps"], agg[k]["best"]))
    d = dict(agg[cfg]); d["config"] = cfg
    return d


def method_of(arm_name, rec):
    """Which of the three methods produced this arm: from the ARGV, never the filename.

        "controller"   -- the reactive controller learner (run_controller_flappy.py)
        "in-model"     -- the in-model controller: planner run with `--controller on`
        "transition"   -- the transition-model planner (operators + world model + search)

    WHY ARGV. I first classified in-model arms by name, treating `_v5` and `_v45` as the
    marker. `flappy_cont_v45` has **no `--controller on`**: v45 is a version tag, not a
    method. Three arms were filed under a method they never ran. The reported number
    survived by luck -- v5's median (1) beat v45's (0) so config selection picked the right
    one anyway -- which is exactly the kind of near-miss that stays invisible until it
    doesn't. Measured 2026-08-17: **exactly one config in the corpus uses `--controller on`**
    (`flappy_cont_v5`, 3 reps), so the in-model column is one game wide and any name-based
    rule that finds more than that is wrong.
    """
    argv = " ".join(((rec or {}).get("meta") or {}).get("argv") or [])
    if "run_controller_flappy" in argv or "ctrl" in arm_name:
        return "controller"
    if "--controller on" in argv:
        return "in-model"
    return "transition"


def model_of(rec):
    """The backend that produced this arm, from `meta.model` or argv. Never assumed.

    The corpus is NOT one backend: 160 haiku, 52 claude-opus-4-8, 2 gpt-4o, 1 unrecorded,
    and on four games the two methods ran on different models -- so a cell's backend has to
    travel with its number.
    """
    import re
    meta = (rec or {}).get("meta") or {}
    if meta.get("model"):
        return meta["model"]
    m = re.search(r"--model\s+(\S+)", " ".join(meta.get("argv") or []))
    return m.group(1) if m else "UNRECORDED"


def episode_steps(ep):
    """Env steps in ONE episode: `ticks` (controller) or `env_steps` (planner/WorldCoder).

    The fifth appearance of this schema split in one session. The controller writes `ticks`;
    `run_autonomous_survive.py` and the WorldCoder baseline write `env_steps`. Summing only
    `ticks` reported **0 env steps for three of four methods** in the august17 pilot, exactly
    as it once scored every older planner record as zero interaction and made the planner
    look infinitely sample-efficient. Any new consumer must call this, not the dict.
    """
    if not isinstance(ep, dict):
        return 0
    v = ep.get("ticks")
    if isinstance(v, (int, float)) and v:
        return int(v)
    v = ep.get("env_steps")
    return int(v) if isinstance(v, (int, float)) and v else 0
