# Provenance

This file records which code produced each number in the paper, and how the files here
differ from the ones that ran.

Each run in the paper wrote a record containing the commit it ran from and the SHA-256
(first 16 hex digits) of its runner, prompt and environment files. The files here were
recovered from those commits and checked against those hashes.

## The runner

`scripts/run_controller_flappy.py` is the runner behind 11 of the 18 Atari cells and 3 of
the 5 Flappy Bird cells, whose hash is `83f3971da1dfee8a`. Only one command-line option was
edited for this release. Nothing that plays a game changed.

The other seven Atari cells ran two other runner versions. Both differ from this one only in
code those games never execute or in instrumentation that does not affect play:

| runner | cells | difference from `83f3971` |
|---|---|---|
| `d9d15ef` | Pong ×3 | Freeway environment construction only. Pong never runs it. |
| `88add1d` | Asterix rep 0–1, Fishing Derby rep 0–1 | Adds a score logged at tick 834 and an episode-termination label. Logging only. |

The other two Flappy Bird cells ran a later version, described under Figure 3B below.

## Prompts

The runner reads `abstraction_prompts/general/controller_only.txt` in `llm` mode. Three
wordings of that prompt produced the reported results:

| file | hash | used by |
|---|---|---|
| `controller_only.txt` | `8bc172abf7a1a2f1` | 14 Atari cells |
| `controller_only_no_orientation.txt` | `408a76ea5c222816` | Asterix rep 0–1, Fishing Derby rep 0–1 |
| `controller_only_flappy.txt` | `3de76274d82ea675` | Flappy Bird |

The three differ only slightly. `408a76` lacks the paragraph describing
`<entity>_orientation`. `3de762` rewords one sentence about import errors. The version
reproduced in the paper's appendix is `3de762`.

To re-run synthesis with a particular wording, copy that file over `controller_only.txt`.
Replaying a champion makes no language-model calls, so it does not use the prompt at all.

## Environments

The Pong, Freeway and Flappy Bird environments, and the generic OCAtari environment used by
8 Atari cells, match their recorded hashes exactly.

Four cells (Asterix rep 0–1, Fishing Derby rep 0–1) ran a generic-environment version
(`44c5154`) that was never committed and cannot be recovered. Their recorded action sets are
identical to the other runs of the same game, and the shipped version differs by optional
arguments that default to off. The two shipped champions drawn from these cells, Asterix
rep 1 and Fishing Derby rep 0, replay to their reported scores under the shipped
environment.

## The `theorycoder/` package

`theorycoder/` holds only the five modules the runner imports: `runmeta`, `llm_client`,
`missions`, `abstractions/objapi` and `abstractions/trajectory`. In the research codebase the
package's `__init__.py` files also imported TheoryCoder's planning code, which the runner never
calls. That code is left out, and the two `__init__.py` files no longer import anything.

## The 21–0 Pong controller

`champions/pong_best.json` is not a number in the paper's table. It is the champion of the best
of the three Pong runs behind the paper's +18, episode 18 of run 1, which scored +21. It ran on
the same runner, prompt and environment as the other Pong runs (hashes in the file) and replays
to 21 in the same 4,286 steps.

## Randomness

Every reported Atari and Flappy Bird run used environment seed 42. The three runs per game
differ because language-model sampling differs between calls, not because of seeding.

## MuJoCo

For each MuJoCo task, `claude-opus-4-8` wrote six feature maps and the best one was kept.
The paper reports the median of three fits of that map, and `champions/mujoco/` holds the
median fit. Each fit is
scored as the median over ten test episodes, seeds 4000 to 4009.

The fits recorded their scores but not their weights. The fits are seeded, so the weights were
re-created by re-running the median fit with its recorded seed. The re-run reproduces the
recorded return, forward displacement, episode length and environment-step count exactly. It
was done in the environment the fits originally ran in: Python 3.10, numpy 1.26.4,
mujoco 3.11.0 and gymnasium 1.3.0. Under Python 3.13 and numpy 2.3, the same re-run gave
different results on four of the five tasks tried, so these versions matter for fitting.
Replaying the released weights passes in a fresh environment built from `requirements.txt`.

Each task's JSON file records where its controller came from: the source run, the library
index within it, the fit seed, and the search settings.

`scripts/replay_mujoco.py` is a standalone copy of the original evaluator. It reproduces the
same observation rounding, feature normalisation, action clipping and termination.

`scripts/fit_mujoco.py` is a copy of the fitter that produced those refits. With the settings
the paper used, it computes the same thing as the version that first fitted the feature maps
when they were drawn. Run with each released controller's fit seed, it re-creates all ten
controllers' weights bit for bit, in a fresh environment built from `requirements.txt`.

`scripts/synthesize_mujoco.py` rebuilds the prompts the paper's runs sent. They are
byte-identical to the logged prompts on eight tasks. The logs for the other two were
overwritten by later experiments: on HalfCheetah by runs with a different prompt version, whose
responses do not match the reported feature maps, and on Humanoid Standup by a 28-term run. In
both, the transitions shown are identical.

## Flappy Bird under changed physics (Figure 3B)

`scripts/flappy_transfer.py` combines two research scripts: the zero-shot sweep and the
constant refit. It uses the same runner and the same environment, whose
physics options set each change. The three champions are the Flappy Bird runs whose champion
flies on the base game: runs 0, 2 and 3. Runs 1 and 4 never did. Run 3 was synthesized with a
later version of the runner (hash `28f124c62a1ea802`). It replays exactly in this one.

The script reproduces every cell of Figure 3B exactly. For the eight cells that are refit, the
steps survived after refitting and the search's environment-step counts also match the
original runs.

The `--save` option is new in this release. It writes each refit to
`champions/flappy_transfer/refits.json`: the constants that moved, and the refit program. The
program is the champion's own source with the moved numbers written where they stand, so it
keeps the model's comments. The script checks that this program survives exactly as long as
the refit did. The project page shows these programs as diffs.

## Decision latency

The paper's 11.4 µs used the timing protocol in `scripts/measure_latency.py`, 300 states per
game timed five times each, on the same machine. It timed one champion per game from runs with
the same configuration. For Pong that was the 21–0 controller in `champions/pong_best.py`. For
the other five games it was a champion not released here. The released champions give about
11 µs.
