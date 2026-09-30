"""Build docs/index.html, the project page.

The controllers shown on the page are read from champions/ at build time, so the page
always shows exactly the code that ships in the repository.

    python tools/build_page.py
"""
import ast
import difflib
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAMP = os.path.join(ROOT, "champions")
OUT = os.path.join(ROOT, "docs", "index.html")

REPO = "https://github.com/ZerghamAhmed/code-to-control"
SITE = "https://zerghamahmed.github.io/code-to-control"
DESCRIPTION = ("A language model writes a game-playing controller as an ordinary Python program. "
               "Choosing an action takes one function call, with no language-model call and no planning.")

GAMES = [  # (file stem, display name, note shown under the score)
    ("pong", "Pong", None),
    ("spaceinvaders", "Space Invaders", None),
    ("asterix", "Asterix", None),
    ("breakout", "Breakout", None),
    ("fishingderby", "Fishing Derby", None),
    ("freeway", "Freeway", None),
    ("flappy", "Flappy Bird", "The full episode survives all 20,000 steps."),
]

# Extra programs shown in the Atari code viewer, after the seven games.
EXTRA_PROGRAMS = [("pong_best", "Pong 21\u20130")]

# Flappy Bird physics changes, as scripts/flappy_transfer.py names them, and how the page
# names them. The exact change, from the script's own label, is shown beside the name.
CHANGE_NAMES = {
    "heavy": "Heavier gravity", "drag": "Drag", "shift": "Heavier gravity mid-flight",
    "narrow": "Narrower gap", "floaty": "Weaker flap", "moon": "Light gravity",
    "fast": "Faster pipes", "inverted": "Gravity reversed",
}
# Clips rendered by tools/render_flappy_variant.py: (clip, run, change, "unchanged" or "refit").
UNCHANGED_CLIPS = [
    ("flappy_heavy", 1, "heavy", "unchanged"),
    ("flappy_moon", 1, "moon", "unchanged"),
    ("flappy_narrow", 1, "narrow", "unchanged"),
    ("flappy_inverted", 1, "inverted", "unchanged"),
]
HALF_SPEED = {"flappy_inverted"}

# MuJoCo tasks in display order. A task appears once champions/mujoco/<stem>.json exists.
# Locomotion tasks also show how far the body travelled, since return alone can hide that.
TASKS = [  # (file stem, display name, locomotion)
    ("halfcheetah", "HalfCheetah", True),
    ("swimmer", "Swimmer", True),
    ("ant", "Ant", True),
    ("hopper", "Hopper", True),
    ("walker2d", "Walker2d", True),
    ("humanoid", "Humanoid", True),
    ("humanoidstandup", "Humanoid Standup", False),
    ("invertedpendulum", "Inverted Pendulum", False),
    ("inverteddoublependulum", "Inverted Double Pendulum", False),
    ("reacher", "Reacher", False),
]

ABSTRACT = (
    "Recent LLM-based approaches to control either invoke a language model to select "
    "actions or synthesize world models that require planning at every decision, "
    "introducing latency that can limit real-time use. We introduce Code to Control, an "
    "approach that synthesizes Python controllers which execute directly as policies. "
    "Code to Control separates program structure from parameters. An LLM synthesizes the "
    "controller structure, while derivative-free search fits its parameters for "
    "continuous control using feedback from the environment. Once learned, the resulting "
    "controllers require neither LLM inference nor planning at decision time, enabling "
    "real-time gameplay and, under our timing protocol, faster action selection than a "
    "PPO policy. Across a suite of Atari games, Flappy Bird, and MuJoCo tasks, Code to "
    "Control outperforms planning-based program synthesis methods, remains competitive "
    "with deep reinforcement learning while using fewer environment interactions, "
    "transfers across substantial changes in environment dynamics, and scales to complex "
    "locomotion tasks."
)

BIBTEX = """@article{ahmed2026codetocontrol,
  title  = {Code to Control: Synthesizing Parameterized Reactive Controllers},
  author = {Ahmed, Zergham and Tenenbaum, Joshua B. and Bates, Chris and Gershman, Samuel J.},
  year   = {2026}
}"""


def code_block(src, lang="python"):
    return f'<pre><code class="language-{lang}">{html.escape(src.rstrip())}</code></pre>'


def video(path, cls):
    """path is relative to static/videos and static/images/posters, without the extension."""
    return (f'<video autoplay muted loop playsinline class="{cls}" '
            f'poster="./static/images/posters/{path}.png">'
            f'<source src="./static/videos/{path}.mp4" type="video/mp4"></video>')


def num(x, decimals):
    s = f"{abs(x):,.{decimals}f}"
    return f"&minus;{s}" if x < 0 else s


def load_game(stem):
    meta = json.load(open(os.path.join(CHAMP, f"{stem}.json")))
    src = open(os.path.join(CHAMP, f"{stem}.py")).read()
    score = meta["reported_score"]
    score_txt = f"{score:+.0f}".replace("-", "&minus;") if stem in ("pong", "fishingderby") \
        else f"{score:,.0f}"
    return src, f"score {score_txt}"


def load_task(stem, locomotion):
    meta = json.load(open(os.path.join(CHAMP, "mujoco", f"{stem}.json")))
    src = open(os.path.join(CHAMP, "mujoco", f"{stem}.py")).read()
    r = meta["reported_return"]
    score = f"return {num(r, 0 if abs(r) >= 10000 else 2)}"
    if locomotion:
        score += f" &middot; {num(meta['displacement_m'], 2)}&nbsp;m forward"
    note = None
    if meta["episode_steps"] < meta["cap"] / 2:
        secs = meta["episode_steps"] * meta["dt"]
        note = f"The median episode ends after {secs:.1f}&nbsp;s, when the body falls."
    return meta, src, score, note


def headline():
    """The strongest results, one clip each. Every number is read from champions/."""
    best = json.load(open(os.path.join(CHAMP, "pong_best.json")))
    si = json.load(open(os.path.join(CHAMP, "spaceinvaders.json")))
    fl = json.load(open(os.path.join(CHAMP, "flappy.json")))
    hc = json.load(open(os.path.join(CHAMP, "mujoco", "halfcheetah.json")))
    won = int(best["reported_score"])
    items = [
        ("pong_best", 480, 630, "Pong", f"21&ndash;{21 - won}", "",
         "code", "pong_best"),
        ("spaceinvaders", 480, 630, "Space Invaders", f"{si['reported_score']:,.0f}", "points",
         "code", "spaceinvaders"),
        ("flappy", 420, 600, "Flappy Bird", f"{int(fl['rails']['--max-ticks']):,} steps",
         "without a crash", "code", "flappy"),
        ("mujoco/halfcheetah", 480, 360, "HalfCheetah", f"{hc['displacement_m']:.2f}&nbsp;m",
         f"at {hc['displacement_m'] / (hc['episode_steps'] * hc['dt']):.2f}&nbsp;m/s",
         "code-mujoco", "halfcheetah"),
    ]
    out = []
    for clip, w, h, name, big, small, viewer, stem in items:
        out.append(f"""
        <div class="headline-item">
          <p class="headline-name">{name}</p>
          <video autoplay muted loop playsinline class="headline-video" width="{w}" height="{h}"
                 poster="./static/images/posters/{clip}.png"><source src="./static/videos/{clip}.mp4" type="video/mp4"></video>
          <a class="read-code headline-score" href="#{viewer}" data-game="{stem}">{big}</a>
          {f'<p class="headline-small">{small}</p>' if small else ""}
        </div>""")
    return "".join(out)


def card(clip, name, score, note, link_text, viewer, stem):
    extra = f'<p class="is-size-7 has-text-grey">{note}</p>' if note else ""
    return f"""
        <div class="column is-one-third-desktop is-half-tablet is-half-mobile">
          <div class="game-card">
            {video(clip, "game-video")}
            <p class="game-title">{name}</p>
            <p class="game-score">{score}</p>
            {extra}
            <a class="read-code" href="#{viewer}" data-game="{stem}">{link_text}</a>
          </div>
        </div>"""


def code_viewer(viewer, entries, lang="python"):
    """entries: (stem, name, src, header html[, html shown above the header])."""
    tabs, panes = [], []
    for i, (stem, name, src, head, *above) in enumerate(entries):
        on = i == 0
        tabs.append(f'<button class="button is-small is-rounded code-tab{" is-dark" if on else ""}" '
                    f'data-game="{stem}" aria-pressed="{str(on).lower()}">{name}</button>')
        panes.append(f"""
      <div class="code-pane{" is-active" if on else ""}" data-game="{stem}">{"".join(above)}
        <p class="code-head">{head}</p>
        {code_block(src, lang)}
      </div>""")
    return f"""
    <div id="{viewer}" class="code-viewer">
      <div class="buttons is-centered code-tabs">
        {"".join(tabs)}
      </div>{"".join(panes)}
    </div>"""


def games_section():
    cards, entries = [], []
    for stem, name, note in GAMES:
        src, score = load_game(stem)
        cards.append(card(stem, name, score, note, "View the controller", "code", stem))
        entries.append((stem, name, src,
                        f'{name} &middot; {score} &middot; '
                        f'<a href="{REPO}/blob/main/champions/{stem}.py">champions/{stem}.py</a>'))
    for stem, name in EXTRA_PROGRAMS:
        src, score = load_game(stem)
        entries.append((stem, name, src,
                        f'{name} &middot; the best of the three Pong runs &middot; {score} &middot; '
                        f'<a href="{REPO}/blob/main/champions/{stem}.py">champions/{stem}.py</a>'))
    return f"""
<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3 has-text-centered">Learned Controllers</h2>
    <div class="content has-text-centered">
      <p>Each controller is shown exactly as produced by Code to Control, with scores reported
         as the median over three runs.</p>
    </div>
    <div class="columns is-multiline is-centered is-mobile">{"".join(cards)}
    </div>{code_viewer("code", entries)}
  </div>
</section>"""


def mujoco_section():
    present = [t for t in TASKS if os.path.exists(os.path.join(CHAMP, "mujoco", f"{t[0]}.json"))]
    if not present:
        return ""
    cards, entries, terms = [], [], set()
    for stem, name, locomotion in present:
        meta, src, score, note = load_task(stem, locomotion)
        terms.add(meta["n_terms"])
        rows, cols = len(meta["weights"]), len(meta["weights"][0])
        cards.append(card(f"mujoco/{stem}", name, score, note,
                          "View the feature map", "code-mujoco", stem))
        base = f"{REPO}/blob/main/champions/mujoco/{stem}"
        entries.append((stem, name, src,
                        f'{name} &middot; {score} &middot; {meta["n_terms"]} features, '
                        f'fitted {rows}&times;{cols} matrix in '
                        f'<a href="{base}.json">{stem}.json</a> &middot; '
                        f'<a href="{base}.py">champions/mujoco/{stem}.py</a>'))
    n = f"{terms.pop()} numbers" if len(terms) == 1 else "a vector of numbers"
    return f"""
<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3 has-text-centered">Continuous Control in MuJoCo</h2>
    <div class="content has-text-centered">
      <p>Here the controller outputs torques. The model writes a Python feature map that turns
         the observation into {n}, and derivative-free search fits the matrix that maps them to
         torques.</p>
    </div>
    <div class="columns is-multiline is-centered is-mobile">{"".join(cards)}
    </div>
    <div class="content has-text-centered is-size-7 has-text-grey mujoco-note">
      <p>Returns are medians over ten test episodes.</p>
    </div>{code_viewer("code-mujoco", entries)}
  </div>
</section>"""


def transfer_data():
    """VARIANTS, EXPECTED, CAP and RUNS from scripts/flappy_transfer.py, the script that checks
    every cell of Figure 3B. They are read as literals, so the page imports no game code."""
    tree = ast.parse(open(os.path.join(ROOT, "scripts", "flappy_transfer.py")).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("VARIANTS", "EXPECTED", "CAP", "RUNS"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


QUANTITIES = {"GRAV": "gravity", "FLAP": "flap strength", "GAP": "gap", "PIPE_SPEED": "pipe speed"}


def base_physics():
    """The base game's constants, read as literals from flappy_env.py."""
    out = {}
    for node in ast.parse(open(os.path.join(ROOT, "flappy_env.py")).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t, v = node.targets[0], node.value
            pairs = zip(t.elts, v.elts) if isinstance(t, ast.Tuple) and isinstance(v, ast.Tuple) \
                else [(t, v)]
            for n, x in pairs:
                try:
                    out[n.id] = ast.literal_eval(x)
                except (ValueError, AttributeError):
                    pass
    return out


def change_detail(physics, schedule, base):
    """A change as multiples of the base game's values, e.g. 'gravity ×1.5'. Empty where a
    multiple would not read: added drag has no base value, and a flipped sign is named."""
    when, sets = "", physics or {}
    if schedule:
        when, sets = f" from step {schedule[0]['at_tick']:,}", schedule[0]["set"]
    parts = []
    for k, v in sets.items():
        r = v / base[k] if k in QUANTITIES else -1
        if r < 0:
            return ""
        parts.append(f"{QUANTITIES[k]} &times;{r:.1f}" if abs(r - round(r, 1)) < 1e-9
                     else f"{QUANTITIES[k]} &times;{r:.2f}")
    return ", ".join(parts) + when


def change_name(name, detail):
    """The page's name for a change, with the change as multiples beside it."""
    if not detail:
        return CHANGE_NAMES[name]
    return f'{CHANGE_NAMES[name]} <span class="change-detail">{detail}</span>'


def diff(old, new):
    """The whole program as a diff: unchanged lines as context, changed lines as -/+ pairs."""
    lines = difflib.unified_diff(old.rstrip().split("\n"), new.rstrip().split("\n"),
                                 lineterm="", n=10 ** 6)
    return "\n".join(line for line in lines if not line.startswith(("---", "+++", "@@")))


def transfer_clip(clip, run, name, how, cell, cap, width, pair=False):
    """One clip. In a before-and-after pair both clips share a change, and so a sky colour,
    and the title says which side of the refit each one is."""
    z, f = cell[run, name]
    steps = z if how == "unchanged" else f
    fate = ("survives the full episode" if steps == cap else
            f"crashes at step {steps:,}" if how == "unchanged" else f"survives {steps:,} steps")
    what = "Program unchanged" if how == "unchanged" else "Constants refit"
    speed = ", shown at half speed" if clip in HALF_SPEED else ""
    title = CHANGE_NAMES[name]
    if pair:
        title = "Before refitting" if how == "unchanged" else "After refitting"
    return f"""
        <div class="column {width}">
          <div class="game-card">
            {video(clip, "game-video")}
            <p class="game-title">{title}</p>
            <p class="clip-caption">{what}. It {fate}{speed}.</p>
          </div>
        </div>"""


def transfer_section():
    d = transfer_data()
    cap, runs, expected = d["CAP"], d["RUNS"], d["EXPECTED"]
    labels = {name: label for name, label, _, _ in d["VARIANTS"]}
    base = base_physics()
    details = {name: change_detail(physics, schedule, base)
               for name, _, physics, schedule in d["VARIANTS"]}
    cell = {(r, name): expected.get((r, name), (cap, None)) for r in runs for name in labels}
    refits = json.load(open(os.path.join(CHAMP, "flappy_transfer", "refits.json")))["refits"]
    for c in refits:   # the saved refits are the cells of the table
        assert cell[c["run"], c["change"]] == (c["zero_shot"], c["refit"]), c["change"]

    row1 = "".join(transfer_clip(*c, cell, cap, "is-one-quarter-tablet is-half-mobile")
                   for c in UNCHANGED_CLIPS)
    # One tab per refit that reaches the end of the episode: its clips, then its diff.
    full = [c for c in refits if c["refit"] == cap]
    one_run = len({c["run"] for c in full}) == 1
    entries = []
    for c in full:
        run, name = c["run"], c["change"]
        src = json.load(open(os.path.join(CHAMP, "flappy_transfer", f"run{run}.json")))["champion_src"]
        n = len(c["constants"])
        pair = "".join(transfer_clip(f"flappy_{name}_{side}", run, name, how, cell, cap,
                                     "is-one-quarter-tablet is-half-mobile", pair=True)
                       for side, how in (("frozen", "unchanged"), ("refit", "refit")))
        for side in ("frozen", "refit"):
            assert os.path.exists(os.path.join(ROOT, "docs", "static", "videos",
                                               f"flappy_{name}_{side}.mp4")), name
        entries.append((f"refit-{run}-{name}",
                        CHANGE_NAMES[name] if one_run else f"Run {run}: {CHANGE_NAMES[name].lower()}",
                        diff(src, c["program"]),
                        f'Run {run} &middot; {n} constant{"s" if n != 1 else ""} changed '
                        f'&middot; original: '
                        f'<a href="{REPO}/blob/main/champions/flappy_transfer/run{run}.py">'
                        f'champions/flappy_transfer/run{run}.py</a>',
                        f"""
        <div class="columns is-centered is-mobile transfer-clips">{pair}
        </div>"""))

    def steps(x):   # a tick for the full episode, so the table carries only the crashes
        return "&#10003;" if x == cap else f"{x:,}"

    rows = []
    for name, label in labels.items():
        tds = []
        for r in runs:
            z, f = cell[r, name]
            end = z if f is None else f
            txt = steps(z) if f is None else \
                f'{steps(z)} <span class="refit">&rarr;&nbsp;{steps(f)}</span>'
            tds.append(f'<td class="{"full" if end == cap else "part"}">{txt}</td>')
        rows.append(f"<tr><td>{change_name(name, details[name])}</td>{''.join(tds)}</tr>")
    heads = "".join(f'<th><a href="{REPO}/blob/main/champions/flappy_transfer/run{r}.py">Run {r}</a></th>'
                    for r in runs)
    return f"""
<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3 has-text-centered">Transfer and Parameter Refitting</h2>
    <div class="content has-text-centered">
      <p>After a controller is learned, we change Flappy Bird's gravity and other dynamics. The
         controller adapts in one of two ways. Often the program is robust enough to survive as
         written. Otherwise our method refits only its numeric constants, with no
         language-model calls. Reversed gravity is the exception since it needs a new program.</p>
    </div>
    <h3 class="title is-5 has-text-centered transfer-sub">Replayed unchanged</h3>
    <div class="columns is-centered is-mobile is-multiline transfer-clips">{row1}
    </div>
    <h3 class="title is-5 has-text-centered transfer-sub">Refitting the constants</h3>
    <div class="content has-text-centered">
      <p>Refitting adapts the program's constants, which lets it survive the full episode
         again. Each tab plays the same stretch of the episode with the original constants and
         with the refit ones. The code below shows that only numbers changed.</p>
    </div>{code_viewer("code-refit", entries, lang="diff")}
    <div class="table-container transfer-table">
      <table class="table is-narrow is-fullwidth">
        <thead><tr><th>Physics change</th>{heads}</tr></thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </div>
    <div class="content has-text-centered is-size-7 has-text-grey mujoco-note">
      <p>Steps survived, before &rarr; after numerical parameter refitting. &#10003; is the
         full {cap:,}-step episode.</p>
    </div>
  </div>
</section>"""


def build():
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{DESCRIPTION}">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Code to Control: Synthesizing Parameterized Reactive Controllers">
  <meta property="og:description" content="{DESCRIPTION}">
  <meta property="og:url" content="{SITE}/">
  <meta property="og:image" content="{SITE}/static/images/og.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="Code to Control: Synthesizing Parameterized Reactive Controllers">
  <meta name="twitter:description" content="{DESCRIPTION}">
  <meta name="twitter:image" content="{SITE}/static/images/og.png">
  <title>Code to Control</title>
  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro" rel="stylesheet">
  <link rel="stylesheet" href="./static/css/bulma.min.css">
  <link rel="stylesheet" href="./static/css/fontawesome.all.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="./static/css/index.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <link rel="icon" href="./static/favicon.svg">
  <script defer src="./static/js/fontawesome.all.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <script>
    function showCode(viewer, game) {{
      const root = document.getElementById(viewer);
      root.querySelectorAll('.code-tab').forEach(b => {{
        const on = b.dataset.game === game;
        b.classList.toggle('is-dark', on);
        b.setAttribute('aria-pressed', on);
      }});
      root.querySelectorAll('.code-pane').forEach(p =>
        p.classList.toggle('is-active', p.dataset.game === game));
    }}
    function play(v) {{
      const p = v.play();
      if (p) p.catch(() => {{}});
    }}
    document.addEventListener('DOMContentLoaded', () => {{
      if (window.hljs) hljs.highlightAll();
      document.querySelectorAll('.code-viewer').forEach(root =>
        root.querySelectorAll('.code-tab').forEach(b =>
          b.addEventListener('click', () => showCode(root.id, b.dataset.game))));
      document.querySelectorAll('.read-code').forEach(a =>
        a.addEventListener('click', () => showCode(a.getAttribute('href').slice(1), a.dataset.game)));
      // Play clips while they are on screen and pause them otherwise. A tap pauses or resumes,
      // which also starts playback where the browser blocks autoplay (e.g. iOS Low Power Mode).
      const videos = document.querySelectorAll('video');
      const seen = 'IntersectionObserver' in window ? new IntersectionObserver(es => es.forEach(e => {{
        if (!e.isIntersecting) e.target.pause();
        else if (!e.target.dataset.held) play(e.target);
      }}), {{ rootMargin: '100px' }}) : null;
      videos.forEach(v => {{
        v.muted = true;
        if (seen) seen.observe(v); else play(v);
        v.addEventListener('click', () => {{
          if (v.paused) {{ delete v.dataset.held; play(v); }}
          else {{ v.dataset.held = '1'; v.pause(); }}
        }});
      }});
    }});
  </script>
  <style>
    pre {{ font-size: 0.78rem; line-height: 1.45; border-radius: 6px; padding: 0 !important;
           background: #f6f8fa !important; }}
    pre code.hljs {{ padding: 0.9rem 1rem; background: #f6f8fa; }}
    video {{ cursor: pointer; }}
    .teaser .hero-body {{ padding-top: 1.5rem; padding-bottom: 1.5rem; }}
    .headline-strip {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 1.25rem; }}
    .headline-item {{ display: flex; flex-direction: column; align-items: center; }}
    .headline-name {{ font-family: monospace; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.35rem; }}
    .headline-video {{ height: 230px; width: auto; border-radius: 6px; image-rendering: pixelated; }}
    .headline-score {{ font-family: monospace; font-weight: 700; font-size: 1.15rem; margin-top: 0.35rem; }}
    /* Bulma's .hero.is-light makes its links inherit grey text; keep these blue. */
    .teaser a.headline-score, .teaser .teaser-caption a {{ color: #3273dc !important; }}
    .headline-small {{ font-family: monospace; color: #6b7280; font-size: 0.8rem; max-width: 180px; }}
    .teaser-caption {{ max-width: 640px; margin: 1.25rem auto 0 !important; }}
    .teaser-caption p + p {{ margin-top: 0.75rem; }}
    .game-card {{ text-align: center; margin-bottom: 1.5rem; }}
    .game-video {{ width: 100%; max-width: 280px; border-radius: 6px; image-rendering: pixelated; }}
    .game-title {{ font-weight: 600; font-size: 1.1rem; margin-top: 0.3rem; }}
    .game-score {{ color: #6b7280; }}
    .mujoco-note {{ max-width: 640px; margin: 0 auto 0.5rem; }}
    .clip-caption {{ color: #6b7280; font-size: 0.85rem; max-width: 240px; margin: 0 auto; }}
    .transfer-table {{ max-width: 720px; margin: 2rem auto 0.75rem; }}
    .transfer-table th, .transfer-table td {{ text-align: center !important;
      font-variant-numeric: tabular-nums; vertical-align: middle; }}
    .transfer-table th:first-child, .transfer-table td:first-child {{ text-align: left !important; }}
    .transfer-table td.full {{ background: #e9f6ec; }}
    .transfer-table td.part {{ background: #fdf3e1; }}
    .transfer-table .refit {{ white-space: nowrap; font-weight: 600; }}
    .transfer-sub {{ margin: 1.75rem 0 0.9rem !important; }}
    .change-detail {{ color: #6b7280; font-size: 0.85em; margin-left: 0.3rem; }}
    .transfer-table .change-detail {{ display: block; margin-left: 0; }}
    html {{ scroll-behavior: smooth; }}
    .code-viewer {{ scroll-margin-top: 1rem; margin-top: 1rem; }}
    .code-pane {{ display: none; }}
    .code-pane.is-active {{ display: block; }}
    .code-pane pre {{ max-height: 36rem; overflow: auto; }}
    .code-head {{ color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem; text-align: center; }}
    .figure {{ width: 100%; max-width: 760px; }}
    .figure-small {{ width: 100%; max-width: 460px; }}
    .title-hero .hero-body {{ padding-bottom: 1.5rem; }}
    .affiliations {{ margin-top: 0.3rem; color: #4a4a4a; }}
    .affiliations .author-block {{ margin: 0 0.5rem; }}
    @media screen and (max-width: 768px) {{
      .title.is-1.publication-title {{ font-size: 2rem; }}
      .publication-authors {{ font-size: 1.1rem !important; }}
      .publication-authors.affiliations {{ font-size: 0.9rem !important; }}
      .teaser .subtitle {{ font-size: 1.05rem; }}
      .headline-strip {{ gap: 0.75rem; }}
      .headline-item {{ width: calc(50% - 0.5rem); }}
      .headline-video {{ height: auto; width: 100%; max-height: 190px; object-fit: contain; }}
      .game-title {{ font-size: 1rem; }}
      .game-score {{ font-size: 0.85rem; }}
      .clip-caption {{ font-size: 0.75rem; }}
      .transfer-table {{ font-size: 0.8rem; }}
    }}
  </style>
  <noscript><style>.code-pane {{ display: block; }}</style></noscript>
</head>
<body>

<section class="hero title-hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-1 publication-title">Code to Control: Synthesizing Parameterized Reactive Controllers</h1>
          <div class="is-size-5 publication-authors">
            <span class="author-block">Zergham Ahmed<sup>1</sup>,</span>
            <span class="author-block">Joshua B. Tenenbaum<sup>2</sup>,</span>
            <span class="author-block">Chris Bates<sup>1,3</sup>,</span>
            <span class="author-block">Samuel J. Gershman<sup>1</sup></span>
          </div>
          <div class="is-size-6 publication-authors affiliations">
            <span class="author-block"><sup>1</sup>Harvard University</span>
            <span class="author-block"><sup>2</sup>Massachusetts Institute of Technology</span>
            <span class="author-block"><sup>3</sup>Florida Institute for Human and Machine Cognition</span>
          </div>
          <div class="column has-text-centered">
            <div class="publication-links">
              <!-- Uncomment once the arXiv ID exists.
              <span class="link-block">
                <a href="https://arxiv.org/abs/XXXX.XXXXX" class="external-link button is-normal is-rounded is-dark">
                  <span class="icon"><i class="ai ai-arxiv"></i></span><span>arXiv</span>
                </a>
              </span>
              -->
              <span class="link-block">
                <a href="{REPO}" class="external-link button is-normal is-rounded is-dark">
                  <span class="icon"><i class="fab fa-github"></i></span><span>Code</span>
                </a>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="hero is-light is-small teaser">
  <div class="hero-body">
    <div class="container is-max-desktop has-text-centered">
      <div class="headline-strip">{headline()}
      </div>
      <div class="subtitle teaser-caption">
        <p>Code to Control learns executable controllers directly from interaction. A language
           model writes each controller's structure, and derivative-free search fits its
           numerical parameters when useful. The learned controllers act directly as policies
           across Atari, Flappy Bird, and MuJoCo.</p>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified"><p>{ABSTRACT}</p></div>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Code to Control Overview</h2>
        <img src="./static/images/concept.png" class="figure" alt="Code to Control schema">
        <div class="content has-text-justified" style="margin-top:1rem">
          <p>Our method writes a controller, plays episodes, and uses the trajectories to revise
             the program. Its numerical parameters can be fit separately from environment
             feedback when useful, especially in continuous control. There the program is a
             feature map &phi;(o,&nbsp;t), and search fits the matrix W that maps features to
             actuator commands.</p>
        </div>
      </div>
    </div>
  </div>
</section>
{games_section()}
{transfer_section()}
{mujoco_section()}

<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Enabling Real-Time Gameplay</h2>
        <div class="content">
          <p>Code to Control selects an action in 11.4&nbsp;&micro;s, faster than PPO at
             76.5&nbsp;&micro;s. ReAct takes 1.29&nbsp;s per action.</p>
        </div>
        <img src="./static/images/latency.png" class="figure-small" alt="Median decision latency on a log scale">
        <p class="is-size-7 has-text-grey">Median decision latency, log scale.
          &dagger;WorldCoder: planning time per action.</p>
      </div>
    </div>
  </div>
</section>





<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>{html.escape(BIBTEX)}</code></pre>
  </div>
</section>

<footer class="footer">
  <div class="container">
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content has-text-centered">
          <p>This page is adapted from the
            <a href="https://github.com/nerfies/nerfies.github.io">Nerfies project page</a>,
            licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative
            Commons Attribution-ShareAlike 4.0 International License</a>.</p>
        </div>
      </div>
    </div>
  </div>
</footer>

</body>
</html>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(page)
    print(f"wrote {os.path.relpath(OUT, ROOT)}  ({len(page):,} bytes)")


if __name__ == "__main__":
    build()
