"""Standardized OBJECT API for synthesized theories (the transfer substrate).

A read-only object view over the canonical state dict
(``entity -> [[x, y]]``, ``entity_size -> [[w, h]]``, ``entity_velocity ->
[[dx, dy]]``). Synthesized predicates, operators, and world models are written
against THIS interface, so they transfer across games unchanged.

Core attributes follow PoE-World's Obj (category, x/y, w/h, dx/dy, touches)
for OCAtari compatibility; `.past()` / `.approaching()` add the operator-level
vocabulary (subgoal completion / threat anticipation).

The dict remains ground truth: Obj is a VIEW for reading geometry — a
transition_model must still mutate the state dict itself.

API_DOC below is injected into the synthesis prompt verbatim.
"""

API_DOC = '''
OBJECT API (provided — call `objects(state)`; do NOT reimplement):
    objs = objects(state)            # ObjList over every entity in the state
    o = objs.get("player")           # first object of a category (None if absent)
    cars = objs.all("car")           # list of all objects of a category
    for o in objs: ...               # iterate every object

    Obj attributes (read-only view; mutate the STATE DICT, not the Obj):
      o.category  o.x  o.y  (center, px)   o.w  o.h   o.dx  o.dy  (per-tick velocity)
      o.left_side  o.right_side  o.top_side  o.bottom_side   (box edges)
    Obj methods:
      o.touches(other, margin=0)   -> True if the two boxes overlap (contact!)
      o.past(other)                -> True if `other` is FULLY behind o along
                                      other's own travel direction (disengaged —
                                      use in commit-operator goals: a hazard is
                                      only "passed" when past() holds, NOT at the
                                      scoring instant)
      o.approaching(other)         -> True if other's velocity is carrying it
                                      toward o
      o.distance_to(other)         -> Euclidean center distance
'''


# Neutral variant, selected by `--neutral-objapi`. Same functions and signatures;
# describes only what each one COMPUTES. The default API_DOC above additionally tells
# the model how to USE them ("disengaged", "commit-operator goals", "NOT at the
# scoring instant") — which restates the prompt's COMPLETION SAFETY rule inside the
# API reference, and so leaks past any prompt variant that removes that rule.
API_DOC_NEUTRAL = '''
OBJECT API (provided — call `objects(state)`; do NOT reimplement):
    objs = objects(state)            # ObjList over every entity in the state
    o = objs.get("player")           # first object of a category (None if absent)
    cars = objs.all("car")           # list of all objects of a category
    for o in objs: ...               # iterate every object

    Obj attributes (read-only view; mutate the STATE DICT, not the Obj):
      o.category  o.x  o.y  (center, px)   o.w  o.h   o.dx  o.dy  (per-tick velocity)
      o.left_side  o.right_side  o.top_side  o.bottom_side   (box edges)
    Obj methods:
      o.touches(other, margin=0)   -> True if the two boxes overlap
      o.past(other)                -> True if `other` is fully behind o along
                                      other's own travel direction
      o.approaching(other)         -> True if other's velocity is carrying it
                                      toward o
      o.distance_to(other)         -> Euclidean center distance
'''


class Obj:
    __slots__ = ("category", "index", "x", "y", "w", "h", "dx", "dy")

    def __init__(self, category, index, x, y, w=0.0, h=0.0, dx=0.0, dy=0.0):
        self.category = category
        self.index = index
        self.x, self.y = float(x), float(y)
        self.w, self.h = float(w), float(h)
        self.dx, self.dy = float(dx), float(dy)

    # ---- box edges (PoE-World naming) ----
    @property
    def left_side(self):
        return self.x - self.w / 2.0

    @property
    def right_side(self):
        return self.x + self.w / 2.0

    @property
    def top_side(self):
        return self.y - self.h / 2.0

    @property
    def bottom_side(self):
        return self.y + self.h / 2.0

    # ---- contact / relational vocabulary ----
    def touches(self, other, margin=0.0):
        """AABB overlap — the universal contact primitive."""
        return (abs(self.x - other.x) < (self.w + other.w) / 2.0 + margin and
                abs(self.y - other.y) < (self.h + other.h) / 2.0 + margin)

    def past(self, other):
        """`other` is FULLY disengaged from self along other's travel direction.
        If `other` isn't moving horizontally, falls back to vertical, then to
        simple non-overlap."""
        if other.dx > 0:
            return other.left_side > self.right_side
        if other.dx < 0:
            return other.right_side < self.left_side
        if other.dy > 0:
            return other.top_side > self.bottom_side
        if other.dy < 0:
            return other.bottom_side < self.top_side
        return not self.touches(other)

    def approaching(self, other):
        """`other`'s velocity is carrying it toward self."""
        return ((other.x - self.x) * other.dx + (other.y - self.y) * other.dy) < 0

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __repr__(self):
        return (f"Obj({self.category}[{self.index}] x={self.x:.1f} y={self.y:.1f} "
                f"w={self.w:.0f} h={self.h:.0f} dx={self.dx:.1f} dy={self.dy:.1f})")


class ObjList:
    def __init__(self, objs):
        self._objs = list(objs)

    def get(self, category):
        for o in self._objs:
            if o.category == category:
                return o
        return None

    def all(self, category):
        return [o for o in self._objs if o.category == category]

    def nearest(self, to, category=None):
        cands = self.all(category) if category else [o for o in self._objs if o is not to]
        cands = [o for o in cands if o is not to]
        return min(cands, key=lambda o: to.distance_to(o), default=None)

    def __iter__(self):
        return iter(self._objs)

    def __len__(self):
        return len(self._objs)


_META_KEYS = {"score", "won", "lost"}


def objects(state):
    """Build an ObjList view over every coordinate entity in the state dict."""
    out = []
    for key, val in state.items():
        k = str(key)
        if k.startswith("_") or k in _META_KEYS or k.endswith("_size") \
                or k.endswith("_velocity"):
            continue
        if not (isinstance(val, list) and val and isinstance(val[0], (list, tuple))
                and len(val[0]) >= 2):
            continue
        sizes = state.get(k + "_size", [])
        vels = state.get(k + "_velocity", [])
        for i, pos in enumerate(val):
            # POLYGON entities are skipped, not fatal. Most envs report a position as
            # [x, y] scalars, but VirtualTools reports tools and obstacle outlines as
            # VERTEX LISTS ([[x0,y0], [x1,y1], ...]), so pos[0] is itself a list and
            # float() raised TypeError here — which crashed the run outright, from
            # inside `format_trajectory_table`, AFTER the episode had already been
            # played and its GIF written. That is why virtualtools has zero runs on
            # disk despite being wired up for a long time.
            #
            # This view is a CONVENIENCE over the state dict, not the state itself:
            # the agent still receives every polygon field verbatim in `get_obs()`.
            # So the correct behaviour for an entity this view cannot express is to
            # omit it, exactly as it already omits non-coordinate entries above —
            # never to take down a run that has otherwise completed.
            if not (isinstance(pos, (list, tuple)) and len(pos) >= 2
                    and isinstance(pos[0], (int, float))
                    and isinstance(pos[1], (int, float))):
                continue
            w, h = (sizes[i] if i < len(sizes) else (0.0, 0.0))[:2]
            dx, dy = (vels[i] if i < len(vels) else (0.0, 0.0))[:2]
            if not (isinstance(w, (int, float)) and isinstance(h, (int, float))):
                w, h = 0.0, 0.0
            if not (isinstance(dx, (int, float)) and isinstance(dy, (int, float))):
                dx, dy = 0.0, 0.0
            out.append(Obj(k, i, pos[0], pos[1], w, h, dx, dy))
    return ObjList(out)


#: names injected into every synthesized module's namespace
NAMESPACE = {"objects": objects, "Obj": Obj, "ObjList": ObjList}
