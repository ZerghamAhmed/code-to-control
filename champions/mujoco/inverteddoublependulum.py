import math

def terms(state, t):
    def g(k):
        try:
            v = state[k][0]
            if v is None or not math.isfinite(v):
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    cx = g('cart_x')
    s1 = g('sin_pole1_angle')
    s2 = g('sin_pole2_angle')
    c1 = g('cos_pole1_angle')
    c2 = g('cos_pole2_angle')
    cv = g('cart_velocity')
    w1 = g('pole1_angular_velocity')
    w2 = g('pole2_angular_velocity')

    def clip(x, lo=-3.0, hi=3.0):
        return lo if x < lo else (hi if x > hi else x)

    out = [
        1.0,                              # constant bias
        clip(cx),                         # cart position
        cv,                               # cart velocity
        s1,                               # pole1 angle (sin)
        s2,                               # pole2 angle (sin)
        c1 - c2,                          # cos difference
        clip(w1 / 5.0),                   # pole1 angular velocity (scaled)
        clip(w2 / 5.0),                   # pole2 angular velocity (scaled)
        s1 * c1,                          # nonlinear coupling pole1
        s2 * c2,                          # nonlinear coupling pole2
        s1 - s2,                          # inter-pole angle difference
        clip((w1 - w2) / 5.0),            # relative angular rate
        math.tanh(cv + clip(w1 / 5.0)),   # saturating combined velocity
        math.sin(0.3 * t),                # periodic term
    ]
    return [float(x) if math.isfinite(x) else 0.0 for x in out]
