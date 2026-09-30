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

    x = g('cart_x')
    a = g('pole_angle')
    xv = g('cart_velocity')
    av = g('pole_angular_velocity')

    # clip helper for saturation
    def sat(v):
        return math.tanh(v)

    out = [
        1.0,                      # constant bias
        x,                        # cart position
        a,                        # pole angle
        xv,                       # cart velocity
        av,                       # pole angular velocity
        sat(a * 3.0),             # saturated angle
        a * av,                   # angle * angular velocity (product)
        x * xv,                   # position * velocity (product)
        a + 0.5 * av,             # angle + rate combo (PD-like)
        x + 0.5 * xv,             # position + velocity combo
        math.sin(0.3 * t),        # periodic term 1
        math.cos(0.11 * t),       # periodic term 2, different freq
        sat(xv),                  # saturated velocity
        a * a * (1.0 if a >= 0 else -1.0),  # signed quadratic angle (growth)
    ]
    return [o if math.isfinite(o) else 0.0 for o in out]
