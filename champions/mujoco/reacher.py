import math

def terms(state, t):
    def g(k, d=0.0):
        try:
            v = state[k][0]
            if v is None or not math.isfinite(v):
                return d
            return float(v)
        except Exception:
            return d

    c0 = g('cos_joint0_angle', 1.0)
    c1 = g('cos_joint1_angle', 1.0)
    s0 = g('sin_joint0_angle', 0.0)
    s1 = g('sin_joint1_angle', 0.0)
    w0 = g('joint0_angular_velocity', 0.0)
    w1 = g('joint1_angular_velocity', 0.0)
    ex = g('fingertip_to_target_x', 0.0)
    ey = g('fingertip_to_target_y', 0.0)

    # normalise velocities to O(1)
    nw0 = math.tanh(w0 / 10.0)
    nw1 = math.tanh(w1 / 10.0)

    out = [
        1.0,                          # constant bias
        c0,                           # raw cos joint0
        c1,                           # raw cos joint1
        s0,                           # raw sin joint0
        s1,                           # raw sin joint1
        nw0,                          # joint0 velocity (normalised)
        nw1,                          # joint1 velocity (normalised)
        ex * 10.0,                    # fingertip error x (scaled)
        ey * 10.0,                    # fingertip error y (scaled)
        c0 * s1 - c1 * s0,            # angle difference sin term (coupling)
        s0 * ex * 10.0 + c0 * ey * 10.0,  # error projected via joint0
        nw0 * nw1,                    # velocity product coupling
        math.sin(0.3 * t),            # periodic term freq A
        math.cos(0.11 * t),           # periodic term freq B (different freq/phase)
    ]

    return [float(v) if math.isfinite(v) else 0.0 for v in out]
