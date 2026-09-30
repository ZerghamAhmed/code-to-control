import math

def terms(state, t):
    def g(k):
        try:
            v = state[k][0]
            if v is None:
                return 0.0
            v = float(v)
            if not math.isfinite(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    def clip(x, lo=-3.0, hi=3.0):
        if x < lo: return lo
        if x > hi: return hi
        return x

    h    = g('torso_height')
    ta   = g('torso_angle')
    tha  = g('thigh_angle')
    la   = g('leg_angle')
    fa   = g('foot_angle')
    vx   = g('torso_velocity_x')
    vz   = g('torso_velocity_z')
    tav  = g('torso_angular_velocity')
    thav = g('thigh_angular_velocity')
    lav  = g('leg_angular_velocity')
    fav  = g('foot_angular_velocity')

    out = []
    # 0: constant bias
    out.append(1.0)
    # 1: torso height deviation from upright (~1.25 nominal)
    out.append(clip(h - 1.25))
    # 2: torso angle (posture)
    out.append(clip(ta))
    # 3: thigh angle
    out.append(clip(tha))
    # 4: leg angle
    out.append(clip(la))
    # 5: foot angle
    out.append(clip(fa))
    # 6: torso angular velocity (normalised)
    out.append(clip(tav / 4.0))
    # 7: thigh angular velocity (normalised)
    out.append(clip(thav / 4.0))
    # 8: leg + foot angular velocity combo (normalised)
    out.append(clip((lav + fav) / 4.0))
    # 9: horizontal velocity
    out.append(clip(vx))
    # 10: product of torso angle and angular velocity (damping-ish)
    out.append(clip(ta * tav))
    # 11: periodic term
    out.append(math.sin(2.0 * math.pi * t / 20.0))
    # 12: second periodic term, different freq + phase
    out.append(math.cos(2.0 * math.pi * t / 7.0 + 0.5))
    # 13: saturating nonlinearity of posture (angle sum)
    out.append(math.tanh(ta + tha + la))

    return [float(x) if math.isfinite(x) else 0.0 for x in out]
