import math

def terms(state, t):
    def g(k):
        try:
            v = state[k][0]
        except Exception:
            return 0.0
        if v is None or not math.isfinite(v):
            return 0.0
        return float(v)

    def clip(x, lo=-1.0, hi=1.0):
        return lo if x < lo else (hi if x > hi else x)

    # Raw postural fields
    th = g('torso_height')
    ta = g('torso_angle')
    rt = g('right_thigh_angle')
    lt = g('left_thigh_angle')
    rl = g('right_leg_angle')
    ll = g('left_leg_angle')

    # Velocities (scaled to O(1); angular velocities saturate at +-10)
    tvx = g('torso_velocity_x')
    tvz = g('torso_velocity_z')
    tav = g('torso_angular_velocity') / 10.0
    rtav = g('right_thigh_angular_velocity') / 10.0
    ltav = g('left_thigh_angular_velocity') / 10.0

    out = []
    # 0: constant bias
    out.append(1.0)
    # 1: torso upright error (height offset from ~1.25 nominal)
    out.append(clip(th - 1.25, -2.0, 2.0))
    # 2: torso lean angle
    out.append(clip(ta, -3.0, 3.0))
    # 3: torso angular velocity (normalised)
    out.append(clip(tav, -1.5, 1.5))
    # 4: left-right thigh difference (drives symmetric gait)
    out.append(clip(rt - lt, -3.0, 3.0))
    # 5: left-right leg difference
    out.append(clip(rl - ll, -3.0, 3.0))
    # 6: forward velocity
    out.append(clip(tvx, -3.0, 3.0))
    # 7: vertical velocity
    out.append(clip(tvz, -3.0, 3.0))
    # 8: periodic drive (primary gait frequency)
    out.append(math.sin(2.0 * math.pi * t / 20.0))
    # 9: periodic drive, phase-shifted (quadrature)
    out.append(math.cos(2.0 * math.pi * t / 20.0))
    # 10: periodic drive, slower frequency
    out.append(math.sin(2.0 * math.pi * t / 47.0))
    # 11: product of lean and angular velocity (restoring/damping coupling)
    out.append(clip(ta * tav, -3.0, 3.0))
    # 12: thigh angular-velocity asymmetry
    out.append(clip(rtav - ltav, -2.0, 2.0))
    # 13: saturating height-times-lean (fall indicator via tanh-like)
    out.append(math.tanh(ta * (1.25 - th)))

    # Final safety pass
    safe = []
    for v in out:
        if not math.isfinite(v):
            v = 0.0
        safe.append(float(v))
    return safe
