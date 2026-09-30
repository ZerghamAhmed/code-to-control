import math

def terms(state, t):
    def g(k):
        try:
            v = state[k][0]
        except Exception:
            return 0.0
        if v is None:
            return 0.0
        try:
            f = float(v)
        except Exception:
            return 0.0
        if not math.isfinite(f):
            return 0.0
        return f

    # raw positions
    rootz = g('rootz')
    rooty = g('rooty')
    bthigh = g('bthigh')
    bshin = g('bshin')
    bfoot = g('bfoot')
    fthigh = g('fthigh')
    fshin = g('fshin')
    ffoot = g('ffoot')

    # velocities
    vx = g('rootx_velocity')
    vz = g('rootz_velocity')
    wy = g('rooty_angular_velocity')
    wbt = g('bthigh_angular_velocity')
    wbs = g('bshin_angular_velocity')
    wbf = g('bfoot_angular_velocity')
    wft = g('fthigh_angular_velocity')
    wfs = g('fshin_angular_velocity')
    wff = g('ffoot_angular_velocity')

    def sat(x, s=1.0):
        return math.tanh(x / s)

    out = []
    # 0: constant bias
    out.append(1.0)
    # 1: root height (posture)
    out.append(sat(rootz, 0.2))
    # 2: root pitch angle
    out.append(sat(rooty, 0.3))
    # 3: sum of back-leg joint angles
    out.append(sat(bthigh + bshin + bfoot, 1.0))
    # 4: sum of front-leg joint angles
    out.append(sat(fthigh + fshin + ffoot, 1.0))
    # 5: forward velocity (drives reward)
    out.append(sat(vx, 1.0))
    # 6: vertical velocity
    out.append(sat(vz, 1.0))
    # 7: pitch angular velocity
    out.append(sat(wy, 2.0))
    # 8: combined back-leg angular velocities
    out.append(sat(wbt + wbs + wbf, 10.0))
    # 9: combined front-leg angular velocities
    out.append(sat(wft + wfs + wff, 10.0))
    # 10: coupling term (posture x forward velocity)
    out.append(sat(rooty * vx, 0.5))
    # 11: periodic gait clock
    out.append(math.sin(0.3 * t))
    # 12: periodic gait clock, second frequency / phase
    out.append(math.cos(0.15 * t))
    # 13: front-back leg antisymmetry (differential drive)
    out.append(sat((fthigh - bthigh) + 0.5 * (fshin - bshin), 1.0))

    return [x if math.isfinite(x) else 0.0 for x in out]
