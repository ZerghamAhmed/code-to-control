import math

def terms(state, t):
    def g(k, d=0.0):
        try:
            v = state.get(k, d)
            if isinstance(v, list):
                v = v[0] if v else d
            v = float(v)
            if not math.isfinite(v):
                return d
            return v
        except Exception:
            return d

    def clip(x, lo=-5.0, hi=5.0):
        if not math.isfinite(x):
            return 0.0
        return lo if x < lo else (hi if x > hi else x)

    # raw pose fields (normalised-ish)
    root_z = g("root_z", 1.0)
    abdomen_y = g("abdomen_y")
    r_knee = g("right_knee")
    l_knee = g("left_knee")
    r_hip_y = g("right_hip_y")
    l_hip_y = g("left_hip_y")

    # velocities
    rvz = g("root_velocity_z")
    ravy = g("root_angular_velocity_y")
    aby_av = g("abdomen_y_angular_velocity")
    r_knee_av = g("right_knee_angular_velocity")
    l_knee_av = g("left_knee_angular_velocity")

    out = []
    # 1: constant bias
    out.append(1.0)
    # 2: upright height error (saturating)
    out.append(clip(root_z - 1.2))
    # 3: torso pitch
    out.append(clip(abdomen_y))
    # 4: knee difference (asymmetry)
    out.append(clip(0.5 * (r_knee - l_knee)))
    # 5: hip_y sum (both legs forward/back)
    out.append(clip(0.3 * (r_hip_y + l_hip_y)))
    # 6: vertical velocity (falling), clipped
    out.append(clip(0.3 * rvz))
    # 7: angular velocity y (tumbling), tanh-saturated
    out.append(math.tanh(0.2 * ravy))
    # 8: abdomen angular velocity, tanh
    out.append(math.tanh(0.1 * aby_av))
    # 9: knee angular velocity sum
    out.append(math.tanh(0.1 * (r_knee_av + l_knee_av)))
    # 10: periodic term (gait clock)
    out.append(math.sin(0.3 * t))
    # 11: periodic term, different freq / phase
    out.append(math.cos(0.15 * t))
    # 12: product of pitch and vertical velocity (coupling)
    out.append(clip(math.tanh(abdomen_y) * math.tanh(0.3 * rvz)))
    # 13: quadratic height penalty (saturating bowl)
    dz = root_z - 1.2
    out.append(clip(-(dz * dz)))
    # 14: slow saturating growth toward 1
    out.append(math.tanh(0.01 * t))

    return [float(x) if math.isfinite(x) else 0.0 for x in out]
