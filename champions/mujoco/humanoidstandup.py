import math

def terms(state, t):
    def g(k, d=0.0):
        try:
            v = state.get(k, d)
            if isinstance(v, list):
                v = v[0]
            v = float(v)
            if not math.isfinite(v):
                return d
            return v
        except Exception:
            return d

    def clip(x, lo=-5.0, hi=5.0):
        if x < lo: return lo
        if x > hi: return hi
        return x

    def sat(x):
        # smooth saturating nonlinearity, O(1)
        return math.tanh(x)

    # raw postural state
    rz = g('root_z')
    ab_y = g('abdomen_y')
    ab_x = g('abdomen_x')
    rhy = g('right_hip_y')
    lhy = g('left_hip_y')
    rk = g('right_knee')
    lk = g('left_knee')
    qw = g('root_quat_w', 1.0)
    qx = g('root_quat_x')
    qz = g('root_quat_z')

    # velocities
    rvx = g('root_velocity_x')
    rvz = g('root_velocity_z')
    ravx = g('root_angular_velocity_x')
    ravy = g('root_angular_velocity_y')
    abyv = g('abdomen_y_angular_velocity')
    rhyv = g('right_hip_y_angular_velocity')
    lhyv = g('left_hip_y_angular_velocity')

    out = []

    # 1. constant bias
    out.append(1.0)

    # 2. root height deviation from nominal (~0.1) -- upright drive
    out.append(clip((rz - 0.1) * 10.0))

    # 3. torso pitch (abdomen_y) -- posture
    out.append(clip(ab_y))

    # 4. torso roll (abdomen_x)
    out.append(clip(ab_x))

    # 5. difference of hips (gait antisymmetry)
    out.append(clip(rhy - lhy))

    # 6. sum of knees (crouch level), saturated
    out.append(sat(rk + lk))

    # 7. forward velocity, saturated
    out.append(sat(rvx))

    # 8. vertical velocity
    out.append(clip(rvz))

    # 9. root angular velocity (roll+pitch) mixed
    out.append(clip((ravx + ravy) * 0.3))

    # 10. abdomen pitch rate damping term
    out.append(clip(abyv * 0.2))

    # 11. hip velocity antisymmetry (gait phase feedback)
    out.append(clip((rhyv - lhyv) * 0.2))

    # 12. periodic term (walking clock)
    out.append(math.sin(0.3 * t))

    # 13. second periodic term, different freq + phase
    out.append(math.cos(0.15 * t + 0.5))

    # 14. product: tilt x forward velocity (coupling), plus quaternion lean
    out.append(clip(ab_y * sat(rvx) + (qx + qz) * qw))

    # ensure finite
    return [float(x) if math.isfinite(x) else 0.0 for x in out]
