import math

def terms(state, t):
    def g(k):
        try:
            v = state[k][0] if isinstance(state[k], (list, tuple)) else state[k]
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

    rz = g('root_z')
    qw = g('root_quat_w')
    qx = g('root_quat_x')
    qy = g('root_quat_y')
    qz = g('root_quat_z')

    h1, h2, h3, h4 = g('hip_1'), g('hip_2'), g('hip_3'), g('hip_4')
    a1, a2, a3, a4 = g('ankle_1'), g('ankle_2'), g('ankle_3'), g('ankle_4')

    vx, vy, vz = g('root_velocity_x'), g('root_velocity_y'), g('root_velocity_z')
    wx, wy, wz = g('root_angular_velocity_x'), g('root_angular_velocity_y'), g('root_angular_velocity_z')

    hv1 = g('hip_1_angular_velocity')
    av1 = g('ankle_1_angular_velocity')
    hv2 = g('hip_2_angular_velocity')
    av2 = g('ankle_2_angular_velocity')
    hv3 = g('hip_3_angular_velocity')
    av3 = g('ankle_3_angular_velocity')
    hv4 = g('hip_4_angular_velocity')
    av4 = g('ankle_4_angular_velocity')

    out = []
    # 1. constant bias
    out.append(1.0)
    # 2. raw height, centered near typical standing ~0.9
    out.append(clip(rz - 0.9))
    # 3. tilt proxy: deviation of upright quaternion
    out.append(clip(1.0 - qw))
    # 4. combined joint posture (mean hip)
    out.append(clip(0.25 * (h1 + h2 + h3 + h4)))
    # 5. mean ankle posture
    out.append(clip(0.25 * (a1 + a2 + a3 + a4)))
    # 6. linear velocity magnitude (saturated)
    out.append(math.tanh(math.sqrt(vx*vx + vy*vy + vz*vz)))
    # 7. angular velocity of body, tanh-squashed sum
    out.append(math.tanh(wx + wy + wz))
    # 8. periodic term, base frequency
    out.append(math.sin(0.2 * t))
    # 9. periodic term, second frequency + phase shift
    out.append(math.cos(0.07 * t + 0.5))
    # 10. product/coupling: tilt * forward velocity
    out.append(clip(qx * vx + qy * vy))
    # 11. joint-velocity aggregate (tanh)
    out.append(math.tanh(0.1 * (hv1 + hv2 + hv3 + hv4)))
    # 12. ankle-velocity aggregate (tanh)
    out.append(math.tanh(0.1 * (av1 + av2 + av3 + av4)))
    # 13. quaternion yaw component (raw)
    out.append(clip(qz))
    # 14. saturating growth in t
    out.append(math.tanh(0.01 * t))

    return [float(x) if math.isfinite(x) else 0.0 for x in out]
