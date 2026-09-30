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

    ta = g('tip_angle')
    r1 = g('rotor1_angle')
    r2 = g('rotor2_angle')
    vx = g('tip_velocity_x')
    vy = g('tip_velocity_y')
    wt = g('tip_angular_velocity')
    w1 = g('rotor1_angular_velocity')
    w2 = g('rotor2_angular_velocity')

    def clip(x, lo=-3.0, hi=3.0):
        return max(lo, min(hi, x))

    out = []
    # 1. constant bias
    out.append(1.0)
    # 2. tip angle
    out.append(clip(ta))
    # 3. sin of tip angle (gravity-like restoring)
    out.append(math.sin(ta))
    # 4. rotor angle difference
    out.append(clip(r1 - r2))
    # 5. tip angular velocity
    out.append(clip(wt))
    # 6. rotor1 angular velocity (scaled)
    out.append(clip(0.2 * w1))
    # 7. rotor2 angular velocity (scaled)
    out.append(clip(0.2 * w2))
    # 8. tip vy
    out.append(clip(vy))
    # 9. tip vx
    out.append(clip(vx))
    # 10. product tip_angle * tip_angular_velocity
    out.append(clip(ta * wt))
    # 11. difference of rotor angular velocities (scaled)
    out.append(clip(0.15 * (w1 - w2)))
    # 12. periodic term
    out.append(math.sin(0.3 * t))
    # 13. second periodic term, different freq + phase
    out.append(math.cos(0.11 * t + 1.0))
    # 14. saturating combined velocity magnitude
    out.append(math.tanh(0.3 * (abs(wt) + abs(vy))))

    return [float(x) if math.isfinite(x) else 0.0 for x in out]
