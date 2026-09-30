# controller.py
import math

def policy(state):
    objs = objects(state)
    player = objs.get("player")
    
    if player is None:
        return "noop"
    
    aliens = objs.all("alien")
    bullets = objs.all("bullet")
    
    # THREAT ASSESSMENT: detect incoming bullets
    incoming_threats = []
    for bullet in bullets:
        if bullet.approaching(player):
            speed = math.sqrt(bullet.dx**2 + bullet.dy**2)
            if speed > 0.1:
                tti = player.distance_to(bullet) / speed
                if tti < 30:
                    incoming_threats.append((bullet, tti))
    
    incoming_threats.sort(key=lambda x: x[1])
    
    # EVASION: dodge immediate threats
    if incoming_threats:
        closest_bullet, tti = incoming_threats[0]
        if tti < 20:
            if closest_bullet.x > player.x:
                return "left"
            else:
                return "right"
    
    # TARGETING: engage aliens
    if aliens:
        closest_alien = min(aliens, key=lambda a: player.distance_to(a))
        
        if closest_alien.y < player.y - 30:  # Alien is above us
            dx = closest_alien.x - player.x
            horiz_gap = abs(dx)
            
            if horiz_gap < 12:
                # Aligned — fire!
                return "fire"
            elif horiz_gap < 50 and closest_alien.y > -50:
                if horiz_gap < 30:
                    # Very close, fire at angle
                    if dx > 0:
                        return "rightfire"
                    else:
                        return "leftfire"
                else:
                    # Move to align
                    if dx > 0:
                        return "right"
                    else:
                        return "left"
            else:
                # Far alien — move to align slowly
                if dx > 0:
                    return "right"
                else:
                    return "left"
    
    # NO THREATS, NO ALIENS: stay safe
    # Shield zone is x ≈ 42–106. Player at x=49 is safe.
    # Only recenter if we drift far from the safe zone (> 20 px from shields).
    shield_left = 42
    shield_right = 106
    shield_mid = (shield_left + shield_right) / 2  # 74
    
    safe_zone_left = shield_left - 10
    safe_zone_right = shield_right + 10
    
    if player.x < safe_zone_left:
        # Drifted too far left, move right to safety
        return "right"
    elif player.x > safe_zone_right:
        # Drifted too far right, move left to safety
        return "left"
    else:
        # Within safe zone — hold position
        return "noop"
