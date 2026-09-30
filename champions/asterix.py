# controller.py
import math

def policy(state):
    objs = objects(state)
    player = objs.get("player")
    
    if player is None:
        return "noop"
    
    consumables = objs.all("consumable")
    hazards = objs.all("car") + objs.all("obstacle")
    
    # If no consumables left, we've won — STOP MOVING
    if not consumables or state.get("won", False):
        return "noop"
    
    # Check for approaching hazards and evade preemptively
    for hazard in hazards:
        # Use larger safety margin and check approach velocity
        if hazard.approaching(player) and player.distance_to(hazard) < 80:
            # Approaching hazard within danger zone — evade
            if hazard.x > player.x:
                # Hazard to the right, move left
                if hazard.y > player.y:
                    return "upleft"
                else:
                    return "downleft"
            else:
                # Hazard to the left, move right
                if hazard.y > player.y:
                    return "upright"
                else:
                    return "downright"
        
        # Also handle contact/near-contact
        if player.touches(hazard, margin=10):
            if hazard.x > player.x:
                if hazard.y > player.y:
                    return "upleft"
                else:
                    return "downleft"
            else:
                if hazard.y > player.y:
                    return "upright"
                else:
                    return "downright"
    
    # Find the closest consumable
    target = None
    min_distance = float('inf')
    for c in consumables:
        dist = player.distance_to(c)
        if dist < min_distance:
            min_distance = dist
            target = c
    
    if target is None:
        return "noop"
    
    # Calculate direction to target
    dx = target.x - player.x
    dy = target.y - player.y
    
    # Larger threshold to reduce oscillation
    horizontal_threshold = 5.0
    
    # Move toward target
    if abs(dy) > abs(dx):
        # Vertical component dominates
        if dy > 0:  # Target below
            if dx > horizontal_threshold:
                return "downright"
            elif dx < -horizontal_threshold:
                return "downleft"
            else:
                return "down"
        else:  # Target above
            if dx > horizontal_threshold:
                return "upright"
            elif dx < -horizontal_threshold:
                return "upleft"
            else:
                return "up"
    else:
        # Horizontal component dominates
        if dx > horizontal_threshold:
            return "right"
        elif dx < -horizontal_threshold:
            return "left"
        else:
            if dy > 0:
                return "down"
            elif dy < 0:
                return "up"
            else:
                return "noop"
