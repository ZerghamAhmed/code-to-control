# controller.py
import math

def policy(state):
    objs = objects(state)
    
    # Get player one's hook
    hook = objs.get("playeronehook")
    if hook is None:
        return "noop"
    
    # Get all fish
    fish_list = objs.all("fish")
    if not fish_list:
        return "noop"
    
    # Find nearest fish
    nearest_fish = None
    min_dist = float('inf')
    for fish in fish_list:
        dist = hook.distance_to(fish)
        if dist < min_dist:
            min_dist = dist
            nearest_fish = fish
    
    if nearest_fish is None:
        return "noop"
    
    # Calculate direction to nearest fish
    dx = nearest_fish.x - hook.x
    dy = nearest_fish.y - hook.y
    
    angle = math.atan2(dy, dx)
    angle_deg = math.degrees(angle) % 360
    
    # Map angle to direction and fire action
    if angle_deg < 22.5 or angle_deg >= 337.5:
        direction = "right"
        fire_action = "rightfire"
    elif angle_deg < 67.5:
        direction = "downright"
        fire_action = "downrightfire"
    elif angle_deg < 112.5:
        direction = "down"
        fire_action = "downfire"
    elif angle_deg < 157.5:
        direction = "downleft"
        fire_action = "downleftfire"
    elif angle_deg < 202.5:
        direction = "left"
        fire_action = "leftfire"
    elif angle_deg < 247.5:
        direction = "upleft"
        fire_action = "upleftfire"
    elif angle_deg < 292.5:
        direction = "up"
        fire_action = "upfire"
    else:
        direction = "upright"
        fire_action = "uprightfire"
    
    # Fire when close enough AND reasonably well-aligned
    if min_dist < 120:
        angle_to_cardinal = min(
            abs(angle_deg),
            abs(angle_deg - 45),
            abs(angle_deg - 90),
            abs(angle_deg - 135),
            abs(angle_deg - 180),
            abs(angle_deg - 225),
            abs(angle_deg - 270),
            abs(angle_deg - 315),
            abs(360 - angle_deg)
        )
        
        # Fire if within 45 degrees of a cardinal/diagonal direction
        if angle_to_cardinal < 45:
            return fire_action
    
    # Move toward the nearest fish
    return direction
