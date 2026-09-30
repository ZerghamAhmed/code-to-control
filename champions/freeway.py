# controller.py
import math

def policy(state):
    """
    Frogger-like game with stationary cars blocking the road.
    
    Strategy:
    - Move UP toward y=0 (goal) when possible
    - When blocked by stationary cars and stuck, move DOWN to reset
    - Check all chickens to detect if progress is truly blocked
    """
    objs = objects(state)
    
    chickens = objs.all("chicken")
    if not chickens:
        return "noop"
    
    cars = objs.all("car")
    
    # === COLLISION AVOIDANCE ===
    # If touching a car, move down immediately
    for chicken in chickens:
        for car in cars:
            if chicken.touches(car, margin=3):
                return "down"
    
    # === PROXIMITY THREAT ===
    # If any car is very close and approaching, escape down
    for chicken in chickens:
        for car in cars:
            dist = chicken.distance_to(car)
            if dist < 12 and car.approaching(chicken):
                return "down"
    
    # === STUCK DETECTION ===
    # Track the best y-position achieved by any chicken
    if "_best_y" not in state:
        state["_best_y"] = min(c.y for c in chickens)
        state["_stuck_ticks"] = 0
    
    current_best_y = min(c.y for c in chickens)
    
    if current_best_y < state["_best_y"]:
        # Progress made: reset stuck counter
        state["_best_y"] = current_best_y
        state["_stuck_ticks"] = 0
    else:
        # No progress
        state["_stuck_ticks"] += 1
    
    # If stuck for too long, reset by moving down
    if state["_stuck_ticks"] >= 4:
        state["_stuck_ticks"] = 0
        state["_best_y"] = min(c.y for c in chickens)
        return "down"
    
    # === DEFAULT: MOVE UP ===
    return "up"
