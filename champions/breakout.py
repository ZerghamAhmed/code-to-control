# controller.py
import math

def policy(state):
    objs = objects(state)
    player = objs.get("player")
    
    if player is None:
        return "noop"
    
    # Look for a ball in the state
    ball = objs.get("ball")
    
    # If no ball exists yet, fire to start the game
    if ball is None:
        return "fire"
    
    # Ball exists - track it and position paddle defensively
    dx = ball.x - player.x
    dy = ball.y - player.y
    
    # Safety margin: if ball is approaching from above or is close to paddle level,
    # move aggressively to get under it
    
    # If ball is moving downward and is already close to or below paddle, center under it urgently
    if ball.dy > 0 and dy > -10:  # dy > -10 means ball is below or very close to paddle
        # Ball is coming down at us; position paddle under its x
        if abs(dx) > 5:
            return "left" if dx < 0 else "right"
        else:
            return "noop"
    
    # Normal case: ball is above us, track its horizontal motion
    if ball.y < player.y:
        # Ball approaching horizontally toward us from the left
        if ball.dx < 0 and dx < 0:
            return "left"
        # Ball approaching horizontally toward us from the right
        if ball.dx > 0 and dx > 0:
            return "right"
        # Ball above but may be heading to either side; position under it
        if abs(dx) > 10:
            return "left" if dx < 0 else "right"
    
    # Default: do nothing
    return "noop"
