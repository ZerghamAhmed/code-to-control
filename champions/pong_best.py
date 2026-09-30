# controller.py
import math

def policy(state):
    objs = objects(state)
    player = objs.get("player")
    
    if player is None:
        return 'noop'
    
    # Primary objective: track ball
    ball = objs.get("ball")
    
    if ball is not None:
        # Lookahead 1 tick to anticipate ball motion (more responsive)
        lookahead_ticks = 1
        predicted_ball_y = ball.y + ball.dy * lookahead_ticks
        
        # Vertical distance from player center to predicted ball position
        dy = predicted_ball_y - player.y
        
        # Tighter threshold to reduce oscillation while maintaining responsiveness
        threshold = 7
        
        if dy > threshold:
            return 'down'
        elif dy < -threshold:
            return 'up'
        else:
            return 'noop'
    
    # No ball present: hold position
    return 'noop'
