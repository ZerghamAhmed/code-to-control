# controller.py
import math

def policy(state):
    objs = objects(state)
    player = objs.get("player")
    
    if player is None:
        return "noop"
    
    # Check if there's a ball to track
    ball = objs.get("ball")
    
    if ball is None:
        # No ball yet; hold center position
        return "noop"
    
    # Predict where the ball will be in the near future
    # The ball moves with velocity (dx, dy), so in ~10 ticks it will move by (10*dx, 10*dy)
    # Lead the ball prediction by a few ticks to account for reaction latency
    lead_ticks = 5  # Predict 5 ticks ahead
    predicted_ball_y = ball.y + ball.dy * lead_ticks
    
    player_center = player.y
    deadzone = 4.0  # Tighter deadzone for more responsive control
    
    if player_center < predicted_ball_y - deadzone:
        # Player is above where the ball will be; move down
        return "down"
    elif player_center > predicted_ball_y + deadzone:
        # Player is below where the ball will be; move up
        return "up"
    else:
        # Player is near the predicted ball position; hold position
        return "noop"
