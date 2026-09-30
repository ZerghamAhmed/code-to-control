# controller.py
import math

def policy(state):
    bird = objects(state).get("bird")
    pipes = objects(state).all("pipe")
    
    if bird is None:
        return "noop"
    
    if len(pipes) < 2:
        return "noop"
    
    bird_x = bird.x
    bird_y = bird.y
    bird_dy = bird.dy
    
    # Find the next pipe pair ahead (or current) relative to bird
    # Pipes come in pairs: [0,1], [2,3], etc. (top, bottom)
    # Find the pair where the bird will soon need to fit through
    next_pair_idx = 0
    min_distance = float('inf')
    
    for i in range(0, len(pipes) - 1, 2):
        top_pipe = pipes[i]
        bottom_pipe = pipes[i + 1]
        
        # Use the rightmost edge of the pipe pair as reference
        pipe_right = max(top_pipe.x + top_pipe.w / 2.0, bottom_pipe.x + bottom_pipe.w / 2.0)
        
        # We care about pipes that are ahead of or very close to the bird
        if pipe_right >= bird_x - 50:  # Lookahead margin
            distance = pipe_right - bird_x
            if distance < min_distance:
                min_distance = distance
                next_pair_idx = i
    
    top_pipe = pipes[next_pair_idx]
    bottom_pipe = pipes[next_pair_idx + 1]
    
    # Calculate gap boundaries
    gap_top = top_pipe.y + top_pipe.h / 2.0      # bottom edge of top pipe
    gap_bottom = bottom_pipe.y - bottom_pipe.h / 2.0  # top edge of bottom pipe
    gap_center = (gap_top + gap_bottom) / 2.0
    
    # Safety margins (in pixels)
    safe_margin = 15.0
    
    # Flap if bird is below safe center OR falling toward bottom pipe
    # Don't flap if already above the gap center (avoid over-rising)
    if bird_y > gap_center + safe_margin and bird_dy >= -2.0:
        return "flap"
    
    return "noop"
