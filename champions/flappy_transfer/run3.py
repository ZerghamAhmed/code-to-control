# controller.py
import math

def policy(state):
    bird = objects(state).get("bird")
    
    # Safety check
    if bird is None:
        return "noop"
    
    # Get next pipe (pipes move left, so find the closest one ahead)
    pipes = objects(state).all("pipe")
    
    if not pipes:
        return "noop"
    
    # Find the next pipe approaching from the right
    # Pipes come in pairs (top and bottom), we need to identify pairs by x-position
    next_pipe = None
    for pipe in pipes:
        if pipe.x > bird.x - 100:  # Pipe ahead or slightly behind but still relevant
            if next_pipe is None or pipe.x < next_pipe.x:
                next_pipe = pipe
    
    if next_pipe is None:
        # No upcoming pipes, try to stay in middle
        target_y = 280
    else:
        # Determine if this is a top or bottom pipe by examining y position
        # Top pipes have smaller y values (near ceiling)
        # Bottom pipes have larger y values (near ground)
        
        # Find both pipes at the same x-position to identify the gap
        same_x_pipes = [p for p in pipes if abs(p.x - next_pipe.x) < 5]
        
        if len(same_x_pipes) >= 2:
            # Sort by y position
            same_x_pipes.sort(key=lambda p: p.y)
            top_pipe = same_x_pipes[0]
            bottom_pipe = same_x_pipes[1]
            
            # Gap center
            gap_center = (top_pipe.y + top_pipe.h / 2 + bottom_pipe.y - bottom_pipe.h / 2) / 2
            target_y = gap_center
        else:
            target_y = 280
    
    # Simple control: if below target, flap; otherwise noop
    # Add hysteresis to avoid oscillation
    margin = 30
    
    if bird.y > target_y + margin:
        # Bird is too low, flap to go up
        return "flap"
    else:
        # Bird is at good height or too high, let gravity pull it down
        return "noop"
