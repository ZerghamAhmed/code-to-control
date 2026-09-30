# controller.py
import math

def policy(state):
    objs = objects(state)
    bird = objs.get("bird")
    
    if bird is None:
        return "noop"
    
    pipes = objs.all("pipe")
    bird_y = bird.y
    bird_dy = bird.dy
    bird_x = bird.x
    
    # AGGRESSIVE ceiling avoidance: keep bird well away from top
    # At y < 80, never flap; at y < 100 only in emergency
    if bird_y < 80:
        return "noop"
    
    if bird_y < 100 and bird_dy < 0:
        return "noop"
    
    # Hard ground bounds
    if bird_y > 500:
        return "flap"
    
    if pipes:
        # Filter pipes ahead of the bird
        pipes_ahead = [p for p in pipes if p.x > bird_x - 50]
        
        if pipes_ahead:
            pipes_ahead.sort(key=lambda p: p.x)
            
            # Identify pipe pair: upper and lower pipe at similar x
            next_upper = None
            next_lower = None
            
            for i in range(len(pipes_ahead)):
                for j in range(i + 1, len(pipes_ahead)):
                    p1, p2 = pipes_ahead[i], pipes_ahead[j]
                    if abs(p1.x - p2.x) < 30:
                        if p1.y < p2.y:
                            next_upper, next_lower = p1, p2
                        else:
                            next_upper, next_lower = p2, p1
                        break
                if next_upper:
                    break
            
            if next_upper and next_lower:
                # Gap geometry
                upper_bottom = next_upper.y + next_upper.h / 2.0
                lower_top = next_lower.y - next_lower.h / 2.0
                gap_center = (upper_bottom + lower_top) / 2.0
                gap_height = lower_top - upper_bottom
                safe_margin = max(20, gap_height * 0.25)
                
                dist_to_pipe = next_upper.x - bird_x
                
                # Close range: precise emergency control only
                if dist_to_pipe < 80:
                    if bird_y < upper_bottom + safe_margin:
                        return "noop"
                    if bird_y > lower_top - safe_margin:
                        return "flap"
                    return "noop"
                
                # Moderate range: steer toward center, but conservatively
                elif dist_to_pipe < 180:
                    target = gap_center
                    if bird_y > target + 30:
                        return "flap"
                    if bird_dy > 3.0 and bird_y > target - 20:
                        return "noop"
                    return "noop"
                
                # Far range: position in middle of gap proactively
                else:
                    target_y = gap_center
                    if bird_y > target_y + 50:
                        return "flap"
                    if bird_dy > 0.5:
                        return "noop"
                    return "noop"
    
    # No pipe ahead; maintain mid-altitude conservatively
    if bird_y > 280:
        return "flap"
    if bird_dy > 1.5:
        return "noop"
    return "noop"
