import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

class ControlState:
    def __init__(self):
        self.iteration = 0 # TODO Used for debuggin dont forget to comment out the prints
        self.last_closest_idx = 0
        
ctrl_state = ControlState()

def get_closest_point_index(pos: ArrayLike, path: ArrayLike) -> int:
    """Find index of closest point on path to current position."""
    
    dists = np.sum((path - pos) ** 2, axis=1)
    return np.argmin(dists)

def get_lookahead_point(pos: ArrayLike, path: ArrayLike, start_idx: int, 
                        lookahead: float) -> tuple:
    """Get lookahead point based on the approximate lookahead distance."""

    n = len(path)
    
    for i in range(1, min(n, 100)):
        idx = (start_idx + i) % n
        dist = np.linalg.norm(path[idx] - pos)
        
        if dist >= lookahead:
            return path[idx], idx
    
    # Fallback in case we cant find a point > lookahead distance
    idx = (start_idx + 30) % n
    return path[idx], idx

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""

    return np.arctan2(np.sin(angle), np.cos(angle))

def pure_pursuit_control(pos: ArrayLike, heading: float, 
                        target: ArrayLike, wheelbase: float) -> float:
    """Calculate steering angle using Pure Pursuit algorithm."""

    # Vector detla to target
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    
    # Distance to target  
    ld = np.sqrt(dx**2 + dy**2)
    
    # Avoid turning when the point is too close
    if ld < 1.0:
        return 0.0
    
    # Transform to vehicle frame, rotate by heading
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    x_veh = dx * cos_h + dy * sin_h
    y_veh = -dx * sin_h + dy * cos_h
    
    # Alpha angle in vehicle frame
    alpha = np.arctan2(y_veh, x_veh)
    
    # Pure Pursuit formula
    delta = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)
    
    return delta

def curvature_three_points(p1: ArrayLike, p2: ArrayLike, p3: ArrayLike) -> float:
    """
    Estimate geometric curvature (1 / radius) from three points triangle using the circumcircle. 
    Returns curvature in 1/m.
    """

    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)

    # Side lengths of the triangle
    a, b, c = np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p2), np.linalg.norm(p3 - p1)

    # Tiny triangle -> treat as straight
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0

    # Area via cross product
    area = abs(np.cross(p2 - p1, p3 - p1)) * 0.5
    if area < 1e-6:
        return 0.0

    # Circumradius R = (a * b * c) / (4 * A), curvature κ = 1 / R
    R = (a * b * c) / (4.0 * area)

    # If area too small -> treat as straight
    if R < 1e-6:
        return 0.0

    return 1.0 / R

def estimate_path_curvature(path: ArrayLike, idx: int, window_distance_m: float = 150.0) -> float:
    """
    Estimate curvature ahead on the path, updated to use the max curve ahead as our reference point instead of average to catch sharp and short turns.

    Each point along the path will also call the curvature three points for computing the curvature angle.
    """

    path = np.asarray(path)
    n = len(path)
    max_curvature = 0.0

    # Accumulated distance along the path ahead of idx
    dist_acc = 0.0
    i = 1

    # Safety: don't loop more than one full lap even on weird inputs
    while dist_acc < window_distance_m and i + 1 < n * 2:
        idx1 = (idx + i - 1) % n
        idx2 = (idx + i) % n
        idx3 = (idx + i + 1) % n

        p1 = path[idx1]
        p2 = path[idx2]
        p3 = path[idx3]

        # Curvature at p2 based on the local circle through p1, p2, p3
        kappa = curvature_three_points(p1, p2, p3)
        if kappa > max_curvature:
            max_curvature = kappa

        # Advance distance using segment length between p1 and p2
        seg_len = np.linalg.norm(p2 - p1)
        dist_acc += seg_len

        i += 1
        if i > n:
            break

    return max_curvature


def compute_target_velocity(path: ArrayLike, idx: int, base_vel: float, current_vel: float,
                            desired_steering: float, max_steering: float) -> float:
    """
    Sliding window that computes target velocity based on three things
    1. Max curvature angle in the window
    2. Steering angle change
    3. Current velocity
    """

    # Choose how far ahead in distance to scan, based on current velocity
    if current_vel > 80.0:
        window_distance_m = 250.0
    elif current_vel > 60.0:
        window_distance_m = 150.0
    else:
        window_distance_m = 100.0

    curvature = estimate_path_curvature(path, idx, window_distance_m=window_distance_m)

    # Use how much steering the high-level controller is asking for (normalized).
    steer_ratio = abs(desired_steering) / max_steering if max_steering > 1e-6 else 0.0

    if curvature < 0.01: # Almost striaght / close to no curve
        curvature_factor = 1.0
        steering_factor = 1.0
    elif curvature < 0.03: # Medium turn
        curvature_factor = 0.8
        if steer_ratio < 0.25: # Almost no steering
            steering_factor = 0.8
        elif steer_ratio < 0.5:
            steering_factor = 0.7
        elif steer_ratio < 0.7:
            steering_factor = 0.6
        else:
            steering_factor = 0.3 # Near max steering, keep speed low
    elif curvature < 0.07: # Sharp turn
        curvature_factor = 0.5
        if steer_ratio < 0.25:
            steering_factor = 0.4
        elif steer_ratio < 0.5:
            steering_factor = 0.3
        elif steer_ratio < 0.7:
            steering_factor = 0.2
        else:
            steering_factor = 0.1     # near max steering, keep speed low
    elif curvature < 0.35: # Sharper turn
        curvature_factor = 0.15
        if steer_ratio < 0.25:
            steering_factor = 0.9
        else:
            steering_factor = 0.7
    else: # Anything above this is just a crazy turn, we want to keep the speed minimal here
        curvature_factor = 0.05
        if steer_ratio < 0.25:
            steering_factor = 0.9
        else:
            steering_factor = 0.7

    # Final target speed is limited by the combination of the two factors
    factor = curvature_factor * steering_factor
    vel = base_vel * factor

    # print(f"{'*'*50} \n Current curvature factor: {curvature_factor} \n Current steering factor: {steering_factor} \n {'*'*50}")
    return vel


def lower_controller(state: ArrayLike, desired: ArrayLike, 
                     parameters: ArrayLike) -> ArrayLike:
    """
    Lower-level controller producing [steering_rate, acceleration].
    state: [sx, sy, delta, v, phi]
    desired: [delta_ref, v_ref]
    parameters: vehicle parameters
    """

    current_steering = state[2]
    current_velocity = state[3]
    
    desired_steering = desired[0]
    desired_velocity = desired[1]
    
    max_steer_rate = parameters[9]
    min_steer_rate = parameters[7]
    max_accel = parameters[10]
    min_accel = parameters[8]
    
    # Steering control (simple P controller)
    steer_error = normalize_angle(desired_steering - current_steering)
    Kp_steer = 6.8
    if current_velocity < 20:
        Kp_steer *= 1.5
    steering_rate = Kp_steer * steer_error
    steering_rate = np.clip(steering_rate, min_steer_rate, max_steer_rate)
    
    # Velocity control (also simple P controller)
    vel_error = desired_velocity - current_velocity
    Kp_vel = 3.5
    acceleration = Kp_vel * vel_error
    
    # Clip to limits
    acceleration = np.clip(acceleration, min_accel, max_accel)
    
    return np.array([steering_rate, acceleration])

def controller(state: ArrayLike, parameters: ArrayLike, 
               racetrack: RaceTrack) -> ArrayLike:
    """
    Main controller that produces [desired_steering, desired_velocity].
    state: [sx, sy, delta, v, phi]
    parameters: vehicle parameters
    racetrack: RaceTrack object
    """
    
    ctrl_state.iteration += 1
    
    # State
    sx, sy = state[0], state[1]
    delta = state[2]
    v = state[3]
    phi = state[4]
    
    pos = np.array([sx, sy])
    
    # Parameters
    wheelbase = parameters[0]      # 3.6m
    max_steering = parameters[4]   # 0.9 rad
    max_velocity = parameters[5]   # 100 m/s
    max_accel = parameters[10]     # 20 m/s²
    
    # Path, using just the centerline
    path = racetrack.centerline
    
    # Find closest point
    closest_idx = get_closest_point_index(pos, path)
    
    # Make sure we don't jump backwards on the track
    if closest_idx < ctrl_state.last_closest_idx - 50:
        closest_idx = ctrl_state.last_closest_idx
    elif closest_idx > ctrl_state.last_closest_idx:
        ctrl_state.last_closest_idx = closest_idx
    
    # Dymamic lookahead distance calculation based on current speed
    speed = max(abs(v), 1.0)
    lookahead = np.clip(3.0 + 0.6 * speed, 10.0, 25.0)
    
    # Get target point
    target_point, target_idx = get_lookahead_point(pos, path, closest_idx, lookahead)
    
    # Pure Pursuit steering
    desired_steering = pure_pursuit_control(pos, phi, target_point, wheelbase)
    desired_steering = np.clip(desired_steering, -max_steering, max_steering)
    
    # Target velocity based on curvature and steering demand
    desired_velocity = compute_target_velocity(
        path, target_idx, max_velocity, v,
        desired_steering, max_steering
    )

    
    # Debug
    # if ctrl_state.iteration % 1 == 0:
    #     curv = estimate_path_curvature(path, target_idx)
    #     # Check distance to start line
    #     start_pos = path[0]
    #     dist_to_start = np.linalg.norm(pos - start_pos)
    #     print(f"Iter {ctrl_state.iteration}: Pos=({sx:.1f},{sy:.1f}), V={v:.1f}->{desired_velocity:.1f}, " +
    #           f"δ={np.degrees(delta):.1f}°->{np.degrees(desired_steering):.1f}°, Curv={curv:.3f}, LD={lookahead:.1f}m, " +
    #           f"Dist2Start={dist_to_start:.1f}m")
    
    desired = np.array([desired_steering, desired_velocity])
    
    return desired