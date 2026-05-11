import numpy as np

# ---------------------------------------------------------------------------
# Observation layout (259-element vector from MetaDrive LidarStateObservation)
# ---------------------------------------------------------------------------
# [0]   lateral_to_left       — distance to left road boundary
# [1]   lateral_to_right      — distance to right road boundary
# [2]   heading_diff          — heading error relative to road direction
# [3]   speed
# [4]   steering              — current wheel steering angle
# [5]   last_action_steer
# [6]   last_action_throttle
# [7]   yaw_rate
# [8]   lateral_pos_in_lane
# Navigation checkpoints:
# [9]   ckpt1 forward projection
# [10]  ckpt1 side projection
# [11]  ckpt1 radius
# [12]  ckpt1 clockwise
# [13]  ckpt1 angle
# [14]  ckpt2 forward projection
# [15]  ckpt2 side projection
# [16]  ckpt2 radius
# [17]  ckpt2 clockwise
# [18]  ckpt2 angle
# [19:259] 240 lidar rays
# ---------------------------------------------------------------------------

LIDAR_START = 19
LIDAR_COUNT = 240

# Speed limits
MAX_SPEED_STRAIGHT = 0.60
MAX_SPEED_TURN = 0.40
MAX_SPEED_RECOVERY = 0.30
MAX_SPEED_DANGER = 0.16
SPEED_KP = 1.20

# Lidar sector widths (in ray count)
FRONT_WIDTH = 18
FRONT_SIDE_WIDTH = 28
SIDE_WIDTH = 40

# Safety distance thresholds (normalised, 0=wall, 1=clear)
DANGER_DIST = 0.11
CAUTION_DIST = 0.30
ROAD_EDGE_CAUTION = 0.25
ROAD_EDGE_DANGER = 0.14
ROAD_EDGE_RELEASE = 0.29
YAW_DAMP_GAIN = 0.65

# Steering
STEER_MAX = 0.60
SMOOTH = 0.35

# Throttle
THROTTLE_CRUISE = 0.40
THROTTLE_CAUTION = 0.18
THROTTLE_BRAKE = -0.55

# Navigation
NAV_TURN_STEER = 0.25

# Debug (all disabled — set True locally for telemetry during development)
DEBUG = False
DEBUG_EVERY = 5
DEBUG_EVENT_WARNINGS = False
DEBUG_RAW_OBS = False
DEBUG_FULL_LIDAR = False


class TeacherSolution:
    """
    Rule-based driving controller. Handles lane keeping, obstacle avoidance,
    and navigation command following. Used as the throttle provider in the
    hybrid agent and as the label source during behavioural cloning data
    collection.
    """

    def __init__(self, game):
        self._game = game
        self._prev_steer = 0.0
        self._step = 0
        self._last_cmd = "forward"
        self._prev_boundary_error = 0.0
        self._edge_recovering = False
        self._prev_road_center_error = 0.0

    @property
    def config(self):
        return {"image_observation": False}

    def _lidar_regions(self, obs):
        lidar = obs[LIDAR_START:LIDAR_START + LIDAR_COUNT]
        n = LIDAR_COUNT

        front = np.concatenate([lidar[:FRONT_WIDTH], lidar[n - FRONT_WIDTH:]])
        front_right = lidar[FRONT_WIDTH:FRONT_WIDTH + FRONT_SIDE_WIDTH]
        right = lidar[FRONT_WIDTH + FRONT_SIDE_WIDTH:FRONT_WIDTH + FRONT_SIDE_WIDTH + SIDE_WIDTH]
        front_left = lidar[n - FRONT_WIDTH - FRONT_SIDE_WIDTH:n - FRONT_WIDTH]
        left = lidar[n - FRONT_WIDTH - FRONT_SIDE_WIDTH - SIDE_WIDTH:n - FRONT_WIDTH - FRONT_SIDE_WIDTH]

        return {
            "front": float(np.min(front)),
            "front_left": float(np.min(front_left)),
            "front_right": float(np.min(front_right)),
            "left": float(np.min(left)),
            "right": float(np.min(right)),
            "front_mean": float(np.mean(front)),
            "front_left_mean": float(np.mean(front_left)),
            "front_right_mean": float(np.mean(front_right)),
            "left_mean": float(np.mean(left)),
            "right_mean": float(np.mean(right)),
            "raw": lidar,
        }

    def do_iteration(self, simulator_output, user_input=None):
        obs = simulator_output.get("observation", None)
        info = simulator_output.get("info", {}) or {}

        if not isinstance(obs, np.ndarray) or len(obs) < 259:
            return user_input if user_input is not None else [0.0, 0.25]

        # Raw state
        left_boundary = float(obs[0])
        right_boundary = float(obs[1])
        heading = float(obs[2])
        speed = float(obs[3])
        yaw_rate = float(obs[7])
        lateral = float(obs[8])

        # Navigation command
        cmd = str(info.get("navigation_command", "forward")).lower()

        # Camera perception / future-risk input
        perception = info.get("perception") or {}
        perception_available = isinstance(perception, dict) and bool(perception)
        perception_best_action = perception.get("best_action")
        perception_action_risks = perception.get("action_risks", {}) or {}
        perception_go_risk = float(perception_action_risks.get("go", 0.0))
        perception_max_ego_iou = float(perception.get("max_ego_iou", 0.0) or 0.0)
        perception_danger = bool(perception.get("danger", False))

        # Lidar
        lidar = self._lidar_regions(obs)
        min_front = lidar["front"]
        min_front_left = lidar["front_left"]
        min_front_right = lidar["front_right"]
        min_left = lidar["left"]
        min_right = lidar["right"]

        # -------------------------------------------------------------------
        # 1. Lane keeping
        # -------------------------------------------------------------------
        road_width = max(left_boundary + right_boundary, 1e-6)
        boundary_error = (left_boundary - right_boundary) / road_width
        boundary_rate = boundary_error - self._prev_boundary_error

        CENTER_DEADBAND = 0.04
        if abs(boundary_error) < CENTER_DEADBAND:
            effective_error = 0.0
        else:
            effective_error = np.sign(boundary_error) * (abs(boundary_error) - CENTER_DEADBAND)

        lane_lateral = float(np.clip(effective_error * 0.34 + boundary_rate * 0.25, -0.20, 0.20))
        yaw_damp_s = float(np.clip(-yaw_rate * 0.45, -0.10, 0.10))
        lane_s = lane_lateral + yaw_damp_s
        self._prev_boundary_error = boundary_error

        # -------------------------------------------------------------------
        # 2. Navigation
        # -------------------------------------------------------------------
        if cmd == "left":
            nav_s = NAV_TURN_STEER
        elif cmd == "right":
            nav_s = -NAV_TURN_STEER
        else:
            nav_s = 0.0

        # -------------------------------------------------------------------
        # 3. Road-edge recovery
        # -------------------------------------------------------------------
        edge_danger = (
            left_boundary < ROAD_EDGE_DANGER or right_boundary < ROAD_EDGE_DANGER
        )
        edge_caution = (
            left_boundary < ROAD_EDGE_CAUTION or right_boundary < ROAD_EDGE_CAUTION
        )

        if edge_danger:
            self._edge_recovering = True
        elif min(left_boundary, right_boundary) > ROAD_EDGE_RELEASE:
            self._edge_recovering = False

        recovery_s = 0.0
        if self._edge_recovering or edge_caution:
            if left_boundary < right_boundary:
                recovery_s = -0.30  # too close to left → steer right
            else:
                recovery_s = 0.30   # too close to right → steer left

        # -------------------------------------------------------------------
        # 4. Obstacle avoidance (lidar)
        # -------------------------------------------------------------------
        front_blocked = min_front < DANGER_DIST
        front_caution = min_front < CAUTION_DIST

        avoid_s = 0.0
        if front_blocked or front_caution:
            if min_front_right < min_front_left:
                avoid_s = 0.20   # obstacle on right → steer left
            else:
                avoid_s = -0.20  # obstacle on left → steer right

        # -------------------------------------------------------------------
        # 5. Combine steering
        # -------------------------------------------------------------------
        if self._edge_recovering or edge_danger:
            raw_s = recovery_s
        elif front_blocked:
            raw_s = avoid_s + lane_s * 0.3
        else:
            raw_s = lane_s + nav_s + avoid_s

        raw_s = float(np.clip(raw_s, -STEER_MAX, STEER_MAX))
        steer = float(np.clip(
            self._prev_steer * SMOOTH + raw_s * (1.0 - SMOOTH),
            -STEER_MAX, STEER_MAX,
        ))
        self._prev_steer = steer

        # -------------------------------------------------------------------
        # 6. Throttle
        # -------------------------------------------------------------------
        turning = cmd in ("left", "right")
        max_speed = MAX_SPEED_TURN if turning else MAX_SPEED_STRAIGHT

        if edge_danger or self._edge_recovering:
            max_speed = MAX_SPEED_RECOVERY

        # Perception-based danger override
        if perception_available and (perception_danger or perception_max_ego_iou > 0.1):
            max_speed = min(max_speed, MAX_SPEED_DANGER)

        if front_blocked:
            throttle = THROTTLE_BRAKE
        elif front_caution or edge_caution:
            throttle = THROTTLE_CAUTION
        else:
            speed_error = max_speed - speed
            throttle = float(np.clip(speed_error * SPEED_KP, -1.0, THROTTLE_CRUISE))

        # Further reduce throttle based on perception risk score
        if perception_available and perception_best_action in ("slow_down", "brake", "stop"):
            risk_scale = max(0.0, 1.0 - perception_go_risk * 3.0)
            throttle = float(np.clip(throttle * risk_scale, -1.0, 1.0))

        self._step += 1
        return [steer, throttle]
