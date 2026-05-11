import sys
import time
import numpy as np
from pathlib import Path

from solution import Solution
from teacher_solution import TeacherSolution
from logger import ActionLogger, DatasetWriter
from control import ControlState
from perception_risk_module import PerceptionRiskModule

from metadrive.obs.state_obs import LidarStateObservation
from metadrive.component.sensors.rgb_camera import RGBCamera

try:
    import pygame
except ImportError:
    sys.exit("pygame not found. Install with:  pip install pygame")

try:
    from metadrive import MetaDriveEnv
except ImportError:
    sys.exit("MetaDrive not found. Install with:  pip install metadrive-simulator")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    # Control ramping
    "STEER_STEP": 0.05,
    "STEER_DECAY": 0.07,
    "THROTTLE_STEP": 0.05,
    "THROTTLE_DECAY": 0.02,
    "BRAKE_VALUE": -1,
    "RESET_ON_SPACE": False,
    # Simulation
    "MAX_STEPS": 10_000,
    "LOG_FILE": "user_input_log.json",
    # Data collection
    # Modes: "model" | "human" | "dagger" | "human_assist"
    "DATA_MODE": "model",
    "DATASET_FILE": f"datasets/driving_{int(time.time())}.jsonl",
    # Perception
    "USE_PERCEPTION": True,
    "PERCEPTION_MODEL_PATH": "models/yolo_carla_best.pt",
    # Camera / frame settings
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 360,
    # Adaptive perception: run more often when lidar detects nearby objects
    "PERCEPTION_ACTIVE_EVERY": 5,       # frames between YOLO runs when active
    "PERCEPTION_IDLE_EVERY": 60,        # frames between YOLO runs when idle
    "PERCEPTION_ACTIVE_HOLD_FRAMES": 40,
    "LIDAR_WAKE_NEAR": 0.45,            # lidar threshold to trigger active mode
    "LIDAR_WAKE_DELTA": 0.05,           # lidar change threshold
    # MetaDrive environment
    "ENV_CONFIG": {
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.3,
        "num_scenarios": 8,
        "start_seed": 293,
        "map": "SOC",           # Straight, rOundabout, Curve
        "accident_prob": 0.0,
        "decision_repeat": 1,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BAR_WIDTH = 40


def _bar(value: float, low: float = -1.0, high: float = 1.0) -> str:
    """Return an ASCII bar centred on zero."""
    fraction = (value - low) / (high - low)
    pos = max(0, min(BAR_WIDTH - 1, int(fraction * BAR_WIDTH)))
    bar = ["-"] * BAR_WIDTH
    bar[BAR_WIDTH // 2] = "|"
    bar[pos] = "#"
    return "[" + "".join(bar) + "]"


def print_controls():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MetaDrive Keyboard Controller                   ║
╠══════════════════════════════════════════════════════════════╣
║  A / D    Steer left / right  (auto-center decay on release) ║
║  W        Accelerate          (auto-decay on release)        ║
║  S        Brake / cut throttle (instant 0, stays 0 on rls)   ║
║  SPACE    Reset steering & throttle instantly (if enabled)   ║
║  Q / ESC  Quit and save log                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Steer step   : 0.05 / frame   Steer decay  : 0.07 / frame   ║
║  Throttle step: 0.05 / frame   Throttle dec : 0.02 / frame   ║
╚══════════════════════════════════════════════════════════════╝
""")


def extract_config_from_solution(solution: Solution) -> dict:
    solution_config = solution.config
    result_config = {}

    if "image_observation" in solution_config:
        result_config = {"image_observation": solution_config["image_observation"]}

    if "sensors" in solution_config:
        result_config["sensors"] = solution_config["sensors"]

    vehicle = solution_config.get("vehicle_config", {})
    vehicle_config = {k: vehicle[k] for k in "image_source" if k in vehicle}
    if vehicle_config:
        result_config["vehicle_config"] = vehicle_config

    return result_config


def get_lidar_observation(env):
    vehicle = env.agent
    lidar_sensor = env.engine.get_sensor("lidar")
    lidar_config = vehicle.config.get("lidar")
    cloud_points, _ = lidar_sensor.perceive(
        vehicle,
        env.engine.physics_world.dynamic_world,
        lidar_config.get("num_lasers"),
        lidar_config.get("distance"),
        vehicle.config.get("show_lidar", False),
    )
    return np.array(cloud_points)


def sim_out_to_dict(sim_out):
    obs, reward, terminated, truncated, info = sim_out
    return {
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class Game:

    def __init__(self, config=CONFIG, intercepts=[]):
        self._prev_lidar_signature = None
        self._perception_active_until = -1
        self._last_perception_output = None
        self.loggers = []
        self.handlers = {}
        self.intercepts = intercepts
        self.config = config
        self.dataset_writer = DatasetWriter(self.config["DATASET_FILE"])
        self.episode_id = 0

    def _lidar_activity(self, obs):
        """
        Checks whether lidar sees something nearby or has changed significantly.
        Uses obs[19:259] — the 240 lidar rays in the state observation.
        Returns (triggered: bool, details: dict).
        """
        if not isinstance(obs, np.ndarray) or len(obs) < 259:
            return False, {}

        lidar = obs[19:259]
        n = len(lidar)
        front_width = 18
        front_side_width = 28

        front = np.concatenate([lidar[:front_width], lidar[n - front_width:]])
        front_right = lidar[front_width:front_width + front_side_width]
        front_left = lidar[n - front_width - front_side_width:n - front_width]

        signature = np.array([
            float(np.min(front)),
            float(np.min(front_left)),
            float(np.min(front_right)),
            float(np.mean(front)),
            float(np.mean(front_left)),
            float(np.mean(front_right)),
        ], dtype=np.float32)

        near = min(signature[0], signature[1], signature[2]) < float(self.config["LIDAR_WAKE_NEAR"])
        changed = False
        max_delta = 0.0

        if self._prev_lidar_signature is not None:
            max_delta = float(np.max(np.abs(signature - self._prev_lidar_signature)))
            changed = max_delta > float(self.config["LIDAR_WAKE_DELTA"])

        self._prev_lidar_signature = signature

        return near or changed, {
            "near": near,
            "changed": changed,
            "max_delta": max_delta,
            "front_min": float(signature[0]),
            "front_left_min": float(signature[1]),
            "front_right_min": float(signature[2]),
        }

    def _get_rgb_frame(self, env):
        camera = env.engine.get_sensor("rgb_camera")
        return camera.perceive(to_float=False)

    def start(self):
        print_controls()

        pygame.init()
        pg_screen = pygame.display.set_mode((520, 60))
        pygame.display.set_caption("MetaDrive Controls (focus here for input)")

        control = ControlState(self.config)
        step = 0
        running = True

        solution = Solution(self)
        teacher_solution = TeacherSolution(self)
        perception_module = None

        if self.config.get("USE_PERCEPTION", False):
            perception_module = PerceptionRiskModule(
                model_path=self.config["PERCEPTION_MODEL_PATH"],
                conf=0.10,
                imgsz=640,
                predict_ahead=1,
                min_history=5,
                max_history=10,
            )

        env_config = dict(self.config["ENV_CONFIG"])
        env_config.update(extract_config_from_solution(solution))

        existing_sensors = dict(env_config.get("sensors", {}))
        existing_sensors["rgb_camera"] = (
            RGBCamera,
            self.config["FRAME_WIDTH"],
            self.config["FRAME_HEIGHT"],
        )
        env_config.update({
            "agent_observation": LidarStateObservation,
            "image_observation": True,
            "norm_pixel": False,
            "sensors": existing_sensors,
        })
        self.config["ENV_CONFIG"] = env_config

        print("[MetaDrive] Initializing environment …")
        env = MetaDriveEnv(env_config)
        obs, info = env.reset()
        print("[MetaDrive] Environment ready.\n")

        sim_out = {
            "observation": obs,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": info,
        }

        try:
            while running and step < self.config["MAX_STEPS"]:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type in self.handlers:
                        self.handlers[event.type](event)

                keys = pygame.key.get_pressed()

                if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                    running = False
                    break

                steering, throttle = control.update(keys)
                user_input = [steering, throttle]

                obs = sim_out["observation"]
                if isinstance(obs, dict) and "lidar" not in obs:
                    obs["lidar"] = get_lidar_observation(env)
                for intercept in self.intercepts:
                    obs = intercept(obs)
                sim_out["observation"] = obs

                # --- Adaptive perception ---
                perception_output = self._last_perception_output
                lidar_triggered, lidar_activity = self._lidar_activity(obs)

                if lidar_triggered:
                    self._perception_active_until = max(
                        self._perception_active_until,
                        step + int(self.config["PERCEPTION_ACTIVE_HOLD_FRAMES"]),
                    )

                perception_active = step <= self._perception_active_until

                if perception_module is not None:
                    run_every = (
                        int(self.config["PERCEPTION_ACTIVE_EVERY"]) if perception_active
                        else int(self.config["PERCEPTION_IDLE_EVERY"])
                    )
                    if step % run_every == 0:
                        frame_bgr = self._get_rgb_frame(env)
                        if frame_bgr is not None:
                            raw = perception_module.process_frame(frame_bgr)
                            perception_output = {
                                "best_action": raw.get("best_action"),
                                "action_risks": raw.get("action_risks", {}),
                                "detections": raw.get("detections", []),
                                "avg_prediction_iou": raw.get("avg_prediction_iou"),
                                "ego_box": raw.get("ego_box"),
                                "max_ego_iou": raw.get("max_ego_iou", 0.0),
                                "danger": raw.get("danger", False),
                                "danger_threshold": raw.get("danger_threshold"),
                                "danger_objects": raw.get("danger_objects", []),
                                "ran_this_frame": True,
                                "perception_active": perception_active,
                                "lidar_activity": lidar_activity,
                            }
                            self._last_perception_output = perception_output
                elif perception_output is not None:
                    perception_output = dict(perception_output)
                    perception_output["ran_this_frame"] = False
                    perception_output["perception_active"] = perception_active
                    perception_output["lidar_activity"] = lidar_activity

                info = sim_out.get("info", {}) or {}
                info["perception"] = perception_output
                sim_out["info"] = info

                # --- Action selection ---
                obs_before = obs
                info_before = sim_out.get("info", {}) or {}

                model_action = solution.do_iteration(sim_out, user_input=user_input)
                teacher_action = teacher_solution.do_iteration(sim_out, user_input=user_input)

                data_mode = str(self.config.get("DATA_MODE", "model")).lower()
                human_override = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

                if data_mode == "human":
                    next_action = user_input
                    training_action = user_input
                    action_source = "human"
                elif data_mode == "controller":
                    next_action = teacher_action
                    training_action = teacher_action
                    action_source = "controller"
                elif data_mode == "dagger":
                    next_action = model_action
                    training_action = teacher_action
                    action_source = "dagger_controller"
                elif data_mode == "human_assist":
                    if human_override:
                        next_action = [float(user_input[0]), float(teacher_action[1])]
                        training_action = next_action
                        action_source = "human_correction"
                    else:
                        next_action = model_action
                        training_action = teacher_action
                        action_source = "dagger_controller"
                elif data_mode == "model":
                    next_action = model_action
                    training_action = model_action
                    action_source = "model"
                else:
                    raise ValueError(
                        f"DATA_MODE must be one of: human, controller, dagger, human_assist, model. Got: {data_mode!r}"
                    )

                next_sim_out = sim_out_to_dict(env.step(next_action))

                # --- Dataset logging ---
                self.dataset_writer.log(
                    episode_id=self.episode_id,
                    step=step,
                    source=action_source,
                    observation=obs_before,
                    action_steering=float(training_action[0]),
                    action_throttle=float(training_action[1]),
                    executed_steering=float(next_action[0]),
                    executed_throttle=float(next_action[1]),
                    human_override=bool(human_override),
                    human_steering=float(user_input[0]),
                    human_throttle=float(user_input[1]),
                    model_steering=float(model_action[0]),
                    model_throttle=float(model_action[1]),
                    teacher_steering=float(teacher_action[0]),
                    teacher_throttle=float(teacher_action[1]),
                    steering_disagreement=float(abs(model_action[0] - teacher_action[0])),
                    throttle_disagreement=float(abs(model_action[1] - teacher_action[1])),
                    navigation_command=info_before.get("navigation_command", "forward"),
                    reward=float(next_sim_out.get("reward", 0.0)),
                    terminated=bool(next_sim_out.get("terminated", False)),
                    truncated=bool(next_sim_out.get("truncated", False)),
                    next_observation=next_sim_out.get("observation"),
                    next_info=next_sim_out.get("info", {}),
                )

                sim_out = next_sim_out

                # --- Logger notify ---
                info = sim_out.get("info", {})
                self._notify_loggers(
                    step,
                    user_steering=steering,
                    user_throttle=throttle,
                    action_steering=float(next_action[0]),
                    action_throttle=float(next_action[1]),
                    reward=float(sim_out.get("reward", 0.0)),
                    terminated=bool(sim_out.get("terminated", False)),
                    truncated=bool(sim_out.get("truncated", False)),
                    speed=info.get("speed"),
                    velocity=info.get("velocity"),
                    position=info.get("position"),
                    crash=info.get("crash", False),
                    out_of_road=info.get("out_of_road", False),
                    arrive_dest=info.get("arrive_dest", False),
                    info=info,
                )

                # --- Pygame HUD ---
                pg_screen.fill((20, 20, 20))
                font = pygame.font.SysFont("monospace", 13)
                pg_screen.blit(
                    font.render(
                        f"S:{steering:+.2f}  T:{throttle:+.2f}  step:{step}",
                        True,
                        (200, 230, 200),
                    ),
                    (8, 22),
                )
                pygame.display.flip()

                # --- Episode reset ---
                if sim_out["terminated"] or sim_out["truncated"]:
                    self.episode_id += 1
                    print(f"\n[Episode ended at step {step}] Resetting …")
                    running = False
                    env.reset()
                    control.reset()

                step += 1

        except KeyboardInterrupt:
            print("\n[Interrupted by Ctrl+C]")

        finally:
            print("\n[Shutting down …]")
            env.close()
            pygame.quit()
            self._save_loggers()
            self._summarize_loggers()
            self.dataset_writer.close()

    def subscribe_logger(self, logger: ActionLogger):
        self.loggers.append(logger)

    def _notify_loggers(self, step: int, **kwargs):
        for logger in self.loggers:
            logger.log(step, **kwargs)

    def _save_loggers(self):
        for logger in self.loggers:
            logger.save()

    def _summarize_loggers(self):
        for logger in self.loggers:
            logger.summary()

    def subscribe_event_handler(self, event_type, handler):
        """
        Subscribe a callback for a pygame event type.

        Example:
            def on_keydown(event):
                if event.key == pygame.K_f:
                    print("F pressed")
            game.subscribe_event_handler(pygame.KEYDOWN, on_keydown)
        """
        self.handlers[event_type] = handler

    def unsubscribe_event_handler(self, event_type):
        self.handlers.pop(event_type, None)
