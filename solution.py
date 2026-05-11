import numpy as np
import torch

from teacher_solution import TeacherSolution
from training.train_bc import DrivingMLP


class Solution:
    """
    Hybrid driving agent: BC neural network for steering, rule-based teacher for throttle.

    The MLP was trained via behavioural cloning on lidar+state observations collected
    from the teacher controller. At inference time, only the first 19 state features
    are used for steering — lidar rays are masked out so the model generalises purely
    from road geometry. Throttle is always delegated to TeacherSolution.
    """

    def __init__(self, game):
        self._game = game
        self.device = torch.device("cpu")
        self.teacher = TeacherSolution(game)

        checkpoint_path = "models/1.pt"
        # weights_only=False is required because the checkpoint stores NumPy
        # normalisation arrays alongside the model state dict.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = DrivingMLP(input_dim=checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.obs_mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float32)

    @property
    def config(self):
        return {"image_observation": False}

    def do_iteration(self, simulator_output, user_input=None):
        obs = simulator_output.get("observation", None)

        if not isinstance(obs, np.ndarray) or len(obs) != 259:
            return [0.0, 0.0]

        obs_for_model = obs.astype(np.float32).copy()
        # Mask obstacle/lidar channels — steering should depend only on road geometry.
        obs_for_model[19:] = 1.0

        obs_norm = (obs_for_model - self.obs_mean) / self.obs_std
        x = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.model(x).cpu().numpy()[0]

        model_steer = float(np.clip(action[0], -1.0, 1.0))

        teacher_action = self.teacher.do_iteration(simulator_output, user_input=user_input)
        controller_throttle = float(np.clip(teacher_action[1], -1.0, 1.0))

        return [model_steer, controller_throttle]
