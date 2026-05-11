import pygame


class ControlState:
    """Manages steering/throttle values with ramping & asymmetric release."""

    def __init__(self, cfg: dict):
        self.steer_step = cfg["STEER_STEP"]
        self.steer_decay = cfg["STEER_DECAY"]
        self.throttle_step = cfg["THROTTLE_STEP"]
        self.throttle_decay = cfg["THROTTLE_DECAY"]
        self.brake_value = cfg["BRAKE_VALUE"]
        self.reset_on_space = cfg["RESET_ON_SPACE"]

        self.steering = 0.0
        self.throttle = 0.0
        self._brake_active = False  # tracks whether S was just released

    def reset(self):
        self.steering = 0.0
        self.throttle = 0.0
        self._brake_active = False

    def update(self, keys):
        """
        Call once per frame with the current pygame key state dict.
        Returns (steering, throttle) clamped to [-1, 1].
        """
        steer_left = keys[pygame.K_a]
        steer_right = keys[pygame.K_d]
        gas = keys[pygame.K_w]
        brake = keys[pygame.K_s]
        space = keys[pygame.K_SPACE]

        # --- SPACE: instant reset ---
        if space and self.reset_on_space:
            self.reset()
            return self.steering, self.throttle

        # --- STEERING ---
        if steer_right and not steer_left:
            self.steering = max(-1.0, self.steering - self.steer_step)
        elif steer_left and not steer_right:
            self.steering = min(1.0, self.steering + self.steer_step)
        else:
            # Slow decay toward 0
            if self.steering > 0:
                self.steering = max(0.0, self.steering - self.steer_decay)
            elif self.steering < 0:
                self.steering = min(0.0, self.steering + self.steer_decay)

        # --- THROTTLE / BRAKE ---
        if brake:
            # S pressed: hard kill throttle, apply brake value
            self.throttle = self.brake_value
            self._brake_active = True
        elif gas:
            # W pressed: ignore _brake_active flag, ramp up
            self._brake_active = False
            self.throttle = min(1.0, self.throttle + self.throttle_step)
        else:
            if self._brake_active:
                # S was just released → snap to 0, no memory of old throttle
                self.throttle = 0.0
                self._brake_active = False
            else:
                # Neither gas nor brake: slow decay toward 0
                if self.throttle > 0:
                    self.throttle = max(0.0, self.throttle - self.throttle_decay)
                elif self.throttle < 0:
                    self.throttle = min(0.0, self.throttle + self.throttle_decay)

        # Safety clamp
        self.steering = max(-1.0, min(1.0, self.steering))
        self.throttle = max(-1.0, min(1.0, self.throttle))

        return self.steering, self.throttle
