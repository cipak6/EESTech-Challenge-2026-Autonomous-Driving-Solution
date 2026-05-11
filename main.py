"""
EESTech Challenge 2026 — Team Solution Entry Point
====================================================
Starts the MetaDrive simulation with our hybrid BC + rule-based agent,
perception pipeline, and dataset collection.

Controls:
    A / D     — Steer left / right
    W         — Accelerate
    S         — Brake
    Q / ESC   — Quit and save log
    LSHIFT    — Human override (in human_assist mode)
"""

import time
from datetime import datetime
import os

from logger import ActionLogger
from game import Game


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join("logs", f"drive_log_{timestamp}.json")

    game = Game()
    game.subscribe_logger(ActionLogger(log_filename))
    game.start()


if __name__ == "__main__":
    main()
