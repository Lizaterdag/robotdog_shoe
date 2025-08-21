import math
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any

# =========================================================
# Config
# =========================================================
@dataclass
class SACConfig:
    # Observation / action
    obs_dim: int
    act_dim: int

    # Control + episode
    control_hz: float = 50.0
    episodes: int = 50
    episode_seconds: float = 180.0
    warmup_seconds: float = 1.0
    reset_cooldown_s: float = 1.0

    # Sound
    db_calibration_offset: float = 120
    device_index: int = 10

    # Reward weights
    w_spl: float = 1.0
    w_band: float = 0.5
    w_foot_impact: float = 0.1
    w_joint_vel: float = 0.01
    w_energy: float = 0.001
    w_posture: float = 0.2
    w_bonus_upright: float = 0.5
    target_speed_mps: float = 0.0

    # SAC hyperparameters
    gamma: float = 0.995
    tau: float = 0.005            # target smoothing coefficient
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1_000_000
    start_steps: int = 2_000       # steps of random policy for better coverage
    updates_per_step: int = 1      # gradient updates per environment step
    policy_update_freq: int = 1    # update policy every N critic updates

    # Entropy temperature (learned)
    alpha_init: float = 0.2
    target_entropy_scale: float = 0.5  # multiplies -act_dim (0.5 is conservative)

    # Action bounding
    max_abs_torque: float = 18.0
    action_clip: float = 1.0  # env expects actions in [-1, 1]

    # Safety / posture
    max_joint_vel: float = 10.0
    base_pitch_roll_limit: float = math.radians(25)
    base_height_limits: Tuple[float, float] = (0.20, 0.40)

    # Device / seeds / IO
    device: str = "cpu"
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    run_name: str = "sac_go1_quiet"

    # Plotting
    plot_file: str = "training_curves.png"  # updated every episode