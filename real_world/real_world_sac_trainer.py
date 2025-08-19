
"""
real_world_sac_trainer.py

Soft Actor-Critic (SAC) training loop tailored for a Unitree Go1 running in
low-level mode to minimize acoustic noise while maintaining balance and gentle
locomotion.

Changes from PPO version:
- Uses off-policy SAC with twin Q-critics and target networks
- Adds a large replay buffer with save/load to resume after failures
- Generates/updates live plots after every episode:
    (1) episode reward
    (2) dB(A) values
    (3) low-band RMS values
"""
from __future__ import annotations
import os
import time
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid GUI requirements
import matplotlib.pyplot as plt

from go1_interface import Go1QuietInterface

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
    episode_seconds: float = 10.0
    warmup_seconds: float = 1.0
    reset_cooldown_s: float = 1.0

    # Reward weights (kept from PPO config for continuity)
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


# =========================================================
# Running Normalization
# =========================================================
class RunningNorm:
    def __init__(self, shape, eps=1e-5):
        self.shape = shape
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.shape[-1] != self.shape:
            raise ValueError(f"Obs dimension mismatch: expected {self.shape}, got {x.shape[-1]}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            return  # skip invalid updates
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor):
        if x.shape[-1] != self.shape:
            raise ValueError(f"Obs dimension mismatch in normalize: expected {self.shape}, got {x.shape[-1]}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            return x  # bypass normalization for bad inputs
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


# =========================================================
# Networks
# =========================================================
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Gaussian policy with Tanh squashing: a = tanh(mu + std * eps)
    Returns actions in [-1, 1] and log probs corrected for tanh.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden = 256
        self.net = mlp([obs_dim, hidden, hidden, 2 * act_dim], activation=nn.Tanh)
        self.act_dim = act_dim
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, obs):
        h = self.net(obs)
        mu, log_std = torch.split(h, self.act_dim, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std, log_std

    def sample(self, obs):
        mu, std, log_std = self.forward(obs)
        # Reparameterization trick
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        # Log prob with tanh correction
        log_prob = (-0.5 * ((pre_tanh - mu) / (std + 1e-8))**2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # Tanh correction: sum log(1 - tanh(x)^2)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, log_prob

    @torch.no_grad()
    def act(self, obs):
        a, _ = self.sample(obs)
        return a


class Critic(nn.Module):
    """ Twin Q networks (Q1 and Q2) """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden = 256
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.Tanh)
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.Tanh)

    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        q1 = self.q1(xu)
        q2 = self.q2(xu)
        return q1.squeeze(-1), q2.squeeze(-1)


# =========================================================
# Replay Buffer (with save/load for resume)
# =========================================================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs[idxs],
                    acts=self.acts[idxs],
                    rews=self.rews[idxs],
                    next_obs=self.next_obs[idxs],
                    dones=self.dones[idxs])

    def save(self, path):
        np.savez_compressed(
            path,
            obs=self.obs[:self.size],
            acts=self.acts[:self.size],
            rews=self.rews[:self.size],
            next_obs=self.next_obs[:self.size],
            dones=self.dones[:self.size],
            size=self.size,
            ptr=self.ptr,
            max_size=self.max_size,
        )

    def load(self, path):
        data = np.load(path, allow_pickle=False)
        size = int(data["size"])
        self.obs[:size] = data["obs"]
        self.acts[:size] = data["acts"]
        self.rews[:size] = data["rews"]
        self.next_obs[:size] = data["next_obs"]
        self.dones[:size] = data["dones"]
        self.size = size
        self.ptr = int(data["ptr"])
        self.max_size = int(data["max_size"])


# =========================================================
# Observation / Reward (kept from PPO version)
# =========================================================
def obs_vector(obs: Dict[str, Any]) -> np.ndarray:
    base_rpy = obs["base_rpy"]
    r, p, y = base_rpy
    ori = [math.sin(y), math.cos(y), math.sin(p), math.cos(p), math.sin(r), math.cos(r)]
    gyro = obs["base_gyro"]
    acc = obs["base_acc"]
    q = obs["q"]
    dq = obs["dq"]
    contacts = obs["contacts"]
    mic = obs["mic_feats"]
    target_v = [obs.get("target_vx", 0.0)]
    vec = np.concatenate([ori, gyro, acc, q, dq, contacts, mic, target_v]).astype(np.float32)
    if not np.isfinite(vec).all():
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def compute_reward(prev: Dict[str, Any], curr: Dict[str, Any], cfg: SACConfig) -> Tuple[float, Dict[str, float]]:
    db_a = float(curr["mic_feats"][0])
    low_band = float(curr["mic_feats"][1])
    r_spl = -cfg.w_spl * db_a
    r_band = -cfg.w_band * low_band

    contact_impulse = float(curr.get("contact_impulse", 0.0))
    r_impact = -cfg.w_foot_impact * contact_impulse

    dq = np.asarray(curr["dq"], dtype=np.float32)
    r_dq = -cfg.w_joint_vel * float(np.linalg.norm(dq, ord=1))

    tau = np.asarray(curr.get("tau", np.zeros_like(dq)), dtype=np.float32)
    r_energy = -cfg.w_energy * float(np.sum(np.abs(tau * dq)))

    roll, pitch, _ = curr["base_rpy"]
    pr = cfg.w_posture * ( - (abs(roll) + abs(pitch)) )

    alive = cfg.w_bonus_upright if not curr.get("fall", False) else -1.0

    vx = float(curr.get("base_vel_x", 0.0))
    r_track = -0.1 * (vx - cfg.target_speed_mps) ** 2

    total = r_spl + r_band + r_impact + r_dq + r_energy + pr + alive + r_track
    info = {
        "r_spl": r_spl,
        "r_band": r_band,
        "r_impact": r_impact,
        "r_dq": r_dq,
        "r_energy": r_energy,
        "r_posture": pr,
        "r_alive": alive,
        "r_track": r_track,
        "db_a": db_a,
        "low_band": low_band,
    }
    return float(total), info


# =========================================================
# SAC Trainer
# =========================================================
class SACTrainer:
    def __init__(self, cfg: SACConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Networks
        self.actor = Actor(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.critic = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.critic_targ = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Entropy temperature (learned)
        self.log_alpha = torch.tensor(np.log(cfg.alpha_init), device=self.device, requires_grad=True)
        # Conservative target entropy: a fraction of -act_dim
        self.target_entropy = -cfg.act_dim * cfg.target_entropy_scale
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.lr)

        # Running normalization
        self.obs_norm = RunningNorm(cfg.obs_dim)

        # Replay buffer
        self.replay = ReplayBuffer(cfg.obs_dim, cfg.act_dim, size=cfg.replay_size)

        # Book-keeping
        self.total_env_steps = 0
        self.updates = 0

        # IO setup
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        # Plotting state
        self.rewards = []
        self.db_vals = []
        self.band_vals = []
        self.steps = []

        # Prepare figure (written to file each episode)
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 10))
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ---------------------------
    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ---------------------------
    def save_ckpt(self, suffix="latest"):
        path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.run_name}_{suffix}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_targ": self.critic_targ.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "obs_norm": {
                "mean": self.obs_norm.mean,
                "var": self.obs_norm.var,
                "count": self.obs_norm.count,
            },
            "total_env_steps": self.total_env_steps,
            "updates": self.updates,
            "cfg": asdict(self.cfg),
        }, path)
        # Save replay buffer
        self.replay.save(os.path.join(self.cfg.ckpt_dir, f"{self.cfg.run_name}_{suffix}_replay.npz"))

        # Save curves as JSON for quick glance
        curves = {
            "rewards": self.rewards,
            "db": self.db_vals,
            "low_band": self.band_vals,
            "steps": self.steps,
        }
        with open(os.path.join(self.cfg.ckpt_dir, f"{self.cfg.run_name}_{suffix}_curves.json"), "w") as f:
            json.dump(curves, f)

    # ---------------------------
    def load_ckpt_if_exists(self, suffix="latest"):
        path = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.run_name}_{suffix}.pt")
        if not os.path.isfile(path):
            return False
        do_load = input("Existing replay buffer found. Do you want to load it? (y/n)")
        if "n" in do_load:
            return False
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.critic_targ.load_state_dict(data["critic_targ"])
        self.actor_opt.load_state_dict(data["actor_opt"])
        self.critic_opt.load_state_dict(data["critic_opt"])
        self.log_alpha = data["log_alpha"].to(self.device).requires_grad_()
        self.alpha_opt.load_state_dict(data["alpha_opt"])

        on = data.get("obs_norm", None)
        if on is not None:
            self.obs_norm.mean = on["mean"]
            self.obs_norm.var = on["var"]
            self.obs_norm.count = on["count"]

        self.total_env_steps = int(data.get("total_env_steps", 0))
        self.updates = int(data.get("updates", 0))

        # Load replay
        rpath = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.run_name}_{suffix}_replay.npz")
        if os.path.isfile(rpath):
            self.replay.load(rpath)

        # Curves are optional
        cpath = os.path.join(self.cfg.ckpt_dir, f"{self.cfg.run_name}_{suffix}_curves.json")
        if os.path.isfile(cpath):
            try:
                with open(cpath, "r") as f:
                    curves = json.load(f)
                    self.rewards = curves.get("rewards", [])
                    self.db_vals = curves.get("db", [])
                    self.band_vals = curves.get("low_band", [])
                    self.steps = curves.get("steps", [])
            except Exception:
                pass
        print(f"[SAC] Loaded checkpoint from {path}")
        return True

    # ---------------------------
    def soft_update_targets(self):
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_targ.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)

    # ---------------------------
    def update(self, batch):
        cfg = self.cfg
        self.updates += 1

        obs = torch.as_tensor(batch["obs"], device=self.device)
        acts = torch.as_tensor(batch["acts"], device=self.device)
        rews = torch.as_tensor(batch["rews"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        dones = torch.as_tensor(batch["dones"], device=self.device)

        # Normalize observations
        obs_n = self.obs_norm.normalize(obs)
        next_obs_n = self.obs_norm.normalize(next_obs)

        # ---------------- Critic update ----------------
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs_n)
            q1_t, q2_t = self.critic_targ(next_obs_n, next_a)
            q_targ = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target = rews + cfg.gamma * (1.0 - dones) * q_targ

        q1, q2 = self.critic(obs_n, acts)
        loss_q = ((q1 - target).pow(2).mean() + (q2 - target).pow(2).mean())

        self.critic_opt.zero_grad(set_to_none=True)
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ---------------- Actor and alpha update (delayed) ----------------
        if self.updates % cfg.policy_update_freq == 0:
            a_pi, logp_pi = self.actor.sample(obs_n)
            q1_pi, q2_pi = self.critic(obs_n, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            loss_pi = (self.alpha * logp_pi - q_pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            # Temperature (alpha) update
            loss_alpha = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            loss_alpha.backward()
            self.alpha_opt.step()

            # Soft update targets
            self.soft_update_targets()

    # ---------------------------
    @torch.no_grad()
    def select_action(self, obs_np: np.ndarray, stochastic=True) -> np.ndarray:
        obs_t = torch.from_numpy(obs_np).to(self.device)
        obs_t = self.obs_norm.normalize(obs_t)
        if stochastic:
            a = self.actor.act(obs_t.unsqueeze(0)).squeeze(0)
        else:
            # Deterministic action: tanh(mu) with zero-noise path
            mu, _, _ = self.actor.forward(obs_t.unsqueeze(0))
            a = torch.tanh(mu).squeeze(0)
        return a.clamp(-self.cfg.action_clip, self.cfg.action_clip).cpu().numpy()

    def update_plots(self):
        # Save separate SVG plots for reward, db, and low-band RMS
        outdir = self.cfg.ckpt_dir

        # Reward
        fig = plt.figure()
        plt.plot(self.rewards)
        plt.title("Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        fig.savefig(os.path.join(outdir, "reward.svg"))
        plt.close(fig)

        # dB(A)
        fig = plt.figure()
        plt.plot(self.db_vals)
        plt.title("dB(A)")
        plt.xlabel("Episode")
        plt.ylabel("dB(A)")
        fig.savefig(os.path.join(outdir, "db.svg"))
        plt.close(fig)

        # Low-band RMS
        fig = plt.figure()
        plt.plot(self.band_vals)
        plt.title("Low-band RMS")
        plt.xlabel("Episode")
        plt.ylabel("RMS")
        fig.savefig(os.path.join(outdir, "low_band.svg"))
        plt.close(fig)

        # Episode steps
        fig = plt.figure()
        plt.plot(self.steps)
        plt.title("Episode steps")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        fig.savefig(os.path.join(outdir, "episode_steps.svg"))
        plt.close(fig)


# =========================================================
# Training Loop
# =========================================================
def train_on_robot(env: Go1QuietInterface, cfg: SACConfig, resume: bool = True):
    # Seeding
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    trainer = SACTrainer(cfg)
    if resume:
        trainer.load_ckpt_if_exists("latest")

    steps_per_ep = int(cfg.episode_seconds * cfg.control_hz)
    warmup_steps = int(cfg.warmup_seconds * cfg.control_hz)
    dt = 1.0 / cfg.control_hz

    ep = 0
    last_db, last_band = 0.0, 0.0

    while True:
        if ep != 0:
            input(f"Checkpoint ep{ep}. Check battery and hang Go1 level with all four legs extended. Press Enter to continue...")
        raw = env.reset()
        prev = raw
        obs = obs_vector(raw)
        t0 = time.time()
        step = 0
        ep_return = 0.0
        time.sleep(cfg.reset_cooldown_s)

        while step < steps_per_ep:
            obs_t = torch.from_numpy(obs).to(device)
            if step < warmup_steps:
                trainer.obs_norm.update(obs_t.unsqueeze(0))

            # Action selection (initial random steps)
            if trainer.total_env_steps < cfg.start_steps:
                act = np.random.uniform(low=-cfg.action_clip, high=cfg.action_clip, size=cfg.act_dim).astype(np.float32)
            else:
                act = trainer.select_action(obs, stochastic=True)
            act = np.clip(act, -cfg.action_clip, cfg.action_clip)

            # Step env
            nxt, _, done_flag, _ = env.step(act)

            # Reward
            r, info = compute_reward(prev, nxt, cfg)
            last_db, last_band = info["db_a"], info["low_band"]

            # Store transition
            next_obs_vec = obs_vector(nxt)
            trainer.replay.add(obs, act, r, next_obs_vec, float(done_flag))

            # Accumulate
            ep_return += r
            step += 1
            trainer.total_env_steps += 1
            prev = nxt
            obs = next_obs_vec

            # SAC updates (if enough data)
            if trainer.replay.size >= cfg.batch_size:
                for _ in range(cfg.updates_per_step):
                    batch = trainer.replay.sample(cfg.batch_size)
                    trainer.update(batch)

            if done_flag:
                break

            # Real-time pacing
            elapsed = time.time() - t0
            target = step * dt
            sleep = target - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # Episode end bookkeeping
        ep += 1
        trainer.rewards.append(float(ep_return))
        trainer.db_vals.append(float(last_db))
        trainer.band_vals.append(float(last_band))
        trainer.steps.append(float(step))

        print(f"Episode {ep} return={ep_return:.2f} steps={step}, db={last_db:.2f}, low band rms={last_band:.4f}")

        # Update plots and save checkpoints
        trainer.update_plots()
        trainer.save_ckpt("latest")




# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # NOTE: adjust device_index to your microphone index
    env = Go1QuietInterface(pose_file="real_world/stand_pose.npy", device_index=13)
    env.start_mic()

    try:
        first_obs = env.reset()
        obs_dim = obs_vector(first_obs).shape[0]
        act_dim = 12

        cfg = SACConfig(obs_dim=obs_dim, act_dim=act_dim)

        # Save an initial copy of config
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        with open(os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        train_on_robot(env, cfg, resume=True)
    finally:
        env.stop_mic()
