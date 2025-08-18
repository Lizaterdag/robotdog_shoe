"""
ppo_quiet_go1.py

Proximal Policy Optimization (PPO) training loop tailored for a Unitree Go1
running in low-level mode to minimize acoustic noise while maintaining balance
and gentle locomotion.

This version integrates the Go1QuietInterface (robot + mic) and uses an updated
reward with both A-weighted SPL and low-frequency footstep band RMS.
"""
from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd

from go1_interface import Go1QuietInterface

# ============================
# Config
# ============================
@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int
    control_hz: float = 50.0
    episode_seconds: float = 10.0
    warmup_seconds: float = 1.0
    reset_cooldown_s: float = 1.0
    w_spl: float = 1.0
    w_band: float = 0.5
    w_foot_impact: float = 0.1
    w_joint_vel: float = 0.01
    w_energy: float = 0.001
    w_posture: float = 0.2
    w_bonus_upright: float = 0.5
    target_speed_mps: float = 0.0
    rollout_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 10
    gamma: float = 0.995
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    action_std_init: float = 0.2
    action_std_min: float = 0.05
    action_std_decay: float = 0.9995
    max_abs_torque: float = 18.0
    max_joint_vel: float = 10.0
    base_pitch_roll_limit: float = math.radians(25)
    base_height_limits: Tuple[float, float] = (0.20, 0.40)
    device: str = "cpu"


# ============================
# Running Normalization
# ============================
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

# ============================
# Actor‑Critic Network
# ============================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, action_std_init=0.2):
        super().__init__()
        hidden = 256
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * math.log(action_std_init))

    def forward(self, x):
        z = self.shared(x)
        return self.mu(z), self.v(z)

    def dist(self, x):
        mu, _ = self.forward(x)
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            mu = torch.zeros_like(mu)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def value(self, x):
        _, v = self.forward(x)
        return v.squeeze(-1)

# ============================
# Rollout Buffer
# ============================
class RolloutBuffer:
    def __init__(self, steps: int, obs_dim: int, act_dim: int, device: str):
        self.obs = torch.zeros((steps, obs_dim), device=device)
        self.acts = torch.zeros((steps, act_dim), device=device)
        self.rews = torch.zeros(steps, device=device)
        self.dones = torch.zeros(steps, device=device)
        self.vals = torch.zeros(steps, device=device)
        self.logps = torch.zeros(steps, device=device)
        self.ptr = 0
        self.max = steps

    def add(self, obs, act, rew, done, val, logp):
        if self.ptr >= self.max:
            return  # prevent overflow
        i = self.ptr
        self.obs[i] = obs
        self.acts[i] = act
        self.rews[i] = rew
        self.dones[i] = done
        self.vals[i] = val
        self.logps[i] = logp
        self.ptr += 1

    def full(self):
        return self.ptr >= self.max

    def reset(self):
        self.ptr = 0

# ============================
# Observation / Reward
# ============================
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

def compute_reward(prev: Dict[str, Any], curr: Dict[str, Any], cfg: PPOConfig) -> Tuple[float, Dict[str, float]]:
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

# ============================
# PPO Trainer
# ============================
class PPOTrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = ActorCritic(cfg.obs_dim, cfg.act_dim, cfg.action_std_init).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.obs_norm = RunningNorm(cfg.obs_dim)

    def update(self, buf: RolloutBuffer, last_val: torch.Tensor):
        cfg = self.cfg
        with torch.no_grad():
            rews = buf.rews
            vals = buf.vals
            dones = buf.dones
            adv = torch.zeros_like(rews)
            lastgaelam = 0
            for t in reversed(range(buf.ptr)):
                next_nonterminal = 1.0 - (dones[t] if t < buf.ptr - 1 else 0.0)
                next_value = last_val if t == buf.ptr - 1 else vals[t+1]
                delta = rews[t] + cfg.gamma * next_value * next_nonterminal - vals[t]
                lastgaelam = delta + cfg.gamma * cfg.lam * next_nonterminal * lastgaelam
                adv[t] = lastgaelam
            ret = adv + vals
        obs = buf.obs[:buf.ptr]
        acts = buf.acts[:buf.ptr]
        old_logps = buf.logps[:buf.ptr]
        adv = (adv[:buf.ptr] - adv[:buf.ptr].mean()) / (adv[:buf.ptr].std() + 1e-8)
        ret = ret[:buf.ptr]

        batch_size = obs.shape[0]
        minibatch_size = batch_size // cfg.num_minibatches
        idxs = torch.randperm(batch_size, device=self.device)

        for _ in range(cfg.update_epochs):
            for mb in range(cfg.num_minibatches):
                mb_idx = idxs[mb*minibatch_size:(mb+1)*minibatch_size]
                mb_obs = obs[mb_idx]
                mb_acts = acts[mb_idx]
                mb_old_logps = old_logps[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]

                dist = self.net.dist(mb_obs)
                logps = dist.log_prob(mb_acts).sum(-1)
                ratio = torch.exp(logps - mb_old_logps)
                clip_adv = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                loss_pi = -(torch.min(ratio * mb_adv, clip_adv)).mean()
                v = self.net.value(mb_obs)
                loss_v = 0.5 * (mb_ret - v).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()

                loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

        with torch.no_grad():
            self.net.log_std.data = torch.clamp(self.net.log_std.data, math.log(self.cfg.action_std_min), 2.0)

    def act(self, obs_np: np.ndarray) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.from_numpy(obs_np).to(self.device)
        obs_t = self.obs_norm.normalize(obs_t)
        dist = self.net.dist(obs_t)
        act_t = dist.sample()
        logp = dist.log_prob(act_t).sum(-1)
        val = self.net.value(obs_t)
        return act_t.cpu().numpy(), float(logp.item()), float(val.item())

# ============================
# Training Loop
# ============================
def train_on_robot(env: Go1QuietInterface, cfg: PPOConfig):
    device = torch.device(cfg.device)
    trainer = PPOTrainer(cfg)
    buf = RolloutBuffer(cfg.rollout_steps, cfg.obs_dim, cfg.act_dim, device)

    steps_per_ep = int(cfg.episode_seconds * cfg.control_hz)
    warmup_steps = int(cfg.warmup_seconds * cfg.control_hz)
    dt = 1.0 / cfg.control_hz

    ep = 0
    while True:
        input(f"Checkpoint ep{ep}. Check battery levels and make sure the go1 hangs level with all four legs extended. Press Enter to continue...")
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

            act, logp, val = trainer.act(obs)
            act = np.clip(act, -1.0, 1.0)

            nxt, _, done_flag, _ = env.step(act)

            r, _ = compute_reward(prev, nxt, cfg)

            buf.add(
                obs=torch.from_numpy(obs).to(device),
                act=torch.from_numpy(act).to(device),
                rew=torch.tensor(r, device=device),
                done=torch.tensor(1.0 if done_flag else 0.0, device=device),
                val=torch.tensor(val, device=device),
                logp=torch.tensor(logp, device=device),
            )

            ep_return += r
            step += 1
            prev = nxt
            obs = obs_vector(nxt)

            if done_flag:
                break

            elapsed = time.time() - t0
            target = step * dt
            sleep = target - elapsed
            if sleep > 0:
                time.sleep(sleep)

        with torch.no_grad():
            last_val = torch.tensor(trainer.net.value(trainer.obs_norm.normalize(torch.from_numpy(obs).to(device))).item(), device=device)

        if buf.full() or step < steps_per_ep:
            trainer.update(buf, last_val)
            buf.reset()

        ep += 1
        print(f"Episode {ep} return={ep_return:.2f} steps={step}, db={obs[-3]}, low band rms={obs[-2]}")


# ============================
# Mic utility to auto-detect supported sample rate
# ============================
def pick_supported_rate(device_index: int, candidates=(16000, 22050, 32000, 44100, 48000)) -> int:
    for rate in candidates:
        try:
            sd.check_input_settings(device=device_index, samplerate=rate)
            print(f"[Mic] Using supported sample rate: {rate} Hz")
            return rate
        except Exception:
            continue
    print("[Mic] Falling back to 44100 Hz (default).")
    return 44100

# ============================
# Main
# ============================
if __name__ == "__main__":
    sample_rate = pick_supported_rate(13)
    env = Go1QuietInterface(pose_file="real_world/stand_pose.npy", device_index=13, sample_rate=sample_rate)
    env.start_mic()
    first_obs = env.reset()
    obs_dim = obs_vector(first_obs).shape[0]
    act_dim = 12
    cfg = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)

    try:
        train_on_robot(env, cfg)
    finally:
        env.stop_mic()
