import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

from gymnasium import spaces
from mujoco import MjModel, MjData, Renderer


class QuietDogEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = MjModel.from_xml_path("go1_with_shoes.xml")
        self.data = MjData(self.model)
        self.sim_steps = 5  # Number of sim steps per env step
        self.render_mode = render_mode
        self.viewer = None

        self.max_episode_steps = 1000
        self.current_step = 0


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.nq + self.model.nv,),
            dtype=np.float32,
        )

        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        #used for contact force measurement
        self.foot_body_ids = [
            self.model.body(name).id for name in ["FR_shoe", "FL_shoe", "RR_shoe", "RL_shoe"]
        ]

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self):
        #forward velocity x-pos
        forward_vel = self.data.qvel[0]

        #sum of squared contact forces on all 4 legs
        total_impact = sum(
            np.linalg.norm(self.data.cfrc_ext[bid][:3]) ** 2 for bid in self.foot_body_ids
        )

        energy_penalty = np.sum(np.square(self.data.ctrl))

        reward = forward_vel - 0.01 * total_impact - 0.001 * energy_penalty
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
        else:
            self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False  # Add your own logic here (e.g., fall detection)
        truncated = self.current_step >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return

        if self.viewer is None:
            self.viewer = Renderer(self.model)

        self.viewer.update_scene(self.data)
        img = self.viewer.render()
        return img


    def close(self):
        if self.viewer:
            self.viewer.close()
