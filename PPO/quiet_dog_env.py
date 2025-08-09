import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

from gymnasium import spaces
from mujoco import MjModel, MjData, Renderer


class QuietDogEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, xml_path="go1_with_shoes.xml"):
        super().__init__()
        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)

        self.sim_steps = 5
        self.max_episode_steps = 1000
        self.current_step = 0
        self.render_mode = render_mode
        self.viewer = None


        assert self.model.nu == 12, f"Expected 12 actuators, got {self.model.nu}"
        self.ctrl_range = np.array(self.model.actuator_ctrlrange, dtype=np.float32)


        self.act_joint_id = self.model.actuator_trnid[:, 0].astype(int)
        self.jnt_qposadr = self.model.jnt_qposadr.copy().astype(int)


        CANON = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        pairs = []
        for i, j_id in enumerate(self.act_joint_id):
            name = self.model.joint(int(j_id)).name
            if name not in CANON:
                raise ValueError(f"Actuator {i} targets unexpected joint '{name}'")
            addr = int(self.jnt_qposadr[int(j_id)])
            pairs.append((CANON.index(name), i, addr))
        pairs.sort(key=lambda x: x[0])

        self.canon_names = CANON
        self.act_indices_in_model_order = np.array([p[1] for p in pairs], dtype=int)
        self.act_qpos_addr = np.array([p[2] for p in pairs], dtype=int)

        def _canon_to_model(a):
            out = np.empty_like(a)
            out[self.act_indices_in_model_order] = a
            return out
        self._canon_to_model = _canon_to_model

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # observation = qpos + qvel + last_action + v_cmd
        self.include_last_action = True
        self.include_command = True
        base_dim = self.model.nq + self.model.nv
        extra = (12 if self.include_last_action else 0) + (1 if self.include_command else 0)
        obs_dim = base_dim + extra
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Per-joint delta scaling
        self.per_joint_delta = np.array([0.08, 0.15, 0.10] * 4, dtype=np.float32)

       
        self.settle_steps = 60
        self._settle = 0

        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_action_diff = np.zeros_like(self.last_action)

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.trunk_id = self.model.body("trunk").id
        self.foot_site_ids = [self.model.site(n).id for n in ["FR_1", "FL_1", "RR_1", "RL_1"]]
        self.foot_body_ids = [self.model.body(n).id for n in ["FR_shoe", "FL_shoe", "RR_shoe", "RL_shoe"]]

        #encouragin bottom-of-shoe contact
        self.floor_geom_id = self.model.geom("floor").id
        self.silicone_geom_ids = {
            self.model.geom("FR_silicone").id,
            self.model.geom("FL_silicone").id,
            self.model.geom("RR_silicone").id,
            self.model.geom("RL_silicone").id,
        }

        thigh_bodies = ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]
        thigh_geom_ids = []
        for bname in thigh_bodies:
            bid = self.model.body(bname).id
            adr = self.model.body_geomadr[bid]
            num = self.model.body_geomnum[bid]
            thigh_geom_ids.extend(range(adr, adr + num))
        self.thigh_geom_ids = set(thigh_geom_ids)
        self._thigh_contact_streak = 0
        self._thigh_contact_streak_limit = 12  

        self.v_cmd = 0.4
        self.v_cmd_min = 0.0
        self.v_cmd_max = 0.6
        self.v_sigma = 0.2
        self.stall_speed = 0.05
        self.no_progress_steps = 0
        self.no_progress_limit = 200

    def _read_actuated_qpos(self) -> np.ndarray:
        return self.data.qpos[self.act_qpos_addr].copy()

    def _get_obs(self) -> np.ndarray:
        base = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        pieces = [base]
        if self.include_last_action:
            pieces.append(self.last_action.astype(np.float32))
        if self.include_command:
            pieces.append(np.array([self.v_cmd], dtype=np.float32))
        return np.concatenate(pieces, dtype=np.float32)

    def _is_fallen(self) -> bool:
        z = float(self.data.qpos[2])
        R = self.data.xmat[self.trunk_id].reshape(3, 3)
        up_z = float(R[2, 2])
        return (z < 0.20) or (up_z < 0.6)

    def _compute_reward(self) -> float:
        # encouraging standin upright
        R = self.data.xmat[self.trunk_id].reshape(3, 3)
        up_z = float(R[2, 2])
        roll_pitch_pen = 1.0 - up_z

        #velocities
        vx, vy, _ = self.data.qvel[0:3]
        _, _, wz = self.data.qvel[3:6]
        vx = float(vx)
        side_vel_pen = abs(float(vy))
        yaw_rate_pen = abs(float(wz))

        #impact shoe body
        impact = 0.0
        for bid in self.foot_body_ids:
            f = self.data.cfrc_ext[bid, :3]
            impact += float(np.dot(f, f))

        
        q = self._read_actuated_qpos()
        abd = q[[0, 3, 6, 9]]
        abd_sym_pen = float(np.sum(np.abs(abd)))

        lat_spread_pen = 0.0
        for sid in self.foot_site_ids:
            lat_spread_pen += abs(float(self.data.site_xpos[sid][1]))

       
        ctrl_pen = float(np.sum(np.square(self.data.ctrl)))
        rate_pen = float(np.sum(np.square(self.last_action_diff)))

       
        silicone_contact_count = 0.0
        silicone_normal_force = 0.0
        thigh_floor_force = 0.0
        thigh_floor_count = 0.0
        fbuf = np.zeros(6, dtype=np.float64)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            
            if ((g1 in self.silicone_geom_ids and g2 == self.floor_geom_id) or
                (g2 in self.silicone_geom_ids and g1 == self.floor_geom_id)):
                silicone_contact_count += 1.0
                mujoco.mj_contactForce(self.model, self.data, i, fbuf)
                silicone_normal_force += max(0.0, float(fbuf[2]))

           
            if ((g1 in self.thigh_geom_ids and g2 == self.floor_geom_id) or
                (g2 in self.thigh_geom_ids and g1 == self.floor_geom_id)):
                thigh_floor_count += 1.0
                mujoco.mj_contactForce(self.model, self.data, i, fbuf)
                thigh_floor_force += max(0.0, float(fbuf[2]))

       
        shoe_contact_reward = 0.05 * silicone_contact_count + 0.0001 * silicone_normal_force
        thigh_contact_pen = 0.5 * thigh_floor_count + 0.001 * thigh_floor_force  # tune: 0.3–1.0 & 0.0005–0.003

        
        track = np.exp(-0.5 * ((vx - self.v_cmd) / self.v_sigma) ** 2)
        track_reward = 1.5 * float(track)

        
        gait_shape = np.exp(-0.5 * ((silicone_contact_count - 2.0) / 0.75) ** 2)
        gait_reward = 0.2 * float(gait_shape)

        
        stall_pen = 0.02 if abs(vx) < self.stall_speed else 0.0

        reward = 0.0
        reward += 0.8 * up_z
        reward += 1.0 * vx
        reward += track_reward
        reward += gait_reward
        reward += shoe_contact_reward
        reward -= 0.2 * roll_pitch_pen
        reward -= 0.1 * side_vel_pen
        reward -= 0.05 * yaw_rate_pen
        reward -= 0.01 * impact
        reward -= 0.05 * abd_sym_pen
        reward -= 0.02 * lat_spread_pen
        reward -= 0.001 * ctrl_pen
        reward -= 0.002 * rate_pen
        reward -= stall_pen
        reward -= thigh_contact_pen     

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        
        self.v_cmd = float(np.random.uniform(self.v_cmd_min, self.v_cmd_max))
        self.no_progress_steps = 0
        self._thigh_contact_streak = 0

        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
            self.data.qpos[2] += 0.03
        else:
            self.data.qpos[:] = 0.0
            self.data.qpos[2] = 0.48

        self.data.qvel[:] = 0.0

        q_init = self._read_actuated_qpos()
        if self.model.nkey > 0 and self.model.key_ctrl.shape[0] > 0:
            self.data.ctrl[:] = self.model.key_ctrl[0]
        else:
            self.data.ctrl[:] = self._canon_to_model(q_init)

        self.last_action[:] = 0.0
        self.last_action_diff[:] = 0.0

        self._settle = self.settle_steps
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        if self._settle > 0:
            self._settle -= 1
            for _ in range(self.sim_steps):
                mujoco.mj_step(self.model, self.data)
            self.current_step += 1
            self.last_action_diff = -self.last_action
            self.last_action[:] = 0.0
            obs = self._get_obs()
            terminated = False
            truncated = self.current_step >= self.max_episode_steps
            return obs, 0.0, terminated, truncated, {}

        
        q_now = self._read_actuated_qpos()
        q_des = q_now + self.per_joint_delta * action

        #clip to canonical cotrl ranges
        low_canon = self.ctrl_range[self.act_indices_in_model_order, 0]
        high_canon = self.ctrl_range[self.act_indices_in_model_order, 1]
        q_des = np.clip(q_des, low_canon, high_canon)

        self.data.ctrl[:] = self._canon_to_model(q_des)

        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        new_last = action.copy()
        self.last_action_diff = new_last - self.last_action
        self.last_action = new_last


        vx = float(self.data.qvel[0])
        if abs(vx) < self.stall_speed:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        
        thigh_touch = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if ((g1 in self.thigh_geom_ids and g2 == self.floor_geom_id) or
                (g2 in self.thigh_geom_ids and g1 == self.floor_geom_id)):
                thigh_touch = True
                break
        if thigh_touch:
            self._thigh_contact_streak += 1
        else:
            self._thigh_contact_streak = 0

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_fallen() or (self.no_progress_steps >= self.no_progress_limit) \
                     or (self._thigh_contact_streak >= self._thigh_contact_streak_limit)
        truncated = self.current_step >= self.max_episode_steps
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.viewer is None:
            self.viewer = Renderer(self.model)
        self.viewer.update_scene(self.data)
        return self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
