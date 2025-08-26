import numpy as np
import time, sys, math, json, os
import sounddevice as sd
import scipy.signal
from typing import Tuple, Dict, Any
from sac_config import SACConfig
from optitrack import Optitrack
sys.path.append('/home/lilly/robot_shoe/unitree_legged_sdk/lib/python/amd64/')
import robot_interface as sdk


class Go1QuietEnv:
    LOWLEVEL = 0xff
    CTRL_HZ_DEFAULT = 50.0
    DT_MIN = 1.0 / 400.0
    DT_MAX = 1.0 / 20.0

    TORQUE_SCALE = 0.1  # fraction of MAX_TORQUE

    ACTION_SCALE = np.array([
        0.35, 0.70, 0.60,   # FL: abd, hip, knee
        0.35, 0.70, 0.60,   # FR
        0.35, 0.70, 0.60,   # RL
        0.35, 0.70, 0.60    # RR
    ], dtype=np.float32)

    MAX_TORQUE = np.array([
        10.0, 20.0, 20.0,  # front-left leg
        10.0, 20.0, 20.0,  # front-right leg
        10.0, 20.0, 20.0,  # rear-left leg
        10.0, 20.0, 20.0   # rear-right leg
    ], dtype=np.float32)

    # PD gains (soft)
    KP_STAND = 12.0
    KD_STAND = 1.0

    # Policy PD gains (a bit stiffer)
    KP_POLICY = 20.0
    KD_POLICY = 1.2

    # Torque clamp
    TORQUE_SCALE = 0.0  

    #Foot contact threshold (N)
    CONTACT_THRESH = 5.0

    # Supervisor thresholds
    NEED_CONTACTS = 3               # >= 3 feet in contact…
    CONTACT_STREAK_TO_WALK = 10     
    AIRBORNE_MAX_STEPS = 2          

    # Action filtering
    ACTION_LP_ALPHA = 0.3           # EMA smoothing for policy actions


    def __init__(self, pose_file, device_index=10, db_calibration_offset=0, sample_rate=48000, control_hz=CTRL_HZ_DEFAULT):
        # --- Robot setup ---
        self.nominal_q = np.load(pose_file).astype(float)
        assert self.nominal_q.size == 12, "pose file must contain 12 joint angles"
        self.dt = float(np.clip(1.0 / control_hz, self.DT_MIN, self.DT_MAX))
        self._step = 0
        self.device_index = device_index
        self.db_calibration_offset = db_calibration_offset
        self.sample_rate = sample_rate
        self.samples_per_step = int(sample_rate * self.dt)

        # state for supervisor
        self.mode = "GROUND"  # "GROUND" or "WALK"
        self.contact_streak = 0
        self.airborne_steps = 0
        self.prev_contacts = np.zeros(4, dtype=bool)
        self.prev_position = None  # for velocity estimate
        self.prev_time = None

        # action filter state
        self.prev_action = np.zeros(12, dtype=np.float32)

        # UDP/SDK objects
        self.udp = sdk.UDP(self.LOWLEVEL, 8080, "192.168.123.10", 8007)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)

        # mic
        self.mic_buf = np.zeros((0,), dtype=np.float32)
        self.stream = None
        self.low_band_b, self.low_band_a = scipy.signal.butter(
            4, [50/(sample_rate/2), 200/(sample_rate/2)], btype='band')

        # external tracking
        self.optitrack = Optitrack()

        # logs
        self.positions = {}
        self.db_vals = {}
        self.rms_vals = {}
        self.observations = {}

    def design_a_weighting(self, fs: int):
        f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
        A1000 = 1.9997
        NUMs = [(2*math.pi*f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
        DENs = np.polymul([1, 4*math.pi*f4, (2*math.pi*f4)**2], [1, 4*math.pi*f1, (2*math.pi*f1)**2])
        DENs = np.polymul(np.polymul(DENs, [1, 2*math.pi*f3]), [1, 2*math.pi*f2])
        from scipy.signal import bilinear
        return bilinear(NUMs, DENs, fs)

    def start_mic(self):
        print(f"[MIC] starting mic index={self.device_index}, offset={self.db_calibration_offset}")

        def callback(indata, frames, time_info, status):
            if status:
                print("[MIC] Status:", status)
            self.mic_buf = np.concatenate((self.mic_buf, indata[:, 0]))

        self.stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            callback=callback)
        self.stream.start()

    def stop_mic(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def compute_db_and_rms(self):
        mic_samples = self._consume_mic_window()
        if mic_samples.size == 0:
            rms = 0.0
            db_a = -np.inf
            low_band_rms = 0.0
        else:
            rms = np.sqrt(np.mean(mic_samples**2))
            db = 20*np.log10(rms+1e-12) + self.db_calibration_offset
            b, a = self.design_a_weighting(self.sample_rate)
            weighted = scipy.signal.lfilter(b, a, mic_samples)
            rms_a = np.sqrt(np.mean(weighted**2))
            db_a = 20*np.log10(rms_a+1e-12) + self.db_calibration_offset
            low = scipy.signal.lfilter(self.low_band_b, self.low_band_a, mic_samples)
            low_band_rms = np.sqrt(np.mean(low**2))
        return db_a, low_band_rms

    def _consume_mic_window(self):
        if self.mic_buf.size >= self.samples_per_step:
            window = self.mic_buf[:self.samples_per_step]
            self.mic_buf = self.mic_buf[self.samples_per_step:]
            return window
        return np.zeros((0,), dtype=np.float32)
    
    # ------------------------ RESET/STAND ------------------------
    def reset(self):
        # Stand up with ramp (keeps original behavior)
        # KP0, KD0 = 4.0, 0.3
        # KPH, KDH = 12.0, 1.0
        # RAMP_SEC, HOLD_SEC = 3.0, 1.0
        # t0 = time.time()
        # while True:
        #     now = time.time()
        #     t = now - t0
        #     self.udp.Recv()
        #     self.udp.GetRecv(self.state)
        #     if t < RAMP_SEC:
        #         alpha = t / RAMP_SEC
        #         kp = KP0 + alpha * (KPH - KP0)
        #         kd = KD0 + alpha * (KDH - KD0)
        #         for i in range(12):
        #             q_target = self.state.motorState[i].q * (1 - alpha) + self.pose[i] * alpha
        #             m = self.cmd.motorCmd[i]
        #             m.q, m.dq, m.Kp, m.Kd, m.tau = q_target, 0.0, kp, kd, 0.0
        #     elif t < RAMP_SEC + HOLD_SEC:
        #         for i in range(12):
        #             m = self.cmd.motorCmd[i]
        #             m.q, m.dq, m.Kp, m.Kd, m.tau = self.pose[i], 0.0, KPH, KDH, 0.0
        #     else:
        #         break
        #     self.safe.PowerProtect(self.cmd, self.state, 1)
        #     self.udp.SetSend(self.cmd)
        #     self.udp.Send()
        #     time.sleep(self.dt)

        # Flush mic buffer so first step starts fresh
        # self.mic_buf = np.zeros((0,), dtype=np.float32)
        # # reset airborne counter
        # self.airborne_steps = 0

        # # Read a fresh state and return initial obs
        # self.udp.Recv()
        # self.udp.GetRecv(self.state)
        
        # return self._get_obs(-1)

        def reset(self):
            self.mic_buf = np.zeros((0,), dtype=np.float32)
            self._step = 0
            self.mode = "GROUND"
            self.contact_streak = 0
            self.airborne_steps = 0
            self.prev_contacts[:] = False
            self.prev_position = None
            self.prev_time = None
            self.prev_action[:] = 0.0

            # Read state
            self.udp.Recv()
            self.udp.GetRecv(self.state)

            # Immediately command nominal stand
            for i in range(12):
                m = self.cmd.motorCmd[i]
                m.q = float(self.nominal_q[i])
                m.dq = 0.0
                m.Kp = self.KP_STAND
                m.Kd = self.KD_STAND
                m.tau = 0.0
            self.safe.PowerProtect(self.cmd, self.state, 1)
            self.udp.SetSend(self.cmd)
            self.udp.Send()

            time.sleep(self.dt)

            return self._get_obs(-1)

    # ------------------------ STEP ------------------------
    # def step(self, action, episode):
    #     # First update robot state BEFORE computing targets
    #     self.udp.Recv()
    #     self.udp.GetRecv(self.state)

    #     current_q = np.array([m.q for m in self.state.motorState[:12]])

    #     # Compute new targets
    #     q_targets = current_q + np.clip(action, -1.0, 1.0) * self.SCALE
    #     tau_targets = np.clip(action, -1.0, 1.0) * self.MAX_TORQUE * self.TORQUE_SCALE

    #     for i in range(12):
    #         m = self.cmd.motorCmd[i]
    #         # hybrid control: PD target with small torque injection
    #         m.q, m.dq, m.Kp, m.Kd, m.tau = q_targets[i], 0.0, 12.0, 1.0, float(tau_targets[i])

    #     self.safe.PowerProtect(self.cmd, self.state, 1)
    #     self.udp.SetSend(self.cmd)
    #     self.udp.Send()

    #     # Wait for the effect of the command, then read state again so obs reflect new contact/forces
    #     time.sleep(self.dt)
    #     self.udp.Recv()
    #     self.udp.GetRecv(self.state)

    #     self._step += 1

    #     obs = self._get_obs(episode)

    #     # airborne detection: require N consecutive steps with no contact
    #     contact_count = float(np.array(obs["contacts"]).sum())
    #     if contact_count == 0:
    #         self.airborne_steps += 1
    #     else:
    #         self.airborne_steps = 0

    #     done_tilt = self._check_tilt(obs)
    #     done_airborne = (self.airborne_steps >= self.airborne_max_steps)
    #     done = bool(done_tilt or done_airborne)

    #     return obs, 0.0, done, {}

    def step(self, action, episode):
        # Read current state first
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        # Get fresh contacts
        contacts_bool = self._read_contacts_bool()
        contact_count = int(np.sum(contacts_bool))

        # Supervisor transitions
        if self.mode == "GROUND":
            if contact_count >= self.NEED_CONTACTS:
                self.contact_streak += 1
                if self.contact_streak >= self.CONTACT_STREAK_TO_WALK:
                    self.mode = "WALK"
                    self.contact_streak = 0
            else:
                self.contact_streak = 0
        else:  # WALK
            if contact_count == 0:
                self.airborne_steps += 1
                if self.airborne_steps >= self.AIRBORNE_MAX_STEPS:
                    self.mode = "GROUND"
                    self.airborne_steps = 0
            else:
                self.airborne_steps = 0

        # Compute command according to mode
        if self.mode == "GROUND":
            # Ignore policy: hold nominal stance (feet down)
            q_targets = self.nominal_q.copy()
            kp, kd = self.KP_STAND, self.KD_STAND
            tau_targets = np.zeros(12, dtype=np.float32)
        else:
            # WALK: smooth the action and map to absolute targets around nominal
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            a = self.ACTION_LP_ALPHA * a + (1.0 - self.ACTION_LP_ALPHA) * self.prev_action
            self.prev_action = a

            q_targets = self.nominal_q + self.ACTION_SCALE * a
            kp, kd = self.KP_POLICY, self.KD_POLICY

            if self.TORQUE_SCALE > 0.0:
                # optional torque injection (rarely needed for first trials)
                tau_targets = np.clip(a, -1.0, 1.0) * self.TORQUE_SCALE
                tau_targets = tau_targets.astype(np.float32)
            else:
                tau_targets = np.zeros(12, dtype=np.float32)

        # Send to robot
        for i in range(12):
            m = self.cmd.motorCmd[i]
            m.q = float(q_targets[i])
            m.dq = 0.0
            m.Kp = kp
            m.Kd = kd
            m.tau = float(tau_targets[i])
        self.safe.PowerProtect(self.cmd, self.state, 1)
        self.udp.SetSend(self.cmd)
        self.udp.Send()

        # Wait, then read state again so obs reflect new contacts & motion
        time.sleep(self.dt)
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        self._step += 1

        obs = self._get_obs(episode)

        # Terminations: tilt or fully airborne for a couple steps
        done_tilt = self._check_tilt(obs)
        if int(np.sum(obs["contacts"])) == 0:
            self.airborne_steps += 1
        else:
            self.airborne_steps = 0
        done_airborne = (self.airborne_steps >= self.AIRBORNE_MAX_STEPS)

        done = bool(done_tilt or done_airborne)
        return obs, 0.0, done, {}

    # ------------------------ OBSERVATION ------------------------
    def _get_obs(self, episode):
        roll = float(self.state.imu.rpy[0])
        pitch = float(self.state.imu.rpy[1])
        yaw = float(self.state.imu.rpy[2])
        gyro = np.array(self.state.imu.gyroscope, dtype=np.float32)
        acc = np.array(self.state.imu.accelerometer, dtype=np.float32)
        q = np.array([m.q for m in self.state.motorState[:12]], dtype=np.float32)
        dq = np.array([m.dq for m in self.state.motorState[:12]], dtype=np.float32)

        # torques (estimated if available)
        taus = []
        for m in self.state.motorState[:12]:
            tau_val = getattr(m, 'tauEst', None)
            if tau_val is None:
                tau_val = getattr(m, 'tau', 0.0)
            taus.append(tau_val)
        tau = np.array(taus, dtype=np.float32)

        # foot force contacts
        contacts_bool = self._read_contacts_bool()
        contacts = contacts_bool.astype(np.float32)

        # mic features
        db_a, low_band_rms = self.compute_db_and_rms()

        # OptiTrack pos & velocity (x forward)
        position = self._read_position()
        base_vel_x = 0.0
        tnow = time.time()
        if self.prev_position is not None and self.prev_time is not None:
            dt = max(1e-3, tnow - self.prev_time)
            base_vel_x = float((position[0] - self.prev_position[0]) / dt)
        self.prev_position = position.copy()
        self.prev_time = tnow

        # crude contact impulse proxy: sum of changes in foot contact binary
        contact_impulse = float(np.sum(np.abs(contacts_bool.astype(int) - self.prev_contacts.astype(int))))
        self.prev_contacts = contacts_bool

        observations = {
            "base_rpy": (roll, pitch, yaw),
            "base_gyro": gyro,
            "base_acc": acc,
            "q": q,
            "dq": dq,
            "tau": tau,
            "contacts": contacts,
            "mic_feats": np.array([db_a, low_band_rms], dtype=np.float32),
            "position": position,
            "base_vel_x": base_vel_x,
            "contact_impulse": contact_impulse,
            "mode": self.mode,
            "fall": False,
        }

        # store traces
        self._append_or_add(self.positions, episode, position)
        self._append_or_add(self.db_vals, episode, db_a)
        self._append_or_add(self.rms_vals, episode, low_band_rms)
        if episode in self.observations:
            self.observations[episode][self._step] = observations
        else:
            self.observations[episode] = {self._step: observations}

        return observations

    def obs_vector(self, obs: Dict[str, Any]) -> np.ndarray:
        r, p, y = obs["base_rpy"]
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

    # ------------------------ REWARD ------------------------
    def compute_reward(self, prev, curr, cfg: SACConfig):
        # --- sound penalties ---
        db_a = float(curr["mic_feats"][0])
        quiet_threshold = 40.0
        loud_threshold = 80.0
        db_norm = np.clip((db_a - quiet_threshold) / (loud_threshold - quiet_threshold), 0, 1)
        r_spl = -0.1 * cfg.w_spl * db_norm

        low_band = float(curr["mic_feats"][1])
        r_band = -0.05 * cfg.w_band * np.clip(low_band / 50.0, 0, 1)

        # --- impacts / energy / smoothness ---
        contact_impulse = float(curr.get("contact_impulse", 0.0))
        r_impact = -0.05 * cfg.w_foot_impact * np.tanh(contact_impulse)

        dq = np.asarray(curr["dq"][:12], dtype=np.float32)
        r_dq = -0.001 * cfg.w_joint_vel * np.mean(np.abs(dq))

        tau = np.asarray(curr.get("tau", np.zeros_like(dq)), dtype=np.float32)
        r_energy = -0.0005 * cfg.w_energy * float(np.mean(np.abs(tau * dq)))

        # --- posture ---
        roll, pitch, _ = curr["base_rpy"]
        pr = -0.05 * cfg.w_posture * (abs(roll) + abs(pitch))

        # --- forward motion ---
        vx = float(curr.get("base_vel_x", 0.0))
        r_track = cfg.w_movement * vx

        r_move = 0.0
        if prev is not None and ("position" in prev) and ("position" in curr):
            try:
                dx = float(curr['position'][0] - prev['position'][0])
                dy = float(curr['position'][1] - prev['position'][1])
                r_move = cfg.w_movement * (dx + abs(dy))
            except Exception:
                r_move = 0.0

        # --- ground contact ---
        contacts = np.array(curr["contacts"])
        contact_count = float(np.sum(contacts))
        r_contact = cfg.w_contact * contact_count

        # airborne penalty vs upright bonus
        r_airborne = -cfg.w_airborne if contact_count == 0 else cfg.w_bonus_upright

        # --- alternating gait bonus (simple diagonal pattern) ---
        fl, fr, rl, rr = contacts
        diag1 = fl + rr
        diag2 = fr + rl
        r_gait = cfg.w_gait * float(((diag1 == 2) and (diag2 == 0)) or ((diag2 == 2) and (diag1 == 0)))

        # --- joint limit penalty ---
        q = np.asarray(curr["q"][:12], dtype=np.float32)
        q_min, q_max = -1.5, 1.5
        r_joint_limit = -0.1 * float(np.sum((q < q_min) | (q > q_max)))

        total = (
            r_spl + r_band + r_impact + r_dq + r_energy + pr +
            r_airborne + r_track + r_move + r_contact + r_gait + r_joint_limit
        )

        info = dict(
            r_spl=r_spl, r_band=r_band, r_impact=r_impact,
            r_dq=r_dq, r_energy=r_energy, r_posture=pr,
            r_alive=r_airborne, r_track=r_track, r_move=r_move,
            r_contact=r_contact, r_gait=r_gait, r_joint_limit=r_joint_limit,
            db_a=db_a, low_band=low_band, vx=vx, mode=curr.get("mode", "NA")
        )
        return float(total), info

    # ------------------------ SAFETY ------------------------
    def _check_tilt(self, obs):
        roll, pitch, _ = obs["base_rpy"]
        return (abs(roll) > np.radians(25)) or (abs(pitch) > np.radians(25))

    # ------------------------ HELPERS ------------------------
    def _read_contacts_bool(self) -> np.ndarray:
        """Return boolean array (4,) for foot contact using foot force sensors."""
        try:
            ff = np.array([fs for fs in self.state.footForce], dtype=np.float32)
            return ff > self.CONTACT_THRESH
        except Exception:
            return np.zeros((4,), dtype=bool)

    def _read_position(self) -> np.ndarray:
        try:
            pos = self.optitrack.optiTrackGetPos()
            if isinstance(pos, (tuple, list)):
                return np.array(pos[0], dtype=np.float32)
            return np.array(pos, dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _append_or_add(self, dict_obj, key, val):
        if key in dict_obj:
            dict_obj[key].append(val)
        else:
            dict_obj[key] = [val]

    # ------------------------ SAVE ------------------------
    def save_observations(self, run_name, dir):
        os.makedirs(dir, exist_ok=True)
        pos = {k: np.array(v).tolist() for k, v in self.positions.items()}
        with open(os.path.join(dir, f"{run_name}_positions.json"), "w") as f:
            json.dump(pos, f)

        dbs = {k: np.array(v).tolist() for k, v in self.db_vals.items()}
        with open(os.path.join(dir, f"{run_name}_db.json"), "w") as f:
            json.dump(dbs, f)

        rms = {k: np.array(v).tolist() for k, v in self.rms_vals.items()}
        with open(os.path.join(dir, f"{run_name}_rms.json"), "w") as f:
            json.dump(rms, f)

        obs = {}
        for ep, epdata in self.observations.items():
            for key, val in epdata.items():
                data = {}
                for k, v in val.items():
                    data[k] = np.array(v).tolist()
                obs[key] = data

        with open(os.path.join(dir, f"{run_name}_obs.json"), "w") as f:
            json.dump(obs, f)