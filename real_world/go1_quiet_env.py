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
    SCALE = 0.05  # rad max delta per step
    TORQUE_SCALE = 0.1  # fraction of MAX_TORQUE
    MAX_TORQUE = np.array([
        10.0, 20.0, 20.0,  # front-left leg
        10.0, 20.0, 20.0,  # front-right leg
        10.0, 20.0, 20.0,  # rear-left leg
        10.0, 20.0, 20.0   # rear-right leg
    ], dtype=np.float32)

    def __init__(self, pose_file, device_index=10, db_calibration_offset=0, sample_rate=48000, control_hz=50.0):
        # --- Robot setup ---
        self.pose = np.load(pose_file).astype(float)
        assert self.pose.size == 12, "pose file must contain 12 joint angles"
        self.dt = 1.0 / control_hz
        self._step = 0
        self.device_index = device_index
        self.db_calibration_offset = db_calibration_offset
        self.sample_rate = sample_rate
        self.samples_per_step = int(sample_rate * self.dt)
        self.optitrack = Optitrack()

        # UDP/SDK objects
        self.udp = sdk.UDP(self.LOWLEVEL, 8080, "192.168.123.10", 8007)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)

        # mic buffer
        self.mic_buf = np.zeros((0,), dtype=np.float32)
        self.stream = None

        # Design low-pass for footstep band
        self.low_band_b, self.low_band_a = scipy.signal.butter(
            4, [50/(sample_rate/2), 200/(sample_rate/2)], btype='band')

        # data storage
        self.positions = {}
        self.db_vals = {}
        self.rms_vals = {}
        self.observations = {}

        # airborne termination counter
        self.airborne_steps = 0
        self.airborne_max_steps = 2  # require N consecutive steps with no contact

    # ------------------------ MIC ------------------------
    def design_a_weighting(self, fs: int):
        """Return (b, a) IIR filter coefficients approximating A-weighting."""
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
            # append new mono samples
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
            # overall RMS -> dB
            rms = np.sqrt(np.mean(mic_samples**2))
            db = 20*np.log10(rms+1e-12) + self.db_calibration_offset
            # A-weighting filter
            b, a = self.design_a_weighting(self.sample_rate)
            weighted = scipy.signal.lfilter(b, a, mic_samples)
            rms_a = np.sqrt(np.mean(weighted**2))
            db_a = 20*np.log10(rms_a+1e-12) + self.db_calibration_offset
            # low-frequency band RMS
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
        self.mic_buf = np.zeros((0,), dtype=np.float32)
        # reset airborne counter
        self.airborne_steps = 0

        # Read a fresh state and return initial obs
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        
        return self._get_obs(-1)

    # ------------------------ STEP ------------------------
    def step(self, action, episode):
        # First update robot state BEFORE computing targets
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        current_q = np.array([m.q for m in self.state.motorState[:12]])

        # Compute new targets
        q_targets = current_q + np.clip(action, -1.0, 1.0) * self.SCALE
        tau_targets = np.clip(action, -1.0, 1.0) * self.MAX_TORQUE * self.TORQUE_SCALE

        for i in range(12):
            m = self.cmd.motorCmd[i]
            # hybrid control: PD target with small torque injection
            m.q, m.dq, m.Kp, m.Kd, m.tau = q_targets[i], 0.0, 12.0, 1.0, float(tau_targets[i])

        self.safe.PowerProtect(self.cmd, self.state, 1)
        self.udp.SetSend(self.cmd)
        self.udp.Send()

        # Wait for the effect of the command, then read state again so obs reflect new contact/forces
        time.sleep(self.dt)
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        self._step += 1

        obs = self._get_obs(episode)

        # airborne detection: require N consecutive steps with no contact
        contact_count = float(np.array(obs["contacts"]).sum())
        if contact_count == 0:
            self.airborne_steps += 1
        else:
            self.airborne_steps = 0

        done_tilt = self._check_tilt(obs)
        done_airborne = (self.airborne_steps >= self.airborne_max_steps)
        done = bool(done_tilt or done_airborne)

        return obs, 0.0, done, {}

    # ------------------------ OBSERVATION ------------------------
    def _get_obs(self, episode):
        # Basic proprio
        roll = float(self.state.imu.rpy[0])
        pitch = float(self.state.imu.rpy[1])
        yaw = float(self.state.imu.rpy[2])
        gyro = np.array(self.state.imu.gyroscope)
        acc = np.array(self.state.imu.accelerometer)
        q = np.array([m.q for m in self.state.motorState[:12]])
        dq = np.array([m.dq for m in self.state.motorState[:12]])
        # try to read estimated torques if available
        taus = []
        for m in self.state.motorState[:12]:
            tau_val = getattr(m, 'tauEst', None)
            if tau_val is None:
                tau_val = getattr(m, 'tau', 0.0)
            taus.append(tau_val)
        taus = np.array(taus, dtype=np.float32)

        # contact detection from foot force sensors
        try:
            contacts = np.array([fs for fs in self.state.footForce], dtype=np.float32) > 5.0
        except Exception:
            # fallback if structure differs
            contacts = np.zeros((4,), dtype=np.float32)

        # Mic features over last control window
        db_a, low_band_rms = self.compute_db_and_rms()

        # Optitrack position (may return (pos, quat) or just pos)
        try:
            pos = self.optitrack.optiTrackGetPos()
            if isinstance(pos, tuple) or isinstance(pos, list):
                position = np.array(pos[0], dtype=np.float32)
            else:
                position = np.array(pos, dtype=np.float32)
        except Exception:
            position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        observations = {
            "base_rpy": (roll, pitch, yaw),
            "base_gyro": gyro,
            "base_acc": acc,
            "q": q,
            "dq": dq,
            "tau": taus,
            "contacts": contacts.astype(np.float32),
            "mic_feats": np.array([db_a, low_band_rms], dtype=np.float32),
            "position": position,
            "fall": False
        }

        # store time-series data
        self._append_or_add(self.positions, episode, position)
        self._append_or_add(self.db_vals, episode, db_a)
        self._append_or_add(self.rms_vals, episode, low_band_rms)
        if episode in self.observations.keys():
            self.observations[episode][self._step] = observations
        else:
            self.observations[episode] = {self._step: observations}

        return observations

    def obs_vector(self, obs: Dict[str, Any]) -> np.ndarray:
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

    # ------------------------ REWARD ------------------------
    def compute_reward(self, prev, curr, cfg: SACConfig):
        # --- sound penalties (normalized) ---
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

        # --- posture & upright bonus / airborne penalty ---
        roll, pitch, _ = curr["base_rpy"]
        pr = -0.05 * cfg.w_posture * (abs(roll) + abs(pitch))

        # --- forward motion / movement ---
        vx = float(curr.get("base_vel_x", 0.0))
        r_track = cfg.w_movement * vx

        # use optitrack displacement if available
        r_move = 0.0
        if prev is not None and ("position" in prev) and ("position" in curr):
            try:
                dx = float(curr['position'][0] - prev['position'][0])
                dy = float(curr['position'][1] - prev['position'][1])
                r_move = cfg.w_movement * (dx + abs(dy))
            except Exception:
                r_move = 0.0

        # --- ground contact ---
        contacts = np.array(curr["contacts"])  # shape (4,)
        contact_count = float(np.sum(contacts))
        r_contact = cfg.w_contact * contact_count

        # airborne penalty (termination handled in step)
        if contact_count == 0:
            r_airborne = -cfg.w_airborne
        else:
            r_airborne = cfg.w_bonus_upright

        # --- alternating gait reward ---
        fl, fr, rl, rr = contacts
        diag1 = fl + rr
        diag2 = fr + rl
        # reward when one diagonal pair is down and the other up
        r_gait = cfg.w_gait * float(((diag1 == 2) and (diag2 == 0)) or ((diag2 == 2) and (diag1 == 0)))

        # --- joint limit penalty (prevent extreme joint angles / crossing) ---
        q = np.asarray(curr["q"][:12], dtype=np.float32)
        q_min, q_max = -1.5, 1.5
        r_joint_limit = -0.1 * float(np.sum((q < q_min) | (q > q_max)))

        # --- total reward ---
        total = (
            r_spl + r_band + r_impact + r_dq + r_energy + pr +
            r_airborne + r_track + r_move + r_contact + r_gait + r_joint_limit
        )

        info = dict(
            r_spl=r_spl, r_band=r_band, r_impact=r_impact,
            r_dq=r_dq, r_energy=r_energy, r_posture=pr,
            r_alive=r_airborne, r_track=r_track, r_move=r_move,
            r_contact=r_contact, r_gait=r_gait, r_joint_limit=r_joint_limit,
            db_a=db_a, low_band=low_band, vx=vx
        )
        return float(total), info

    # ------------------------ SAFETY ------------------------
    def _check_tilt(self, obs):
        roll, pitch, _ = obs["base_rpy"]
        if abs(roll) > np.radians(25) or abs(pitch) > np.radians(25):
            return True
        return False

    def _check_safety(self, obs):
        # kept for compatibility; main termination uses tilt + airborne counter
        return self._check_tilt(obs)

    # ------------------------ UTILS ------------------------
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
