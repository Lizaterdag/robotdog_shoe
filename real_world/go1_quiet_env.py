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
    def __init__(self, pose_file, device_index, sample_rate=48000, control_hz=50.0):
        # --- Robot setup ---
        self.pose = np.load(pose_file).astype(float)
        assert self.pose.size == 12, "pose file must contain 12 joint angles"
        self.dt = 1.0 / control_hz
        self._step = 0
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.samples_per_step = int(sample_rate * self.dt)
        self.optitrack = Optitrack()

        LOWLEVEL = 0xff
        self.udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.cmd   = sdk.LowCmd()
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


    def design_a_weighting(self, fs: int):
        """Return (b, a) IIR filter coefficients approximating A-weighting."""
        f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
        A1000 = 1.9997
        NUMs = [(2*math.pi*f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
        DENs = np.polymul([1, 4*math.pi*f4, (2*math.pi*f4)**2], [1, 4*math.pi*f1, (2*math.pi*f1)**2])
        DENs = np.polymul(np.polymul(DENs, [1, 2*math.pi*f3]), [1, 2*math.pi*f2])
        b, a = NUMs, DENs
        from scipy.signal import bilinear
        return bilinear(b, a, fs)

    def start_mic(self):
        def callback(indata, frames, time_info, status):
            if status:
                print("[Mic] Status:", status)
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

    def reset(self):
        # Stand up with ramp
        KP0, KD0 = 4.0, 0.3
        KPH, KDH = 12.0, 1.0
        RAMP_SEC, HOLD_SEC = 3.0, 1.0
        t0 = time.time()
        while True:
            now = time.time()
            t = now - t0
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            if t < RAMP_SEC:
                alpha = t / RAMP_SEC
                kp = KP0 + alpha * (KPH - KP0)
                kd = KD0 + alpha * (KDH - KD0)
                for i in range(12):
                    q_target = self.state.motorState[i].q * (1 - alpha) + self.pose[i] * alpha
                    m = self.cmd.motorCmd[i]
                    m.q, m.dq, m.Kp, m.Kd, m.tau = q_target, 0.0, kp, kd, 0.0
            elif t < RAMP_SEC + HOLD_SEC:
                for i in range(12):
                    m = self.cmd.motorCmd[i]
                    m.q, m.dq, m.Kp, m.Kd, m.tau = self.pose[i], 0.0, KPH, KDH, 0.0
            else:
                break
            self.safe.PowerProtect(self.cmd, self.state, 1)
            self.udp.SetSend(self.cmd)
            self.udp.Send()
            time.sleep(self.dt)

        # Flush mic buffer so first step starts fresh
        self.mic_buf = np.zeros((0,), dtype=np.float32)

        return self._get_obs(-1)

    def step(self, action, episode):
        # Apply PD target as pose + scaled delta
        SCALE = 0.05  # rad max delta
        KPH, KDH = 12.0, 1.0
        q_targets = self.pose + np.clip(action, -1.0, 1.0) * SCALE
        self.udp.Recv()
        self.udp.GetRecv(self.state)
        for i in range(12):
            m = self.cmd.motorCmd[i]
            m.q, m.dq, m.Kp, m.Kd, m.tau = q_targets[i], 0.0, KPH, KDH, 0.0
        self.safe.PowerProtect(self.cmd, self.state, 1)
        self.udp.SetSend(self.cmd)
        self.udp.Send()

        time.sleep(self.dt)
        self._step += 1

        obs = self._get_obs(episode)
        done = self._check_safety(obs)
        return obs, 0.0, done, {}  # reward handled externally

    def _get_obs(self, episode):
        # Basic proprio
        roll = self.state.imu.rpy[0]
        pitch = self.state.imu.rpy[1]
        yaw = self.state.imu.rpy[2]
        gyro = np.array(self.state.imu.gyroscope)
        acc = np.array(self.state.imu.accelerometer)
        q = np.array([m.q for m in self.state.motorState])
        dq = np.array([m.dq for m in self.state.motorState])
        contacts = np.array([fs for fs in self.state.footForce], dtype=np.float32) > 5.0

        # Mic features over last control window
        mic_samples = self._consume_mic_window()
        if mic_samples.size == 0:
            rms = 0.0
            db_a = -np.inf
            low_band_rms = 0.0
        else:
            # overall RMS -> dB
            rms = np.sqrt(np.mean(mic_samples**2))
            db = 20*np.log10(rms+1e-12)
            # A-weighting filter
            b, a = self.design_a_weighting(self.sample_rate)
            weighted = scipy.signal.lfilter(b, a, mic_samples)
            rms_a = np.sqrt(np.mean(weighted**2))
            db_a = 20*np.log10(rms_a+1e-12)
            # low-frequency band RMS
            low = scipy.signal.lfilter(self.low_band_b, self.low_band_a, mic_samples)
            low_band_rms = np.sqrt(np.mean(low**2))

        mic_feats = np.array([db_a, low_band_rms], dtype=np.float32)

        position = self.optitrack.optiTrackGetPos()

        observations = {
            "base_rpy": (roll, pitch, yaw),
            "base_gyro": gyro,
            "base_acc": acc,
            "q": q,
            "dq": dq,
            "contacts": contacts.astype(np.float32),
            "mic_feats": mic_feats,
            "position": position,
            "fall": False  # replace with tilt check if desired
        }

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

    def compute_reward(self, prev: Dict[str, Any], curr: Dict[str, Any], cfg: SACConfig) -> Tuple[float, Dict[str, float]]:
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
    
    def save_observations(self, run_name, dir):
        pos = {k: np.array(v).tolist() for k,v in self.positions.items()}
        pos_path = os.path.join(dir, f"{run_name}_positions.json")
        with open(pos_path, "w") as f:
            json.dump(pos, f)

        dbs = {k: np.array(v).tolist() for k,v in self.db_vals.items()}
        db_path = os.path.join(dir, f"{run_name}_db.json")
        with open(db_path, "w") as f:
            json.dump(dbs, f)

        rms = {k: np.array(v).tolist() for k,v in self.rms_vals.items()}
        rms_path = os.path.join(dir, f"{run_name}_rms.json")
        with open(rms_path, "w") as f:
            json.dump(rms, f)

        obs = {}
        for ep, epdata in self.observations.items():
            for key,val in epdata.items():
                data = {}
                for k,v in val.items():
                    data[k] = np.array(v).tolist()
                obs[key] = data

        with open(os.path.join(dir, f"{run_name}_obs.json"), "w") as f:
            json.dump(obs, f)


    def _consume_mic_window(self):
        if self.mic_buf.size >= self.samples_per_step:
            window = self.mic_buf[:self.samples_per_step]
            self.mic_buf = self.mic_buf[self.samples_per_step:]
            return window
        else:
            # Not enough samples yet
            return np.zeros((0,), dtype=np.float32)

    def _check_safety(self, obs):
        roll, pitch, _ = obs["base_rpy"]
        if abs(roll) > np.radians(25) or abs(pitch) > np.radians(25):
            return True
        return False


    def _append_or_add(self, dict, key, val):
        if key in dict:
            dict[key].append(val)
        else:
            dict[key] = [val]