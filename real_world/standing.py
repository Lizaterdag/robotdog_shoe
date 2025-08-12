import sys, time, os, math, numpy as np
sys.path.append('/home/lilly/robot_shoe/unitree_legged_sdk/lib/python/amd64/')
import robot_interface as sdk

# ────────── parameters ──────────────────────────────────────
POSE_FILE   = 'real_world/stand_pose.npy'
RAMP_SEC    = 3.0            # limp → pose
HOLD_SEC    = 5.0            # pose hold
FREQ_HZ     = 1.2            # sweep frequency
KP0, KD0    = 4.0,  0.3      # soft gains at start
KPH, KDH    = 12.0, 1.0      # holding gains
# ─────────────────────────────────────────────────────────────


def execute_go1_movement_routine():
    if not os.path.isfile(POSE_FILE):
        raise FileNotFoundError(f'{POSE_FILE} missing – run save_stand_pose.py first')
    POSE = np.load(POSE_FILE).astype(float)
    assert POSE.size == 12, "pose file must contain 12 joint angles"

    # joint map
    d = {f'{leg}_{j}': i
        for i,(leg,j) in enumerate([(l,k) for l in ('FR','FL','RR','RL') for k in range(3)])}

    LOWLEVEL = 0xff
    DT       = 0.002
    udp  = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)

    cmd   = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    t0       = time.time()

    print("► standup_a.py running – robot must be in LOW-LEVEL and hanging.")

    while True:
        loop_t = time.time()
        udp.Recv();  udp.GetRecv(state)

        t = loop_t - t0

        # 1) soft ramp from limp to stand pose
        if t < RAMP_SEC:
            α  = t / RAMP_SEC
            kp = KP0 + α * (KPH - KP0)
            kd = KD0 + α * (KDH - KD0)
            for i in range(12):
                q_target = state.motorState[i].q * (1 - α) + POSE[i] * α
                m = cmd.motorCmd[i]
                m.q, m.dq, m.Kp, m.Kd, m.tau = q_target, 0.0, kp, kd, 0.0

        # 2) hold pose (no swing yet)
        elif t < RAMP_SEC + HOLD_SEC:
            for i in range(12):
                m = cmd.motorCmd[i]
                m.q, m.dq, m.Kp, m.Kd, m.tau = POSE[i], 0.0, KPH, KDH, 0.0

        # 3) execute the walking part
        else:
            pass

        if t > 1.0:  safe.PowerProtect(cmd, state, 1)
        udp.SetSend(cmd);  udp.Send()

        time.sleep(max(0.0, DT - (time.time() - loop_t)))


if __name__ == "__main__":
    execute_go1_movement_routine()