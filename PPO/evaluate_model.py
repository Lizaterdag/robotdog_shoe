# evaluate_model.py  — no CLI args needed
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from quiet_dog_env import QuietDogEnv

MODEL_PATH = "quiet_dog_ppo_model_foot_contact2.zip"
VECNORM_PATH = "quiet_dog_vecnormalize_foot_contact2.pkl"  
VIDEO_DIR = "./eval_videos"
VIDEO_NAME_PREFIX = "ppo_eval5"
N_EPS = 5
VIDEO_LENGTH = 2000      
RENDER_MODE = "rgb_array"  # rgb_array for video, "human" for live viewer, or None
DETERMINISTIC = True


def main():
   
    venv = DummyVecEnv([lambda: Monitor(QuietDogEnv(render_mode=RENDER_MODE))])

    if VECNORM_PATH and os.path.isfile(VECNORM_PATH):
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, training=False, norm_obs=False, norm_reward=False)


    if RENDER_MODE == "rgb_array":
        os.makedirs(VIDEO_DIR, exist_ok=True)
        venv = VecVideoRecorder(
            venv,
            video_folder=VIDEO_DIR,
            record_video_trigger=lambda ep_id: True, 
            video_length=VIDEO_LENGTH,
            name_prefix=VIDEO_NAME_PREFIX,
        )


    model = PPO.load(MODEL_PATH, env=venv, print_system_info=True)


    mean_reward, std_reward = evaluate_policy(
        model, venv, n_eval_episodes=N_EPS, deterministic=DETERMINISTIC
    )
    print(f"Mean reward: {mean_reward:.3f} | Std: {std_reward:.3f}")

    venv.close()

if __name__ == "__main__":
    main()
