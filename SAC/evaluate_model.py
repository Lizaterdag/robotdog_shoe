from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.evaluation import evaluate_policy

from quiet_dog_env import QuietDogEnv
from stable_baselines3 import SAC



env = QuietDogEnv(render_mode="rgb_array")
env = Monitor(env)
env = RecordVideo(env, video_folder="./eval_videos", name_prefix="eval", episode_trigger=lambda x: True)
env = DummyVecEnv([lambda: env])

model = SAC.load("/Users/liza/robotdog_shoe/SAC/quiet_dog_model.zip", env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

print(f"Mean reward: {mean_reward}, Std: {std_reward}")

env.close()
