from stable_baselines3 import SAC
from quiet_dog_env import QuietDogEnv

env = QuietDogEnv(render_mode=None)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("quiet_dog_model")
