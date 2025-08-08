from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from quiet_dog_env import QuietDogEnv

def make_env():
    return Monitor(QuietDogEnv(render_mode=None))

if __name__ == "__main__":
    venv = DummyVecEnv([make_env])
    # Normalize obs & rewards for more stable PPO training
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        n_epochs=10,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tb_logs_ppo",
    )

    model.learn(total_timesteps=1_000_000)

    model.save("quiet_dog_ppo_model")
    venv.save("quiet_dog_vecnormalize.pkl")
    venv.close()
