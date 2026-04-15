import os
import torch
import numpy as np

from src.methods import load_model
from src.envs import make_single_walker_env


# Configuración--
ALGO = "sac"   # "ppo" o "sac"
ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DATE = "Apr01_17_17_22"  
LOAD_STEP =  5_000_000
MODEL_PATH = f"runs/{MODEL_DATE}/{ALGO}_walker2d_step{LOAD_STEP}.pt"
VIDEO_DIR = f"runs/{MODEL_DATE}/videos_eval"
N_EPISODES = 20

SEED = 42
IMAGE_SIZE = 84
FRAME_STACK = 4
REWARD_SHAPING = True
TERMINATE_WHEN_UNHEALTHY = True
HEALTHY_Z_RANGE = (0.8, 2.0)
DETERMINISTIC = False

USE_DISCRETE_ACTIONS = False  

os.makedirs(VIDEO_DIR, exist_ok=True)

def main():
    env = make_single_walker_env(
        env_id=ENV_ID,
        seed=SEED,
        image_size=IMAGE_SIZE,
        frame_stack=FRAME_STACK,
        reward_shaping=REWARD_SHAPING,
        terminate_when_unhealthy=TERMINATE_WHEN_UNHEALTHY,
        healthy_z_range=HEALTHY_Z_RANGE,
        record_video_folder=VIDEO_DIR,
        video_prefix=f"{ALGO}_video_{LOAD_STEP}steps",
        use_discrete_actions=USE_DISCRETE_ACTIONS,
    )

    model = load_model(
        algo=ALGO,
        model_path=MODEL_PATH,
        env=None,
        device=DEVICE,
    )

    returns = []
    lengths = []

    try:
        for ep in range(N_EPISODES):
            obs, _ = env.reset(seed=SEED + ep)
            done = False
            ep_return = 0.0
            ep_len = 0

            print(f"\n[Episode {ep + 1}] Starting...")

            while not done:
                action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_return += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)

            returns.append(ep_return)
            lengths.append(ep_len)

            print(f"Episode {ep + 1} return: {ep_return:.2f}, len: {ep_len}")

            if done:
                print("----EPISODE END----")
                print("step:", ep_len)
                z = env.unwrapped.data.qpos[1]
                angle = env.unwrapped.data.qpos[2]
                print("torso height (z):", z)
                print("torso angle:", angle)
                print("is healthy:", env.unwrapped.is_healthy)

        print("-" * 60)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"Mean length: {mean_length:.2f} ± {std_length:.2f}")

    finally:
        env.close()

    print(f"Videos saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()