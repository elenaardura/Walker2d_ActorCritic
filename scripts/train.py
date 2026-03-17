import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.envs import make_vec_walker_env, make_single_walker_env
from src.methods import build_model
from scripts.utils import seed_everything, make_run_dir, save_config, save_experiment_to_excel


# -----------------------------
# Configuración / hiperparámetros
# -----------------------------
ALGO = "ppo"                      # "ppo" o "sac"
ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIMESTEPS = 5_000_000
SEED = 42
NUM_ENVS = 4                      # PPO: 4 suele ir bien; SAC: mejor 1
IMAGE_SIZE = 84
FRAME_STACK = 4

REWARD_SHAPING = True
TERMINATE_WHEN_UNHEALTHY = True
HEALTHY_Z_RANGE = (0.8, 2.0)

EVAL_EVERY = 100_000
CHECKPOINT_EVERY = 100_000
N_EVAL_EPISODES = 10

EXPERIMENT_XLSX = "runs/experiments.xlsx"

# Si quieres reusar un directorio fijo, comenta la línea de abajo y fija MODEL_DIR manualmente
MODEL_DIR = str(make_run_dir("runs"))
TB_DIR = str(make_run_dir("runs") + "/PPO_0")

os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_model(model, n_episodes=10):
    eval_env = make_single_walker_env(
        env_id=ENV_ID,
        seed=SEED + 10_000,
        image_size=IMAGE_SIZE,
        frame_stack=FRAME_STACK,
        reward_shaping=REWARD_SHAPING,
        terminate_when_unhealthy=TERMINATE_WHEN_UNHEALTHY,
        healthy_z_range=HEALTHY_Z_RANGE,
    )

    rewards = []

    try:
        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=SEED + 10_000 + ep)
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)

            rewards.append(ep_reward)

    finally:
        eval_env.close()

    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    seed_everything(SEED)

    global NUM_ENVS

    if ALGO == "sac":
        NUM_ENVS = 1

    writer = SummaryWriter(TB_DIR)

    config = {
        "algo": ALGO,
        "env_id": ENV_ID,
        "device": DEVICE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "image_size": IMAGE_SIZE,
        "frame_stack": FRAME_STACK,
        "reward_shaping": REWARD_SHAPING,
        "terminate_when_unhealthy": TERMINATE_WHEN_UNHEALTHY,
        "healthy_z_range": HEALTHY_Z_RANGE,
        "eval_every": EVAL_EVERY,
        "checkpoint_every": CHECKPOINT_EVERY,
        "n_eval_episodes": N_EVAL_EPISODES,
    }
    save_config(config, MODEL_DIR)

    env = make_vec_walker_env(
        env_id=ENV_ID,
        seed=SEED,
        n_envs=NUM_ENVS,
        image_size=IMAGE_SIZE,
        frame_stack=FRAME_STACK,
        reward_shaping=REWARD_SHAPING,
        terminate_when_unhealthy=TERMINATE_WHEN_UNHEALTHY,
        healthy_z_range=HEALTHY_Z_RANGE,
        monitor_path=os.path.join(MODEL_DIR, "train_monitor.csv"),
    )

    model = build_model(
        algo=ALGO,
        env=env,
        seed=SEED,
        tensorboard_log=MODEL_DIR,
        device=DEVICE,
    )

    steps_done = 0
    avg_eval_reward = np.nan

    try:
        while steps_done < TOTAL_TIMESTEPS:
            chunk = min(EVAL_EVERY, TOTAL_TIMESTEPS - steps_done)

            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=True,
            )

            steps_done += chunk

            avg_eval_reward, std_eval_reward = evaluate_model(model, N_EVAL_EPISODES)
            print(
                f"[Eval] steps={steps_done} | avg_reward={avg_eval_reward:.2f} ± {std_eval_reward:.2f}"
            )

            writer.add_scalar("eval/avg_reward", avg_eval_reward, steps_done)
            writer.add_scalar("eval/std_reward", std_eval_reward, steps_done)

            if steps_done % CHECKPOINT_EVERY == 0 or steps_done >= TOTAL_TIMESTEPS:
                ckpt_path = os.path.join(MODEL_DIR, f"{ALGO}_walker2d_step{steps_done}.pt")
                model.save(ckpt_path)
                print(f"[Checkpoint] Saved: {ckpt_path}")

        final_path = os.path.join(MODEL_DIR, f"{ALGO}_walker2d.pt")
        model.save(final_path)
        print(f"[OK] Final model saved: {final_path}")

    finally:
        env.close()
        writer.close()

    row = {
        "model_dir": MODEL_DIR[5:] if MODEL_DIR.startswith("runs/") else MODEL_DIR,
        "algo": ALGO,
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "num_envs": NUM_ENVS,
        "image_size": IMAGE_SIZE,
        "frame_stack": FRAME_STACK,
        "reward_shaping": REWARD_SHAPING,
        "terminate_when_unhealthy": TERMINATE_WHEN_UNHEALTHY,
        "avg_eval_reward": avg_eval_reward,
        "comments": "PPO/SAC visual Walker2d-v5",
    }

    save_experiment_to_excel(row, EXPERIMENT_XLSX)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")


if __name__ == "__main__":
    main()