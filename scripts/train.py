import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.callbacks import BaseCallback

from src.envs import make_vec_walker_env, make_single_walker_env
from src.methods import build_model
from scripts.utils import seed_everything, make_run_dir, save_config, save_experiment_to_excel


# ============================================================
# Callback personalizado para monitorear acciones
# ============================================================
class ActionMonitorCallback(BaseCallback):
    """
    Monitorea las acciones tomadas por el modelo durante el entrenamiento.
    Las guarda en TensorBoard cada N pasos.
    """
    def __init__(self, writer: SummaryWriter, log_freq: int = 1000):
        super().__init__()
        self.writer = writer
        self.log_freq = log_freq
        self.action_history = []

    def _on_step(self) -> bool:
        """Se llama después de cada step del entorno."""
        # Obtener las acciones del último step
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'buf_actions'):
            actions = self.model.env.buf_actions
            self.action_history.append(actions.copy())

        # Loguear cada log_freq steps
        if self.num_timesteps % self.log_freq == 0 and len(self.action_history) > 0:
            actions_array = np.array(self.action_history)
            
            # Estadísticas por dimensión
            for i in range(actions_array.shape[1]):
                mean_action = float(np.mean(actions_array[:, i]))
                std_action = float(np.std(actions_array[:, i]))
                min_action = float(np.min(actions_array[:, i]))
                max_action = float(np.max(actions_array[:, i]))
                
                self.writer.add_scalar(f"train/action_{i}_mean", mean_action, self.num_timesteps)
                self.writer.add_scalar(f"train/action_{i}_std", std_action, self.num_timesteps)
                self.writer.add_scalar(f"train/action_{i}_min", min_action, self.num_timesteps)
                self.writer.add_scalar(f"train/action_{i}_max", max_action, self.num_timesteps)
            
            # Estadísticas globales
            mean_all = float(np.mean(actions_array))
            std_all = float(np.std(actions_array))
            self.writer.add_scalar(f"train/action_mean_all", mean_all, self.num_timesteps)
            self.writer.add_scalar(f"train/action_std_all", std_all, self.num_timesteps)
            
            self.action_history = []

        return True


# -----------------------------
# Configuración / hiperparámetros
# -----------------------------
ALGO = "sac"                      # "ppo" o "sac"
ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_TIMESTEPS = 5_000_000 # antes 5M (cambio a 10M a partir del 24 a las 10:35). Los del SAC a 2M y cambio a 5M a las 12:29 del 27 
SEED = 42
NUM_ENVS = 1 # antes 4 (cambio +batch 256 y capas 512) ,luego 8 para SAC 1                     # PPO: 4 suele ir bien; SAC: mejor 1
IMAGE_SIZE = 84
FRAME_STACK = 4

REWARD_SHAPING = True
TERMINATE_WHEN_UNHEALTHY = True
HEALTHY_Z_RANGE = (0.8, 2.0)

# ✅ Usar acciones discretas
USE_DISCRETE_ACTIONS = False

EVAL_EVERY = 50_000
CHECKPOINT_EVERY = 50_000
N_EVAL_EPISODES = 10

EXPERIMENT_XLSX = "runs/experiments.xlsx"

# Si quieres reusar un directorio fijo, comenta la línea de abajo y fija MODEL_DIR manualmente
MODEL_DIR = str(make_run_dir("runs"))
TB_DIR = os.path.join(MODEL_DIR, "SAC_0")

# os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_model(model, n_episodes=10, use_discrete_actions=False):
    eval_env = make_single_walker_env(
        env_id=ENV_ID,
        seed=SEED + 10_000,
        image_size=IMAGE_SIZE,
        frame_stack=FRAME_STACK,
        reward_shaping=REWARD_SHAPING,
        terminate_when_unhealthy=TERMINATE_WHEN_UNHEALTHY,
        healthy_z_range=HEALTHY_Z_RANGE,
use_discrete_actions=use_discrete_actions,  # ✅ Pasar parámetro
    )

    shaped_rewards = []
    base_rewards = []

    speed_bonus_list = []
    alive_bonus_list = []
    height_pen_list = []
    angle_pen_list = []
    smooth_pen_list = []

    try:
        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=SEED + 10_000 + ep)
            done = False
            
            ep_shaped_reward = 0.0
            ep_base_reward = 0.0

            ep_speed_bonus = 0.0
            ep_alive_bonus = 0.0
            ep_height_pen = 0.0
            ep_angle_pen = 0.0
            ep_smooth_pen = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                ep_shaped_reward += float(reward)
                ep_base_reward += float(info.get("debug/base", reward))

                ep_speed_bonus += float(info.get("debug/speed_bonus", 0.0))
                ep_alive_bonus += float(info.get("debug/alive_bonus", 0.0))
                ep_height_pen += float(info.get("debug/height_pen", 0.0))
                ep_angle_pen += float(info.get("debug/angle_pen", 0.0))
                ep_smooth_pen += float(info.get("debug/smooth_pen", 0.0))
                done = bool(terminated or truncated)

            shaped_rewards.append(ep_shaped_reward)
            base_rewards.append(ep_base_reward)

            speed_bonus_list.append(ep_speed_bonus)
            alive_bonus_list.append(ep_alive_bonus)
            height_pen_list.append(ep_height_pen)
            angle_pen_list.append(ep_angle_pen)
            smooth_pen_list.append(ep_smooth_pen)

    finally:
        eval_env.close()

    return {
        "mean_shaped_reward": float(np.mean(shaped_rewards)),
        "std_shaped_reward": float(np.std(shaped_rewards)),
        "mean_base_reward": float(np.mean(base_rewards)),
        "std_base_reward": float(np.std(base_rewards)),
        "mean_speed_bonus": float(np.mean(speed_bonus_list)),
        "mean_alive_bonus": float(np.mean(alive_bonus_list)),
        "mean_height_pen": float(np.mean(height_pen_list)),
        "mean_angle_pen": float(np.mean(angle_pen_list)),
        "mean_smooth_pen": float(np.mean(smooth_pen_list)),
    }


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
        "use_discrete_actions": USE_DISCRETE_ACTIONS,  # ✅ Guardar en config
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
use_discrete_actions=USE_DISCRETE_ACTIONS,  # ✅ Pasar parámetro
    )

    model = build_model(
        algo=ALGO,
        env=env,
        seed=SEED,
        tensorboard_log=MODEL_DIR,
        device=DEVICE,
    )

    # Añadir el callback personalizado
    action_monitor_callback = ActionMonitorCallback(writer=writer)

    steps_done = 0
    avg_eval_reward = np.nan

    try:
        while steps_done < TOTAL_TIMESTEPS:
            chunk = min(EVAL_EVERY, TOTAL_TIMESTEPS - steps_done)

            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=True,
                callback=action_monitor_callback,  # Usar el callback
            )

            steps_done += chunk

# ✅ Pasar use_discrete_actions a evaluate_model
            eval_stats = evaluate_model(
                model,
                N_EVAL_EPISODES,
                use_discrete_actions=USE_DISCRETE_ACTIONS,
            )
            print(
                f"[Eval] steps={steps_done} | "
                f"shaped={eval_stats['mean_shaped_reward']:.2f} � {eval_stats['std_shaped_reward']:.2f} | "
                f"base={eval_stats['mean_base_reward']:.2f} � {eval_stats['std_base_reward']:.2f}"
            )

            writer.add_scalar("eval/shaped_reward_mean", eval_stats["mean_shaped_reward"], steps_done)
            writer.add_scalar("eval/shaped_reward_std", eval_stats["std_shaped_reward"], steps_done)

            writer.add_scalar("eval/base_reward_mean", eval_stats["mean_base_reward"], steps_done)
            writer.add_scalar("eval/base_reward_std", eval_stats["std_base_reward"], steps_done)

            writer.add_scalar("eval/shaping/speed_bonus_mean", eval_stats["mean_speed_bonus"], steps_done)
            writer.add_scalar("eval/shaping/alive_bonus_mean", eval_stats["mean_alive_bonus"], steps_done)
            writer.add_scalar("eval/shaping/height_pen_mean", eval_stats["mean_height_pen"], steps_done)
            writer.add_scalar("eval/shaping/angle_pen_mean", eval_stats["mean_angle_pen"], steps_done)
            writer.add_scalar("eval/shaping/smooth_pen_mean", eval_stats["mean_smooth_pen"], steps_done)

            avg_eval_reward = eval_stats["mean_shaped_reward"]

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
"use_discrete_actions": USE_DISCRETE_ACTIONS,  # ✅ Guardar en Excel
        "avg_eval_reward": avg_eval_reward,
        "comments": "PPO/SAC visual Walker2d-v5",
    }

    save_experiment_to_excel(row, EXPERIMENT_XLSX)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")


if __name__ == "__main__":
    main()