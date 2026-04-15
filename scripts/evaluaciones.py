import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.envs import make_single_walker_env
from src.methods import load_model

import gc

gc.collect()
torch.cuda.empty_cache()

DEFAULT_ENV_ID = "Walker2d-v5"
DEFAULT_IMAGE_SIZE = 84
DEFAULT_FRAME_STACK = 4
DEFAULT_REWARD_SHAPING = True
DEFAULT_TERMINATE_WHEN_UNHEALTHY = True
DEFAULT_HEALTHY_Z_RANGE = (0.8, 2.0)
DEFAULT_USE_DISCRETE_ACTIONS = False
DEFAULT_DETERMINISTIC = True


def extract_step(path: Path) -> int:
    """
    Extrae el numero de step de nombres tipo:
      - ppo_walker2d_step500000.pt
      - sac_walker2d_step1350000.zip
      - sac_walker2d_step1350000.pt.zip
    """
    m = re.search(r"step(\d+)(?:\.pt)?(?:\.pt)?$", path.name)
    if m is None:
        return -1
    return int(m.group(1))


def find_checkpoints(run_dir: Path, algo: str) -> list[Path]:
    patterns = [
        f"{algo}_walker2d_step*.pt",
        f"{algo}_walker2d_step*.pt",
        f"{algo}_walker2d_step*.pt",
        f"{algo}_walker2d_step*",
    ]

    seen = set()
    ckpts = []
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                ckpts.append(path)

    ckpts = [p for p in ckpts if extract_step(p) >= 0]
    ckpts = sorted(ckpts, key=extract_step)
    return ckpts


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_from_config(config: dict, seed: int, record_video_folder: str | None = None, video_prefix: str = "eval"):
    return make_single_walker_env(
        env_id=config.get("env_id", DEFAULT_ENV_ID),
        seed=seed,
        image_size=config.get("image_size", DEFAULT_IMAGE_SIZE),
        frame_stack=config.get("frame_stack", DEFAULT_FRAME_STACK),
        reward_shaping=config.get("reward_shaping", DEFAULT_REWARD_SHAPING),
        terminate_when_unhealthy=config.get("terminate_when_unhealthy", DEFAULT_TERMINATE_WHEN_UNHEALTHY),
        healthy_z_range=tuple(config.get("healthy_z_range", DEFAULT_HEALTHY_Z_RANGE)),
        record_video_folder=record_video_folder,
        video_prefix=video_prefix,
        use_discrete_actions=config.get("use_discrete_actions", DEFAULT_USE_DISCRETE_ACTIONS),
    )


def summarize(values):
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0))


def evaluate_checkpoint(
    algo: str,
    model_path: Path,
    config: dict,
    n_episodes: int,
    seed: int,
    device: str,
    deterministic: bool,
    record_video_folder: str | None = None,
    video_prefix: str = "eval",
):
    env = build_env_from_config(
        config=config,
        seed=seed,
        record_video_folder=record_video_folder,
        video_prefix=video_prefix,
    )

    model = load_model(
        algo=algo,
        model_path=model_path,
        env=None,
        device=device,
    )

    metrics = defaultdict(list)

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + 10_000 + ep)
            done = False
            ep_return = 0.0
            ep_len = 0

            reward_forward_sum = 0.0
            reward_survive_sum = 0.0
            reward_ctrl_sum = 0.0
            height_pen_sum = 0.0
            angle_pen_sum = 0.0
            smooth_pen_sum = 0.0
            alive_bonus_sum = 0.0

            terminated = False
            truncated = False

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_return += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)

                reward_forward_sum += float(info.get("debug/reward_forward", 0.0))
                reward_survive_sum += float(info.get("debug/reward_survive", 0.0))
                reward_ctrl_sum += float(info.get("debug/reward_ctrl", 0.0))
                height_pen_sum += float(info.get("debug/height_pen", 0.0))
                angle_pen_sum += float(info.get("debug/angle_pen", 0.0))
                smooth_pen_sum += float(info.get("debug/smooth_pen", 0.0))
                alive_bonus_sum += float(info.get("debug/alive_bonus", 0.0))

            data = env.unwrapped.data
            final_x = float(data.qpos[0]) if len(data.qpos) > 0 else np.nan
            final_z = float(data.qpos[1]) if len(data.qpos) > 1 else np.nan
            final_angle = float(data.qpos[2]) if len(data.qpos) > 2 else np.nan
            final_abs_angle = abs(final_angle) if not np.isnan(final_angle) else np.nan
            final_vx = float(data.qvel[0]) if len(data.qvel) > 0 else np.nan
            final_healthy = float(getattr(env.unwrapped, "is_healthy", False))

            metrics["return"].append(ep_return)
            metrics["episode_length"].append(ep_len)
            metrics["terminated"].append(float(terminated))
            metrics["truncated"].append(float(truncated))

            metrics["reward_forward_sum"].append(reward_forward_sum)
            metrics["reward_survive_sum"].append(reward_survive_sum)
            metrics["reward_ctrl_sum"].append(reward_ctrl_sum)
            metrics["height_pen_sum"].append(height_pen_sum)
            metrics["angle_pen_sum"].append(angle_pen_sum)
            metrics["smooth_pen_sum"].append(smooth_pen_sum)
            metrics["alive_bonus_sum"].append(alive_bonus_sum)

            metrics["final_x"].append(final_x)
            metrics["final_torso_height"].append(final_z)
            metrics["final_torso_angle"].append(final_angle)
            metrics["final_abs_torso_angle"].append(final_abs_angle)
            metrics["final_forward_velocity"].append(final_vx)
            metrics["final_healthy"].append(final_healthy)

    finally:
        env.close()

    summary = {}
    for key, values in metrics.items():
        mean, std = summarize(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std

    summary["_raw_returns"] = np.asarray(metrics["return"], dtype=np.float32)
    summary["_raw_lengths"] = np.asarray(metrics["episode_length"], dtype=np.float32)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Directorio de la corrida, p.ej. runs/Apr01_17_17_22")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"], help="Algoritmo a evaluar")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--every", type=int, default=1, help="Evalua 1 de cada N checkpoints")
    parser.add_argument("--limit", type=int, default=0, help="Maximo numero de checkpoints a evaluar; 0 = todos")
    parser.add_argument("--deterministic", action="store_true", help="Fuerza evaluacion determinista")
    parser.add_argument("--record_best_video", action="store_true", help="Graba video del mejor checkpoint")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"No existe el directorio: {run_dir}"

    algo = args.algo.lower()
    config = load_run_config(run_dir)
    if "algo" in config and config["algo"].lower() != algo:
        print(f"[Aviso] config.json indica algo={config['algo']}, pero se evaluar� con --algo={algo}")

    deterministic = args.deterministic or config.get("algo", algo).lower() == "ppo"

    ckpts = find_checkpoints(run_dir, algo)
    if args.every > 1:
        ckpts = ckpts[::args.every]
    if args.limit > 0:
        ckpts = ckpts[:args.limit]
    if len(ckpts) == 0:
        raise RuntimeError(f"No se han encontrado checkpoints para {algo} en {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = run_dir / f"eval_{algo}_checkpoints_tb_{timestamp}"
    csv_path = run_dir / f"checkpoint_eval_summary_{algo}_{timestamp}.csv"

    writer = SummaryWriter(str(tb_dir))
    rows = []

    print(f"Evaluando {len(ckpts)} checkpoints de {algo}...")
    print(f"TensorBoard logs: {tb_dir}")
    print(f"CSV resumen: {csv_path}")

    best_step = None
    best_return = -np.inf
    best_ckpt = None

    for ckpt in tqdm(ckpts):
        step = extract_step(ckpt)
        summary = evaluate_checkpoint(
            algo=algo,
            model_path=ckpt,
            config=config,
            n_episodes=args.n_episodes,
            seed=args.seed,
            device=args.device,
            deterministic=deterministic,
        )

        writer.add_scalar("eval/return_mean", summary["return_mean"], step)
        writer.add_scalar("eval/return_std", summary["return_std"], step)
        writer.add_scalar("eval/episode_length_mean", summary["episode_length_mean"], step)
        writer.add_scalar("eval/episode_length_std", summary["episode_length_std"], step)
        writer.add_scalar("eval/terminated_rate", summary["terminated_mean"], step)
        writer.add_scalar("eval/truncated_rate", summary["truncated_mean"], step)

        writer.add_scalar("eval/final_x_mean", summary["final_x_mean"], step)
        writer.add_scalar("eval/final_forward_velocity_mean", summary["final_forward_velocity_mean"], step)
        writer.add_scalar("eval/final_torso_height_mean", summary["final_torso_height_mean"], step)
        writer.add_scalar("eval/final_abs_torso_angle_mean", summary["final_abs_torso_angle_mean"], step)
        writer.add_scalar("eval/final_healthy_rate", summary["final_healthy_mean"], step)

        writer.add_scalar("eval_reward_terms/reward_forward_sum_mean", summary["reward_forward_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/reward_survive_sum_mean", summary["reward_survive_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/reward_ctrl_sum_mean", summary["reward_ctrl_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/height_pen_sum_mean", summary["height_pen_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/angle_pen_sum_mean", summary["angle_pen_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/smooth_pen_sum_mean", summary["smooth_pen_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/alive_bonus_sum_mean", summary["alive_bonus_sum_mean"], step)

        writer.add_histogram("eval_distributions/returns", summary["_raw_returns"], step)
        writer.add_histogram("eval_distributions/episode_lengths", summary["_raw_lengths"], step)

        row = {
            "checkpoint": ckpt.name,
            "step": step,
            "algo": algo,
            "return_mean": summary["return_mean"],
            "return_std": summary["return_std"],
            "episode_length_mean": summary["episode_length_mean"],
            "episode_length_std": summary["episode_length_std"],
            "terminated_rate": summary["terminated_mean"],
            "truncated_rate": summary["truncated_mean"],
            "final_x_mean": summary["final_x_mean"],
            "final_x_std": summary["final_x_std"],
            "final_forward_velocity_mean": summary["final_forward_velocity_mean"],
            "final_forward_velocity_std": summary["final_forward_velocity_std"],
            "final_torso_height_mean": summary["final_torso_height_mean"],
            "final_torso_height_std": summary["final_torso_height_std"],
            "final_abs_torso_angle_mean": summary["final_abs_torso_angle_mean"],
            "final_abs_torso_angle_std": summary["final_abs_torso_angle_std"],
            "final_healthy_rate": summary["final_healthy_mean"],
            "reward_forward_sum_mean": summary["reward_forward_sum_mean"],
            "reward_survive_sum_mean": summary["reward_survive_sum_mean"],
            "reward_ctrl_sum_mean": summary["reward_ctrl_sum_mean"],
            "height_pen_sum_mean": summary["height_pen_sum_mean"],
            "angle_pen_sum_mean": summary["angle_pen_sum_mean"],
            "smooth_pen_sum_mean": summary["smooth_pen_sum_mean"],
            "alive_bonus_sum_mean": summary["alive_bonus_sum_mean"],
        }
        rows.append(row)

        if summary["return_mean"] > best_return:
            best_return = summary["return_mean"]
            best_step = step
            best_ckpt = ckpt

        print(
            f"[step={step}] "
            f"return={summary['return_mean']:.2f}�{summary['return_std']:.2f} | "
            f"len={summary['episode_length_mean']:.1f}�{summary['episode_length_std']:.1f} | "
            f"healthy_end={summary['final_healthy_mean']:.2f}"
        )

    writer.close()

    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(csv_path, index=False)

    print("\nResumen final")
    print(f"Mejor checkpoint por return_mean: step={best_step} | return_mean={best_return:.2f}")
    print(f"CSV guardado en: {csv_path}")
    print(f"TensorBoard logdir: {tb_dir}")

    if args.record_best_video and best_ckpt is not None:
        video_dir = run_dir / f"best_{algo}_video_{timestamp}"
        print(f"\nGrabando v�deo del mejor checkpoint en: {video_dir}")
        _ = evaluate_checkpoint(
            algo=algo,
            model_path=best_ckpt,
            config=config,
            n_episodes=1,
            seed=args.seed,
            device=args.device,
            deterministic=deterministic,
            record_video_folder=str(video_dir),
            video_prefix=f"best_{algo}_step{best_step}",
        )


if __name__ == "__main__":
    main()
