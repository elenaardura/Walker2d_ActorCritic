from __future__ import annotations

import argparse

from src.envs import make_vec_walker_env
from src.methods import build_model
from scripts.utils import build_callbacks, make_run_dir, save_args, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento PPO/SAC en Walker2d-v5 con observación RGB.")

    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--env-id", type=str, default="Walker2d-v5")

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--frame-stack", type=int, default=4)

    parser.add_argument("--n-envs", type=int, default=None)

    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--no-terminate-when-unhealthy", action="store_true")

    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--runs-dir", type=str, default="runs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.n_envs is None:
        args.n_envs = 4 if args.algo == "ppo" else 1

    if args.algo == "sac" and args.n_envs != 1:
        print("[INFO] Para SAC dejo n_envs=1 en este baseline.")
        args.n_envs = 1

    run_dir = make_run_dir(
        root=args.runs_dir,
        algo=args.algo,
        env_id=args.env_id,
        seed=args.seed,
    )
    save_args(args, run_dir)

    terminate_when_unhealthy = not args.no_terminate_when_unhealthy

    train_env = make_vec_walker_env(
        env_id=args.env_id,
        seed=args.seed,
        n_envs=args.n_envs,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        reward_shaping=args.reward_shaping,
        terminate_when_unhealthy=terminate_when_unhealthy,
        monitor_path=run_dir / "train_monitor.csv",
    )

    eval_env = make_vec_walker_env(
        env_id=args.env_id,
        seed=args.seed + 10_000,
        n_envs=1,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        reward_shaping=args.reward_shaping,
        terminate_when_unhealthy=terminate_when_unhealthy,
        monitor_path=run_dir / "eval_monitor.csv",
    )

    model = build_model(
        algo=args.algo,
        env=train_env,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
    )

    eval_freq = max(args.eval_freq // args.n_envs, 1)
    checkpoint_freq = max(args.checkpoint_freq // args.n_envs, 1)

    callbacks = build_callbacks(
        eval_env=eval_env,
        run_dir=run_dir,
        eval_freq=eval_freq,
        checkpoint_freq=checkpoint_freq,
        save_replay_buffer=(args.algo == "sac"),
        n_eval_episodes=5,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        model.save(str(run_dir / "final_model"))
        print(f"[OK] Modelo final guardado en: {run_dir / 'final_model.zip'}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()