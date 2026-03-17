from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.envs import make_single_walker_env
from src.methods import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación PPO/SAC en Walker2d-v5 con observación RGB.")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac"])
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--env-id", type=str, default="Walker2d-v5")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--frame-stack", type=int, default=4)

    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--no-terminate-when-unhealthy", action="store_true")

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    terminate_when_unhealthy = not args.no_terminate_when_unhealthy

    env = make_single_walker_env(
        env_id=args.env_id,
        seed=args.seed,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        reward_shaping=args.reward_shaping,
        terminate_when_unhealthy=terminate_when_unhealthy,
        record_video_folder=args.video_dir,
        video_prefix=f"{args.algo}_eval",
    )

    model = load_model(
        algo=args.algo,
        model_path=args.model_path,
        env=None,
        device=args.device,
    )

    returns = []
    lengths = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            ep_return = 0.0
            ep_len = 0

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_return += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)

            returns.append(ep_return)
            lengths.append(ep_len)
            print(f"Episode {ep + 1:02d} | return={ep_return:.2f} | len={ep_len}")

        print("-" * 60)
        print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

        if args.video_dir is not None:
            print(f"Vídeos guardados en: {Path(args.video_dir).resolve()}")

    finally:
        env.close()


if __name__ == "__main__":
    main()