from __future__ import annotations

import json
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def make_run_dir(
    root: str | Path,
    algo: str,
    env_id: str,
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{algo}_{env_id}_seed{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_args(args: Namespace, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


def build_callbacks(
    eval_env,
    run_dir: str | Path,
    eval_freq: int,
    checkpoint_freq: int,
    save_replay_buffer: bool = False,
    n_eval_episodes: int = 5,
) -> CallbackList:
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix="agent",
        save_replay_buffer=save_replay_buffer,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    return CallbackList([checkpoint_cb, eval_cb])