import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def make_run_dir(root: str | Path = "runs") -> Path:
    run_dir = Path(root) / datetime.now().strftime("%b%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(config: dict, run_dir: str | Path, filename: str = "config.json") -> None:
    run_dir = Path(run_dir)
    with open(run_dir / filename, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_experiment_to_excel(row_dict, filename="runs/experiments.xlsx"):
    new_df = pd.DataFrame([row_dict])

    if not os.path.isfile(filename):
        new_df.to_excel(filename, index=False, engine="openpyxl")
    else:
        with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            try:
                start_row = writer.book["Sheet1"].max_row
            except KeyError:
                start_row = 0

            new_df.to_excel(
                writer,
                index=False,
                header=False,
                startrow=start_row,
                sheet_name="Sheet1",
            )