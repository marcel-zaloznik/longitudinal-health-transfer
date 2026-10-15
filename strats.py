#!/usr/bin/env python3
"""STraTS adaptation for longitudinal multi-target regression and dataset transfer.

Based on Tipirneni and Reddy (2022) and public STraTS implementations.
See THIRD_PARTY_NOTICES.md for attribution.
"""

import argparse
import ast
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prepare_data import validate_dataset, validate_folds
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Default configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
ERRORS_DIR = ROOT / "results" / "errors"
SEED = 1234

DATA_PATH = str(ROOT / "data" / "dataset.csv")
PAIRS_PATH = str(ROOT / "data" / "input_output_pairs.csv")
SPLIT_PATH = str(ROOT / "data" / "split5k.json")
TARGET_TRAIN_SUBSETS_PATH = str(ROOT / "data" / "target_train_subsets_by_dataset_fold_percent.json")

FINAL_ERROR_PATH = (ERRORS_DIR / "strats_main.csv")

# Experiment mode:
#   target_only   -> no source pretraining, fine-tune only on target dataset
#   topk_pretrain -> pretrain on top-k source datasets from ORIGIN_MAP, then fine-tune target
#   single_source -> pretrain on one named source dataset, then fine-tune target
#   pooled        -> train from scratch on target train + source datasets; validate/test on target only
MODE = "topk_pretrain"

# Use 1..6 source datasets according to ORIGIN_MAP for topk_pretrain/pooled.
N_ORIGINS = 6

# Source dataset for MODE="single_source".
SOURCE_DATASET: Optional[str] = None

# 100 for full target training data, or 10/30/50/70/90.
# This reduces only target fine-tuning training people. Validation/test stay unchanged.
TRAIN_DATA_PERCENT = 100

# Model size.
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_HEADS = 4
FF_MULT = 2

DROPOUT = 0.20
ATTENTION_DROPOUT = 0.20
VARIABLE_DROPOUT = 0.15
USE_LAYER_NORM = False

# Prefixes can become long because every visit contributes up to 20 triplets.
# The experiments retain the most recent observed triplets.
# Set to None to disable truncation.
MAX_OBS: Optional[int] = 256

PRETRAIN_BATCH_SIZE = 256
FINETUNE_BATCH_SIZE = 256

PRETRAIN_LR = 5e-4
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-5

MAX_PRETRAIN_EPOCHS = 1000
MAX_FINETUNE_EPOCHS = 1000
PATIENCE = 10
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0

# Useful for smoke tests.
MAX_PRETRAIN_RECORDS: Optional[int] = None
MAX_FINETUNE_RECORDS: Optional[int] = None

NUM_WORKERS = min(8, os.cpu_count() or 1)
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2

EPS = 1e-8

# Terminal output control.
# 0 hides per-epoch logs; 25/50/100 prints only every N epochs.
EPOCH_PRINT_EVERY = 50

TARGET_COLS = [
    "Height",
    "Weight",
    "Waist_circumference",
    "BMI",
    "Systolic_blood_pressure",
    "Diastolic_blood_pressure",
    "Resting_heart_rate",
    "Glucose",
    "Total_cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Smoking",
    "Fitness_speed",
    "PA_h_per_w",
    "Alcohol",
    "Diabetes",
    "Sleep",
    "Diet_HHS",
    "Diet_T2P",
]

NO_OBS_TOKEN = "__NO_OBS__"
INPUT_VARIABLES = TARGET_COLS + [NO_OBS_TOKEN]
NO_OBS_VAR_ID = len(INPUT_VARIABLES) - 1

TARGET_DIM = len(TARGET_COLS)
INPUT_VAR_DIM = len(INPUT_VARIABLES)

# demo/static vector:
# 0 sex_male
# 1 sex_female
# 2 target_age
# 3 input_age
# 4 forecast_horizon = target_age - input_age
# 5 number_of_input_visits
# 6 number_of_observed_triplets
DEMO_DIM = 7
DEMO_CONT_IDXS = [2, 3, 4, 5, 6]

ERROR_COLUMNS = [
    "method",
    "seed",
    "dataset",
    "fold",
    "person_id",
    "attribute",
    "error",
    "input_age",
    "output_age",
    "horizon",
    "n_input_visits",
    "source_config",
    "source_dataset",
    "n_pretrain_datasets",
]

ORIGIN_MAP = {
    "ABCD": [
        "YFS",
        "BE_KUL_Wave_1B",
        "SLOfit_lifelong",
        "SLOfit_students",
        "BE_KUL_Wave_1A",
        "BE_KUL_Wave_23456",
    ],
    "BE_KUL_Wave_1A": [
        "YFS",
        "BE_KUL_Wave_1B",
        "SLOfit_students",
        "SLOfit_lifelong",
        "ABCD",
        "BE_KUL_Wave_23456",
    ],
    "BE_KUL_Wave_1B": [
        "YFS",
        "SLOfit_students",
        "BE_KUL_Wave_1A",
        "SLOfit_lifelong",
        "ABCD",
        "BE_KUL_Wave_23456",
    ],
    "BE_KUL_Wave_23456": [
        "YFS",
        "SLOfit_lifelong",
        "ABCD",
        "BE_KUL_Wave_1B",
        "BE_KUL_Wave_1A",
        "SLOfit_students",
    ],
    "SLOfit_lifelong": [
        "YFS",
        "BE_KUL_Wave_23456",
        "SLOfit_students",
        "BE_KUL_Wave_1B",
        "BE_KUL_Wave_1A",
        "ABCD",
    ],
    "SLOfit_students": [
        "SLOfit_lifelong",
        "BE_KUL_Wave_1B",
        "BE_KUL_Wave_1A",
        "YFS",
        "ABCD",
        "BE_KUL_Wave_23456",
    ],
    "YFS": [
        "BE_KUL_Wave_23456",
        "SLOfit_lifelong",
        "BE_KUL_Wave_1B",
        "ABCD",
        "BE_KUL_Wave_1A",
        "SLOfit_students",
    ],
}


# =============================================================================
# General utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def clone_state_dict(model_or_state_dict):
    state_dict = (
        model_or_state_dict
        if isinstance(model_or_state_dict, dict)
        else model_or_state_dict.state_dict()
    )
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def make_adamw(params, lr: float, weight_decay: float, device: torch.device):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device.type == "cuda":
        kwargs["fused"] = True

    try:
        return torch.optim.AdamW(params, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(params, **kwargs)


def parse_index_tuple(cell) -> Tuple[int, ...]:
    if isinstance(cell, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in cell)

    if pd.isna(cell):
        return tuple()

    text = str(cell).strip()

    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple, np.ndarray)):
            return tuple(int(x) for x in value)
        return (int(value),)
    except Exception:
        return tuple(int(x) for x in re.findall(r"-?\d+", text))


def finite_float_or_nan(x) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return math.nan

    return value if np.isfinite(value) else math.nan


def safe_stem(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "PersonID" in df.columns:
        df["PersonID"] = df["PersonID"].astype(str)

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str)

    categorical_maps = {
        "Smoking": {"No": 0, "Quit/strike": 1, "Yes": 2},
        "Diabetes": {"No": 0, "Yes": 1},
    }

    for col, mapping in categorical_maps.items():
        if col in df.columns:
            original = df[col]
            mapped = original.map(mapping)
            df[col] = pd.to_numeric(mapped.where(mapped.notna(), original), errors="coerce")

    for col in ["Age"] + TARGET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def maybe_limit(records: List[dict], n: Optional[int], seed: int) -> List[dict]:
    if n is None or len(records) <= n:
        return records

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), n, replace=False)
    idx.sort()
    return [records[i] for i in idx]


def load_split5k(path: str) -> List[set]:
    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return [set(map(str, fold_ids)) for fold_ids in split]


def load_target_train_subsets(path: str) -> Optional[dict]:
    if TRAIN_DATA_PERCENT == 100:
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_errors(df_errors: pd.DataFrame, output_path: Path) -> int:
    if df_errors.empty:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_errors.to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )
    return len(df_errors)


# =============================================================================
# Statistics and record construction
# =============================================================================

def compute_value_stats(df_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Train-only standardization for observed input values.
    No imputation is performed.
    """
    stats = {}

    for col in TARGET_COLS:
        values = (
            pd.to_numeric(df_train[col], errors="coerce")
            .dropna()
            .astype(float)
            .to_numpy()
        )

        if values.size == 0:
            stats[col] = {"mean": 0.0, "std": 1.0, "n": 0}
            continue

        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std < EPS:
            std = 1.0

        stats[col] = {
            "mean": float(values.mean()),
            "std": std,
            "n": int(values.size),
        }

    return stats


def z_value(value: float, col: str, stats: Dict[str, Dict[str, float]]) -> float:
    st = stats.get(col, {"mean": 0.0, "std": 1.0})
    return (float(value) - float(st["mean"])) / float(st["std"])


def sex_one_hot(sex) -> Tuple[float, float]:
    sex = str(sex)
    return (
        1.0 if sex == "Male" else 0.0,
        1.0 if sex == "Female" else 0.0,
    )


def build_records(
    df_input: pd.DataFrame,
    pairs: pd.DataFrame,
    value_stats: Dict[str, Dict[str, float]],
    target_df: Optional[pd.DataFrame],
    max_obs: Optional[int],
) -> List[dict]:
    """
    Converts input_output_pairs into STraTS records.

    Each observed scalar in the input prefix becomes one triplet:
        value = train-standardized scalar value
        time  = age_of_measurement - target_age
        vari  = target attribute id

    The output target is the raw 20-dimensional vector from output_row.
    """
    if target_df is None:
        target_df = df_input

    if not df_input.index.equals(target_df.index):
        raise ValueError("df_input and target_df indices must match.")

    people = set(df_input["PersonID"].astype(str).unique())
    pairs_df = pairs[pairs["PersonID"].astype(str).isin(people)].copy()

    col_to_var = {col: i for i, col in enumerate(TARGET_COLS)}

    records: List[dict] = []
    person_cache: Dict[str, dict] = {}

    for pair in pairs_df.itertuples(index=False):
        pid = str(pair.PersonID)
        input_rows = tuple(int(x) for x in pair.input_rows)
        output_row = int(pair.output_row)

        if len(input_rows) == 0:
            continue

        if pid not in person_cache:
            p_df = (
                df_input[df_input["PersonID"].astype(str) == pid]
                .sort_values("Age", kind="mergesort")
            )
            if p_df.empty:
                continue

            t_df = target_df.loc[p_df.index]
            row_ids = p_df.index.to_list()

            person_cache[pid] = {
                "rowpos": {row_id: i for i, row_id in enumerate(row_ids)},
                "ages": p_df["Age"].astype(float).to_numpy(),
                "x": p_df[TARGET_COLS]
                    .apply(pd.to_numeric, errors="coerce")
                    .astype(float),
                "y": t_df[TARGET_COLS]
                    .apply(pd.to_numeric, errors="coerce")
                    .astype(float),
                "sex": sex_one_hot(t_df["Sex"].iloc[0]),
            }

        cached = person_cache[pid]

        if output_row not in cached["rowpos"]:
            continue

        if any(row_id not in cached["rowpos"] for row_id in input_rows):
            continue

        input_positions = [cached["rowpos"][row_id] for row_id in input_rows]
        output_position = cached["rowpos"][output_row]

        target_age = float(cached["ages"][output_position])
        input_age = float(cached["ages"][input_positions[-1]])
        horizon = target_age - input_age

        times_raw = []
        varis = []
        values = []

        for pos in input_positions:
            relative_time = float(cached["ages"][pos]) - target_age
            row = cached["x"].iloc[pos]

            for col in TARGET_COLS:
                val = finite_float_or_nan(row[col])
                if math.isnan(val):
                    continue

                times_raw.append(relative_time)
                varis.append(col_to_var[col])
                values.append(z_value(val, col, value_stats))

        # STraTS expects at least one token.
        if len(values) == 0:
            times_raw = [0.0]
            varis = [NO_OBS_VAR_ID]
            values = [0.0]

        # Keep the most recent triplets when prefixes are too long.
        if max_obs is not None and len(values) > max_obs:
            keep = np.argsort(np.asarray(times_raw), kind="mergesort")[-max_obs:]
            keep.sort()
            times_raw = [times_raw[i] for i in keep]
            varis = [varis[i] for i in keep]
            values = [values[i] for i in keep]

        male, female = cached["sex"]

        raw_demo = np.asarray(
            [
                male,
                female,
                target_age,
                input_age,
                horizon,
                len(input_positions),
                len(values),
            ],
            dtype=np.float32,
        )

        records.append(
            {
                "values": np.asarray(values, dtype=np.float32),
                "times_raw": np.asarray(times_raw, dtype=np.float32),
                "varis": np.asarray(varis, dtype=np.int64),
                "raw_demo": raw_demo,
                "target_raw": cached["y"]
                    .iloc[output_position][TARGET_COLS]
                    .to_numpy(dtype=np.float32),
                "person_id": pid,
                "input_row": int(input_rows[-1]),
                "output_row": int(output_row),
                "input_age": float(input_age),
                "target_age": float(target_age),
                "horizon": float(horizon),
                "n_input_visits": int(len(input_positions)),
            }
        )

    return records


def compute_time_stats(records: List[dict]) -> Dict[str, float]:
    """
    Scale relative times by a train-only horizon scale.
    This keeps the continuous time embedding input numerically stable.
    """
    if len(records) == 0:
        return {"horizon_scale": 1.0}

    horizons = np.asarray([r["horizon"] for r in records], dtype=float)
    finite_horizons = horizons[np.isfinite(horizons)]

    if finite_horizons.size == 0:
        scale = 1.0
    else:
        scale = float(np.nanstd(finite_horizons))

    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0

    return {"horizon_scale": scale}


def apply_time_stats(records: List[dict], stats: Dict[str, float]) -> None:
    scale = max(float(stats["horizon_scale"]), EPS)
    for record in records:
        record["times"] = (record["times_raw"] / scale).astype(np.float32)


def compute_demo_stats(records: List[dict]) -> Dict[int, Dict[str, float]]:
    stats = {idx: {"mean": 0.0, "std": 1.0} for idx in DEMO_CONT_IDXS}

    if len(records) == 0:
        return stats

    raw = np.asarray([r["raw_demo"] for r in records], dtype=float)

    for idx in DEMO_CONT_IDXS:
        values = raw[:, idx]
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std < EPS:
            std = 1.0

        stats[idx] = {"mean": float(values.mean()), "std": std}

    return stats


def apply_demo_stats(records: List[dict], stats: Dict[int, Dict[str, float]]) -> None:
    for record in records:
        demo = np.asarray(record["raw_demo"], dtype=np.float32).copy()

        for idx in DEMO_CONT_IDXS:
            demo[idx] = (demo[idx] - stats[idx]["mean"]) / stats[idx]["std"]

        demo[~np.isfinite(demo)] = 0.0
        record["demo"] = demo.astype(np.float32)


def compute_target_stats(records: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Target standardization for optimization.

    Uses unique output rows, not duplicated prefix-future pairs.
    Test errors are later computed on the original raw scale.
    """
    unique_records = []
    seen_output_rows = set()

    for record in records:
        output_row = int(record.get("output_row", -1))
        if output_row in seen_output_rows:
            continue
        seen_output_rows.add(output_row)
        unique_records.append(record)

    stats = {}

    for j, col in enumerate(TARGET_COLS):
        values = [finite_float_or_nan(r["target_raw"][j]) for r in unique_records]
        values = np.asarray(
            [v for v in values if not math.isnan(v)],
            dtype=np.float64,
        )

        if values.size == 0:
            stats[col] = {"mean": 0.0, "std": 1.0, "n": 0}
            continue

        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std < EPS:
            std = 1.0

        stats[col] = {
            "mean": float(values.mean()),
            "std": std,
            "n": int(values.size),
        }

    return stats


def standardize_targets(
    records: List[dict],
    target_stats: Dict[str, Dict[str, float]],
) -> None:
    for record in records:
        y = np.asarray(record["target_raw"], dtype=np.float32).copy()
        mask = np.isfinite(y).astype(np.float32)

        for j, col in enumerate(TARGET_COLS):
            st = target_stats[col]
            if st["n"] <= 0 or not np.isfinite(y[j]):
                y[j] = np.nan
                mask[j] = 0.0
            else:
                y[j] = (y[j] - st["mean"]) / st["std"]

        record["target"] = y.astype(np.float32)
        record["target_mask"] = mask.astype(np.float32)


def set_eval_targets(records: List[dict]) -> None:
    """
    For test records, keep target values raw.
    Predictions are de-standardized before error calculation.
    """
    for record in records:
        y = np.asarray(record["target_raw"], dtype=np.float32).copy()
        record["target"] = y
        record["target_mask"] = np.isfinite(y).astype(np.float32)


def prepare_train_val(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    pairs: pd.DataFrame,
    seed: int,
    max_train_records: Optional[int] = None,
):
    value_stats = compute_value_stats(df_train)

    train_records = build_records(df_train, pairs, value_stats, df_train, max_obs=MAX_OBS)
    train_records = maybe_limit(train_records, max_train_records, seed)
    val_records = build_records(df_val, pairs, value_stats, df_val, max_obs=MAX_OBS)

    time_stats = compute_time_stats(train_records)
    demo_stats = compute_demo_stats(train_records)

    for records in (train_records, val_records):
        apply_time_stats(records, time_stats)
        apply_demo_stats(records, demo_stats)

    target_stats = compute_target_stats(train_records)
    standardize_targets(train_records, target_stats)
    standardize_targets(val_records, target_stats)

    return train_records, val_records, target_stats


def prepare_finetune(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    pairs: pd.DataFrame,
    seed: int,
    max_train_records: Optional[int] = None,
    stats_df: Optional[pd.DataFrame] = None,
):
    """
    Build fine-tuning records.

    df_train is the actual training data.
    stats_df is the data used for value/time/demo/target standardization.

    For ordinary target-only and source-pretrained fine-tuning:
        stats_df = df_train

    For pooled control:
        df_train = target train + source datasets
        stats_df = target train only

    This prevents pooled output scaling from being dominated by source cohorts.
    """
    if stats_df is None:
        stats_df = df_train

    value_stats = compute_value_stats(stats_df)

    train_records = build_records(df_train, pairs, value_stats, df_train, max_obs=MAX_OBS)
    train_records = maybe_limit(train_records, max_train_records, seed)
    val_records = build_records(df_val, pairs, value_stats, df_val, max_obs=MAX_OBS)
    test_records = build_records(df_test, pairs, value_stats, df_test, max_obs=MAX_OBS)

    if stats_df is df_train:
        stats_records = train_records
    else:
        stats_records = build_records(stats_df, pairs, value_stats, stats_df, max_obs=MAX_OBS)

    time_stats = compute_time_stats(stats_records)
    demo_stats = compute_demo_stats(stats_records)

    for records in (train_records, val_records, test_records):
        apply_time_stats(records, time_stats)
        apply_demo_stats(records, demo_stats)

    target_stats = compute_target_stats(stats_records)

    standardize_targets(train_records, target_stats)
    standardize_targets(val_records, target_stats)
    set_eval_targets(test_records)

    return train_records, val_records, test_records, target_stats


# =============================================================================
# Dataset and collate
# =============================================================================

class STRATSDataset(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return {
            "values": torch.tensor(record["values"], dtype=torch.float32),
            "times": torch.tensor(record["times"], dtype=torch.float32),
            "varis": torch.tensor(record["varis"], dtype=torch.long),
            "demo": torch.tensor(record["demo"], dtype=torch.float32),
            "target": torch.tensor(record["target"], dtype=torch.float32),
            "target_mask": torch.tensor(record["target_mask"], dtype=torch.float32),
        }


def collate_fn(batch):
    values = [b["values"] for b in batch]
    times = [b["times"] for b in batch]
    varis = [b["varis"] for b in batch]

    lengths = torch.tensor([x.size(0) for x in values], dtype=torch.long)
    max_len = int(lengths.max())

    obs_mask = (
        torch.arange(max_len).unsqueeze(0)
        < lengths.unsqueeze(1)
    ).float()

    values = pad_sequence(values, batch_first=True, padding_value=0.0)
    times = pad_sequence(times, batch_first=True, padding_value=0.0)
    varis = pad_sequence(varis, batch_first=True, padding_value=NO_OBS_VAR_ID)

    demo = torch.stack([b["demo"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    target_mask = torch.stack([b["target_mask"] for b in batch])

    return values, times, varis, obs_mask, demo, target, target_mask


def make_loader(
    records: List[dict],
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )

    if NUM_WORKERS > 0:
        kwargs.update(
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
        )

    return DataLoader(STRATSDataset(records), **kwargs)


# =============================================================================
# STraTS model
# =============================================================================

class CVE(nn.Module):
    """Continuous value embedding used for both time and observed scalar value."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        intermediate_dim = max(1, int(math.sqrt(hidden_dim)))

        self.W1 = nn.Parameter(torch.empty(1, intermediate_dim))
        self.b1 = nn.Parameter(torch.zeros(intermediate_dim))
        self.W2 = nn.Parameter(torch.empty(intermediate_dim, hidden_dim))

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        h = torch.tanh(
            torch.matmul(x, self.W1)
            + self.b1.view(1, 1, -1)
        )
        return torch.matmul(h, self.W2)


class FusionAttention(nn.Module):
    """Attention pooling over irregular observation tokens."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.u(torch.tanh(self.proj(x))).squeeze(-1)
        mask_bool = mask > 0

        # Do masked softmax in float32 for AMP stability.
        scores_float = scores.float().masked_fill(~mask_bool, -1e30)
        attn = torch.softmax(scores_float, dim=-1)
        attn = attn * mask_bool.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return attn.to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        attention_dropout: float,
        use_layer_norm: bool,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm

        self.mha = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
        )

        if use_layer_norm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = mask <= 0

        attn_out, _ = self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        if self.use_layer_norm:
            x = self.norm1(x + self.dropout(attn_out))
        else:
            x = (x + self.dropout(attn_out)) / 2.0

        ffn_out = self.dropout(self.ffn(x))

        if self.use_layer_norm:
            x = self.norm2(x + ffn_out)
        else:
            x = (x + ffn_out) / 2.0

        return x * mask.unsqueeze(-1)


class STRATSRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.cve_time = CVE(HIDDEN_DIM)
        self.cve_value = CVE(HIDDEN_DIM)
        self.variable_emb = nn.Embedding(INPUT_VAR_DIM, HIDDEN_DIM)
        self.input_dropout = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=HIDDEN_DIM,
                    num_heads=NUM_HEADS,
                    ff_mult=FF_MULT,
                    dropout=DROPOUT,
                    attention_dropout=ATTENTION_DROPOUT,
                    use_layer_norm=USE_LAYER_NORM,
                )
                for _ in range(NUM_LAYERS)
            ]
        )

        self.fusion = FusionAttention(HIDDEN_DIM)

        self.demo_emb = nn.Sequential(
            nn.Linear(DEMO_DIM, 2 * HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
        )

        self.head = nn.Linear(2 * HIDDEN_DIM, TARGET_DIM)

    def _variable_dropout(
        self,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (not self.training) or VARIABLE_DROPOUT <= 0:
            return obs_mask

        batch_size = varis.size(0)
        drop_by_variable = (
            torch.rand((batch_size, INPUT_VAR_DIM), device=varis.device)
            < VARIABLE_DROPOUT
        )

        # Never drop the fallback token.
        drop_by_variable[:, NO_OBS_VAR_ID] = False

        token_drop = drop_by_variable.gather(
            1,
            varis.clamp(0, INPUT_VAR_DIM - 1),
        )

        new_mask = obs_mask * (~token_drop).float()

        # Avoid empty sequences.
        empty = new_mask.sum(dim=1) <= 0
        if empty.any():
            new_mask[empty] = obs_mask[empty]

        return new_mask

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
        demo: torch.Tensor,
    ) -> torch.Tensor:
        obs_mask = self._variable_dropout(varis, obs_mask)

        x = (
            self.cve_time(times)
            + self.cve_value(values)
            + self.variable_emb(varis)
        )

        x = self.input_dropout(x)
        x = x * obs_mask.unsqueeze(-1)

        for block in self.blocks:
            x = block(x, obs_mask)

        attn = self.fusion(x, obs_mask)
        ts_emb = (x * attn.unsqueeze(-1)).sum(dim=1)
        demo_emb = self.demo_emb(demo)

        return self.head(torch.cat([ts_emb, demo_emb], dim=-1))


def new_model(device: torch.device) -> STRATSRegressor:
    return STRATSRegressor().to(device)


# =============================================================================
# Training and evaluation
# =============================================================================

def sse_count(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
):
    mask = (target_mask > 0) & torch.isfinite(targets)

    diff = preds.float() - torch.nan_to_num(targets, nan=0.0).float()
    diff = diff.masked_fill(~mask, 0.0)

    sse = diff.pow(2).sum(dim=0)
    count = mask.sum(dim=0)

    return sse, count


def target_balanced_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    weights: torch.Tensor,
):
    """
    Multi-target masked MSE.
    Weighting prevents dense targets from dominating sparse targets.
    """
    mask = (target_mask > 0) & torch.isfinite(targets)
    if not mask.any():
        return None

    se = (
        preds.float()
        - torch.nan_to_num(targets, nan=0.0).float()
    ).pow(2)
    se = se.masked_fill(~mask, 0.0)

    weights = weights.to(preds.device, se.dtype).view(1, -1)
    denom = (mask.to(se.dtype) * weights).sum()

    if denom.detach().item() == 0:
        return None

    return (se * weights).sum() / denom


def epoch_mse(sse: torch.Tensor, count: torch.Tensor) -> float:
    valid = count > 0
    if not valid.any():
        return float("nan")
    return (sse[valid] / count[valid].float()).mean().item()


def loss_weights(
    target_stats: Dict[str, Dict[str, float]],
    device: torch.device,
) -> torch.Tensor:
    weights = []
    for col in TARGET_COLS:
        n = target_stats[col]["n"]
        weights.append(1.0 / n if n > 0 else 0.0)
    return torch.tensor(weights, device=device, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_records: List[dict],
    val_records: List[dict],
    target_stats: Dict[str, Dict[str, float]],
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    seed_offset: int,
    name: str,
):
    train_loader = make_loader(
        train_records,
        batch_size=batch_size,
        shuffle=True,
        seed=SEED + seed_offset + 1,
        device=device,
    )

    val_loader = make_loader(
        val_records,
        batch_size=batch_size,
        shuffle=False,
        seed=SEED + seed_offset + 2,
        device=device,
    )

    optimizer = make_adamw(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    weights = loss_weights(target_stats, device)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sse = torch.zeros(TARGET_DIM, device=device)
        train_count = torch.zeros(TARGET_DIM, device=device, dtype=torch.long)

        for batch in train_loader:
            values, times, varis, obs_mask, demo, y, ymask = [
                x.to(device, non_blocking=True)
                for x in batch
            ]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(values, times, varis, obs_mask, demo)
                loss = target_balanced_mse(preds, y, ymask, weights)

            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {name}.")

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            with torch.no_grad():
                sse, count = sse_count(preds, y, ymask)
                train_sse += sse
                train_count += count

        model.eval()
        val_sse = torch.zeros(TARGET_DIM, device=device)
        val_count = torch.zeros(TARGET_DIM, device=device, dtype=torch.long)

        with torch.no_grad():
            for batch in val_loader:
                values, times, varis, obs_mask, demo, y, ymask = [
                    x.to(device, non_blocking=True)
                    for x in batch
                ]
                preds = model(values, times, varis, obs_mask, demo)
                sse, count = sse_count(preds, y, ymask)
                val_sse += sse
                val_count += count

        train_loss = epoch_mse(train_sse, train_count)
        val_loss = epoch_mse(val_sse, val_count)

        if EPOCH_PRINT_EVERY > 0 and (
            epoch == 1 or epoch % EPOCH_PRINT_EVERY == 0
        ):
            print(
                f"{name} epoch {epoch}: "
                f"train MSE={train_loss:.6f}, "
                f"val MSE={val_loss:.6f}"
            )

        if not math.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss: {name}.")

        if val_loss + MIN_DELTA < best_val:
            best_val = val_loss
            best_state = clone_state_dict(model)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"{name}: early stopping")
                break

    if best_state is None:
        best_state = clone_state_dict(model)

    model.load_state_dict(best_state)
    return best_state


def collect_errors(
    model: nn.Module,
    test_records: List[dict],
    target_stats: Dict[str, Dict[str, float]],
    origin: str,
    fold: int,
    device: torch.device,
    method_name: str,
    source_config: str,
    source_dataset: str,
    n_pretrain_datasets: int,
    seed: int,
) -> pd.DataFrame:
    if len(test_records) == 0:
        return pd.DataFrame(columns=ERROR_COLUMNS)

    loader = make_loader(
        test_records,
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=False,
        seed=seed + 5000 + fold,
        device=device,
    )

    means = torch.tensor(
        [target_stats[col]["mean"] for col in TARGET_COLS],
        device=device,
        dtype=torch.float32,
    )
    stds = torch.tensor(
        [target_stats[col]["std"] for col in TARGET_COLS],
        device=device,
        dtype=torch.float32,
    )
    usable_targets = torch.tensor(
        [target_stats[col]["n"] > 0 for col in TARGET_COLS],
        device=device,
        dtype=torch.bool,
    )

    frames = []
    offset = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            values, times, varis, obs_mask, demo, y_raw, ymask = [
                x.to(device, non_blocking=True)
                for x in batch
            ]

            batch_size = y_raw.size(0)
            batch_records = test_records[offset:offset + batch_size]
            offset += batch_size

            preds_z = model(values, times, varis, obs_mask, demo)
            preds_raw = preds_z * stds.view(1, -1) + means.view(1, -1)

            valid_mask = (
                (ymask > 0)
                & torch.isfinite(y_raw)
                & usable_targets.view(1, -1)
            )

            errors = preds_raw - torch.nan_to_num(y_raw, nan=0.0)
            if not torch.isfinite(errors[valid_mask]).all():
                raise FloatingPointError(
                    f"Non-finite prediction error for an observed target: {origin}, fold {fold}."
                )
            errors = errors.masked_fill(~valid_mask, float("nan"))

            error_df = pd.DataFrame(errors.cpu().numpy(), columns=TARGET_COLS)
            error_df = error_df.mask(~valid_mask.cpu().numpy())
            error_df.index.name = "row_id"

            long_df = (
                error_df
                .reset_index()
                .melt(
                    id_vars="row_id",
                    var_name="attribute",
                    value_name="error",
                )
                .dropna(subset=["error"])
            )

            if long_df.empty:
                continue

            row_ids = long_df["row_id"].astype(int).to_numpy()

            input_ages = np.asarray([r["input_age"] for r in batch_records], dtype=float)
            output_ages = np.asarray([r["target_age"] for r in batch_records], dtype=float)
            horizons = np.asarray([r["horizon"] for r in batch_records], dtype=float)
            n_input_visits = np.asarray([r["n_input_visits"] for r in batch_records], dtype=int)
            person_ids = np.asarray([str(r["person_id"]) for r in batch_records], dtype=object)

            long_df["method"] = method_name
            long_df["seed"] = seed
            long_df["dataset"] = origin
            long_df["fold"] = fold
            long_df["person_id"] = person_ids[row_ids]
            long_df["input_age"] = input_ages[row_ids]
            long_df["output_age"] = output_ages[row_ids]
            long_df["horizon"] = horizons[row_ids]
            long_df["n_input_visits"] = n_input_visits[row_ids]
            long_df["source_config"] = source_config
            long_df["source_dataset"] = source_dataset
            long_df["n_pretrain_datasets"] = n_pretrain_datasets

            frames.append(long_df[ERROR_COLUMNS])

    if len(frames) == 0:
        return pd.DataFrame(columns=ERROR_COLUMNS)

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Experiment setup helpers
# =============================================================================

def get_source_origins(origin: str, all_origins: List[str]) -> List[str]:
    if MODE == "target_only":
        return []

    if MODE in {"topk_pretrain", "pooled"}:
        if origin not in ORIGIN_MAP:
            raise ValueError(f"Missing ORIGIN_MAP entry for origin={origin!r}")
        if N_ORIGINS < 1 or N_ORIGINS > 6:
            raise ValueError("--n-origins must be between 1 and 6.")
        return ORIGIN_MAP[origin][:N_ORIGINS]

    if MODE == "single_source":
        if SOURCE_DATASET is None:
            raise ValueError("--source-dataset is required when --mode single_source.")
        if SOURCE_DATASET not in all_origins:
            raise ValueError(
                f"source dataset {SOURCE_DATASET!r} is not in available origins: {all_origins}"
            )
        if SOURCE_DATASET == origin:
            return []
        return [SOURCE_DATASET]

    raise ValueError(f"Unknown MODE: {MODE}")


def get_method_metadata(source_origins: List[str]) -> Tuple[str, str, str, int]:
    source_dataset = "|".join(source_origins)
    n_sources = len(source_origins)

    if MODE == "target_only":
        return "strats_no_pretrain", "target_only", "", 0

    if MODE == "topk_pretrain":
        method_name = "strats_main" if n_sources == 6 else f"strats_dataset_pretrain_{n_sources}"
        return method_name, f"top_{n_sources}", source_dataset, n_sources

    if MODE == "single_source":
        source_name = safe_stem(source_origins[0]) if source_origins else "none"
        return f"strats_single_source_{source_name}", "single_source", source_dataset, n_sources

    if MODE == "pooled":
        method_name = "strats_pooled_all_sources" if n_sources == 6 else f"strats_pooled_top_{n_sources}"
        return method_name, f"pooled_top_{n_sources}", source_dataset, n_sources

    raise ValueError(f"Unknown MODE: {MODE}")


def apply_target_train_percent(
    df_train_target: pd.DataFrame,
    origin: str,
    fold: int,
    target_train_subsets: Optional[dict],
) -> pd.DataFrame:
    if target_train_subsets is None:
        return df_train_target

    keep_train_ids = set(
        map(
            str,
            target_train_subsets[origin][str(fold)][str(TRAIN_DATA_PERCENT)],
        )
    )

    return df_train_target[
        df_train_target["PersonID"].astype(str).isin(keep_train_ids)
    ].copy()


def make_pretrain_split(
    df_pretrain: pd.DataFrame,
    split5k: List[set],
    origin_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_pretrain_val = df_pretrain[
        df_pretrain["PersonID"].astype(str).isin(split5k[0])
    ].copy()

    df_pretrain_train = df_pretrain[
        ~df_pretrain["PersonID"].astype(str).isin(split5k[0])
    ].copy()

    # Fallback for cases where split5k[0] has no source people.
    if len(df_pretrain_val) == 0 and len(df_pretrain) > 0:
        ids = df_pretrain["PersonID"].drop_duplicates().astype(str).to_numpy()
        rng = np.random.default_rng(SEED + origin_idx)
        rng.shuffle(ids)
        n_val = max(1, round(0.2 * len(ids)))
        val_ids = set(ids[:n_val])

        df_pretrain_val = df_pretrain[
            df_pretrain["PersonID"].astype(str).isin(val_ids)
        ].copy()
        df_pretrain_train = df_pretrain[
            ~df_pretrain["PersonID"].astype(str).isin(val_ids)
        ].copy()

    return df_pretrain_train, df_pretrain_val


# =============================================================================
# Main leave-one-dataset-out experiment
# =============================================================================

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {MODE}")
    print(f"Seed: {SEED}")
    print(f"Train data percent: {TRAIN_DATA_PERCENT}")
    print(f"Output path: {FINAL_ERROR_PATH}")
    print(f"Maximum observations: {MAX_OBS}")

    if FINAL_ERROR_PATH.exists():
        raise FileExistsError(f"Output already exists: {FINAL_ERROR_PATH}. Choose another --output path.")

    df = pd.read_csv(DATA_PATH, dtype={"PersonID": str}, low_memory=False)
    validate_dataset(df)
    df = preprocess_dataframe(df)

    split5k = load_split5k(SPLIT_PATH)
    validate_folds(df, split5k)
    target_train_subsets = load_target_train_subsets(TARGET_TRAIN_SUBSETS_PATH)

    pairs = pd.read_csv(PAIRS_PATH, dtype={"PersonID": str})
    pairs["PersonID"] = pairs["PersonID"].astype(str)
    pairs["input_rows"] = pairs["input_rows"].apply(parse_index_tuple)
    pairs["output_row"] = pd.to_numeric(pairs["output_row"], errors="coerce").astype("Int64")
    pairs = pairs.dropna(subset=["output_row"]).copy()
    pairs = pairs[pairs["input_rows"].map(len) > 0].copy()
    pairs["output_row"] = pairs["output_row"].astype(int)

    if set(pairs["PersonID"]) != set(df["PersonID"]):
        raise ValueError("Pair-table participants do not match dataset.csv; run prepare_data.py.")

    all_origins = sorted(df["Origin"].dropna().unique().tolist())
    error_frames = []

    for origin_idx, origin in enumerate(all_origins):
        print("\n" + "=" * 100)
        print(f"Target dataset: {origin}")
        print("=" * 100)

        source_origins = get_source_origins(origin, all_origins)

        if MODE == "single_source" and len(source_origins) == 0:
            print(f"Skipping target={origin} because source_dataset equals target dataset.")
            continue

        method_name, source_config, source_dataset, n_pretrain_datasets = get_method_metadata(source_origins)

        print(f"Method: {method_name}")
        print(f"Source config: {source_config}")
        print(f"Source datasets: {source_origins}")

        df_target = df[df["Origin"].eq(origin)].copy()
        df_source = df[
            df["Origin"].isin(source_origins)
            & ~df["Origin"].eq(origin)
        ].copy()

        best_pretrain_state = None

        if MODE in {"topk_pretrain", "single_source"}:
            df_pretrain_train, df_pretrain_val = make_pretrain_split(
                df_source,
                split5k,
                origin_idx,
            )

            model = new_model(device)

            pretrain_train_records, pretrain_val_records, pretrain_target_stats = prepare_train_val(
                df_pretrain_train,
                df_pretrain_val,
                pairs,
                seed=SEED + 10000 + origin_idx,
                max_train_records=MAX_PRETRAIN_RECORDS,
            )

            print(
                "Pretrain records: "
                f"train={len(pretrain_train_records)}, "
                f"val={len(pretrain_val_records)}"
            )

            if len(pretrain_train_records) > 0 and len(pretrain_val_records) > 0:
                best_pretrain_state = train_model(
                    model=model,
                    train_records=pretrain_train_records,
                    val_records=pretrain_val_records,
                    target_stats=pretrain_target_stats,
                    batch_size=PRETRAIN_BATCH_SIZE,
                    lr=PRETRAIN_LR,
                    max_epochs=MAX_PRETRAIN_EPOCHS,
                    patience=PATIENCE,
                    device=device,
                    seed_offset=1000 * (origin_idx + 1),
                    name=f"pretrain {origin}",
                )
            else:
                raise ValueError(f"Empty pretraining or validation records for {origin}.")

        for fold in range(5):
            print("\n" + "-" * 80)
            print(f"Fine-tune/test: target={origin}, fold={fold}")
            print("-" * 80)

            test_ids = split5k[fold]
            val_ids = split5k[fold - 1]

            df_test = df_target[df_target["PersonID"].astype(str).isin(test_ids)].copy()
            df_val = df_target[df_target["PersonID"].astype(str).isin(val_ids)].copy()

            df_train_target = df_target[
                ~df_target["PersonID"].astype(str).isin(test_ids)
                & ~df_target["PersonID"].astype(str).isin(val_ids)
            ].copy()

            # Reduced-data experiments reduce only target fine-tuning training people.
            # Validation and test folds are unchanged.
            df_train_target = apply_target_train_percent(
                df_train_target,
                origin,
                fold,
                target_train_subsets,
            )

            if MODE == "pooled":
                # Pooled control: train from scratch on target train + source datasets.
                # Validation/test stay target-only.
                # Normalization/statistics stay target-train-only for a fair target-calibrated control.
                df_train = pd.concat([df_train_target, df_source], axis=0).copy()
                stats_df_for_finetune = df_train_target
            else:
                df_train = df_train_target
                stats_df_for_finetune = df_train_target

            train_records, val_records, test_records, fold_target_stats = prepare_finetune(
                df_train,
                df_val,
                df_test,
                pairs,
                seed=SEED + 20000 + 10 * origin_idx + fold,
                max_train_records=MAX_FINETUNE_RECORDS,
                stats_df=stats_df_for_finetune,
            )

            print(
                "Fine-tune records: "
                f"train={len(train_records)}, "
                f"val={len(val_records)}, "
                f"test={len(test_records)}"
            )

            no_signal_targets = [
                col for col in TARGET_COLS
                if fold_target_stats[col]["n"] == 0
            ]
            if len(no_signal_targets) > 0:
                print(f"Targets with no train signal in this fold: {no_signal_targets}")

            if not train_records or not val_records or not test_records:
                raise ValueError(f"Empty training, validation or test records: {origin}, fold {fold}.")

            fold_model = new_model(device)
            if best_pretrain_state is not None:
                fold_model.load_state_dict(best_pretrain_state)

            if len(train_records) > 0 and len(val_records) > 0:
                train_model(
                    model=fold_model,
                    train_records=train_records,
                    val_records=val_records,
                    target_stats=fold_target_stats,
                    batch_size=FINETUNE_BATCH_SIZE,
                    lr=FINETUNE_LR,
                    max_epochs=MAX_FINETUNE_EPOCHS,
                    patience=PATIENCE,
                    device=device,
                    seed_offset=2000 * (origin_idx + 1) + fold,
                    name=f"finetune {origin} fold {fold}",
                )
            else:
                raise ValueError(f"Empty training or validation records: {origin}, fold {fold}.")

            fold_errors = collect_errors(
                model=fold_model,
                test_records=test_records,
                target_stats=fold_target_stats,
                origin=origin,
                fold=fold,
                device=device,
                method_name=method_name,
                source_config=source_config,
                source_dataset=source_dataset,
                n_pretrain_datasets=n_pretrain_datasets,
                seed=SEED,
            )

            if not fold_errors.empty:
                error_frames.append(fold_errors)

            print(f"Collected {len(fold_errors)} error rows for target={origin}, fold={fold}.")

    if error_frames:
        final_errors = pd.concat(error_frames, ignore_index=True)
    else:
        final_errors = pd.DataFrame(columns=ERROR_COLUMNS)

    if final_errors.empty:
        raise ValueError("No observed-target errors were produced.")
    FINAL_ERROR_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_errors.to_csv(FINAL_ERROR_PATH, index=False, mode="x")

    print(
        f"\nFinished. Saved errors to {FINAL_ERROR_PATH} "
        f"with {len(final_errors)} rows."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--mode",
        choices=["target_only", "topk_pretrain", "single_source", "pooled"],
        default=MODE,
    )
    parser.add_argument("--n-origins", type=int, default=N_ORIGINS)
    parser.add_argument("--source-dataset", type=str, default=SOURCE_DATASET)
    parser.add_argument("--train-data-percent", type=int, default=TRAIN_DATA_PERCENT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--pairs-path", type=str, default=PAIRS_PATH)
    parser.add_argument("--split-path", type=str, default=SPLIT_PATH)
    parser.add_argument("--target-train-subsets-path", type=str, default=TARGET_TRAIN_SUBSETS_PATH)

    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--max-obs", type=int, default=MAX_OBS if MAX_OBS is not None else -1)

    parser.add_argument("--pretrain-batch-size", type=int, default=PRETRAIN_BATCH_SIZE)
    parser.add_argument("--finetune-batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    parser.add_argument("--max-pretrain-epochs", type=int, default=MAX_PRETRAIN_EPOCHS)
    parser.add_argument("--max-finetune-epochs", type=int, default=MAX_FINETUNE_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    parser.add_argument("--max-pretrain-records", type=int, default=-1)
    parser.add_argument("--max-finetune-records", type=int, default=-1)
    parser.add_argument("--epoch-print-every", type=int, default=EPOCH_PRINT_EVERY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    return parser.parse_args()


def apply_args(args) -> None:
    global SEED
    global DATA_PATH, PAIRS_PATH, SPLIT_PATH, TARGET_TRAIN_SUBSETS_PATH
    global FINAL_ERROR_PATH
    global MODE, N_ORIGINS, SOURCE_DATASET, TRAIN_DATA_PERCENT
    global HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, MAX_OBS
    global PRETRAIN_BATCH_SIZE, FINETUNE_BATCH_SIZE
    global MAX_PRETRAIN_EPOCHS, MAX_FINETUNE_EPOCHS, PATIENCE
    global MAX_PRETRAIN_RECORDS, MAX_FINETUNE_RECORDS
    global EPOCH_PRINT_EVERY, NUM_WORKERS

    SEED = args.seed
    DATA_PATH = args.data_path
    PAIRS_PATH = args.pairs_path
    SPLIT_PATH = args.split_path
    TARGET_TRAIN_SUBSETS_PATH = args.target_train_subsets_path

    MODE = args.mode
    N_ORIGINS = args.n_origins
    SOURCE_DATASET = args.source_dataset
    TRAIN_DATA_PERCENT = args.train_data_percent

    HIDDEN_DIM = args.hidden_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    MAX_OBS = None if args.max_obs < 0 else args.max_obs

    PRETRAIN_BATCH_SIZE = args.pretrain_batch_size
    FINETUNE_BATCH_SIZE = args.finetune_batch_size
    MAX_PRETRAIN_EPOCHS = args.max_pretrain_epochs
    MAX_FINETUNE_EPOCHS = args.max_finetune_epochs
    PATIENCE = args.patience

    MAX_PRETRAIN_RECORDS = None if args.max_pretrain_records < 0 else args.max_pretrain_records
    MAX_FINETUNE_RECORDS = None if args.max_finetune_records < 0 else args.max_finetune_records

    EPOCH_PRINT_EVERY = args.epoch_print_every
    NUM_WORKERS = args.num_workers

    if args.output is not None:
        FINAL_ERROR_PATH = Path(args.output)
    else:
        pct_suffix = "" if TRAIN_DATA_PERCENT == 100 else f"_pct{TRAIN_DATA_PERCENT}"
        seed_suffix = f"_seed{SEED}"
        cap_suffix = "" if MAX_OBS == 256 else ("_maxobs_none" if MAX_OBS is None else f"_maxobs_{MAX_OBS}")
        run_suffix = f"{cap_suffix}{pct_suffix}{seed_suffix}"

        if MODE == "target_only":
            FINAL_ERROR_PATH = (ERRORS_DIR / f"strats_no_pretrain{run_suffix}.csv")

        elif MODE == "topk_pretrain":
            if N_ORIGINS == 6:
                FINAL_ERROR_PATH = (ERRORS_DIR / f"strats_main{run_suffix}.csv")
            else:
                FINAL_ERROR_PATH = (ERRORS_DIR / f"strats_dataset_pretrain_{N_ORIGINS}{run_suffix}.csv")

        elif MODE == "single_source":
            source_name = safe_stem(SOURCE_DATASET or "none")
            FINAL_ERROR_PATH = (ERRORS_DIR / f"strats_single_source_{source_name}{run_suffix}.csv")

        elif MODE == "pooled":
            if N_ORIGINS == 6:
                FINAL_ERROR_PATH = (ERRORS_DIR / f"strats_pooled_all_sources{run_suffix}.csv")
            else:
                FINAL_ERROR_PATH = (ERRORS_DIR / f"strats_pooled_top_{N_ORIGINS}{run_suffix}.csv")

        else:
            raise ValueError(f"Unknown mode: {MODE}")

    if TRAIN_DATA_PERCENT not in {10, 30, 50, 70, 90, 100}:
        raise ValueError("--train-data-percent must be one of 10, 30, 50, 70, 90, 100.")

    if MODE in {"topk_pretrain", "pooled"} and not (1 <= N_ORIGINS <= 6):
        raise ValueError("--n-origins must be between 1 and 6 for topk_pretrain/pooled.")

    if MODE == "single_source" and not SOURCE_DATASET:
        raise ValueError("--source-dataset is required for --mode single_source.")
    if HIDDEN_DIM <= 0 or NUM_HEADS <= 0 or HIDDEN_DIM % NUM_HEADS:
        raise ValueError("--hidden-dim must be positive and divisible by --num-heads.")
    if min(NUM_LAYERS, PRETRAIN_BATCH_SIZE, FINETUNE_BATCH_SIZE,
           MAX_PRETRAIN_EPOCHS, MAX_FINETUNE_EPOCHS, PATIENCE) < 1:
        raise ValueError("Layer, batch, epoch and patience values must be positive.")
    if NUM_WORKERS < 0 or MAX_OBS == 0:
        raise ValueError("--num-workers must be non-negative; --max-obs must be positive or -1.")
    if MAX_PRETRAIN_RECORDS == 0 or MAX_FINETUNE_RECORDS == 0:
        raise ValueError("Record limits must be positive or -1.")


if __name__ == "__main__":
    args = parse_args()
    apply_args(args)
    main()
