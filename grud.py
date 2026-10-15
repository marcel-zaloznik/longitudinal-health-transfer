#!/usr/bin/env python3
"""GRU-D adaptation for longitudinal multi-target regression.

Based on Che et al. (2018) and public GRU-D implementations.
See THIRD_PARTY_NOTICES.md for attribution.
"""

import argparse
import ast
import copy
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import validate_dataset, validate_folds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
SEED = 1234
TRAIN_DATA_PERCENT = 100
ONLY_ONE_DATASET = True
OUTPUT_PATH = None
DATA_DIR = ROOT / "data"
ERRORS_DIR = ROOT / "results" / "errors"
MAX_EPOCHS = 1000
NUM_WORKERS = min(4, os.cpu_count() or 1)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Reproducibility and device setup
# ----------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_index_tuple(cell):
    if isinstance(cell, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in cell)
    if pd.isna(cell):
        return tuple()

    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, np.ndarray)):
            return tuple(int(x) for x in v)
        return (int(v),)
    except Exception:
        nums = re.findall(r"-?\d+", s)
        return tuple(int(x) for x in nums)


def move_column(df, col_name, new_idx):
    col = df.pop(col_name)
    df.insert(new_idx, col_name, col)
    return df


# ----------------------------------------------------------------------
# Sequence creation
# ----------------------------------------------------------------------

def create_sequences(df, input_output_pairs, static_cols, dynamic_cols):
    """
    Creates variable-length sequences for GRU-D using precomputed
    input/output pairs from prepare_data.py.

    Returns:
        sequences: list of tuples (X_seq, X_last_seq, M_seq, Delta_seq)
        targets: np.ndarray of shape (N_sequences, n_dynamic_features)
        target_masks: np.ndarray of shape (N_sequences, n_dynamic_features)
        metadata: pd.DataFrame with input/output row and age information
    """
    sequences = []
    targets = []
    target_masks = []
    metadata = []

    pairs_df = input_output_pairs[
        input_output_pairs["PersonID"].isin(df["PersonID"].unique())
    ].copy()

    static_dim = len(static_cols)
    person_cache = {}

    for pair in pairs_df.itertuples(index=False):
        person_id = pair.PersonID
        input_rows = tuple(int(x) for x in pair.input_rows)
        output_row = int(pair.output_row)

        if len(input_rows) == 0:
            continue

        if person_id not in person_cache:
            grp = df[df["PersonID"] == person_id].sort_values("Age", kind="mergesort")

            static_vals = grp[static_cols].astype(float).values
            dynamic_vals = grp[dynamic_cols].astype(float).values
            times = grp["Age"].astype(float).to_numpy()
            row_ids = grp.index.to_list()

            dynamic_mask = ~np.isnan(dynamic_vals)
            dynamic_filled = np.nan_to_num(dynamic_vals, nan=0.0)
            static_mask = np.ones_like(static_vals, dtype=float)

            X_all = np.concatenate([static_vals, dynamic_filled], axis=1)
            M_all = np.concatenate([static_mask, dynamic_mask.astype(float)], axis=1)

            X_last = X_all.copy()
            for j in range(static_dim, X_all.shape[1]):
                for i in range(1, len(row_ids)):
                    if M_all[i, j] == 0:
                        X_last[i, j] = X_last[i - 1, j]

            person_cache[person_id] = {
                "rowpos": {row_id: pos for pos, row_id in enumerate(row_ids)},
                "times": times,
                "X_all": X_all,
                "M_all": M_all,
                "X_last": X_last,
                "dynamic_filled": dynamic_filled,
                "dynamic_mask": dynamic_mask,
            }

        cache = person_cache[person_id]

        if output_row not in cache["rowpos"]:
            continue
        if any(row_idx not in cache["rowpos"] for row_idx in input_rows):
            continue

        prefix_pos = [cache["rowpos"][row_idx] for row_idx in input_rows]
        out_src_pos = cache["rowpos"][output_row]
        seq_pos = prefix_pos + [out_src_pos]

        X_seq = cache["X_all"][seq_pos].copy()
        M_seq = cache["M_all"][seq_pos].copy()
        X_last_seq = cache["X_last"][seq_pos].copy()

        # GRU-D requires a variable-specific elapsed time: for each variable,
        # Delta[t, j] is the time since that variable was last observed.
        # Static variables are always treated as observed and are not decayed.
        Delta_seq = np.zeros_like(X_seq)
        seq_times = cache["times"][seq_pos]
        for k in range(1, len(seq_pos)):
            elapsed = seq_times[k] - seq_times[k - 1]
            Delta_seq[k, :static_dim] = 0.0
            for j in range(static_dim, X_seq.shape[1]):
                if M_seq[k - 1, j] == 1:
                    Delta_seq[k, j] = elapsed
                else:
                    Delta_seq[k, j] = Delta_seq[k - 1, j] + elapsed

        out_pos = len(seq_pos) - 1
        # Hide the future dynamic values at the target visit. The target age is
        # still available through the static Age_z component at the final step.
        M_seq[out_pos, static_dim:] = 0.0

        last_prefix_pos = prefix_pos[-1]
        X_seq[out_pos, static_dim:] = 0.0
        X_last_seq[out_pos, static_dim:] = cache["X_last"][
            last_prefix_pos, static_dim:
        ]

        target_vals = cache["dynamic_filled"][out_src_pos].copy()
        target_mask = cache["dynamic_mask"][out_src_pos].astype(float)

        sequences.append((X_seq, X_last_seq, M_seq, Delta_seq))
        targets.append(target_vals)
        target_masks.append(target_mask)
        metadata.append(
            {
                "person_id": str(person_id),
                "input_row": int(input_rows[-1]),
                "output_row": output_row,
                "input_age": float(cache["times"][last_prefix_pos]),
                "output_age": float(cache["times"][out_src_pos]),
                "horizon": float(cache["times"][out_src_pos] - cache["times"][last_prefix_pos]),
                "n_input_visits": int(len(input_rows)),
            }
        )

    if targets:
        targets = np.asarray(targets, dtype=float)
        target_masks = np.asarray(target_masks, dtype=float)
    else:
        targets = np.empty((0, len(dynamic_cols)), dtype=float)
        target_masks = np.empty((0, len(dynamic_cols)), dtype=float)

    metadata = pd.DataFrame(metadata)

    return sequences, targets, target_masks, metadata


# ----------------------------------------------------------------------
# Dataset + custom collate (handles variable-length sequences)
# ----------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Holds variable-length GRU-D sequences.
    Each item: (X_seq, X_last_seq, M_seq, Delta_seq, target, target_mask)
    """

    def __init__(self, sequences, targets, target_masks):
        self.sequences = [
            tuple(torch.as_tensor(a, dtype=torch.float32) for a in seq)
            for seq in sequences
        ]
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.target_masks = torch.as_tensor(target_masks, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq, X_last_seq, M_seq, Delta_seq = self.sequences[idx]
        return X_seq, X_last_seq, M_seq, Delta_seq, self.targets[idx], self.target_masks[idx]


def grud_collate_fn(batch):
    """
    Pads a batch of variable-length sequences to the same length.

    Args:
        batch: list of tuples
          (X_seq, X_last_seq, M_seq, Delta_seq, y, y_mask),
          where each sequence has shape (T_i, D).

    Returns:
        X_padded:      (B, T_max, D)
        X_last_padded: (B, T_max, D)
        M_padded:      (B, T_max, D)
        Delta_padded:  (B, T_max, D)
        seq_mask:      (B, T_max) 1 for real steps, 0 for padding
        y:             (B, n_dynamic_features)
        y_mask:        (B, n_dynamic_features)
    """
    X_list, X_last_list, M_list, Delta_list, y_list, y_mask_list = zip(*batch)

    seq_lengths = torch.tensor([x.size(0) for x in X_list], dtype=torch.long)

    X_padded = pad_sequence(X_list, batch_first=True)
    X_last_padded = pad_sequence(X_last_list, batch_first=True)
    M_padded = pad_sequence(M_list, batch_first=True)
    Delta_padded = pad_sequence(Delta_list, batch_first=True)

    y = torch.stack(y_list, dim=0)
    y_mask = torch.stack(y_mask_list, dim=0)

    max_len = X_padded.size(1)
    seq_mask = (
        torch.arange(max_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
    ).float()

    return X_padded, X_last_padded, M_padded, Delta_padded, seq_mask, y, y_mask


# ----------------------------------------------------------------------
# GRU-D model
# ----------------------------------------------------------------------

class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, X_mean, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.b_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.W_gamma_h = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_gamma_h = nn.Parameter(torch.zeros(hidden_size))

        gate_input_dim = input_size + hidden_size + input_size
        self.z_gate = nn.Linear(gate_input_dim, hidden_size)
        self.r_gate = nn.Linear(gate_input_dim, hidden_size)
        self.h_gate = nn.Linear(gate_input_dim, hidden_size)

        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32))

        nn.init.xavier_uniform_(self.z_gate.weight)
        nn.init.zeros_(self.z_gate.bias)
        nn.init.xavier_uniform_(self.r_gate.weight)
        nn.init.zeros_(self.r_gate.bias)
        nn.init.xavier_uniform_(self.h_gate.weight)
        nn.init.zeros_(self.h_gate.bias)

        self.output_layer = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, X, X_last, M, Delta, seq_mask=None):
        batch_size, seq_len, _ = X.size()
        device = X.device

        h = torch.zeros(batch_size, self.hidden_size, device=device)

        if seq_mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=device)

        for t in range(seq_len):
            mask_t = seq_mask[:, t].unsqueeze(1)

            x_t = X[:, t, :]
            x_last_t = X_last[:, t, :]
            m_t = M[:, t, :]
            delta_t = Delta[:, t, :]

            h_prev = h

            gamma_x = torch.exp(
                -torch.clamp(self.W_gamma_x * delta_t + self.b_gamma_x, min=0.0)
            )

            gamma_h = torch.exp(
                -torch.clamp(delta_t.matmul(self.W_gamma_h.t()) + self.b_gamma_h, min=0.0)
            )

            x_mean = self.X_mean.unsqueeze(0).expand_as(x_t)
            x_hat = m_t * x_t + (1.0 - m_t) * (
                gamma_x * x_last_t + (1.0 - gamma_x) * x_mean
            )

            h_decay = gamma_h * h_prev

            combined = torch.cat([x_hat, h_decay, m_t], dim=1)
            z = torch.sigmoid(self.z_gate(combined))
            r = torch.sigmoid(self.r_gate(combined))

            combined_r = torch.cat([x_hat, r * h_decay, m_t], dim=1)
            h_tilde = torch.tanh(self.h_gate(combined_r))

            h_new = (1.0 - z) * h_decay + z * h_tilde
            h = mask_t * h_new + (1.0 - mask_t) * h_prev

        y_hat = self.output_layer(h)
        return y_hat


# ----------------------------------------------------------------------
# Training/evaluation helpers
# ----------------------------------------------------------------------

def target_balanced_masked_mse(y_pred, y_true, y_mask, target_weights):
    """
    Target-balanced masked MSE.

    Prevents dense targets from dominating sparse targets.
    """
    se = (y_pred - y_true) ** 2
    weighted_mask = y_mask * target_weights.view(1, -1)
    denom = weighted_mask.sum().clamp_min(1e-8)
    return (se * weighted_mask).sum() / denom


def move_batch_to_device(batch):
    return [x.to(DEVICE, non_blocking=True) for x in batch]


def make_loader(dataset, batch_size, shuffle):
    num_workers = NUM_WORKERS

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=grud_collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=num_workers,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    set_all_seeds(SEED)

    print("Using device:", DEVICE)
    print("Seed:", SEED)

    data_dir = DATA_DIR
    errors_dir = ERRORS_DIR

    if OUTPUT_PATH is not None:
        save_path = Path(OUTPUT_PATH)
    else:
        source_tag = "target_only" if ONLY_ONE_DATASET else "pooled_all_datasets"
        pct_tag = "" if TRAIN_DATA_PERCENT == 100 else f"_pct{TRAIN_DATA_PERCENT}"
        save_path = errors_dir / f"gru_d_{source_tag}{pct_tag}_seed{SEED}.csv"
    if save_path.exists():
        raise FileExistsError(f"Output already exists: {save_path}. Choose another --output path.")

    # Data
    df = pd.read_csv(data_dir / "dataset.csv", dtype={"PersonID": str}, low_memory=False)
    validate_dataset(df)

    # Make smoking and diabetes numerical
    df["Smoking"] = df["Smoking"].map({
        "No": 0,
        "Quit/strike": 1,
        "Yes": 2,
    }).astype("Int64")

    df["Diabetes"] = df["Diabetes"].map({
        "No": 0,
        "Yes": 1,
    }).astype("Int64")

    with open(data_dir / "split5k.json", "r", encoding="utf-8") as f:
        split5k = json.load(f)
    validate_folds(df, split5k)

    target_train_subsets = None
    if TRAIN_DATA_PERCENT != 100:
        with open(
            data_dir / "target_train_subsets_by_dataset_fold_percent.json",
            "r",
            encoding="utf-8",
        ) as f:
            target_train_subsets = json.load(f)

    input_output_pairs = pd.read_csv(data_dir / "input_output_pairs.csv", dtype={"PersonID": str})
    input_output_pairs["input_rows"] = input_output_pairs["input_rows"].apply(parse_index_tuple)
    input_output_pairs["output_row"] = pd.to_numeric(
        input_output_pairs["output_row"], errors="coerce"
    ).astype("Int64")
    input_output_pairs = input_output_pairs.dropna(subset=["output_row"]).copy()
    input_output_pairs = input_output_pairs[
        input_output_pairs["input_rows"].map(len) > 0
    ].copy()
    input_output_pairs["output_row"] = input_output_pairs["output_row"].astype(int)

    error_frames = []

    only_one_dataset = ONLY_ONE_DATASET

    for origin in df["Origin"].unique():
        df_origin = df[df["Origin"] == origin].copy()

        for k_split in range(5):
            df_test = df_origin[df_origin["PersonID"].isin(split5k[k_split])].copy()
            df_val = df_origin[df_origin["PersonID"].isin(split5k[k_split - 1])].copy()

            # Train data only from one dataset
            if only_one_dataset:
                df_train = df_origin[~df_origin["PersonID"].isin(split5k[k_split])].copy()
                df_train = df_train[
                    ~df_train["PersonID"].isin(split5k[k_split - 1])
                ].copy()

            # Train also on six other datasets
            else:
                train_mask = (df["Origin"].eq(origin)) & (
                    (df["PersonID"].isin(split5k[k_split]))
                    | (df["PersonID"].isin(split5k[k_split - 1]))
                )
                df_train = df.loc[~train_mask].copy()

            if target_train_subsets is not None:
                keep_target_train_ids = set(
                    map(
                        str,
                        target_train_subsets[origin][str(k_split)][str(TRAIN_DATA_PERCENT)],
                    )
                )

                if only_one_dataset:
                    df_train = df_train[
                        df_train["PersonID"].astype(str).isin(keep_target_train_ids)
                    ].copy()
                else:
                    target_part = df_train[
                        df_train["Origin"].eq(origin)
                        & df_train["PersonID"].astype(str).isin(keep_target_train_ids)
                    ].copy()

                    source_part = df_train[
                        ~df_train["Origin"].eq(origin)
                    ].copy()

                    df_train = pd.concat([target_part, source_part], axis=0).copy()

            df_train = df_train.drop("Origin", axis=1)
            df_val = df_val.drop("Origin", axis=1)
            df_test = df_test.drop("Origin", axis=1)

            df_train = df_train.drop("Country", axis=1)
            df_val = df_val.drop("Country", axis=1)
            df_test = df_test.drop("Country", axis=1)

            df_train = pd.get_dummies(df_train, columns=["Sex"])
            df_val = pd.get_dummies(df_val, columns=["Sex"])
            df_test = pd.get_dummies(df_test, columns=["Sex"])

            if "Sex_Female" not in df_train.columns:
                df_train["Sex_Female"] = False
            if "Sex_Female" not in df_val.columns:
                df_val["Sex_Female"] = False
            if "Sex_Female" not in df_test.columns:
                df_test["Sex_Female"] = False
            if "Sex_Male" not in df_train.columns:
                df_train["Sex_Male"] = False
            if "Sex_Male" not in df_val.columns:
                df_val["Sex_Male"] = False
            if "Sex_Male" not in df_test.columns:
                df_test["Sex_Male"] = False

            df_train = move_column(df_train, "Sex_Male", 1)
            df_train = move_column(df_train, "Sex_Female", 2)
            df_train["Sex_Male"] = df_train["Sex_Male"].astype(int)
            df_train["Sex_Female"] = df_train["Sex_Female"].astype(int)

            df_val = move_column(df_val, "Sex_Male", 1)
            df_val = move_column(df_val, "Sex_Female", 2)
            df_val["Sex_Male"] = df_val["Sex_Male"].astype(int)
            df_val["Sex_Female"] = df_val["Sex_Female"].astype(int)

            df_test = move_column(df_test, "Sex_Male", 1)
            df_test = move_column(df_test, "Sex_Female", 2)
            df_test["Sex_Male"] = df_test["Sex_Male"].astype(int)
            df_test["Sex_Female"] = df_test["Sex_Female"].astype(int)

            # Column definitions: static and dynamic features
            age_mean = pd.to_numeric(df_train["Age"], errors="coerce").mean()
            age_std = pd.to_numeric(df_train["Age"], errors="coerce").std(ddof=0)

            if not np.isfinite(age_std) or age_std == 0:
                age_std = 1.0

            for _df in (df_train, df_val, df_test):
                _df["Age_z"] = (
                    pd.to_numeric(_df["Age"], errors="coerce") - age_mean
                ) / age_std
                _df["Age_z"] = _df["Age_z"].fillna(0.0)

            static_cols = ["Sex_Male", "Sex_Female", "Age_z"]

            dynamic_cols_all = [
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

            # Drop dynamic columns that are completely missing in TRAIN for this origin/split.
            dynamic_cols = [c for c in dynamic_cols_all if df_train[c].notna().any()]
            dropped_dynamic_cols = [c for c in dynamic_cols_all if c not in dynamic_cols]
            for _df in (df_train, df_val, df_test):
                _df.drop(columns=dropped_dynamic_cols, inplace=True, errors="ignore")

            # Standardization of dynamic features based on TRAIN only.
            dynamic_means = df_train[dynamic_cols].astype(float).mean(skipna=True)
            dynamic_stds = df_train[dynamic_cols].astype(float).std(skipna=True)

            dynamic_stds[(dynamic_stds == 0) | (dynamic_stds.isna())] = 1.0

            dynamic_means_arr = dynamic_means[dynamic_cols].values
            dynamic_stds_arr = dynamic_stds[dynamic_cols].values

            for _df in [df_train, df_val, df_test]:
                _df[dynamic_cols] = (
                    _df[dynamic_cols].astype(float) - dynamic_means
                ) / dynamic_stds

            train_seqs, train_targets, train_target_masks, _ = create_sequences(
                df_train, input_output_pairs, static_cols, dynamic_cols
            )
            val_seqs, val_targets, val_target_masks, _ = create_sequences(
                df_val, input_output_pairs, static_cols, dynamic_cols
            )
            test_seqs, test_targets, test_target_masks, test_meta = create_sequences(
                df_test, input_output_pairs, static_cols, dynamic_cols
            )

            if not train_seqs or not val_seqs or not test_seqs:
                raise ValueError(f"Empty training, validation or test sequences: {origin}, fold {k_split}.")

            target_counts = torch.tensor(
                train_target_masks.sum(axis=0),
                dtype=torch.float32,
                device=DEVICE,
            )

            target_weights = torch.where(
                target_counts > 0,
                1.0 / target_counts.clamp_min(1.0),
                torch.zeros_like(target_counts),
            )

            train_dataset = SequenceDataset(train_seqs, train_targets, train_target_masks)
            val_dataset = SequenceDataset(val_seqs, val_targets, val_target_masks)
            test_dataset = SequenceDataset(test_seqs, test_targets, test_target_masks)

            train_loader = make_loader(train_dataset, batch_size=512, shuffle=True)
            val_loader = make_loader(val_dataset, batch_size=512, shuffle=False)
            test_loader = make_loader(test_dataset, batch_size=512, shuffle=False)

            input_dim = len(static_cols) + len(dynamic_cols)
            hidden_dim = 64
            output_dim = len(dynamic_cols)

            X_mean = df_train[static_cols + dynamic_cols].astype(float).mean(skipna=True).values

            model = GRUD(
                input_size=input_dim,
                hidden_size=hidden_dim,
                X_mean=X_mean,
                output_size=output_dim,
            ).to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop with early stopping
            patience = 10
            min_delta = 1e-4
            best_val_loss = float("inf")
            best_model_state = None
            best_epoch = 0
            epochs_no_improve = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                train_loss_sum = 0.0
                train_count = 0.0

                for batch in train_loader:
                    X, X_last, M, Delta, seq_mask, y_true, y_mask = move_batch_to_device(batch)

                    optimizer.zero_grad(set_to_none=True)

                    y_pred = model(X, X_last, M, Delta, seq_mask=seq_mask)

                    mse_loss = target_balanced_masked_mse(
                        y_pred=y_pred,
                        y_true=y_true,
                        y_mask=y_mask,
                        target_weights=target_weights,
                    )

                    if not torch.isfinite(mse_loss):
                        raise FloatingPointError(f"Non-finite training loss: {origin}, fold {k_split}.")
                    mse_loss.backward()
                    optimizer.step()

                    # Unweighted masked MSE for training diagnostics.
                    diff = (y_pred - y_true) * y_mask
                    train_loss_sum += (diff ** 2).sum().item()
                    train_count += y_mask.sum().item()

                train_epoch_loss = train_loss_sum / max(train_count, 1.0)
                _ = train_epoch_loss

                model.eval()
                val_loss_sum = 0.0
                val_count = 0.0

                with torch.inference_mode():
                    for batch in val_loader:
                        X, X_last, M, Delta, seq_mask, y_true, y_mask = move_batch_to_device(batch)

                        y_pred = model(X, X_last, M, Delta, seq_mask=seq_mask)

                        val_loss = target_balanced_masked_mse(
                            y_pred=y_pred,
                            y_true=y_true,
                            y_mask=y_mask,
                            target_weights=target_weights,
                        )

                        val_loss_sum += val_loss.item()
                        val_count += 1.0

                val_epoch_loss = val_loss_sum / max(val_count, 1.0)
                if not np.isfinite(val_epoch_loss):
                    raise FloatingPointError(f"Non-finite validation loss: {origin}, fold {k_split}.")

                if val_epoch_loss + min_delta < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            print(
                f"GRU-D finished: dataset={origin}, fold={k_split}, "
                f"best_epoch={best_epoch}, best_val_mse={best_val_loss:.6f}"
            )

            # Evaluation on test set
            model.eval()
            test_predictions_std = []
            test_actuals_std = []
            test_masks = []

            with torch.inference_mode():
                for batch in test_loader:
                    X, X_last, M, Delta, seq_mask, y_true, y_mask = move_batch_to_device(batch)

                    y_pred = model(X, X_last, M, Delta, seq_mask=seq_mask)
                    test_predictions_std.append(y_pred.cpu())
                    test_actuals_std.append(y_true.cpu())
                    test_masks.append(y_mask.cpu())

            test_predictions_std = torch.cat(test_predictions_std, dim=0).numpy()
            test_actuals_std = torch.cat(test_actuals_std, dim=0).numpy()
            test_masks = torch.cat(test_masks, dim=0).numpy().astype(bool)

            test_predictions = test_predictions_std * dynamic_stds_arr + dynamic_means_arr
            test_actuals = test_actuals_std * dynamic_stds_arr + dynamic_means_arr

            errors = test_predictions - test_actuals
            if not np.isfinite(errors[test_masks]).all():
                raise FloatingPointError(
                    f"Non-finite prediction error for an observed target: {origin}, fold {k_split}."
                )
            err_df = pd.DataFrame(errors, columns=dynamic_cols)
            err_df = err_df.mask(~test_masks)

            err_df.index.name = "row_id"
            err_long = (
                err_df
                .reset_index()
                .melt(
                    id_vars="row_id",
                    var_name="attribute",
                    value_name="error",
                )
                .dropna(subset=["error"])
            )

            if not err_long.empty:
                row_ids = err_long["row_id"].to_numpy()

                err_long["method"] = "gru_d"
                err_long["seed"] = SEED
                err_long["dataset"] = origin
                err_long["fold"] = k_split
                err_long["person_id"] = test_meta["person_id"].astype(str).to_numpy()[row_ids]
                err_long["input_age"] = test_meta["input_age"].to_numpy(dtype=float)[row_ids]
                err_long["output_age"] = test_meta["output_age"].to_numpy(dtype=float)[row_ids]
                err_long["horizon"] = test_meta["horizon"].to_numpy(dtype=float)[row_ids]
                err_long["n_input_visits"] = test_meta["n_input_visits"].to_numpy(dtype=int)[row_ids]

                if only_one_dataset:
                    err_long["source_config"] = "target_only"
                    err_long["source_dataset"] = ""
                    err_long["n_pretrain_datasets"] = 0
                else:
                    err_long["source_config"] = "pooled_all_datasets"
                    err_long["source_dataset"] = "all_other_datasets"
                    err_long["n_pretrain_datasets"] = 6

                error_columns = [
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

                error_frames.append(err_long[error_columns])

    errors_df = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame(
            columns=["error", "attribute", "dataset", "fold", "input_age", "output_age"]
        )
    )

    if errors_df.empty:
        raise ValueError("No observed-target errors were produced.")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    errors_df.to_csv(save_path, index=False, mode="x")
    print(f"Saved errors to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-data-percent", type=int, choices=[10, 30, 50, 70, 90, 100], default=100)
    parser.add_argument("--mode", choices=["target_only", "pooled"], default="target_only")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--errors-dir", type=Path, default=ERRORS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.num_workers < 0:
        parser.error("max-epochs must be positive and num-workers non-negative")
    return args


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    TRAIN_DATA_PERCENT = args.train_data_percent
    ONLY_ONE_DATASET = args.mode == "target_only"
    OUTPUT_PATH = args.output
    DATA_DIR = args.data_dir
    ERRORS_DIR = args.errors_dir
    MAX_EPOCHS = args.max_epochs
    NUM_WORKERS = args.num_workers
    main()
