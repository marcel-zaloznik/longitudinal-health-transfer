#!/usr/bin/env python3
"""Construct forecasting pairs, fixed participant folds and scoring metadata."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 1234
TRAIN_PERCENTAGES = [10, 30, 50, 70, 90, 100]
TARGET_COLS = [
    "Height", "Weight", "Waist_circumference", "BMI",
    "Systolic_blood_pressure", "Diastolic_blood_pressure", "Resting_heart_rate",
    "Glucose", "Total_cholesterol", "HDL", "LDL", "Triglycerides", "Smoking",
    "Fitness_speed", "PA_h_per_w", "Alcohol", "Diabetes", "Sleep", "Diet_HHS", "Diet_T2P",
]


def save_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def validate_dataset(df):
    required = ["Origin", "Country", "PersonID", "Sex", "Age", *TARGET_COLS]
    if list(df.columns) != required:
        raise ValueError("dataset.csv must have the columns and order in metadata/codebook.csv.")
    if df[["Origin", "PersonID", "Sex", "Age"]].isna().any().any():
        raise ValueError("Origin, PersonID, Sex and Age must be observed.")
    if not df["Sex"].isin(["Male", "Female"]).all():
        raise ValueError("Sex must be Male or Female.")
    ages = pd.to_numeric(df["Age"], errors="coerce")
    if not np.isfinite(ages.to_numpy(dtype=float)).all():
        raise ValueError("Age must contain finite numerical values in years.")
    if (df.groupby("PersonID")[["Origin", "Sex"]].nunique() > 1).any().any():
        raise ValueError("Each PersonID must belong to one dataset and have consistent Sex.")
    if df.duplicated(["PersonID", "Age"]).any():
        raise ValueError("Visits for a person must have distinct ages.")
    categorical_values = {
        "Smoking": ["No", "Quit/strike", "Yes"],
        "Diabetes": ["No", "Yes"],
    }
    for col in TARGET_COLS:
        observed = df[col].dropna()
        if col in categorical_values:
            if not observed.isin(categorical_values[col]).all():
                raise ValueError(f"{col} must contain {categorical_values[col]} or missing values.")
        else:
            values = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{col} must contain finite numerical values or missing values.")
    # Validation only: the supplied input is not cleaned, filtered or resampled.


def validate_folds(df, folds):
    if len(folds) != 5:
        raise ValueError("Exactly five participant folds are required.")
    people = [str(pid) for fold in folds for pid in fold]
    if len(people) != len(set(people)) or set(people) != set(df["PersonID"].astype(str)):
        raise ValueError("Participant folds must be disjoint and cover the dataset exactly.")

def build_prefix_future_pairs(
    df: pd.DataFrame,
    person_col: str = "PersonID",
    age_col: str = "Age",
    keep_ages: bool = True,
) -> pd.DataFrame:
    """
    For each PersonID, sort rows by Age and create pairs:
      input = first L rows (L = 1..n-1)
      output = any future row after those inputs
    Total pairs per person with n rows: n*(n-1)/2

    Returns a DataFrame with columns:
      PersonID, input_rows, output_row (+ optional input_ages, output_age)
    """
    if person_col not in df.columns or age_col not in df.columns:
        missing = [c for c in [person_col, age_col] if c not in df.columns]
        raise KeyError(f"Missing columns: {missing}")

    records = []

    # groupby preserves df order unless sort=True; sorting inside each group by Age is what matters here
    for pid, g in df.groupby(person_col, sort=False):
        g_sorted = g.sort_values(age_col, kind="mergesort")  # stable sort
        idx = g_sorted.index.to_list()
        ages = g_sorted[age_col].to_list()

        n = len(idx)
        if n < 2:
            continue

        # output position t = 1..n-1 (0-based)
        for t in range(1, n):
            out_row = idx[t]
            out_age = ages[t]

            # input length L = 1..t
            for L in range(1, t + 1):
                in_rows = tuple(idx[:L])

                rec = {
                    person_col: pid,
                    "input_rows": in_rows,
                    "output_row": out_row,
                }

                if keep_ages:
                    rec["input_ages"] = tuple(ages[:L])
                    rec["output_age"] = out_age

                records.append(rec)

    return pd.DataFrame.from_records(records)


def build_folds(df):
    rng = random.Random(SEED)

    df = df.copy()
    df["PersonID"] = df["PersonID"].astype(str)

    observed_counts = df[TARGET_COLS].notna().sum(axis=1).groupby(df["PersonID"]).sum()
    folds = [[] for _ in range(5)]

    origin_list = sorted(df["Origin"].dropna().unique().tolist())
    rng.shuffle(origin_list)

    for origin in origin_list:
        sex_list = sorted(
            df.loc[df["Origin"].eq(origin), "Sex"]
            .dropna()
            .unique()
            .tolist()
        )
        rng.shuffle(sex_list)

        for sex in sex_list:
            id_list = (
                df.loc[df["Origin"].eq(origin) & df["Sex"].eq(sex), "PersonID"]
                .drop_duplicates()
                .tolist()
            )
            rng.shuffle(id_list)

            fold_counts = [0] * 5

            for pid in id_list:
                n_observed = int(observed_counts.loc[pid])

                fold_idx = int(np.argmin(fold_counts))
                folds[fold_idx].append(str(pid))
                fold_counts[fold_idx] += n_observed
    return folds


def build_train_subsets(df, split5k):
    split5k = [set(map(str, fold_ids)) for fold_ids in split5k]

    target_train_subsets = {}

    for origin_idx, origin in enumerate(sorted(df["Origin"].dropna().unique())):
        target_train_subsets[origin] = {}

        origin_people = (
            df.loc[df["Origin"].eq(origin), "PersonID"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )

        for fold in range(5):
            val_fold = (fold - 1) % 5
            blocked_people = split5k[fold] | split5k[val_fold]

            train_people = sorted(
                pid for pid in origin_people
                if pid not in blocked_people
            )

            rng = np.random.default_rng(SEED + 1000 * origin_idx + fold)
            shuffled = np.asarray(train_people, dtype=object)
            rng.shuffle(shuffled)

            target_train_subsets[origin][str(fold)] = {}

            for pct in TRAIN_PERCENTAGES:
                if pct == 100:
                    keep = shuffled
                else:
                    n_keep = max(1, math.ceil(len(shuffled) * pct / 100))
                    keep = shuffled[:n_keep]

                target_train_subsets[origin][str(fold)][str(pct)] = [
                    str(x) for x in keep.tolist()
                ]
    return target_train_subsets


def build_scoring_metadata(df, input_output_pairs, split5k):
    """Score SDs use unique target visits in the four non-test folds, including validation."""
    df = df.copy()
    input_output_pairs = input_output_pairs.copy()
    df["PersonID"] = df["PersonID"].astype(str)
    input_output_pairs["PersonID"] = input_output_pairs["PersonID"].astype(str)

    split5k = [set(map(str, fold_ids)) for fold_ids in split5k]

    target_cols = list(df.loc[:, "Height":"Diet_T2P"].columns)

    maps = {
        "Smoking": {"No": 0, "Quit/strike": 1, "Yes": 2},
        "Diabetes": {"No": 0, "Yes": 1},
    }

    for col, mapping in maps.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    input_output_pairs["output_row"] = pd.to_numeric(
        input_output_pairs["output_row"],
        errors="coerce",
    )

    input_output_pairs = input_output_pairs.dropna(subset=["output_row"]).copy()
    input_output_pairs["output_row"] = input_output_pairs["output_row"].astype(int)

    PRIMARY_TARGET_PANEL = [
        "Height",
        "Weight",
        "BMI",
        "Waist_circumference",
        "Systolic_blood_pressure",
        "Diastolic_blood_pressure",
        "Resting_heart_rate",
        "Fitness_speed",
        "PA_h_per_w",
    ]

    MIN_TRAIN_OBSERVED_TARGETS = 30
    MIN_TEST_OBSERVED_TARGETS = 30

    target_attribute_stds = {}
    target_attribute_counts = {}
    eligible_dataset_attribute_pairs = {}

    for origin in sorted(df["Origin"].dropna().unique()):
        target_attribute_stds[origin] = {}
        target_attribute_counts[origin] = {}
        eligible_dataset_attribute_pairs[origin] = {}

        origin_mask = df["Origin"].eq(origin)

        for fold, test_person_ids in enumerate(split5k):
            train_row_index = df.index[
                origin_mask & ~df["PersonID"].isin(test_person_ids)
            ]

            test_row_index = df.index[
                origin_mask & df["PersonID"].isin(test_person_ids)
            ]

            train_output_rows = np.sort(
                pd.unique(
                    input_output_pairs.loc[
                        input_output_pairs["output_row"].isin(train_row_index),
                        "output_row",
                    ].astype(int)
                )
            )

            test_output_rows = np.sort(
                pd.unique(
                    input_output_pairs.loc[
                        input_output_pairs["output_row"].isin(test_row_index),
                        "output_row",
                    ].astype(int)
                )
            )

            train_targets = df.loc[train_output_rows, target_cols]
            test_targets = df.loc[test_output_rows, target_cols]

            train_counts = train_targets.notna().sum()
            test_counts = test_targets.notna().sum()

            stds = train_targets.std(skipna=True, ddof=1)

            target_attribute_stds[origin][str(fold)] = {}
            target_attribute_counts[origin][str(fold)] = {}
            eligible_dataset_attribute_pairs[origin][str(fold)] = {}

            for col in target_cols:
                std = stds[col]

                valid_std = (
                    pd.notna(std)
                    and np.isfinite(float(std))
                    and float(std) > 0
                )

                train_n = int(train_counts[col])
                test_n = int(test_counts[col])

                target_attribute_stds[origin][str(fold)][col] = (
                    float(std) if valid_std else None
                )

                target_attribute_counts[origin][str(fold)][col] = {
                    "train_observed_unique_output_rows": train_n,
                    "test_observed_unique_output_rows": test_n,
                }

                eligible_dataset_attribute_pairs[origin][str(fold)][col] = {
                    "eligible": bool(
                        valid_std
                        and train_n >= MIN_TRAIN_OBSERVED_TARGETS
                        and test_n >= MIN_TEST_OBSERVED_TARGETS
                    ),
                    "train_observed_unique_output_rows": train_n,
                    "test_observed_unique_output_rows": test_n,
                    "std": float(std) if valid_std else None,
                }


    return {
        "target_attribute_stds_by_dataset_fold.json": target_attribute_stds,
        "target_attribute_counts_by_dataset_fold.json": target_attribute_counts,
        "eligible_dataset_attribute_pairs.json": eligible_dataset_attribute_pairs,
        "primary_target_panel.json": PRIMARY_TARGET_PANEL,
        "extended_target_panel.json": target_cols,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    df = pd.read_csv(args.data_path, dtype={"PersonID": str}, low_memory=False)
    validate_dataset(df)
    pairs = build_prefix_future_pairs(df, keep_ages=False)
    if pairs.empty:
        raise ValueError("No prefix-to-future pairs can be constructed.")
    folds = build_folds(df)
    flattened = [pid for fold in folds for pid in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(df["PersonID"]):
        raise ValueError("Participant folds must be disjoint and exhaustive.")
    subsets = build_train_subsets(df, folds)
    scoring = build_scoring_metadata(df, pairs, folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "input_output_pairs.csv", index=False)
    save_json(args.output_dir / "split5k.json", folds)
    save_json(args.output_dir / "target_train_subsets_by_dataset_fold_percent.json", subsets)
    for filename, value in scoring.items():
        save_json(args.output_dir / filename, value)
    print(f"Prepared {len(pairs):,} pairs and five folds for {len(flattened):,} people.")
    print(f"Saved private experiment inputs to {args.output_dir}.")


if __name__ == "__main__":
    main()
