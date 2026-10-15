#!/usr/bin/env python3
"""Population/LOCF, linear-regression and random-forest baselines."""

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import validate_dataset, validate_folds
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset.csv"
PAIRS_PATH = ROOT / "data" / "input_output_pairs.csv"
SPLIT_PATH = ROOT / "data" / "split5k.json"
TARGET_TRAIN_SUBSETS_PATH = ROOT / "data" / "target_train_subsets_by_dataset_fold_percent.json"
ERRORS_DIR = ROOT / "results" / "errors"
SEED = 1234
ONLY_ONE_DATASET = True
TRAIN_DATA_PERCENT = 100
AGE_BIN_WIDTH = 2.0
MIN_GROUP_SIZE = 5
METHOD_NAMES = ["train_mean", "age_sex_mean", "locf_age_sex_mean"]
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


def knn_impute_train_only(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    start_col: str = "Sex",
    end_col: str = "Diet_T2P",
    n_neighbors: int = 5,
    weights: str = "distance",  # "uniform" or "distance"
):
    """
    KNN-impute columns from start_col..end_col (inclusive), fitting ONLY on df_train.
    Steps:
      1) train-only standardization (mean/std computed on df_train, skipping NaNs)
      2) KNNImputer fit on standardized df_train subset
      3) transform df_train and df_test
      4) inverse standardization back to original scale
    Notes:
      - Preserves original row indices (including unsorted / gapped indices).
      - Columns that are ALL missing in df_train are ignored (left unchanged in both dfs).
    Returns: (df_train_imputed, df_test_imputed)
    """
    if list(df_train.columns) != list(df_test.columns):
        raise ValueError("df_train and df_test must have the same columns in the same order.")

    cols = df_train.columns
    if start_col not in cols or end_col not in cols:
        raise KeyError(f"Could not find '{start_col}' and/or '{end_col}' in columns.")

    start_i = cols.get_loc(start_col)
    end_i = cols.get_loc(end_col)
    if start_i > end_i:
        raise ValueError(f"Column order is '{start_col}' after '{end_col}'. Check column ordering.")

    block_cols = list(cols[start_i : end_i + 1])

    # Ignore columns that are entirely missing in TRAIN (leave them as-is)
    valid_cols = [c for c in block_cols if df_train[c].notna().any()]
    if not valid_cols:
        return df_train.copy(), df_test.copy()

    # Ensure numeric (if already numeric, this is harmless). Non-numeric coerces to NaN.
    train_block = df_train[valid_cols].apply(pd.to_numeric, errors="coerce")
    test_block = df_test[valid_cols].apply(pd.to_numeric, errors="coerce")

    # Train-only standardization parameters (skip NaNs)
    mu = train_block.mean(skipna=True)
    sigma = train_block.std(skipna=True, ddof=0).replace(0, 1)  # avoid divide-by-zero for constant cols

    X_train = (train_block - mu) / sigma
    X_test = (test_block - mu) / sigma

    # Fit ONLY on df_train
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric="nan_euclidean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Back to original scale
    train_imp = pd.DataFrame(X_train_imp, index=df_train.index, columns=valid_cols) * sigma + mu
    test_imp = pd.DataFrame(X_test_imp, index=df_test.index, columns=valid_cols) * sigma + mu

    # Write back, preserving indices and leaving other columns untouched
    df_train_out = df_train.copy()
    df_test_out = df_test.copy()
    df_train_out.loc[:, valid_cols] = train_imp
    df_test_out.loc[:, valid_cols] = test_imp

    return df_train_out, df_test_out


def fit_predict_one_model_per_target(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model_factory,
    x_start: str = "Sex",
    x_end: str = "Target_age",
    y_start: str = "Height",
    y_end: str = "Diet_T2P",
    min_train_targets: int = 2,
) -> pd.DataFrame:
    """
    Fits one separate model per target attribute.

    Important:
    - Input features must already be KNN-imputed.
    - Target labels must NOT be imputed.
    - For each target, rows with missing y_train[target] are excluded.
    """

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")

    x_cols = list(X_train.loc[:, x_start:x_end].columns)
    x_cols = [
        c for c in x_cols
        if c in X_test.columns and X_train[c].notna().any()
    ]

    if not x_cols:
        raise ValueError("No usable input columns.")

    y_cols = list(y_train.loc[:, y_start:y_end].columns)
    y_cols = [c for c in y_cols if c in y_test.columns]

    Xtr = X_train.loc[:, x_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    Xte = X_test.loc[:, x_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        bad_train = list(pd.DataFrame(Xtr, columns=x_cols).columns[pd.DataFrame(Xtr, columns=x_cols).isna().any()])
        bad_test = list(pd.DataFrame(Xte, columns=x_cols).columns[pd.DataFrame(Xte, columns=x_cols).isna().any()])
        raise ValueError(
            f"NaNs remain in input features after imputation. "
            f"Train bad columns={bad_train}; test bad columns={bad_test}"
        )

    y_pred = pd.DataFrame(np.nan, index=y_test.index, columns=y_test.columns)

    for target in y_cols:
        y = pd.to_numeric(y_train[target], errors="coerce")
        observed = y.notna().to_numpy()

        if observed.sum() < min_train_targets:
            continue

        model = model_factory()
        model.fit(Xtr[observed], y.to_numpy()[observed])

        predictions = np.asarray(model.predict(Xte), dtype=float)
        observed_test = y_test[target].notna().to_numpy()
        if not np.isfinite(predictions[observed_test]).all():
            raise FloatingPointError(f"Non-finite prediction for an observed target: {target}.")
        y_pred[target] = predictions

    return y_pred


def parse_input_rows(cell):
    if isinstance(cell, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in cell)
    if pd.isna(cell):
        return tuple()

    s = str(cell).strip()

    try:
        value = ast.literal_eval(s)

        if isinstance(value, (list, tuple, np.ndarray)):
            return tuple(int(x) for x in value)

        return (int(value),)

    except Exception:
        nums = re.findall(r"-?\d+", s)
        return tuple(int(x) for x in nums)


def make_save_path(method_name):
    source_tag = "target_only" if ONLY_ONE_DATASET else "pooled_all_datasets"
    pct_tag = "" if TRAIN_DATA_PERCENT == 100 else f"_pct{TRAIN_DATA_PERCENT}"
    return ERRORS_DIR / f"persistence_{method_name}_{source_tag}{pct_tag}_seed{SEED}.csv"


def age_bin_id(age_series, width):
    age_numeric = pd.to_numeric(age_series, errors="coerce")
    return np.floor(age_numeric / width).astype("Int64")


def subset_pairs_for_people(pairs, person_ids, n_rows):
    person_ids = set(map(str, person_ids))

    out = pairs[pairs["PersonID"].astype(str).isin(person_ids)].copy()

    out = out[
        out["output_row"].notna()
        & out["last_input_idx"].notna()
    ].copy()

    out["output_row"] = out["output_row"].astype(int)
    out["last_input_idx"] = out["last_input_idx"].astype(int)

    valid = (
        out["output_row"].between(0, n_rows - 1)
        & out["last_input_idx"].between(0, n_rows - 1)
    )

    valid &= out["input_rows_tuple"].apply(
        lambda rows: (
            isinstance(rows, tuple)
            and len(rows) > 0
            and all((0 <= int(r) < n_rows) for r in rows)
        )
    )

    out = out.loc[valid].copy()
    out = out.reset_index(drop=True)

    return out


def group_mean_with_min_count(data, group_cols, target_cols, min_group_size):
    if data.empty:
        return pd.DataFrame(columns=target_cols)

    grouped = data.groupby(group_cols, dropna=False)

    means = grouped[target_cols].mean()
    counts = grouped[target_cols].count()

    means = means.where(counts >= min_group_size)

    return means


def fit_population_stats(df, output_rows, target_cols, age_bin_width, min_group_size):
    output_rows = (
        pd.Series(output_rows)
        .dropna()
        .astype(int)
        .drop_duplicates()
        .to_numpy()
    )

    empty_global = pd.Series(np.nan, index=target_cols, dtype=float)

    if len(output_rows) == 0:
        return {
            "global": empty_global,
            "age_sex": pd.DataFrame(columns=target_cols),
            "sex": pd.DataFrame(columns=target_cols),
            "age": pd.DataFrame(columns=target_cols),
        }

    stats_df = df.loc[output_rows, ["Sex", "Age"] + target_cols].copy()
    stats_df["age_bin"] = age_bin_id(stats_df["Age"], age_bin_width)

    for col in target_cols:
        stats_df[col] = pd.to_numeric(stats_df[col], errors="coerce")

    global_mean = stats_df[target_cols].mean(skipna=True)

    age_sex_mean = group_mean_with_min_count(
        data=stats_df,
        group_cols=["Sex", "age_bin"],
        target_cols=target_cols,
        min_group_size=min_group_size,
    )

    sex_mean = group_mean_with_min_count(
        data=stats_df,
        group_cols=["Sex"],
        target_cols=target_cols,
        min_group_size=min_group_size,
    )

    age_mean = group_mean_with_min_count(
        data=stats_df,
        group_cols=["age_bin"],
        target_cols=target_cols,
        min_group_size=min_group_size,
    )

    return {
        "global": global_mean,
        "age_sex": age_sex_mean,
        "sex": sex_mean,
        "age": age_mean,
    }


def constant_prediction_from_mean(mean_values, n_rows, target_cols):
    values = mean_values.reindex(target_cols).to_numpy(dtype=float)
    arr = np.tile(values, (n_rows, 1))

    return pd.DataFrame(arr, columns=target_cols)


def fill_from_group_means(pred, meta, means, group_cols, target_cols):
    if means is None or means.empty:
        return pred

    means_reset = means.reset_index()

    merged = meta[group_cols].merge(
        means_reset,
        on=group_cols,
        how="left",
    )

    group_values = merged[target_cols].copy()
    group_values.index = pred.index

    return pred.where(pred.notna(), group_values)


def age_sex_prediction_from_stats(stats, output_meta, target_cols, age_bin_width):
    meta = output_meta[["Sex", "Age"]].copy()
    meta["age_bin"] = age_bin_id(meta["Age"], age_bin_width)

    pred = pd.DataFrame(
        np.nan,
        index=output_meta.index,
        columns=target_cols,
        dtype=float,
    )

    # 1. same sex + same age bin
    pred = fill_from_group_means(
        pred=pred,
        meta=meta,
        means=stats["age_sex"],
        group_cols=["Sex", "age_bin"],
        target_cols=target_cols,
    )

    # 2. same sex
    pred = fill_from_group_means(
        pred=pred,
        meta=meta,
        means=stats["sex"],
        group_cols=["Sex"],
        target_cols=target_cols,
    )

    # 3. same age bin
    pred = fill_from_group_means(
        pred=pred,
        meta=meta,
        means=stats["age"],
        group_cols=["age_bin"],
        target_cols=target_cols,
    )

    # 4. global train mean
    global_pred = constant_prediction_from_mean(
        mean_values=stats["global"],
        n_rows=len(output_meta),
        target_cols=target_cols,
    )
    global_pred.index = pred.index

    pred = pred.where(pred.notna(), global_pred)

    return pred


def make_locf_predictions(test_pairs, target_values, target_cols):
    n_targets = len(target_cols)
    out = np.full((len(test_pairs), n_targets), np.nan, dtype=float)

    cache = {}

    for i, rows in enumerate(test_pairs["input_rows_tuple"]):
        rows = tuple(int(r) for r in rows)

        if rows in cache:
            out[i, :] = cache[rows]
            continue

        row_arr = np.asarray(rows, dtype=int)
        values = target_values[row_arr, :]

        observed = ~np.isnan(values)

        last_pos = np.where(
            observed,
            np.arange(values.shape[0])[:, None],
            -1,
        ).max(axis=0)

        pred = np.full(n_targets, np.nan, dtype=float)

        ok = last_pos >= 0
        if ok.any():
            pred[ok] = values[last_pos[ok], np.arange(n_targets)[ok]]

        cache[rows] = pred
        out[i, :] = pred

    return pd.DataFrame(out, columns=target_cols)


def apply_fallback(primary_pred, fallback_pred):
    fallback_pred = fallback_pred.copy()
    fallback_pred.index = primary_pred.index

    return primary_pred.where(primary_pred.notna(), fallback_pred)


def write_error_rows(
    method_name,
    y_pred,
    y_true,
    gt_missing,
    input_age,
    output_age,
    person_id,
    n_input_visits,
    dataset_name,
    fold,
    save_paths,
    rows_written,
    supported_targets,
):
    y_pred = y_pred.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)
    gt_missing = gt_missing.reset_index(drop=True)

    err_df = y_pred - y_true
    evaluated = ~gt_missing.to_numpy() & np.asarray(supported_targets, dtype=bool)[None, :]
    if not np.isfinite(err_df.to_numpy()[evaluated]).all():
        raise FloatingPointError(f"Non-finite prediction error: {dataset_name}, fold {fold}.")

    # Evaluate only observed ground-truth target values.
    err_df = err_df.mask(gt_missing)

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

    if err_long.empty:
        return

    row_ids = err_long["row_id"].astype(int).to_numpy()

    save_path = save_paths[method_name]

    err_long["method"] = save_path.stem
    err_long["seed"] = SEED
    err_long["dataset"] = dataset_name
    err_long["fold"] = fold
    err_long["person_id"] = person_id[row_ids]
    err_long["input_age"] = input_age[row_ids]
    err_long["output_age"] = output_age[row_ids]
    err_long["horizon"] = output_age[row_ids] - input_age[row_ids]
    err_long["n_input_visits"] = n_input_visits[row_ids]

    if ONLY_ONE_DATASET:
        err_long["source_config"] = "target_only"
        err_long["source_dataset"] = ""
        err_long["n_pretrain_datasets"] = 0
    else:
        err_long["source_config"] = "pooled_all_datasets"
        err_long["source_dataset"] = "all_other_datasets"
        err_long["n_pretrain_datasets"] = 6

    err_long[ERROR_COLUMNS].to_csv(
        save_path,
        mode="a",
        header=not save_path.exists(),
        index=False,
    )

    rows_written[method_name] += len(err_long)


def run_regression(model_kind):
    df = pd.read_csv(DATA_PATH, dtype={"PersonID": str}, low_memory=False)
    validate_dataset(df)
    input_output_pairs = pd.read_csv(PAIRS_PATH, dtype={"PersonID": str})
    maps = {
        "Smoking": {"No": 0, "Quit/strike": 1, "Yes": 2},
        "Diabetes": {"No": 0, "Yes": 1},
        "Sex": {"Male": 0, "Female": 1},
    }
    for col, mapping in maps.items():
        df[col] = df[col].map(mapping).astype(float)

    source_tag = "target_only" if ONLY_ONE_DATASET else "pooled_all_datasets"
    pct_tag = "" if TRAIN_DATA_PERCENT == 100 else f"_pct{TRAIN_DATA_PERCENT}"

    if model_kind == "linear":
        METHOD_NAME = "linear_regression_fixed"

        def model_factory():
            return LinearRegression()

    elif model_kind == "random_forest":
        METHOD_NAME = "random_forest_fixed"

        def model_factory():
            return RandomForestRegressor(
                n_estimators=100,
                random_state=SEED,
                n_jobs=-1,
            )

    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    SAVE_PATH = ERRORS_DIR / f"{METHOD_NAME}_{source_tag}{pct_tag}_seed{SEED}.csv"
    if SAVE_PATH.exists():
        raise FileExistsError(f"Output already exists: {SAVE_PATH}. Choose another --errors-dir.")

    ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    df["PersonID"] = df["PersonID"].astype(str)
    input_output_pairs["PersonID"] = input_output_pairs["PersonID"].astype(str)

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split5k = json.load(f)

    validate_folds(df, split5k)
    split5k = [set(map(str, fold_ids)) for fold_ids in split5k]

    with open(
        TARGET_TRAIN_SUBSETS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        target_train_subsets = json.load(f)


    input_output_pairs["input_rows_tuple"] = input_output_pairs["input_rows"].apply(parse_input_rows)
    input_output_pairs["last_input_idx"] = input_output_pairs["input_rows_tuple"].apply(
        lambda rows: rows[-1] if len(rows) > 0 else np.nan
    )

    input_output_pairs["output_row"] = pd.to_numeric(
        input_output_pairs["output_row"],
        errors="coerce",
    )

    input_output_pairs = input_output_pairs.dropna(
        subset=["output_row", "last_input_idx"]
    ).copy()

    input_output_pairs["output_row"] = input_output_pairs["output_row"].astype(int)
    input_output_pairs["last_input_idx"] = input_output_pairs["last_input_idx"].astype(int)

    n_rows = len(df)
    target_cols = list(df.loc[:, "Height":"Diet_T2P"].columns)

    df_missing_values = df.notna().astype(int)


    error_frames = []

    for origin in sorted(df["Origin"].dropna().unique()):
        print(f"Dataset: {origin}")

        df_origin = df[df["Origin"].eq(origin)].copy()

        for fold, test_ids in enumerate(split5k):
            print(f"  Fold {fold}")

            df_test = df_origin[df_origin["PersonID"].isin(test_ids)].copy()

            if ONLY_ONE_DATASET:
                df_train = df_origin[~df_origin["PersonID"].isin(test_ids)].copy()
                source_config = "target_only"
                source_dataset = ""
                n_pretrain_datasets = 0
            else:
                test_mask = df["Origin"].eq(origin) & df["PersonID"].isin(test_ids)
                df_train = df.loc[~test_mask].copy()
                source_config = "pooled_all_datasets"
                source_dataset = "all_other_datasets"
                n_pretrain_datasets = 6

            if target_train_subsets is not None:
                keep_ids = set(
                    target_train_subsets[origin][str(fold)][str(TRAIN_DATA_PERCENT)]
                )

                if ONLY_ONE_DATASET:
                    df_train = df_train[df_train["PersonID"].isin(keep_ids)].copy()
                else:
                    target_part = df_train[
                        df_train["Origin"].eq(origin)
                        & df_train["PersonID"].isin(keep_ids)
                    ]
                    source_part = df_train[~df_train["Origin"].eq(origin)]
                    df_train = pd.concat([target_part, source_part], axis=0)

            train_pairs = input_output_pairs[
                input_output_pairs["PersonID"].isin(df_train["PersonID"].unique())
            ].copy()

            test_pairs = input_output_pairs[
                input_output_pairs["PersonID"].isin(df_test["PersonID"].unique())
            ].copy()

            valid_train = (
                train_pairs["last_input_idx"].between(0, n_rows - 1)
                & train_pairs["output_row"].between(0, n_rows - 1)
            )

            valid_test = (
                test_pairs["last_input_idx"].between(0, n_rows - 1)
                & test_pairs["output_row"].between(0, n_rows - 1)
            )

            train_pairs = train_pairs.loc[valid_train].reset_index(drop=True)
            test_pairs = test_pairs.loc[valid_test].reset_index(drop=True)

            if train_pairs.empty or test_pairs.empty:
                continue

            train_in_idx = train_pairs["last_input_idx"].astype(int).to_numpy()
            train_out_idx = train_pairs["output_row"].astype(int).to_numpy()

            test_in_idx = test_pairs["last_input_idx"].astype(int).to_numpy()
            test_out_idx = test_pairs["output_row"].astype(int).to_numpy()

            # Inputs are allowed to be imputed.
            X_train_raw = df.loc[train_in_idx].reset_index(drop=True)
            X_test_raw = df.loc[test_in_idx].reset_index(drop=True)

            # Targets are NOT imputed.
            y_train = df.loc[train_out_idx, target_cols].reset_index(drop=True).astype(float)
            y_test = df.loc[test_out_idx, target_cols].reset_index(drop=True).astype(float)

            X_train = X_train_raw.copy()
            X_test = X_test_raw.copy()

            X_train["Target_age"] = df.loc[train_out_idx, "Age"].to_numpy(dtype=float)
            X_test["Target_age"] = df.loc[test_out_idx, "Age"].to_numpy(dtype=float)

            X_train_imp, X_test_imp = knn_impute_train_only(
                X_train,
                X_test,
                start_col="Sex",
                end_col="Target_age",
                n_neighbors=5,
                weights="distance",
            )

            y_pred = fit_predict_one_model_per_target(
                X_train=X_train_imp,
                y_train=y_train,
                X_test=X_test_imp,
                y_test=y_test,
                model_factory=model_factory,
            )

            gt_missing = pd.DataFrame(
                df_missing_values.loc[test_out_idx, target_cols].eq(0).to_numpy(),
                index=y_test.index,
                columns=target_cols,
            )

            y_test_eval = y_test.mask(gt_missing)
            y_pred_eval = y_pred.mask(gt_missing)

            err_df = y_pred_eval - y_test_eval
            supported = y_train.notna().sum().ge(2).to_numpy()
            evaluated = ~gt_missing.to_numpy() & supported[None, :]
            if not np.isfinite(err_df.to_numpy()[evaluated]).all():
                raise FloatingPointError(f"Non-finite prediction error: {origin}, fold {fold}.")
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

            if err_long.empty:
                continue

            row_ids = err_long["row_id"].astype(int).to_numpy()

            input_age = df.loc[test_in_idx, "Age"].to_numpy(dtype=float)
            output_age = df.loc[test_out_idx, "Age"].to_numpy(dtype=float)
            horizon = output_age - input_age

            n_input_visits = (
                test_pairs["input_rows_tuple"]
                .map(len)
                .to_numpy(dtype=int)
            )

            person_id = test_pairs["PersonID"].astype(str).to_numpy()

            err_long["method"] = METHOD_NAME
            err_long["seed"] = SEED
            err_long["dataset"] = origin
            err_long["fold"] = fold
            err_long["person_id"] = person_id[row_ids]
            err_long["input_age"] = input_age[row_ids]
            err_long["output_age"] = output_age[row_ids]
            err_long["horizon"] = horizon[row_ids]
            err_long["n_input_visits"] = n_input_visits[row_ids]
            err_long["source_config"] = source_config
            err_long["source_dataset"] = source_dataset
            err_long["n_pretrain_datasets"] = n_pretrain_datasets

            error_frames.append(err_long[ERROR_COLUMNS])

    errors_df = (
        pd.concat(error_frames, ignore_index=True)
        if error_frames
        else pd.DataFrame(columns=ERROR_COLUMNS)
    )

    if errors_df.empty:
        raise ValueError("No observed-target errors were produced.")
    errors_df.to_csv(SAVE_PATH, index=False, mode="x")
    print(f"Saved {SAVE_PATH} with {len(errors_df)} rows.")


def run_persistence():
    df = pd.read_csv(DATA_PATH, dtype={"PersonID": str}, low_memory=False)
    validate_dataset(df)
    input_output_pairs = pd.read_csv(PAIRS_PATH, dtype={"PersonID": str})

    df["PersonID"] = df["PersonID"].astype(str)
    input_output_pairs["PersonID"] = input_output_pairs["PersonID"].astype(str)

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split5k = json.load(f)

    validate_folds(df, split5k)
    split5k = [set(map(str, fold_ids)) for fold_ids in split5k]

    with open(TARGET_TRAIN_SUBSETS_PATH, "r", encoding="utf-8") as f:
        target_train_subsets = json.load(f)

    # =============================================================================
    # Basic preprocessing
    # =============================================================================

    categorical_maps = {
        "Smoking": {
            "No": 0,
            "Quit/strike": 1,
            "Yes": 2,
        },
        "Diabetes": {
            "No": 0,
            "Yes": 1,
        },
        "Sex": {
            "Male": 0,
            "Female": 1,
        },
    }

    for col, mapping in categorical_maps.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_cols = list(df.loc[:, "Height":"Diet_T2P"].columns)

    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # This determines which target values are actually evaluated.
    df_missing_values = df[target_cols].notna().astype(int)

    n_rows = len(df)

    input_output_pairs["input_rows_tuple"] = input_output_pairs["input_rows"].apply(parse_input_rows)
    input_output_pairs["last_input_idx"] = input_output_pairs["input_rows_tuple"].apply(
        lambda rows: rows[-1] if len(rows) > 0 else np.nan
    )

    input_output_pairs["output_row"] = pd.to_numeric(
        input_output_pairs["output_row"],
        errors="coerce",
    )

    input_output_pairs = input_output_pairs.dropna(
        subset=["output_row", "last_input_idx"]
    ).copy()

    input_output_pairs["output_row"] = input_output_pairs["output_row"].astype(int)
    input_output_pairs["last_input_idx"] = input_output_pairs["last_input_idx"].astype(int)

    target_values = df[target_cols].to_numpy(dtype=float)


    # =============================================================================
    # Output files
    # =============================================================================

    save_paths = {
        method_name: make_save_path(method_name)
        for method_name in METHOD_NAMES
    }

    for path in save_paths.values():
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}. Choose another --errors-dir.")


    rows_written = {method_name: 0 for method_name in METHOD_NAMES}

    print("Saving persistence-baseline errors to:")
    for method_name, path in save_paths.items():
        print(f"  {method_name:25s} -> {path}")


    # =============================================================================
    # Main evaluation loop
    # =============================================================================

    for origin in sorted(df["Origin"].dropna().unique()):
        df_origin = df[df["Origin"].eq(origin)].copy()

        print(f"\nDataset: {origin}")

        for fold, test_ids in enumerate(split5k):
            print(f"  Fold {fold}")

            # ---------------------------------------------------------------------
            # Train/test split
            # ---------------------------------------------------------------------

            df_test = df_origin[df_origin["PersonID"].isin(test_ids)].copy()

            if ONLY_ONE_DATASET:
                df_train = df_origin[~df_origin["PersonID"].isin(test_ids)].copy()
            else:
                test_mask = (
                    df["Origin"].eq(origin)
                    & df["PersonID"].isin(test_ids)
                )
                df_train = df.loc[~test_mask].copy()

            if target_train_subsets is not None:
                keep_target_train_ids = set(
                    target_train_subsets[origin][str(fold)][str(TRAIN_DATA_PERCENT)]
                )

                if ONLY_ONE_DATASET:
                    df_train = df_train[
                        df_train["PersonID"].isin(keep_target_train_ids)
                    ].copy()
                else:
                    target_part = df_train[
                        df_train["Origin"].eq(origin)
                        & df_train["PersonID"].isin(keep_target_train_ids)
                    ].copy()

                    source_part = df_train[
                        ~df_train["Origin"].eq(origin)
                    ].copy()

                    df_train = pd.concat([target_part, source_part], axis=0)

            # ---------------------------------------------------------------------
            # Input-output pairs
            # ---------------------------------------------------------------------

            train_pairs = subset_pairs_for_people(
                pairs=input_output_pairs,
                person_ids=df_train["PersonID"].unique(),
                n_rows=n_rows,
            )

            test_pairs = subset_pairs_for_people(
                pairs=input_output_pairs,
                person_ids=df_test["PersonID"].unique(),
                n_rows=n_rows,
            )

            if test_pairs.empty:
                print("    No test pairs. Skipping.")
                continue

            # Use unique output rows for train-only population statistics.
            train_output_rows = (
                train_pairs["output_row"]
                .dropna()
                .astype(int)
                .drop_duplicates()
                .to_numpy()
            )

            test_output_rows = test_pairs["output_row"].astype(int).to_numpy()
            test_last_input_rows = test_pairs["last_input_idx"].astype(int).to_numpy()

            # ---------------------------------------------------------------------
            # True test targets and metadata
            # ---------------------------------------------------------------------

            y_true = (
                df.loc[test_output_rows, target_cols]
                .reset_index(drop=True)
                .astype(float)
            )

            output_meta = (
                df.loc[test_output_rows, ["Sex", "Age"]]
                .reset_index(drop=True)
            )

            input_age = (
                df.loc[test_last_input_rows, "Age"]
                .to_numpy(dtype=float)
            )

            output_age = (
                df.loc[test_output_rows, "Age"]
                .to_numpy(dtype=float)
            )

            person_id = test_pairs["PersonID"].astype(str).to_numpy()

            n_input_visits = (
                test_pairs["input_rows_tuple"]
                .map(len)
                .to_numpy(dtype=int)
            )

            gt_missing = pd.DataFrame(
                df_missing_values.loc[test_output_rows, target_cols].eq(0).to_numpy(),
                index=y_true.index,
                columns=target_cols,
            )

            # ---------------------------------------------------------------------
            # Fit train-only fallback statistics
            # ---------------------------------------------------------------------

            train_stats = fit_population_stats(
                df=df,
                output_rows=train_output_rows,
                target_cols=target_cols,
                age_bin_width=AGE_BIN_WIDTH,
                min_group_size=MIN_GROUP_SIZE,
            )

            # ---------------------------------------------------------------------
            # Population baselines
            # ---------------------------------------------------------------------

            train_mean_pred = constant_prediction_from_mean(
                mean_values=train_stats["global"],
                n_rows=len(test_pairs),
                target_cols=target_cols,
            )

            age_sex_mean_pred = age_sex_prediction_from_stats(
                stats=train_stats,
                output_meta=output_meta,
                target_cols=target_cols,
                age_bin_width=AGE_BIN_WIDTH,
            )

            # ---------------------------------------------------------------------
            # LOCF + age-sex fallback baseline
            # ---------------------------------------------------------------------

            locf_pred = make_locf_predictions(
                test_pairs=test_pairs,
                target_values=target_values,
                target_cols=target_cols,
            )

            locf_age_sex_mean_pred = apply_fallback(
                primary_pred=locf_pred,
                fallback_pred=age_sex_mean_pred,
            )

            # ---------------------------------------------------------------------
            # Save raw errors
            # ---------------------------------------------------------------------

            predictions = {
                "train_mean": train_mean_pred,
                "age_sex_mean": age_sex_mean_pred,
                "locf_age_sex_mean": locf_age_sex_mean_pred,
            }

            for method_name, y_pred in predictions.items():
                write_error_rows(
                    method_name=method_name,
                    y_pred=y_pred,
                    y_true=y_true,
                    gt_missing=gt_missing,
                    input_age=input_age,
                    output_age=output_age,
                    person_id=person_id,
                    n_input_visits=n_input_visits,
                    dataset_name=origin,
                    fold=fold,
                    save_paths=save_paths,
                    rows_written=rows_written,
                    supported_targets=train_stats["global"].reindex(target_cols).notna().to_numpy(),
                )


    for method_name in METHOD_NAMES:
        if rows_written[method_name] == 0:
            raise ValueError(f"No observed-target errors were produced by {method_name}.")


    print("\nDone.")
    print("Rows written:")
    for method_name in METHOD_NAMES:
        print(f"  {method_name:25s}: {rows_written[method_name]}")

    print("\nOutput files:")
    for method_name, path in save_paths.items():
        print(f"  {path}")

def main():
    global SEED, TRAIN_DATA_PERCENT, ONLY_ONE_DATASET, ERRORS_DIR
    global DATA_PATH, PAIRS_PATH, SPLIT_PATH, TARGET_TRAIN_SUBSETS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["persistence", "linear", "random_forest"], default="persistence")
    parser.add_argument("--mode", choices=["target_only", "pooled"], default="target_only")
    parser.add_argument("--train-data-percent", type=int, choices=[10, 30, 50, 70, 90, 100], default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--errors-dir", type=Path, default=ERRORS_DIR)
    args = parser.parse_args()
    SEED = args.seed
    TRAIN_DATA_PERCENT = args.train_data_percent
    ONLY_ONE_DATASET = args.mode == "target_only"
    DATA_PATH = args.data_dir / "dataset.csv"
    PAIRS_PATH = args.data_dir / "input_output_pairs.csv"
    SPLIT_PATH = args.data_dir / "split5k.json"
    TARGET_TRAIN_SUBSETS_PATH = args.data_dir / "target_train_subsets_by_dataset_fold_percent.json"
    ERRORS_DIR = args.errors_dir
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    if args.model == "persistence":
        run_persistence()
    else:
        run_regression(args.model)


if __name__ == "__main__":
    main()
