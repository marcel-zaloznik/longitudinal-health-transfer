#!/usr/bin/env python3
"""Aggregate observed-target errors into participant-, outcome- and dataset-balanced metrics.

Scoring SDs come from unique output visits in the four non-test folds,
including validation participants. They are separate from model-training scalers.
The main score is main_macro_nrmse (primary_9 panel, all horizons).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2.0.2"
CACHE_VERSION = "longitudinal-error-aggregation-v3"

REQUIRED_COLUMNS = [
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

READ_COLUMNS = REQUIRED_COLUMNS.copy()

PERSON_KEYS = [
    "dataset",
    "fold",
    "person_id",
    "attribute",
    "horizon_group",
]

PERSON_SUM_COLUMNS = [
    "sum_normalized_squared_error",
    "sum_absolute_normalized_error",
    "sum_normalized_error",
    "n_errors",
]

HORIZON_ORDER = {"all": 0, "short": 1, "medium": 2, "long": 3}
PANEL_PREFERRED_ORDER = ["primary_9", "common_all_datasets", "extended_20"]

GENERATED_FILENAMES = [
    "model_run_metrics.csv",
    "model_seed_summary.csv",
    "dataset_metrics.csv",
    "dataset_seed_summary.csv",
    "dataset_attribute_metrics.csv",
    "fold_metrics.csv",
    "source_target_metrics.csv",
    "paper_main_table.csv",
    "missing_evaluation_cells.csv",
    "file_inventory.csv",
    "analysis_manifest.json",
    "analysis.log",
]


@dataclass(frozen=True)
class HorizonDefinition:
    mode: str
    short_max: float
    medium_max: float
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    raw_tertile_edges: Optional[Tuple[float, float, float, float]] = None
    n_reference_pairs: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "labels": {
                "short": f"0 <= horizon <= {self.short_max:g}",
                "medium": f"{self.short_max:g} < horizon <= {self.medium_max:g}",
                "long": f"horizon > {self.medium_max:g}",
            },
            "short_max": self.short_max,
            "medium_max": self.medium_max,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "raw_tertile_edges": list(self.raw_tertile_edges) if self.raw_tertile_edges else None,
            "n_reference_pairs": self.n_reference_pairs,
        }


@dataclass
class AuxiliaryData:
    evaluation_lookup: pd.DataFrame
    panels: Dict[str, List[str]]
    expected_datasets: List[str]
    expected_cells: pd.DataFrame
    expected_by_panel_dataset: pd.DataFrame
    fingerprint: str


@dataclass
class ProcessedRun:
    run_metrics: pd.DataFrame
    dataset_metrics: pd.DataFrame
    dataset_attribute_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    missing_cells: pd.DataFrame
    inventory: Dict[str, Any]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def safe_join_unique(values: Iterable[Any], *, empty: str = "") -> str:
    cleaned: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        return empty
    return "|".join(sorted(cleaned))


def single_or_summary(values: Iterable[Any], *, varying_label: str = "<varies>") -> Any:
    unique: List[Any] = []
    for value in values:
        if pd.isna(value):
            continue
        if value not in unique:
            unique.append(value)
    if not unique:
        return np.nan
    if len(unique) == 1:
        return unique[0]
    return varying_label


def parse_seed_from_run_id(run_id: str) -> Optional[int]:
    match = re.search(r"_seed(-?\d+)$", run_id)
    return int(match.group(1)) if match else None


def configuration_id_from_run_id(run_id: str) -> str:
    return re.sub(r"_seed-?\d+$", "", run_id)


def base_configuration_id(configuration_id: str) -> str:
    return re.sub(r"_pct\d+(?=$|_)", "", configuration_id)


def train_percent_from_run_id(run_id: str) -> int:
    match = re.search(r"_pct(\d+)(?:_|$)", run_id)
    return int(match.group(1)) if match else 100


def infer_model_family(run_id: str) -> str:
    name = run_id.lower()
    if name.startswith("strats"):
        return "STraTS"
    if name.startswith("gru_d") or name.startswith("grud"):
        return "GRU-D"
    if name.startswith("linear_regression"):
        return "Linear regression"
    if name.startswith("random_forest"):
        return "Random forest"
    if name.startswith("persistence"):
        return "Persistence baseline"
    return "Other"


def sort_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "panel" in out.columns:
        panel_order = {name: idx for idx, name in enumerate(PANEL_PREFERRED_ORDER)}
        out["__panel_order"] = out["panel"].map(panel_order).fillna(len(panel_order))
    if "horizon_group" in out.columns:
        out["__horizon_order"] = out["horizon_group"].map(HORIZON_ORDER).fillna(99)

    preferred = [
        "configuration_id",
        "run_id",
        "dataset",
        "fold",
        "attribute",
        "__panel_order",
        "__horizon_order",
    ]
    sort_cols = [col for col in preferred if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort")
    return out.drop(columns=["__panel_order", "__horizon_order"], errors="ignore").reset_index(drop=True)


def ensure_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(parents=True, exist_ok=True)


def configure_logging(output_dir: Path, verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_path = output_dir / "analysis.log"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


# -----------------------------------------------------------------------------
# Auxiliary data and horizon definition
# -----------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_evaluation_lookup(stds: Mapping[str, Any], eligible: Mapping[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    datasets = sorted(set(stds) | set(eligible))
    for dataset in datasets:
        std_folds = stds.get(dataset, {})
        eligible_folds = eligible.get(dataset, {})
        folds = sorted(set(std_folds) | set(eligible_folds), key=lambda x: int(x))

        for fold_key in folds:
            std_attrs = std_folds.get(fold_key, {})
            eligible_attrs = eligible_folds.get(fold_key, {})
            attributes = sorted(set(std_attrs) | set(eligible_attrs))

            for attribute in attributes:
                info = eligible_attrs.get(attribute, {}) or {}
                std_value = std_attrs.get(attribute, info.get("std"))
                rows.append(
                    {
                        "dataset": str(dataset),
                        "fold": int(fold_key),
                        "attribute": str(attribute),
                        "attribute_std": std_value,
                        "eligible": bool(info.get("eligible", False)),
                        "train_observed_unique_output_rows": info.get(
                            "train_observed_unique_output_rows"
                        ),
                        "test_observed_unique_output_rows": info.get(
                            "test_observed_unique_output_rows"
                        ),
                        "lookup_present": True,
                    }
                )

    lookup = pd.DataFrame(rows)
    if lookup.empty:
        raise ValueError("The standard-deviation/eligibility lookup is empty.")

    lookup["attribute_std"] = pd.to_numeric(lookup["attribute_std"], errors="coerce")
    lookup["train_observed_unique_output_rows"] = pd.to_numeric(
        lookup["train_observed_unique_output_rows"], errors="coerce"
    )
    lookup["test_observed_unique_output_rows"] = pd.to_numeric(
        lookup["test_observed_unique_output_rows"], errors="coerce"
    )

    duplicate_mask = lookup.duplicated(["dataset", "fold", "attribute"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = lookup.loc[duplicate_mask, ["dataset", "fold", "attribute"]]
        raise ValueError(
            "Duplicate dataset/fold/attribute entries in evaluation lookup: "
            f"{duplicate_rows.head(10).to_dict(orient='records')}"
        )

    return lookup


def derive_common_panel(lookup: pd.DataFrame, extended_panel: Sequence[str]) -> List[str]:
    eligible = lookup[lookup["eligible"].eq(True)].copy()
    all_dataset_fold_pairs = lookup[["dataset", "fold"]].drop_duplicates().shape[0]

    counts = (
        eligible[eligible["attribute"].isin(extended_panel)]
        .groupby("attribute", as_index=False)[["dataset", "fold"]]
        .size()
        .rename(columns={"size": "n_dataset_fold_pairs"})
    )
    common = set(
        counts.loc[
            counts["n_dataset_fold_pairs"].eq(all_dataset_fold_pairs), "attribute"
        ].astype(str)
    )
    return [attribute for attribute in extended_panel if attribute in common]


def build_expected_panel_table(
    lookup: pd.DataFrame,
    panels: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    eligible = lookup[lookup["eligible"].eq(True)].copy()
    rows: List[pd.DataFrame] = []

    for panel_name, attributes in panels.items():
        panel_eligible = eligible[eligible["attribute"].isin(attributes)].copy()
        if panel_eligible.empty:
            continue

        by_dataset = (
            panel_eligible.groupby("dataset", as_index=False)
            .agg(
                n_attributes_expected=("attribute", "nunique"),
                n_fold_attribute_cells_expected=("attribute", "size"),
            )
        )
        by_dataset.insert(0, "panel", panel_name)
        rows.append(by_dataset)

    if not rows:
        return pd.DataFrame(
            columns=[
                "panel",
                "dataset",
                "n_attributes_expected",
                "n_fold_attribute_cells_expected",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def compute_auxiliary_fingerprint(paths: Sequence[Path], extra: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(json.dumps(extra, sort_keys=True, default=json_default).encode("utf-8"))
    return digest.hexdigest()


def load_auxiliary_data(data_dir: Path, horizon: HorizonDefinition) -> AuxiliaryData:
    std_path = data_dir / "target_attribute_stds_by_dataset_fold.json"
    eligible_path = data_dir / "eligible_dataset_attribute_pairs.json"
    primary_path = data_dir / "primary_target_panel.json"
    extended_path = data_dir / "extended_target_panel.json"

    stds = load_json(std_path)
    eligible = load_json(eligible_path)
    primary_panel = [str(x) for x in load_json(primary_path)]
    extended_panel = [str(x) for x in load_json(extended_path)]

    lookup = make_evaluation_lookup(stds, eligible)
    common_panel = derive_common_panel(lookup, extended_panel)

    panels: Dict[str, List[str]] = {
        "primary_9": primary_panel,
        "common_all_datasets": common_panel,
        "extended_20": extended_panel,
    }

    expected_cells = lookup[lookup["eligible"].eq(True)][
        ["dataset", "fold", "attribute"]
    ].drop_duplicates()
    expected_by_panel_dataset = build_expected_panel_table(lookup, panels)
    expected_datasets = sorted(lookup["dataset"].dropna().astype(str).unique().tolist())

    fingerprint = compute_auxiliary_fingerprint(
        [std_path, eligible_path, primary_path, extended_path],
        {"horizon": horizon.as_dict(), "panels": panels, "cache_version": CACHE_VERSION},
    )

    return AuxiliaryData(
        evaluation_lookup=lookup,
        panels=panels,
        expected_datasets=expected_datasets,
        expected_cells=expected_cells,
        expected_by_panel_dataset=expected_by_panel_dataset,
        fingerprint=fingerprint,
    )


def parse_last_input_index(cell: Any) -> Optional[int]:
    if pd.isna(cell):
        return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        return int(cell[-1]) if len(cell) else None

    text = str(cell).strip()
    if not text:
        return None

    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple, np.ndarray)):
            return int(value[-1]) if len(value) else None
        return int(value)
    except Exception:
        numbers = re.findall(r"-?\d+", text)
        return int(numbers[-1]) if numbers else None


def compute_reference_horizon_definition(data_dir: Path) -> HorizonDefinition:
    data_path = data_dir / "dataset.csv"
    pairs_path = data_dir / "input_output_pairs.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Required file not found: {data_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Required file not found: {pairs_path}")

    logging.info("Computing horizon tertiles from all reference prefix-to-future pairs.")
    ages = pd.read_csv(data_path, usecols=["Age"], low_memory=False)["Age"]
    ages = pd.to_numeric(ages, errors="coerce").to_numpy(dtype=float)

    pairs = pd.read_csv(pairs_path, usecols=["input_rows", "output_row"], low_memory=False)
    last_input = pairs["input_rows"].map(parse_last_input_index)
    output_row = pd.to_numeric(pairs["output_row"], errors="coerce")

    valid = last_input.notna() & output_row.notna()
    last_idx = last_input.loc[valid].astype(int).to_numpy()
    output_idx = output_row.loc[valid].astype(int).to_numpy()

    in_bounds = (
        (last_idx >= 0)
        & (last_idx < len(ages))
        & (output_idx >= 0)
        & (output_idx < len(ages))
    )
    last_idx = last_idx[in_bounds]
    output_idx = output_idx[in_bounds]

    horizons = ages[output_idx] - ages[last_idx]
    horizons = horizons[np.isfinite(horizons) & (horizons >= 0)]
    if horizons.size == 0:
        raise ValueError("No valid non-negative horizons could be constructed.")

    _, edges = pd.qcut(horizons, q=3, retbins=True, duplicates="drop")
    if len(edges) != 4:
        raise ValueError(f"Expected four tertile edges, received: {edges.tolist()}")

    edge_tuple = tuple(float(x) for x in edges)
    return HorizonDefinition(
        mode="reference_tertiles",
        short_max=float(edges[1]),
        medium_max=float(edges[2]),
        minimum=float(edges[0]),
        maximum=float(edges[3]),
        raw_tertile_edges=edge_tuple,  # type: ignore[arg-type]
        n_reference_pairs=int(horizons.size),
    )


def resolve_horizon_definition(args: argparse.Namespace, data_dir: Path) -> HorizonDefinition:
    if args.horizon_mode == "fixed":
        if not (math.isfinite(args.short_max) and math.isfinite(args.medium_max)):
            raise ValueError("Fixed horizon boundaries must be finite.")
        if args.short_max < 0 or args.medium_max <= args.short_max:
            raise ValueError("Require 0 <= short-max < medium-max.")
        return HorizonDefinition(
            mode="fixed",
            short_max=float(args.short_max),
            medium_max=float(args.medium_max),
        )
    return compute_reference_horizon_definition(data_dir)


# -----------------------------------------------------------------------------
# Per-file processing
# -----------------------------------------------------------------------------


def cache_path_for(error_path: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(error_path.resolve()).encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", error_path.stem)
    return cache_dir / f"{safe_name}_{digest}.pkl"


def cache_signature(error_path: Path, auxiliary_fingerprint: str) -> Dict[str, Any]:
    stat = error_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "script_version": SCRIPT_VERSION,
        "error_path": str(error_path.resolve()),
        "error_size": stat.st_size,
        "error_mtime_ns": stat.st_mtime_ns,
        "auxiliary_fingerprint": auxiliary_fingerprint,
    }


def load_cached_run(cache_path: Path, signature: Mapping[str, Any]) -> Optional[ProcessedRun]:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("signature") != dict(signature):
            return None
        result = payload.get("result")
        if not isinstance(result, ProcessedRun):
            return None
        result.inventory = dict(result.inventory)
        result.inventory["cache_hit"] = True
        result.inventory["processing_seconds"] = 0.0
        return result
    except Exception as exc:  # pragma: no cover - defensive cache handling
        logging.warning("Ignoring unreadable cache %s: %s", cache_path, exc)
        return None


def save_cached_run(cache_path: Path, signature: Mapping[str, Any], result: ProcessedRun) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(
            {"signature": dict(signature), "result": result},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, cache_path)


def collapse_person_partials(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=PERSON_KEYS + PERSON_SUM_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return combined
    return (
        combined.groupby(PERSON_KEYS, as_index=False, sort=False, dropna=False, observed=True)[
            PERSON_SUM_COLUMNS
        ]
        .sum()
    )


def summarize_dataset_metadata(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(
            columns=[
                "dataset",
                "source_config",
                "source_dataset",
                "n_pretrain_datasets",
                "n_pretrain_datasets_min",
                "n_pretrain_datasets_max",
            ]
        )

    metadata = pd.concat(parts, ignore_index=True).drop_duplicates()
    rows: List[Dict[str, Any]] = []
    for dataset, group in metadata.groupby("dataset", sort=False, dropna=False):
        n_values = pd.to_numeric(group["n_pretrain_datasets"], errors="coerce").dropna()
        n_unique = sorted(set(float(x) for x in n_values.tolist()))
        rows.append(
            {
                "dataset": str(dataset),
                "source_config": single_or_summary(
                    group["source_config"], varying_label="<multiple_within_target>"
                ),
                "source_dataset": single_or_summary(
                    group["source_dataset"], varying_label="<multiple_within_target>"
                ),
                "n_pretrain_datasets": n_unique[0] if len(n_unique) == 1 else np.nan,
                "n_pretrain_datasets_min": min(n_unique) if n_unique else np.nan,
                "n_pretrain_datasets_max": max(n_unique) if n_unique else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_all_horizon_group(person_binned: pd.DataFrame) -> pd.DataFrame:
    if person_binned.empty:
        return person_binned
    all_keys = [key for key in PERSON_KEYS if key != "horizon_group"]
    all_horizons = (
        person_binned.groupby(
            all_keys, as_index=False, sort=False, dropna=False, observed=True
        )[PERSON_SUM_COLUMNS]
        .sum()
    )
    all_horizons["horizon_group"] = "all"
    all_horizons = all_horizons[PERSON_KEYS + PERSON_SUM_COLUMNS]
    return pd.concat([person_binned, all_horizons], ignore_index=True)


def compute_dataset_attribute_metrics(
    person: pd.DataFrame,
    auxiliary: AuxiliaryData,
    dataset_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if person.empty:
        empty = pd.DataFrame()
        return empty, empty

    work = person.copy()
    work["person_normalized_mse"] = (
        work["sum_normalized_squared_error"] / work["n_errors"]
    )
    work["person_normalized_mae"] = (
        work["sum_absolute_normalized_error"] / work["n_errors"]
    )
    work["person_normalized_bias"] = work["sum_normalized_error"] / work["n_errors"]

    cell_group = ["dataset", "attribute", "horizon_group"]
    cell = (
        work.groupby(cell_group, as_index=False, sort=False, dropna=False, observed=True)
        .agg(
            person_balanced_normalized_mse=("person_normalized_mse", "mean"),
            person_balanced_nmae=("person_normalized_mae", "mean"),
            person_balanced_normalized_bias=("person_normalized_bias", "mean"),
            sum_normalized_squared_error=("sum_normalized_squared_error", "sum"),
            sum_absolute_normalized_error=("sum_absolute_normalized_error", "sum"),
            sum_normalized_error=("sum_normalized_error", "sum"),
            n_error_rows=("n_errors", "sum"),
            n_test_people=("person_id", "nunique"),
            n_folds_evaluated=("fold", "nunique"),
            n_person_attribute_units=("person_id", "size"),
        )
    )
    cell["person_balanced_nrmse"] = np.sqrt(
        cell["person_balanced_normalized_mse"]
    )
    cell["person_balanced_absolute_normalized_bias"] = cell[
        "person_balanced_normalized_bias"
    ].abs()
    cell["pair_weighted_nrmse"] = np.sqrt(
        cell["sum_normalized_squared_error"] / cell["n_error_rows"]
    )
    cell["pair_weighted_nmae"] = (
        cell["sum_absolute_normalized_error"] / cell["n_error_rows"]
    )
    cell["pair_weighted_normalized_bias"] = (
        cell["sum_normalized_error"] / cell["n_error_rows"]
    )

    eligible_folds = (
        auxiliary.evaluation_lookup[auxiliary.evaluation_lookup["eligible"].eq(True)]
        .groupby(["dataset", "attribute"], as_index=False)
        .agg(n_eligible_folds_expected=("fold", "nunique"))
    )
    cell = cell.merge(eligible_folds, on=["dataset", "attribute"], how="left")
    cell["all_horizon_fold_coverage"] = np.where(
        cell["horizon_group"].eq("all"),
        cell["n_folds_evaluated"] / cell["n_eligible_folds_expected"],
        np.nan,
    )

    primary = set(auxiliary.panels.get("primary_9", []))
    common = set(auxiliary.panels.get("common_all_datasets", []))
    cell["in_primary_9"] = cell["attribute"].isin(primary)
    cell["in_common_all_datasets"] = cell["attribute"].isin(common)
    cell = cell.merge(dataset_metadata, on="dataset", how="left", validate="many_to_one")

    fold_group = ["dataset", "fold", "attribute", "horizon_group"]
    fold_cell = (
        work.groupby(fold_group, as_index=False, sort=False, dropna=False, observed=True)
        .agg(
            person_balanced_normalized_mse=("person_normalized_mse", "mean"),
            person_balanced_nmae=("person_normalized_mae", "mean"),
            person_balanced_normalized_bias=("person_normalized_bias", "mean"),
            sum_normalized_squared_error=("sum_normalized_squared_error", "sum"),
            sum_absolute_normalized_error=("sum_absolute_normalized_error", "sum"),
            sum_normalized_error=("sum_normalized_error", "sum"),
            n_error_rows=("n_errors", "sum"),
            n_test_people=("person_id", "nunique"),
        )
    )
    fold_cell["person_balanced_nrmse"] = np.sqrt(
        fold_cell["person_balanced_normalized_mse"]
    )
    fold_cell["person_balanced_absolute_normalized_bias"] = fold_cell[
        "person_balanced_normalized_bias"
    ].abs()
    fold_cell["pair_weighted_nrmse"] = np.sqrt(
        fold_cell["sum_normalized_squared_error"] / fold_cell["n_error_rows"]
    )
    fold_cell["pair_weighted_nmae"] = (
        fold_cell["sum_absolute_normalized_error"] / fold_cell["n_error_rows"]
    )
    fold_cell["pair_weighted_normalized_bias"] = (
        fold_cell["sum_normalized_error"] / fold_cell["n_error_rows"]
    )
    return cell, fold_cell


def aggregate_panel_metrics(
    person: pd.DataFrame,
    cell: pd.DataFrame,
    fold_cell: pd.DataFrame,
    auxiliary: AuxiliaryData,
    dataset_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    run_rows: List[Dict[str, Any]] = []

    expected_panel = auxiliary.expected_by_panel_dataset.copy()

    for panel_name, panel_attributes in auxiliary.panels.items():
        panel_set = set(panel_attributes)
        panel_cell = cell[cell["attribute"].isin(panel_set)].copy()
        panel_person = person[person["attribute"].isin(panel_set)].copy()
        panel_fold_cell = fold_cell[fold_cell["attribute"].isin(panel_set)].copy()

        if panel_cell.empty:
            continue

        dataset = (
            panel_cell.groupby(
                ["dataset", "horizon_group"],
                as_index=False,
                sort=False,
                dropna=False,
                observed=True,
            )
            .agg(
                main_macro_nrmse=("person_balanced_nrmse", "mean"),
                main_macro_nmae=("person_balanced_nmae", "mean"),
                main_macro_absolute_normalized_bias=(
                    "person_balanced_absolute_normalized_bias",
                    "mean",
                ),
                mean_signed_normalized_bias=("person_balanced_normalized_bias", "mean"),
                secondary_pair_macro_nrmse=("pair_weighted_nrmse", "mean"),
                secondary_pair_macro_nmae=("pair_weighted_nmae", "mean"),
                secondary_pair_mean_signed_normalized_bias=(
                    "pair_weighted_normalized_bias",
                    "mean",
                ),
                n_attributes_evaluated=("attribute", "nunique"),
                n_dataset_attribute_cells=("attribute", "size"),
                n_error_rows=("n_error_rows", "sum"),
            )
        )

        person_counts = (
            panel_person.groupby(
                ["dataset", "horizon_group"],
                as_index=False,
                sort=False,
                observed=True,
            )
            .agg(
                n_test_people=("person_id", "nunique"),
                n_folds_evaluated=("fold", "nunique"),
            )
        )
        observed_fold_cells = (
            panel_fold_cell.groupby(
                ["dataset", "horizon_group"],
                as_index=False,
                sort=False,
                observed=True,
            )
            .size()
            .rename(columns={"size": "n_fold_attribute_cells_evaluated"})
        )

        dataset = dataset.merge(
            person_counts,
            on=["dataset", "horizon_group"],
            how="left",
            validate="one_to_one",
        )
        dataset = dataset.merge(
            observed_fold_cells,
            on=["dataset", "horizon_group"],
            how="left",
            validate="one_to_one",
        )
        expected_for_panel = expected_panel[expected_panel["panel"].eq(panel_name)].drop(
            columns="panel"
        )
        dataset = dataset.merge(
            expected_for_panel,
            on="dataset",
            how="left",
            validate="many_to_one",
        )
        dataset["evaluation_cell_coverage"] = np.where(
            dataset["horizon_group"].eq("all"),
            dataset["n_fold_attribute_cells_evaluated"]
            / dataset["n_fold_attribute_cells_expected"],
            np.nan,
        )
        dataset["attribute_coverage"] = np.where(
            dataset["horizon_group"].eq("all"),
            dataset["n_attributes_evaluated"] / dataset["n_attributes_expected"],
            np.nan,
        )
        dataset.insert(0, "panel", panel_name)
        dataset = dataset.merge(
            dataset_metadata,
            on="dataset",
            how="left",
            validate="many_to_one",
        )
        dataset_frames.append(dataset)

        fold = (
            panel_fold_cell.groupby(
                ["dataset", "fold", "horizon_group"],
                as_index=False,
                sort=False,
                dropna=False,
                observed=True,
            )
            .agg(
                main_macro_nrmse=("person_balanced_nrmse", "mean"),
                main_macro_nmae=("person_balanced_nmae", "mean"),
                main_macro_absolute_normalized_bias=(
                    "person_balanced_absolute_normalized_bias",
                    "mean",
                ),
                mean_signed_normalized_bias=("person_balanced_normalized_bias", "mean"),
                secondary_pair_macro_nrmse=("pair_weighted_nrmse", "mean"),
                secondary_pair_macro_nmae=("pair_weighted_nmae", "mean"),
                n_attributes_evaluated=("attribute", "nunique"),
                n_error_rows=("n_error_rows", "sum"),
            )
        )
        fold_people = (
            panel_person.groupby(
                ["dataset", "fold", "horizon_group"],
                as_index=False,
                sort=False,
                observed=True,
            )
            .agg(n_test_people=("person_id", "nunique"))
        )
        fold = fold.merge(
            fold_people,
            on=["dataset", "fold", "horizon_group"],
            how="left",
            validate="one_to_one",
        )

        expected_fold = (
            auxiliary.evaluation_lookup[
                auxiliary.evaluation_lookup["eligible"].eq(True)
                & auxiliary.evaluation_lookup["attribute"].isin(panel_set)
            ]
            .groupby(["dataset", "fold"], as_index=False)
            .agg(n_attributes_expected=("attribute", "nunique"))
        )
        fold = fold.merge(
            expected_fold,
            on=["dataset", "fold"],
            how="left",
            validate="many_to_one",
        )
        fold["attribute_coverage"] = np.where(
            fold["horizon_group"].eq("all"),
            fold["n_attributes_evaluated"] / fold["n_attributes_expected"],
            np.nan,
        )
        fold.insert(0, "panel", panel_name)
        fold = fold.merge(
            dataset_metadata,
            on="dataset",
            how="left",
            validate="many_to_one",
        )
        fold_frames.append(fold)

        expected_datasets = set(
            expected_for_panel.loc[
                expected_for_panel["n_attributes_expected"].gt(0), "dataset"
            ].astype(str)
        )
        expected_pairs_total = int(expected_for_panel["n_attributes_expected"].sum())
        expected_fold_cells_total = int(
            expected_for_panel["n_fold_attribute_cells_expected"].sum()
        )

        for horizon_group, group in dataset.groupby("horizon_group", sort=False):
            observed_datasets = set(group["dataset"].astype(str))
            dataset_complete = observed_datasets == expected_datasets
            n_pairs = int(group["n_dataset_attribute_cells"].sum())
            n_fold_cells = int(group["n_fold_attribute_cells_evaluated"].sum())
            all_horizons = horizon_group == "all"
            cell_complete = bool(
                all_horizons
                and dataset_complete
                and n_fold_cells == expected_fold_cells_total
            )

            main_score = float(group["main_macro_nrmse"].mean())
            row = {
                "panel": panel_name,
                "horizon_group": horizon_group,
                "main_macro_nrmse": main_score,
                "main_macro_nmae": float(group["main_macro_nmae"].mean()),
                "main_macro_absolute_normalized_bias": float(
                    group["main_macro_absolute_normalized_bias"].mean()
                ),
                "mean_signed_normalized_bias": float(
                    group["mean_signed_normalized_bias"].mean()
                ),
                "secondary_pair_macro_nrmse": float(
                    group["secondary_pair_macro_nrmse"].mean()
                ),
                "secondary_pair_macro_nmae": float(
                    group["secondary_pair_macro_nmae"].mean()
                ),
                "n_datasets_evaluated": int(group["dataset"].nunique()),
                "n_datasets_expected": len(expected_datasets),
                "dataset_coverage": (
                    len(observed_datasets) / len(expected_datasets)
                    if expected_datasets
                    else np.nan
                ),
                "n_dataset_attribute_pairs_evaluated": n_pairs,
                "n_dataset_attribute_pairs_expected": (
                    expected_pairs_total if all_horizons else np.nan
                ),
                "n_fold_attribute_cells_evaluated": n_fold_cells,
                "n_fold_attribute_cells_expected": (
                    expected_fold_cells_total if all_horizons else np.nan
                ),
                "evaluation_cell_coverage": (
                    n_fold_cells / expected_fold_cells_total
                    if all_horizons and expected_fold_cells_total
                    else np.nan
                ),
                "n_error_rows": int(group["n_error_rows"].sum()),
                "n_test_people": int(group["n_test_people"].sum()),
                "is_complete_dataset_coverage": bool(dataset_complete),
                "is_complete_evaluation_coverage": bool(cell_complete),
                "complete_case_main_macro_nrmse": main_score if cell_complete else np.nan,
            }
            run_rows.append(row)

    dataset_metrics = (
        pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame()
    )
    fold_metrics = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    run_metrics = pd.DataFrame(run_rows)
    return dataset_metrics, fold_metrics, run_metrics


def make_run_metadata(
    error_path: Path,
    method_values: Set[str],
    seed_values: Set[int],
    dataset_metadata: pd.DataFrame,
) -> Dict[str, Any]:
    run_id = error_path.stem
    filename_seed = parse_seed_from_run_id(run_id)

    if len(seed_values) > 1:
        raise ValueError(
            f"{error_path.name} contains multiple seeds: {sorted(seed_values)}. "
            "Keep one run/seed per error CSV."
        )
    seed: Optional[int]
    if seed_values:
        seed = next(iter(seed_values))
    else:
        seed = filename_seed

    method = single_or_summary(method_values, varying_label="<multiple_methods>")
    source_configs = dataset_metadata.get("source_config", pd.Series(dtype=object))
    source_datasets = dataset_metadata.get("source_dataset", pd.Series(dtype=object))
    n_pretrain = pd.to_numeric(
        dataset_metadata.get("n_pretrain_datasets", pd.Series(dtype=float)), errors="coerce"
    ).dropna()

    source_config_summary = single_or_summary(
        source_configs, varying_label="<varies_by_target>"
    )
    source_dataset_summary = single_or_summary(
        source_datasets, varying_label="<varies_by_target>"
    )
    n_pretrain_summary: Any = (
        float(n_pretrain.iloc[0])
        if len(set(n_pretrain.tolist())) == 1 and len(n_pretrain) > 0
        else np.nan
    )

    configuration_id = configuration_id_from_run_id(run_id)
    return {
        "run_id": run_id,
        "configuration_id": configuration_id,
        "base_configuration_id": base_configuration_id(configuration_id),
        "model_family": infer_model_family(run_id),
        "method": method,
        "seed": seed,
        "train_data_percent": train_percent_from_run_id(run_id),
        "source_config_summary": source_config_summary,
        "source_dataset_summary": source_dataset_summary,
        "n_pretrain_datasets_summary": n_pretrain_summary,
        "error_file": error_path.name,
    }


def prepend_run_metadata(df: pd.DataFrame, metadata: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for key, value in reversed(list(metadata.items())):
        out.insert(0, key, value)
    return out


def process_error_file(
    error_path: Path,
    auxiliary: AuxiliaryData,
    horizon: HorizonDefinition,
    chunksize: int,
    cache_dir: Path,
    use_cache: bool,
) -> ProcessedRun:
    signature = cache_signature(error_path, auxiliary.fingerprint)
    cache_path = cache_path_for(error_path, cache_dir)
    if use_cache:
        cached = load_cached_run(cache_path, signature)
        if cached is not None:
            logging.info("Cache hit: %s", error_path.name)
            return cached

    started = time.perf_counter()
    logging.info("Processing: %s", error_path.name)

    header = pd.read_csv(error_path, nrows=0)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(header.columns))
    if missing_columns:
        raise ValueError(
            f"Old/invalid error schema in {error_path.name}; missing columns: {missing_columns}"
        )

    counters: MutableMapping[str, int] = {
        "n_raw_rows": 0,
        "n_evaluated_rows": 0,
        "n_dropped_missing_identity": 0,
        "n_dropped_invalid_numeric": 0,
        "n_dropped_nonfinite": 0,
        "n_dropped_negative_horizon": 0,
        "n_dropped_missing_lookup": 0,
        "n_dropped_ineligible": 0,
        "n_dropped_invalid_std": 0,
        "n_horizon_inconsistencies_corrected": 0,
    }

    method_values: Set[str] = set()
    seed_values: Set[int] = set()
    person_parts: List[pd.DataFrame] = []
    dataset_metadata_parts: List[pd.DataFrame] = []
    partial_rows = 0

    lookup = auxiliary.evaluation_lookup

    reader = pd.read_csv(
        error_path,
        usecols=READ_COLUMNS,
        dtype={"person_id": str},
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_index, chunk in enumerate(reader, start=1):
        counters["n_raw_rows"] += len(chunk)
        if chunk.empty:
            continue
        errors = pd.to_numeric(chunk["error"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(errors).all():
            raise ValueError(f"Non-finite or invalid observed-target error in {error_path.name}.")

        for value in chunk["method"].dropna().astype(str).unique().tolist():
            method_values.add(value)

        seed_numeric = pd.to_numeric(chunk["seed"], errors="coerce").dropna()
        for value in seed_numeric.unique().tolist():
            if math.isfinite(float(value)) and float(value).is_integer():
                seed_values.add(int(value))

        work = chunk.copy()
        for col in [
            "dataset",
            "person_id",
            "attribute",
            "source_config",
            "source_dataset",
        ]:
            work[col] = work[col].fillna("").astype(str).str.strip()

        missing_identity = (
            work["dataset"].eq("")
            | work["person_id"].eq("")
            | work["attribute"].eq("")
        )
        counters["n_dropped_missing_identity"] += int(missing_identity.sum())
        work = work.loc[~missing_identity].copy()
        if work.empty:
            continue

        numeric_cols = [
            "fold",
            "error",
            "input_age",
            "output_age",
            "horizon",
            "n_input_visits",
            "n_pretrain_datasets",
        ]
        for col in numeric_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        valid_fold = work["fold"].notna() & np.isclose(
            work["fold"], np.round(work["fold"]), rtol=0.0, atol=1e-9
        )
        required_numeric = work[["error", "horizon"]].notna().all(axis=1)
        invalid_numeric = ~(valid_fold & required_numeric)
        counters["n_dropped_invalid_numeric"] += int(invalid_numeric.sum())
        work = work.loc[~invalid_numeric].copy()
        if work.empty:
            continue
        work["fold"] = np.round(work["fold"]).astype(int)

        finite = np.isfinite(work["error"].to_numpy(dtype=float)) & np.isfinite(
            work["horizon"].to_numpy(dtype=float)
        )
        counters["n_dropped_nonfinite"] += int((~finite).sum())
        work = work.loc[finite].copy()
        if work.empty:
            continue

        ages_available = (
            work["input_age"].notna()
            & work["output_age"].notna()
            & np.isfinite(work["input_age"])
            & np.isfinite(work["output_age"])
        )
        recalculated = work["output_age"] - work["input_age"]
        inconsistent = ages_available & ~np.isclose(
            work["horizon"], recalculated, rtol=0.0, atol=1e-6
        )
        counters["n_horizon_inconsistencies_corrected"] += int(inconsistent.sum())
        work.loc[ages_available, "horizon"] = recalculated.loc[ages_available]

        negative_horizon = work["horizon"].lt(0)
        counters["n_dropped_negative_horizon"] += int(negative_horizon.sum())
        work = work.loc[~negative_horizon].copy()
        if work.empty:
            continue

        dataset_metadata_parts.append(
            work[
                [
                    "dataset",
                    "source_config",
                    "source_dataset",
                    "n_pretrain_datasets",
                ]
            ].drop_duplicates()
        )

        work = work.merge(
            lookup,
            on=["dataset", "fold", "attribute"],
            how="left",
            validate="many_to_one",
        )

        missing_lookup = work["lookup_present"].isna()
        counters["n_dropped_missing_lookup"] += int(missing_lookup.sum())
        work = work.loc[~missing_lookup].copy()
        if work.empty:
            continue

        ineligible = ~work["eligible"].eq(True)
        counters["n_dropped_ineligible"] += int(ineligible.sum())
        work = work.loc[~ineligible].copy()
        if work.empty:
            continue

        valid_std = (
            work["attribute_std"].notna()
            & np.isfinite(work["attribute_std"])
            & work["attribute_std"].gt(0)
        )
        counters["n_dropped_invalid_std"] += int((~valid_std).sum())
        work = work.loc[valid_std].copy()
        if work.empty:
            continue

        normalized_error = work["error"] / work["attribute_std"]
        work["sum_normalized_squared_error"] = normalized_error.pow(2)
        work["sum_absolute_normalized_error"] = normalized_error.abs()
        work["sum_normalized_error"] = normalized_error
        if not np.isfinite(work["sum_normalized_squared_error"].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite normalized squared error in {error_path.name}.")
        work["n_errors"] = 1
        work["horizon_group"] = np.where(
            work["horizon"].le(horizon.short_max),
            "short",
            np.where(work["horizon"].le(horizon.medium_max), "medium", "long"),
        )

        counters["n_evaluated_rows"] += len(work)
        partial = (
            work.groupby(
                PERSON_KEYS,
                as_index=False,
                sort=False,
                dropna=False,
                observed=True,
            )[PERSON_SUM_COLUMNS]
            .sum()
        )
        person_parts.append(partial)
        partial_rows += len(partial)

        if len(person_parts) >= 8 or partial_rows >= 2_000_000:
            collapsed = collapse_person_partials(person_parts)
            person_parts = [collapsed]
            partial_rows = len(collapsed)

        if chunk_index % 10 == 0:
            logging.debug(
                "%s: read %,d rows; retained %,d",
                error_path.name,
                counters["n_raw_rows"],
                counters["n_evaluated_rows"],
            )

    person_binned = collapse_person_partials(person_parts)
    if person_binned.empty:
        raise ValueError(f"No evaluable error rows remained in {error_path.name}.")
    person = add_all_horizon_group(person_binned)

    dataset_metadata = summarize_dataset_metadata(dataset_metadata_parts)
    metadata = make_run_metadata(error_path, method_values, seed_values, dataset_metadata)

    cell, fold_cell = compute_dataset_attribute_metrics(
        person, auxiliary, dataset_metadata
    )
    dataset_metrics, fold_metrics, run_metrics = aggregate_panel_metrics(
        person, cell, fold_cell, auxiliary, dataset_metadata
    )

    observed_all = person.loc[
        person["horizon_group"].eq("all"), ["dataset", "fold", "attribute"]
    ].drop_duplicates()
    missing_cells = auxiliary.expected_cells.merge(
        observed_all.assign(__observed=True),
        on=["dataset", "fold", "attribute"],
        how="left",
    )
    missing_cells = missing_cells[missing_cells["__observed"].isna()].drop(
        columns="__observed"
    )
    missing_cells["in_primary_9"] = missing_cells["attribute"].isin(
        auxiliary.panels.get("primary_9", [])
    )
    missing_cells["in_common_all_datasets"] = missing_cells["attribute"].isin(
        auxiliary.panels.get("common_all_datasets", [])
    )

    run_metrics = prepend_run_metadata(run_metrics, metadata)
    dataset_metrics = prepend_run_metadata(dataset_metrics, metadata)
    cell = prepend_run_metadata(cell, metadata)
    fold_metrics = prepend_run_metadata(fold_metrics, metadata)
    missing_cells = prepend_run_metadata(missing_cells, metadata)

    elapsed = time.perf_counter() - started
    stat = error_path.stat()
    inventory: Dict[str, Any] = {
        **metadata,
        "status": "ok",
        "cache_hit": False,
        "file_size_bytes": stat.st_size,
        "file_modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).replace(microsecond=0).isoformat(),
        **counters,
        "n_missing_expected_fold_attribute_cells": int(len(missing_cells)),
        "processing_seconds": round(elapsed, 3),
        "message": "",
    }

    result = ProcessedRun(
        run_metrics=run_metrics,
        dataset_metrics=dataset_metrics,
        dataset_attribute_metrics=cell,
        fold_metrics=fold_metrics,
        missing_cells=missing_cells,
        inventory=inventory,
    )
    if use_cache:
        save_cached_run(cache_path, signature, result)
    return result


# -----------------------------------------------------------------------------
# Across-seed summaries and outputs
# -----------------------------------------------------------------------------


def summarize_across_seeds(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    carry_columns: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(
        list(group_columns), sort=False, dropna=False, observed=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = dict(zip(group_columns, keys))

        seeds = pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int)
        row["n_runs"] = int(len(group))
        row["n_seeds"] = int(seeds.nunique())
        row["seeds"] = "|".join(str(x) for x in sorted(seeds.unique().tolist()))

        for column in carry_columns:
            if column in group.columns:
                row[column] = single_or_summary(
                    group[column], varying_label="<varies_across_runs>"
                )

        for column in metric_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) >= 2 else np.nan
            )
            row[f"{column}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{column}_max"] = float(values.max()) if len(values) else np.nan

        for coverage_column in [
            "dataset_coverage",
            "evaluation_cell_coverage",
            "attribute_coverage",
        ]:
            if coverage_column in group.columns:
                values = pd.to_numeric(group[coverage_column], errors="coerce").dropna()
                row[f"{coverage_column}_min"] = (
                    float(values.min()) if len(values) else np.nan
                )

        for bool_column in [
            "is_complete_dataset_coverage",
            "is_complete_evaluation_coverage",
        ]:
            if bool_column in group.columns:
                row[f"all_runs_{bool_column}"] = bool(group[bool_column].fillna(False).all())

        rows.append(row)

    return pd.DataFrame(rows)


def concatenate_frames(results: Sequence[ProcessedRun], attribute: str) -> pd.DataFrame:
    frames = [getattr(result, attribute) for result in results]
    frames = [frame for frame in frames if frame is not None]
    nonempty = [frame for frame in frames if not frame.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    return frames[0].copy() if frames else pd.DataFrame()


def make_failure_inventory(error_path: Path, message: str) -> Dict[str, Any]:
    stat = error_path.stat()
    run_id = error_path.stem
    configuration_id = configuration_id_from_run_id(run_id)
    return {
        "run_id": run_id,
        "configuration_id": configuration_id,
        "base_configuration_id": base_configuration_id(configuration_id),
        "model_family": infer_model_family(run_id),
        "method": np.nan,
        "seed": parse_seed_from_run_id(run_id),
        "train_data_percent": train_percent_from_run_id(run_id),
        "source_config_summary": np.nan,
        "source_dataset_summary": np.nan,
        "n_pretrain_datasets_summary": np.nan,
        "error_file": error_path.name,
        "status": "failed",
        "cache_hit": False,
        "file_size_bytes": stat.st_size,
        "file_modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).replace(microsecond=0).isoformat(),
        "n_raw_rows": np.nan,
        "n_evaluated_rows": np.nan,
        "n_dropped_missing_identity": np.nan,
        "n_dropped_invalid_numeric": np.nan,
        "n_dropped_nonfinite": np.nan,
        "n_dropped_negative_horizon": np.nan,
        "n_dropped_missing_lookup": np.nan,
        "n_dropped_ineligible": np.nan,
        "n_dropped_invalid_std": np.nan,
        "n_horizon_inconsistencies_corrected": np.nan,
        "n_missing_expected_fold_attribute_cells": np.nan,
        "processing_seconds": np.nan,
        "message": message,
    }


def write_outputs(
    output_dir: Path,
    processed: Sequence[ProcessedRun],
    inventory_rows: Sequence[Mapping[str, Any]],
    auxiliary: AuxiliaryData,
    horizon: HorizonDefinition,
    args: argparse.Namespace,
    error_files: Sequence[Path],
) -> Dict[str, int]:
    run_metrics = sort_metric_table(concatenate_frames(processed, "run_metrics"))
    dataset_metrics = sort_metric_table(concatenate_frames(processed, "dataset_metrics"))
    cell_metrics = sort_metric_table(
        concatenate_frames(processed, "dataset_attribute_metrics")
    )
    fold_metrics = sort_metric_table(concatenate_frames(processed, "fold_metrics"))
    missing_cells = sort_metric_table(concatenate_frames(processed, "missing_cells"))
    inventory = pd.DataFrame(inventory_rows)
    if not inventory.empty:
        inventory = inventory.sort_values(["status", "run_id"], kind="mergesort").reset_index(
            drop=True
        )

    run_metric_columns = [
        "main_macro_nrmse",
        "main_macro_nmae",
        "main_macro_absolute_normalized_bias",
        "mean_signed_normalized_bias",
        "secondary_pair_macro_nrmse",
        "secondary_pair_macro_nmae",
        "complete_case_main_macro_nrmse",
    ]
    run_carry = [
        "base_configuration_id",
        "model_family",
        "method",
        "train_data_percent",
        "source_config_summary",
        "source_dataset_summary",
        "n_pretrain_datasets_summary",
        "n_datasets_expected",
        "n_dataset_attribute_pairs_expected",
        "n_fold_attribute_cells_expected",
    ]
    model_seed_summary = summarize_across_seeds(
        run_metrics,
        group_columns=["configuration_id", "panel", "horizon_group"],
        metric_columns=run_metric_columns,
        carry_columns=run_carry,
    )
    model_seed_summary = sort_metric_table(model_seed_summary)

    dataset_metric_columns = [
        "main_macro_nrmse",
        "main_macro_nmae",
        "main_macro_absolute_normalized_bias",
        "mean_signed_normalized_bias",
        "secondary_pair_macro_nrmse",
        "secondary_pair_macro_nmae",
    ]
    dataset_carry = [
        "base_configuration_id",
        "model_family",
        "method",
        "train_data_percent",
        "source_config",
        "source_dataset",
        "n_pretrain_datasets",
        "n_attributes_expected",
        "n_fold_attribute_cells_expected",
    ]
    dataset_seed_summary = summarize_across_seeds(
        dataset_metrics,
        group_columns=["configuration_id", "dataset", "panel", "horizon_group"],
        metric_columns=dataset_metric_columns,
        carry_columns=dataset_carry,
    )
    dataset_seed_summary = sort_metric_table(dataset_seed_summary)

    source_target = dataset_metrics[
        dataset_metrics.get("source_config", pd.Series(dtype=object)).eq("single_source")
    ].copy()
    source_target = sort_metric_table(source_target)

    paper_main = model_seed_summary[
        model_seed_summary.get("panel", pd.Series(dtype=object)).eq("primary_9")
        & model_seed_summary.get("horizon_group", pd.Series(dtype=object)).eq("all")
        & model_seed_summary.get(
            "all_runs_is_complete_evaluation_coverage", pd.Series(dtype=bool)
        ).eq(True)
    ].copy()
    paper_columns = [
        "configuration_id",
        "base_configuration_id",
        "model_family",
        "method",
        "train_data_percent",
        "source_config_summary",
        "n_pretrain_datasets_summary",
        "n_runs",
        "n_seeds",
        "seeds",
        "main_macro_nrmse_mean",
        "main_macro_nrmse_std",
        "main_macro_nmae_mean",
        "main_macro_nmae_std",
        "secondary_pair_macro_nrmse_mean",
        "secondary_pair_macro_nrmse_std",
        "dataset_coverage_min",
        "evaluation_cell_coverage_min",
    ]
    paper_main = paper_main[[col for col in paper_columns if col in paper_main.columns]]
    if "main_macro_nrmse_mean" in paper_main.columns:
        paper_main = paper_main.sort_values(
            ["main_macro_nrmse_mean", "configuration_id"], kind="mergesort"
        ).reset_index(drop=True)

    output_map = {
        "model_run_metrics.csv": run_metrics,
        "model_seed_summary.csv": model_seed_summary,
        "dataset_metrics.csv": dataset_metrics,
        "dataset_seed_summary.csv": dataset_seed_summary,
        "dataset_attribute_metrics.csv": cell_metrics,
        "fold_metrics.csv": fold_metrics,
        "source_target_metrics.csv": source_target,
        "paper_main_table.csv": paper_main,
        "missing_evaluation_cells.csv": missing_cells,
        "file_inventory.csv": inventory,
    }

    row_counts: Dict[str, int] = {}
    for filename, frame in output_map.items():
        atomic_write_csv(frame, output_dir / filename)
        row_counts[filename] = int(len(frame))

    metric_definitions = {
        "main_macro_nrmse": (
            "For each dataset-fold-attribute-person, mean squared error after dividing "
            "by the target-training standard deviation; combine a person's repeated "
            "pairs, average people within each dataset-attribute cell, take the square "
            "root, average eligible attributes within each dataset, then average datasets."
        ),
        "secondary_pair_macro_nrmse": (
            "Normalized RMSE within each dataset-attribute cell with every error row "
            "weighted equally; dataset-attribute RMSE values are then averaged equally "
            "over attributes and datasets. People with more pairs have more influence."
        ),
        "main_macro_nmae": (
            "The person-, attribute-, and dataset-balanced mean absolute error divided "
            "by the fold-specific target-training standard deviation."
        ),
        "main_macro_absolute_normalized_bias": (
            "Absolute person-balanced signed error divided by the fold-specific target "
            "standard deviation, macro-averaged across attributes and datasets."
        ),
    }

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "command": " ".join(sys.argv),
        "paths": {
            "errors_dir": str(args.errors_dir.resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "settings": {
            "file_pattern": args.pattern,
            "chunksize": args.chunksize,
            "cache_enabled": not args.no_cache,
            "strict": args.strict,
        },
        "horizon_definition": horizon.as_dict(),
        "panels": auxiliary.panels,
        "expected_datasets": auxiliary.expected_datasets,
        "metric_definitions": metric_definitions,
        "normalization": (
            "Fold-specific standard deviation from observed values at unique future visits "
            "in the target dataset's four non-test folds, including validation people."
        ),
        "eligibility": (
            "Only dataset-fold-attribute cells marked eligible in "
            "eligible_dataset_attribute_pairs.json are evaluated."
        ),
        "coverage_rule": (
            "A run is globally paper-comparable only when all expected datasets and all "
            "eligible dataset-fold-attribute cells are present for the all-horizons result."
        ),
        "n_error_files_found": len(error_files),
        "n_error_files_succeeded": sum(1 for row in inventory_rows if row.get("status") == "ok"),
        "n_error_files_failed": sum(1 for row in inventory_rows if row.get("status") == "failed"),
        "outputs": row_counts,
    }
    atomic_write_json(manifest, output_dir / "analysis_manifest.json")
    return row_counts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir

    parser = argparse.ArgumentParser(
        description="Aggregate observed-target error CSVs into paper-ready metric tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--errors-dir", type=Path, default=project_dir / "results" / "errors")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results" / "metrics")
    parser.add_argument("--pattern", default="*.csv", help="Error-file glob pattern.")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--horizon-mode",
        choices=["reference-tertiles", "fixed"],
        default="reference-tertiles",
        help=(
            "Use tertiles from all reference input-output pairs, or fixed year boundaries."
        ),
    )
    parser.add_argument("--short-max", type=float, default=3.0)
    parser.add_argument("--medium-max", type=float, default=6.0)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Reprocess every raw error file even when an unchanged cached result exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when one error file is invalid instead of recording and skipping it.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only the first N files; useful for smoke tests.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive.")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive when supplied.")
    if not args.errors_dir.exists():
        raise FileNotFoundError(f"Errors directory not found: {args.errors_dir}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.errors_dir = args.errors_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    try:
        validate_args(args)
        ensure_output_directory(args.output_dir)
        configure_logging(args.output_dir, args.verbose)

        logging.info("Error aggregation v%s", SCRIPT_VERSION)
        logging.info("Errors directory: %s", args.errors_dir)
        logging.info("Output directory: %s", args.output_dir)

        horizon = resolve_horizon_definition(args, args.data_dir)
        logging.info(
            "Horizon bins: short <= %.6g; medium <= %.6g; long above %.6g",
            horizon.short_max,
            horizon.medium_max,
            horizon.medium_max,
        )

        auxiliary = load_auxiliary_data(args.data_dir, horizon)
        logging.info(
            "Panels: %s",
            ", ".join(
                f"{name} ({len(attributes)} targets)"
                for name, attributes in auxiliary.panels.items()
            ),
        )

        error_files = sorted(
            path for path in args.errors_dir.glob(args.pattern) if path.is_file()
        )
        if args.max_files is not None:
            error_files = error_files[: args.max_files]
        if not error_files:
            raise FileNotFoundError(
                f"No error CSV files matched {args.pattern!r} in {args.errors_dir}"
            )

        logging.info("Found %d error files.", len(error_files))
        processed: List[ProcessedRun] = []
        inventory_rows: List[Mapping[str, Any]] = []

        for index, error_path in enumerate(error_files, start=1):
            logging.info("[%d/%d] %s", index, len(error_files), error_path.name)
            try:
                result = process_error_file(
                    error_path=error_path,
                    auxiliary=auxiliary,
                    horizon=horizon,
                    chunksize=args.chunksize,
                    cache_dir=args.output_dir / "cache",
                    use_cache=not args.no_cache,
                )
                invalid_rows = sum(
                    result.inventory.get(key, 0)
                    for key in (
                        "n_dropped_missing_identity", "n_dropped_invalid_numeric",
                        "n_dropped_nonfinite", "n_dropped_negative_horizon",
                        "n_dropped_missing_lookup", "n_dropped_invalid_std",
                    )
                )
                if args.strict and invalid_rows:
                    raise ValueError(f"{invalid_rows} invalid error rows in {error_path.name}.")
                processed.append(result)
                inventory_rows.append(result.inventory)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if args.verbose:
                    logging.exception("Failed to process %s", error_path.name)
                else:
                    logging.error("Failed to process %s: %s", error_path.name, message)
                inventory_rows.append(make_failure_inventory(error_path, message))
                if args.strict:
                    raise

        if not processed:
            inventory = pd.DataFrame(inventory_rows)
            atomic_write_csv(inventory, args.output_dir / "file_inventory.csv")
            raise RuntimeError("No error file was processed successfully; see file_inventory.csv.")

        row_counts = write_outputs(
            output_dir=args.output_dir,
            processed=processed,
            inventory_rows=inventory_rows,
            auxiliary=auxiliary,
            horizon=horizon,
            args=args,
            error_files=error_files,
        )

        logging.info("Finished. Output row counts:")
        for filename, count in row_counts.items():
            logging.info("  %-38s %s", filename, f"{count:,}")

        failures = sum(1 for row in inventory_rows if row.get("status") == "failed")
        if failures:
            logging.warning(
                "%d file(s) failed. Valid files were still summarized; inspect file_inventory.csv.",
                failures,
            )
        return 0

    except Exception as exc:
        # Logging may not yet be configured when argument/path validation fails.
        if logging.getLogger().handlers:
            logging.error("Analysis failed: %s", exc)
        else:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
