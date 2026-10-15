#!/usr/bin/env python3
"""Source suitability, transfer gains, ranking validation and bootstrap intervals.

Run --rank-only without private data; the full analysis requires the paper's
complete experiment grid. Numerical calculations are retained without plotting.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

PRIMARY_PANEL = "primary_9"

COMMON_PANEL = "common_all_datasets"

EXTENDED_PANEL = "extended_20"

ALL_HORIZONS = "all"

TRAIN_PERCENTAGES = [10, 30, 50, 70, 90, 100]

TOPK_PERCENTAGES = [10, 50, 100]

SINGLE_SOURCE_PERCENTAGES = [10, 100]

DEFAULT_BOOTSTRAP_REPS = 10_000

DEFAULT_RANDOM_SEED = 20260728

DATASET_ORDER = [
    "ABCD",
    "BE_KUL_Wave_1A",
    "BE_KUL_Wave_1B",
    "BE_KUL_Wave_23456",
    "SLOfit_lifelong",
    "SLOfit_students",
    "YFS",
]

DATASET_LABELS = {
    "ABCD": "ABCD",
    "BE_KUL_Wave_1A": "LGS boys",
    "BE_KUL_Wave_1B": "LGS girls",
    "BE_KUL_Wave_23456": "Leuven adults",
    "SLOfit_lifelong": "SLOfit lifelong",
    "SLOfit_students": "SLOfit students",
    "YFS": "YFS",
}

ATTRIBUTE_LABELS = {
    "Height": "Height",
    "Weight": "Weight",
    "Waist_circumference": "Waist circumference",
    "BMI": "BMI",
    "Systolic_blood_pressure": "Systolic blood pressure",
    "Diastolic_blood_pressure": "Diastolic blood pressure",
    "Resting_heart_rate": "Resting heart rate",
    "Glucose": "Glucose",
    "Total_cholesterol": "Total cholesterol",
    "HDL": "HDL",
    "LDL": "LDL",
    "Triglycerides": "Triglycerides",
    "Smoking": "Smoking",
    "Fitness_speed": "Fitness",
    "PA_h_per_w": "Physical activity",
    "Alcohol": "Alcohol",
    "Diabetes": "Diabetes",
    "Sleep": "Sleep",
    "Diet_HHS": "Diet (HHS)",
    "Diet_T2P": "Diet (T2P)",
}

PRIMARY_MODEL_ORDER = [
    "persistence_age_sex_mean_target_only",
    "linear_regression_fixed_target_only",
    "random_forest_fixed_target_only",
    "gru_d_target_only",
    "strats_no_pretrain",
    "strats_pooled_all_sources",
    "strats_main",
]

MODEL_LABELS = {
    "persistence_train_mean_target_only": "Training-population mean",
    "persistence_age_sex_mean_target_only": "Age- and sex-specific mean",
    "persistence_locf_age_sex_mean_target_only": "LOCF with age/sex fallback",
    "persistence_train_mean_pooled_all_datasets": "Pooled training-population mean",
    "persistence_age_sex_mean_pooled_all_datasets": "Pooled age- and sex-specific mean",
    "persistence_locf_age_sex_mean_pooled_all_datasets": "Pooled LOCF with age/sex fallback",
    "linear_regression_fixed_target_only": "Linear regression",
    "linear_regression_fixed_pooled_all_datasets": "Pooled linear regression",
    "random_forest_fixed_target_only": "Random forest",
    "random_forest_fixed_pooled_all_datasets": "Pooled random forest",
    "gru_d_target_only": "GRU-D target-only",
    "strats_no_pretrain": "STraTS target-only",
    "strats_pooled_all_sources": "STraTS pooled (all sources)",
    "strats_main": "STraTS pretrained (all sources)",
    "strats_no_pretrain_maxobs_none": "STraTS target-only, no truncation",
    "strats_main_maxobs_none": "STraTS pretrained, no truncation",
}

PANEL_LABELS = {
    PRIMARY_PANEL: "Primary 9-outcome panel",
    COMMON_PANEL: "Three outcomes common to all datasets",
    EXTENDED_PANEL: "Extended 20-outcome panel",
}

SUMMARY_ROWS = {
    "Full percent",
    "Number of people",
    "Min to max age",
    "Age range (mean-std to mean+std)",
    "Average number of measurements per person (std)",
}

TRUE_VALUES = {"yes", "y", "true", "1", "present", "x", "\u2705"}

FALSE_VALUES = {"no", "n", "false", "0", "absent", "", "\u274c"}

NEURAL_MODELS = [
    "gru_d_target_only", "strats_no_pretrain",
    "strats_pooled_all_sources", "strats_main",
]

TRANSFER_MODELS = ["strats_pooled_all_sources", "strats_main"]


@dataclass(frozen=True)
class ProjectPaths:
    metrics_dir: Path
    preprocessing_dir: Path
    data_dir: Path
    src_dir: Path
    data_exports_dir: Path


@dataclass
class StudyData:
    manifest: dict[str, Any]
    model_run: pd.DataFrame
    model_seed: pd.DataFrame
    dataset_metrics: pd.DataFrame
    attribute_metrics: pd.DataFrame
    source_target: pd.DataFrame
    summary_per_dataset: pd.DataFrame
    harmonized_core: pd.DataFrame
    pairs_core: pd.DataFrame
    primary_attributes: list[str]
    extended_attributes: list[str]
    eligibility: dict[str, Any]
    origin_map: dict[str, list[str]]


@dataclass
class DerivedResults:
    dataset_characteristics: pd.DataFrame
    attribute_availability: pd.DataFrame
    horizon_distribution: pd.DataFrame
    source_suitability_rankings: pd.DataFrame
    single_source_gain: pd.DataFrame
    top6_dataset_gain: pd.DataFrame
    attribute_gain: pd.DataFrame
    horizon_gain: pd.DataFrame
    suitability_summary: pd.DataFrame
    suitability_by_target: pd.DataFrame
    model_comparison: pd.DataFrame
    panel_sensitivity: pd.DataFrame
    maxobs_ablation: pd.DataFrame


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_columns(frame: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def parse_origin_map(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ORIGIN_MAP":
                    value = ast.literal_eval(node.value)
                    return {str(k): [str(x) for x in v] for k, v in value.items()}
    raise ValueError(f"Could not find ORIGIN_MAP in {path}")


def load_study_data(paths: ProjectPaths) -> StudyData:
    logging.info("Loading compact metrics and dataset summaries")

    model_run_cols = [
        "run_id",
        "configuration_id",
        "base_configuration_id",
        "model_family",
        "seed",
        "train_data_percent",
        "panel",
        "horizon_group",
        "main_macro_nrmse",
        "main_macro_nmae",
        "n_datasets_evaluated",
        "is_complete_dataset_coverage",
        "is_complete_evaluation_coverage",
    ]
    model_seed_cols = [
        "configuration_id",
        "base_configuration_id",
        "model_family",
        "method",
        "train_data_percent",
        "source_config_summary",
        "source_dataset_summary",
        "n_pretrain_datasets_summary",
        "panel",
        "horizon_group",
        "n_runs",
        "n_seeds",
        "seeds",
        "main_macro_nrmse_mean",
        "main_macro_nrmse_std",
        "main_macro_nmae_mean",
        "main_macro_nmae_std",
        "dataset_coverage_min",
        "evaluation_cell_coverage_min",
        "all_runs_is_complete_dataset_coverage",
        "all_runs_is_complete_evaluation_coverage",
    ]
    dataset_metric_cols = [
        "run_id",
        "configuration_id",
        "base_configuration_id",
        "model_family",
        "seed",
        "train_data_percent",
        "panel",
        "dataset",
        "horizon_group",
        "main_macro_nrmse",
        "main_macro_nmae",
        "n_attributes_evaluated",
        "n_error_rows",
        "n_test_people",
        "evaluation_cell_coverage",
        "attribute_coverage",
        "source_config",
        "source_dataset",
        "n_pretrain_datasets",
    ]
    attribute_metric_cols = [
        "run_id",
        "configuration_id",
        "base_configuration_id",
        "model_family",
        "seed",
        "train_data_percent",
        "dataset",
        "attribute",
        "horizon_group",
        "person_balanced_nrmse",
        "person_balanced_nmae",
        "n_error_rows",
        "n_test_people",
        "n_eligible_folds_expected",
        "all_horizon_fold_coverage",
        "in_primary_9",
        "in_common_all_datasets",
        "source_config",
        "source_dataset",
        "n_pretrain_datasets",
    ]
    source_target_cols = dataset_metric_cols.copy()

    model_run = pd.read_csv(paths.metrics_dir / "model_run_metrics.csv", usecols=model_run_cols)
    model_seed = pd.read_csv(paths.metrics_dir / "model_seed_summary.csv", usecols=model_seed_cols)
    dataset_metrics = pd.read_csv(paths.metrics_dir / "dataset_metrics.csv", usecols=dataset_metric_cols)
    attribute_metrics = pd.read_csv(
        paths.metrics_dir / "dataset_attribute_metrics.csv", usecols=attribute_metric_cols
    )
    source_target = pd.read_csv(
        paths.metrics_dir / "source_target_metrics.csv", usecols=source_target_cols
    )
    summary_per_dataset = pd.read_csv(
        paths.preprocessing_dir / "summary_per_dataset.csv", dtype=str, keep_default_na=False
    )
    harmonized_core = pd.read_csv(
        paths.data_dir / "dataset.csv",
        usecols=["Origin", "Country", "PersonID", "Age"],
        low_memory=False,
    )
    pairs_core = pd.read_csv(
        paths.data_dir / "input_output_pairs.csv",
        usecols=["PersonID", "input_rows", "output_row"],
        low_memory=False,
    )

    return StudyData(
        manifest=read_json(paths.metrics_dir / "analysis_manifest.json"),
        model_run=model_run,
        model_seed=model_seed,
        dataset_metrics=dataset_metrics,
        attribute_metrics=attribute_metrics,
        source_target=source_target,
        summary_per_dataset=summary_per_dataset,
        harmonized_core=harmonized_core,
        pairs_core=pairs_core,
        primary_attributes=[str(x) for x in read_json(paths.data_dir / "primary_target_panel.json")],
        extended_attributes=[str(x) for x in read_json(paths.data_dir / "extended_target_panel.json")],
        eligibility=read_json(paths.data_dir / "eligible_dataset_attribute_pairs.json"),
        origin_map=parse_origin_map(paths.src_dir / "strats.py"),
    )


def validate_study_data(data: StudyData) -> list[str]:
    """Validate schemas, expected experiments, and source-ranking consistency."""
    messages: list[str] = []

    require_columns(
        data.model_seed,
        "model_seed_summary.csv",
        [
            "base_configuration_id",
            "train_data_percent",
            "panel",
            "horizon_group",
            "main_macro_nrmse_mean",
            "n_seeds",
        ],
    )
    require_columns(
        data.dataset_metrics,
        "dataset_metrics.csv",
        [
            "base_configuration_id",
            "seed",
            "train_data_percent",
            "panel",
            "dataset",
            "horizon_group",
            "main_macro_nrmse",
        ],
    )
    require_columns(
        data.attribute_metrics,
        "dataset_attribute_metrics.csv",
        [
            "base_configuration_id",
            "seed",
            "train_data_percent",
            "dataset",
            "attribute",
            "horizon_group",
            "person_balanced_nrmse",
            "in_primary_9",
        ],
    )

    expected_datasets = set(DATASET_ORDER)
    observed_datasets = set(data.manifest.get("expected_datasets", []))
    if observed_datasets != expected_datasets:
        raise ValueError(
            f"Manifest datasets differ from expected datasets. Manifest={sorted(observed_datasets)}"
        )

    if data.primary_attributes != [
        "Height",
        "Weight",
        "BMI",
        "Waist_circumference",
        "Systolic_blood_pressure",
        "Diastolic_blood_pressure",
        "Resting_heart_rate",
        "Fitness_speed",
        "PA_h_per_w",
    ]:
        raise ValueError(
            "primary_target_panel.json does not match the prespecified nine-outcome panel."
        )

    succeeded = int(data.manifest.get("n_error_files_succeeded", -1))
    failed = int(data.manifest.get("n_error_files_failed", -1))
    if failed != 0:
        raise ValueError(f"The error aggregation manifest reports {failed} failed files.")
    messages.append(f"Error files successfully aggregated: {succeeded}; failed: {failed}.")

    primary_all = data.model_seed[
        data.model_seed["panel"].eq(PRIMARY_PANEL)
        & data.model_seed["horizon_group"].eq(ALL_HORIZONS)
    ].copy()

    expected_coverage: dict[str, list[int]] = {
        "gru_d_target_only": TRAIN_PERCENTAGES,
        "linear_regression_fixed_target_only": TRAIN_PERCENTAGES,
        "persistence_age_sex_mean_target_only": TRAIN_PERCENTAGES,
        "strats_no_pretrain": TRAIN_PERCENTAGES,
        "strats_main": TRAIN_PERCENTAGES,
        "strats_pooled_all_sources": TRAIN_PERCENTAGES,
        "random_forest_fixed_target_only": [10, 100],
        "strats_dataset_pretrain_1": TOPK_PERCENTAGES,
        "strats_dataset_pretrain_2": TOPK_PERCENTAGES,
        "strats_dataset_pretrain_3": TOPK_PERCENTAGES,
        "strats_dataset_pretrain_4": TOPK_PERCENTAGES,
        "strats_dataset_pretrain_5": TOPK_PERCENTAGES,
        "strats_pooled_top_1": TOPK_PERCENTAGES,
        "strats_pooled_top_2": TOPK_PERCENTAGES,
        "strats_pooled_top_3": TOPK_PERCENTAGES,
        "strats_pooled_top_4": TOPK_PERCENTAGES,
        "strats_pooled_top_5": TOPK_PERCENTAGES,
        "strats_main_maxobs_none": [100],
        "strats_no_pretrain_maxobs_none": [100],
    }
    for base_id, expected_percentages in expected_coverage.items():
        observed = sorted(
            primary_all.loc[
                primary_all["base_configuration_id"].eq(base_id), "train_data_percent"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if observed != expected_percentages:
            raise ValueError(
                f"Unexpected experiment coverage for {base_id}: expected {expected_percentages}, "
                f"observed {observed}."
            )

    neural_configs = [
        "gru_d_target_only",
        "strats_no_pretrain",
        "strats_main",
        "strats_pooled_all_sources",
        *[f"strats_dataset_pretrain_{k}" for k in range(1, 6)],
        *[f"strats_pooled_top_{k}" for k in range(1, 6)],
    ]
    neural_rows = primary_all[primary_all["base_configuration_id"].isin(neural_configs)]
    if not neural_rows["n_seeds"].eq(3).all():
        bad = neural_rows.loc[~neural_rows["n_seeds"].eq(3), ["base_configuration_id", "n_seeds"]]
        raise ValueError(f"Expected three seeds for neural models. Problem rows:\n{bad}")

    comparable = primary_all[
        primary_all["base_configuration_id"].isin(
            [
                "gru_d_target_only",
                "strats_no_pretrain",
                "strats_main",
                "strats_pooled_all_sources",
            ]
        )
    ]
    if not comparable["all_runs_is_complete_evaluation_coverage"].fillna(False).all():
        raise ValueError("At least one primary comparison run lacks complete evaluation coverage.")

    single = data.source_target[
        data.source_target["panel"].eq(PRIMARY_PANEL)
        & data.source_target["horizon_group"].eq(ALL_HORIZONS)
        & data.source_target["source_config"].eq("single_source")
        & data.source_target["train_data_percent"].isin(SINGLE_SOURCE_PERCENTAGES)
    ].copy()
    observed_pairs = set(zip(single["dataset"], single["source_dataset"]))
    expected_pairs = {(target, source) for target in DATASET_ORDER for source in DATASET_ORDER if source != target}
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(f"Single-source coverage mismatch. Missing={missing}; extra={extra}")
    pair_seed_counts = single.groupby(
        ["train_data_percent", "dataset", "source_dataset"], dropna=False
    )["seed"].nunique()
    if not pair_seed_counts.eq(3).all():
        raise ValueError("Every single-source target/source/percentage cell must contain three seeds.")

    summary_datasets = list(data.summary_per_dataset.columns[1:])
    if summary_datasets != DATASET_ORDER:
        raise ValueError(
            f"Unexpected dataset order in summary_per_dataset.csv: {summary_datasets}"
        )

    messages.append(
        "Primary panel coverage is complete for the principal global comparisons; "
        "single-source experiments contain all 42 directed source-target pairs at 10% and 100%."
    )
    return messages


def present_to_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Feature availability value not recognized: {value!r}")


def parse_range(value: Any) -> tuple[float, float]:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*to\s*(-?\d+(?:\.\d+)?)", str(value))
    if not match:
        raise ValueError(f"Could not parse range: {value!r}")
    first, second = float(match.group(1)), float(match.group(2))
    return min(first, second), max(first, second)


def parse_percent(value: Any) -> float:
    text = str(value).strip()
    number = float(text.replace("%", ""))
    return number / 100.0 if "%" in text or number > 1.0 else number


def parse_first_number(value: Any) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        raise ValueError(f"Could not parse number: {value!r}")
    return float(match.group(0))


def minmax_normalize(values: Iterable[float]) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    low, high = float(series.min()), float(series.max())
    if math.isclose(low, high):
        return pd.Series(1.0, index=series.index, dtype="float64")
    return (series - low) / (high - low)


def overlap_fraction(source_range: tuple[float, float], target_range: tuple[float, float]) -> float:
    source_low, source_high = source_range
    target_low, target_high = target_range
    target_width = target_high - target_low
    if target_width <= 0:
        return 0.0
    overlap = max(0.0, min(source_high, target_high) - max(source_low, target_low))
    return min(1.0, overlap / target_width)


def compute_source_suitability_rankings(summary_per_dataset: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the pre-specified source-suitability score exactly."""
    frame = summary_per_dataset.copy()
    first_col = frame.columns[0]
    frame[first_col] = frame[first_col].astype(str).str.strip()
    for col in frame.columns[1:]:
        frame[col] = frame[col].astype(str).str.strip()

    datasets = list(frame.columns[1:])
    summary = frame.set_index(first_col)
    feature_rows = frame[~frame[first_col].isin(SUMMARY_ROWS)].copy()
    features = feature_rows[first_col].tolist()

    feature_matrix = pd.DataFrame(index=features, columns=datasets, dtype=bool)
    for dataset in datasets:
        feature_matrix[dataset] = feature_rows[dataset].map(present_to_bool).to_numpy()

    full_percent = pd.Series(
        {dataset: parse_percent(summary.loc["Full percent", dataset]) for dataset in datasets}
    )
    n_people = pd.Series(
        {dataset: parse_first_number(summary.loc["Number of people", dataset]) for dataset in datasets}
    )
    avg_visits = pd.Series(
        {
            dataset: parse_first_number(
                summary.loc["Average number of measurements per person (std)", dataset]
            )
            for dataset in datasets
        }
    )
    full_age = {
        dataset: parse_range(summary.loc["Min to max age", dataset]) for dataset in datasets
    }
    core_age = {
        dataset: parse_range(
            summary.loc["Age range (mean-std to mean+std)", dataset]
        )
        for dataset in datasets
    }

    source_quality = (
        0.60 * full_percent
        + 0.25 * minmax_normalize(np.log(n_people)).set_axis(datasets)
        + 0.15 * minmax_normalize(np.log(avg_visits)).set_axis(datasets)
    )

    rows: list[dict[str, Any]] = []
    for target in datasets:
        target_features = feature_matrix[target]
        target_feature_count = int(target_features.sum())
        for source in datasets:
            if source == target:
                continue
            shared = int((feature_matrix[source] & target_features).sum())
            feature_coverage = shared / target_feature_count if target_feature_count else 0.0
            core_overlap = overlap_fraction(core_age[source], core_age[target])
            full_overlap = overlap_fraction(full_age[source], full_age[target])
            age_coverage = 0.7 * core_overlap + 0.3 * full_overlap
            quality = float(source_quality[source])
            score = (
                100.0
                * (feature_coverage ** 0.6)
                * (age_coverage ** 0.4)
                * (0.7 + 0.3 * quality)
            )
            rows.append(
                {
                    "target_dataset": target,
                    "source_dataset": source,
                    "source_suitability_score": score,
                    "F_feature_coverage": feature_coverage,
                    "G_age_coverage": age_coverage,
                    "Q_source_quality": quality,
                    "shared_attributes": shared,
                    "target_attributes": target_feature_count,
                    "core_age_overlap": core_overlap,
                    "full_age_overlap": full_overlap,
                }
            )

    rankings = pd.DataFrame(rows).sort_values(
        [
            "target_dataset",
            "source_suitability_score",
            "F_feature_coverage",
            "G_age_coverage",
            "Q_source_quality",
        ],
        ascending=[True, False, False, False, False],
        kind="mergesort",
    )
    rankings["source_suitability_rank"] = rankings.groupby("target_dataset", sort=False).cumcount() + 1
    return rankings[
        [
            "target_dataset",
            "source_suitability_rank",
            "source_dataset",
            "source_suitability_score",
            "F_feature_coverage",
            "G_age_coverage",
            "Q_source_quality",
            "shared_attributes",
            "target_attributes",
            "core_age_overlap",
            "full_age_overlap",
        ]
    ].reset_index(drop=True)


def validate_ranking_against_origin_map(
    rankings: pd.DataFrame, origin_map: Mapping[str, Sequence[str]]
) -> None:
    for target in DATASET_ORDER:
        observed = (
            rankings.loc[rankings["target_dataset"].eq(target)]
            .sort_values("source_suitability_rank")["source_dataset"]
            .tolist()
        )
        expected = list(origin_map.get(target, []))
        if observed != expected:
            raise ValueError(
                f"The source-suitability ranking for {target} differs from ORIGIN_MAP. "
                f"Computed={observed}; ORIGIN_MAP={expected}."
            )


def eligible_primary_count(
    eligibility: Mapping[str, Any], dataset: str, primary_attributes: Sequence[str]
) -> int:
    folds = eligibility[dataset]
    count = 0
    for attribute in primary_attributes:
        if any(bool(fold_data[attribute]["eligible"]) for fold_data in folds.values()):
            count += 1
    return count


def compute_dataset_characteristics(
    data: StudyData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Table 1 inputs and forecast-horizon summaries from the actual pair file."""
    summary_frame = data.summary_per_dataset.copy()
    first_col = summary_frame.columns[0]
    summary_frame[first_col] = summary_frame[first_col].astype(str).str.strip()
    for col in summary_frame.columns[1:]:
        summary_frame[col] = summary_frame[col].astype(str).str.strip()
    summary = summary_frame.set_index(first_col)
    feature_rows = summary_frame[~summary_frame[first_col].isin(SUMMARY_ROWS)].copy()

    availability = pd.DataFrame(index=feature_rows[first_col].tolist(), columns=DATASET_ORDER)
    for dataset in DATASET_ORDER:
        availability[dataset] = feature_rows[dataset].map(present_to_bool).to_numpy()
    availability.index.name = "attribute"

    harmonized = data.harmonized_core.copy()
    harmonized["PersonID"] = harmonized["PersonID"].astype(str)
    harmonized["Age"] = pd.to_numeric(harmonized["Age"], errors="coerce")
    if harmonized["Age"].isna().any():
        raise ValueError("dataset.csv contains invalid Age values.")

    visits_per_person = (
        harmonized.groupby(["Origin", "PersonID"], sort=False).size().rename("n_visits").reset_index()
    )
    basic = (
        harmonized.groupby("Origin", sort=False)
        .agg(
            country=("Country", lambda x: " / ".join(sorted(set(map(str, x.dropna()))))),
            n_people=("PersonID", "nunique"),
            n_visits=("PersonID", "size"),
            min_age=("Age", "min"),
            max_age=("Age", "max"),
        )
        .reindex(DATASET_ORDER)
    )
    visit_stats = (
        visits_per_person.groupby("Origin")["n_visits"]
        .agg(mean_visits="mean", sd_visits="std")
        .reindex(DATASET_ORDER)
    )
    basic = basic.join(visit_stats)
    basic["sd_visits"] = basic["sd_visits"].fillna(0.0)

    pairs = data.pairs_core.copy()
    pairs["PersonID"] = pairs["PersonID"].astype(str)
    pairs["output_row"] = pd.to_numeric(pairs["output_row"], errors="coerce")
    if pairs["output_row"].isna().any():
        raise ValueError("input_output_pairs.csv contains nonnumeric output_row values.")
    pairs["output_row"] = pairs["output_row"].astype(np.int64)
    last_input = pairs["input_rows"].astype(str).str.extract(r"(\d+)(?!.*\d)", expand=False)
    if last_input.isna().any():
        example = pairs.loc[last_input.isna(), "input_rows"].head().tolist()
        raise ValueError(f"Could not parse the final input row from input_rows. Examples={example}")
    pairs["last_input_row"] = last_input.astype(np.int64)

    n_rows = len(harmonized)
    for index_col in ["output_row", "last_input_row"]:
        if pairs[index_col].lt(0).any() or pairs[index_col].ge(n_rows).any():
            raise IndexError(f"{index_col} contains row indices outside the harmonized dataset.")

    age_array = harmonized["Age"].to_numpy(dtype=float)
    origin_array = harmonized["Origin"].astype(str).to_numpy()
    person_array = harmonized["PersonID"].astype(str).to_numpy()
    output_index = pairs["output_row"].to_numpy(dtype=np.int64)
    input_index = pairs["last_input_row"].to_numpy(dtype=np.int64)
    pair_origin = origin_array[output_index]
    pair_person = person_array[output_index]
    input_origin = origin_array[input_index]
    horizon = age_array[output_index] - age_array[input_index]

    if np.any(pair_origin != input_origin):
        raise ValueError("At least one input-output pair crosses datasets.")
    if np.any(pair_person != pairs["PersonID"].to_numpy(dtype=str)):
        raise ValueError("At least one pair PersonID does not match its output row.")
    if not np.isfinite(horizon).all() or np.any(horizon <= 0):
        raise ValueError("Forecast horizons must be finite and strictly positive.")

    horizon_frame = pd.DataFrame(
        {
            "dataset": pair_origin,
            "horizon_years": horizon,
        }
    )
    horizon_definition = data.manifest.get("horizon_definition", {})
    short_max = float(horizon_definition.get("short_max"))
    medium_max = float(horizon_definition.get("medium_max"))
    horizon_frame["horizon_group"] = pd.cut(
        horizon_frame["horizon_years"],
        bins=[-np.inf, short_max, medium_max, np.inf],
        labels=["short", "medium", "long"],
        right=True,
    )

    horizon_summary = (
        horizon_frame.groupby("dataset", observed=False)["horizon_years"]
        .agg(
            n_pairs="size",
            horizon_min="min",
            horizon_q1=lambda x: x.quantile(0.25),
            horizon_median="median",
            horizon_q3=lambda x: x.quantile(0.75),
            horizon_max="max",
        )
        .reindex(DATASET_ORDER)
    )
    horizon_counts = (
        horizon_frame.groupby(["dataset", "horizon_group"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=DATASET_ORDER, columns=["short", "medium", "long"], fill_value=0)
    )
    horizon_counts.columns = [f"n_{col}" for col in horizon_counts.columns]
    horizon_summary = horizon_summary.join(horizon_counts)
    for group in ["short", "medium", "long"]:
        horizon_summary[f"pct_{group}"] = (
            100.0 * horizon_summary[f"n_{group}"] / horizon_summary["n_pairs"]
        )

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        min_age_summary, max_age_summary = parse_range(summary.loc["Min to max age", dataset])
        core_low, core_high = parse_range(
            summary.loc["Age range (mean-std to mean+std)", dataset]
        )
        reported_people = int(parse_first_number(summary.loc["Number of people", dataset]))
        observed_people = int(basic.loc[dataset, "n_people"])
        if reported_people != observed_people:
            raise ValueError(
                f"Person count mismatch for {dataset}: summary={reported_people}, data={observed_people}."
            )
        if not (
            math.isclose(min_age_summary, float(basic.loc[dataset, "min_age"]), abs_tol=0.15)
            and math.isclose(max_age_summary, float(basic.loc[dataset, "max_age"]), abs_tol=0.15)
        ):
            logging.warning("Age range in summary differs slightly from raw data for %s", dataset)

        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "country": basic.loc[dataset, "country"],
                "n_people": observed_people,
                "n_visits": int(basic.loc[dataset, "n_visits"]),
                "mean_visits": float(basic.loc[dataset, "mean_visits"]),
                "sd_visits": float(basic.loc[dataset, "sd_visits"]),
                "min_age": float(basic.loc[dataset, "min_age"]),
                "max_age": float(basic.loc[dataset, "max_age"]),
                "core_age_low": core_low,
                "core_age_high": core_high,
                "n_available_attributes": int(availability[dataset].sum()),
                "n_eligible_primary_attributes": eligible_primary_count(
                    data.eligibility, dataset, data.primary_attributes
                ),
                "completeness_available_percent": 100.0
                * parse_percent(summary.loc["Full percent", dataset]),
                "n_pairs": int(horizon_summary.loc[dataset, "n_pairs"]),
                "horizon_q1": float(horizon_summary.loc[dataset, "horizon_q1"]),
                "horizon_median": float(horizon_summary.loc[dataset, "horizon_median"]),
                "horizon_q3": float(horizon_summary.loc[dataset, "horizon_q3"]),
                "horizon_min": float(horizon_summary.loc[dataset, "horizon_min"]),
                "horizon_max": float(horizon_summary.loc[dataset, "horizon_max"]),
            }
        )

    characteristics = pd.DataFrame(rows)
    horizon_export = horizon_summary.reset_index().rename(columns={"dataset": "dataset"})
    horizon_export.insert(
        1, "dataset_label", horizon_export["dataset"].map(DATASET_LABELS)
    )
    availability_export = availability.reset_index()
    availability_export["attribute_label"] = availability_export["attribute"].map(ATTRIBUTE_LABELS)
    availability_export = availability_export[
        ["attribute", "attribute_label", *DATASET_ORDER]
    ]
    return characteristics, availability_export, horizon_export


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid]
    y_array = y_array[valid]
    if len(x_array) < 2 or np.allclose(x_array, x_array[0]) or np.allclose(y_array, y_array[0]):
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_rank = pd.Series(x, dtype="float64").rank(method="average").to_numpy()
    y_rank = pd.Series(y, dtype="float64").rank(method="average").to_numpy()
    return pearson_correlation(x_rank, y_rank)


def percentile_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(array, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_mean_interval(
    values: Sequence[float], reps: int, rng: np.random.Generator
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(array), size=(reps, len(array)))
    means = array[indices].mean(axis=1)
    return percentile_interval(means)


def cluster_bootstrap_relative_gain(
    paired: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
    dataset_col: str = "dataset",
    baseline_col: str = "baseline_nrmse",
    transfer_col: str = "transfer_nrmse",
) -> tuple[float, float, float]:
    """Bootstrap target datasets after averaging seeds within each target dataset."""
    by_dataset = (
        paired.groupby(dataset_col, as_index=False)
        .agg(
            baseline_nrmse=(baseline_col, "mean"),
            transfer_nrmse=(transfer_col, "mean"),
        )
        .dropna()
    )
    if by_dataset.empty:
        return float("nan"), float("nan"), float("nan")
    baseline = by_dataset["baseline_nrmse"].to_numpy(dtype=float)
    transfer = by_dataset["transfer_nrmse"].to_numpy(dtype=float)
    point = 100.0 * (baseline.mean() - transfer.mean()) / baseline.mean()
    if len(by_dataset) < 2:
        return float(point), float("nan"), float("nan")
    indices = rng.integers(0, len(by_dataset), size=(reps, len(by_dataset)))
    baseline_boot = baseline[indices].mean(axis=1)
    transfer_boot = transfer[indices].mean(axis=1)
    gain_boot = 100.0 * (baseline_boot - transfer_boot) / baseline_boot
    low, high = percentile_interval(gain_boot)
    return float(point), low, high


def pair_dataset_transfer(
    dataset_metrics: pd.DataFrame,
    transfer_base_id: str,
    percentages: Sequence[int] | None = None,
    panel: str = PRIMARY_PANEL,
    horizon_group: str = ALL_HORIZONS,
) -> pd.DataFrame:
    base = dataset_metrics[
        dataset_metrics["base_configuration_id"].eq("strats_no_pretrain")
        & dataset_metrics["panel"].eq(panel)
        & dataset_metrics["horizon_group"].eq(horizon_group)
    ][["dataset", "seed", "train_data_percent", "main_macro_nrmse"]].rename(
        columns={"main_macro_nrmse": "baseline_nrmse"}
    )
    transfer = dataset_metrics[
        dataset_metrics["base_configuration_id"].eq(transfer_base_id)
        & dataset_metrics["panel"].eq(panel)
        & dataset_metrics["horizon_group"].eq(horizon_group)
    ][["dataset", "seed", "train_data_percent", "main_macro_nrmse"]].rename(
        columns={"main_macro_nrmse": "transfer_nrmse"}
    )
    if percentages is not None:
        base = base[base["train_data_percent"].isin(percentages)]
        transfer = transfer[transfer["train_data_percent"].isin(percentages)]
    merged = transfer.merge(
        base,
        on=["dataset", "seed", "train_data_percent"],
        how="inner",
        validate="one_to_one",
    )
    merged["relative_gain_percent"] = 100.0 * (
        merged["baseline_nrmse"] - merged["transfer_nrmse"]
    ) / merged["baseline_nrmse"]
    return merged


def pair_global_transfer(
    model_run: pd.DataFrame, transfer_base_id: str, percentages: Sequence[int] | None = None
) -> pd.DataFrame:
    base = model_run[
        model_run["base_configuration_id"].eq("strats_no_pretrain")
        & model_run["panel"].eq(PRIMARY_PANEL)
        & model_run["horizon_group"].eq(ALL_HORIZONS)
    ][["seed", "train_data_percent", "main_macro_nrmse"]].rename(
        columns={"main_macro_nrmse": "baseline_nrmse"}
    )
    transfer = model_run[
        model_run["base_configuration_id"].eq(transfer_base_id)
        & model_run["panel"].eq(PRIMARY_PANEL)
        & model_run["horizon_group"].eq(ALL_HORIZONS)
    ][["seed", "train_data_percent", "main_macro_nrmse"]].rename(
        columns={"main_macro_nrmse": "transfer_nrmse"}
    )
    if percentages is not None:
        base = base[base["train_data_percent"].isin(percentages)]
        transfer = transfer[transfer["train_data_percent"].isin(percentages)]
    merged = transfer.merge(
        base,
        on=["seed", "train_data_percent"],
        how="inner",
        validate="one_to_one",
    )
    merged["relative_gain_percent"] = 100.0 * (
        merged["baseline_nrmse"] - merged["transfer_nrmse"]
    ) / merged["baseline_nrmse"]
    return merged


def compute_single_source_gain(data: StudyData) -> pd.DataFrame:
    baseline = data.dataset_metrics[
        data.dataset_metrics["base_configuration_id"].eq("strats_no_pretrain")
        & data.dataset_metrics["panel"].eq(PRIMARY_PANEL)
        & data.dataset_metrics["horizon_group"].eq(ALL_HORIZONS)
        & data.dataset_metrics["train_data_percent"].isin(SINGLE_SOURCE_PERCENTAGES)
    ][["dataset", "seed", "train_data_percent", "main_macro_nrmse"]].rename(
        columns={"main_macro_nrmse": "baseline_nrmse"}
    )
    single = data.source_target[
        data.source_target["panel"].eq(PRIMARY_PANEL)
        & data.source_target["horizon_group"].eq(ALL_HORIZONS)
        & data.source_target["source_config"].eq("single_source")
        & data.source_target["train_data_percent"].isin(SINGLE_SOURCE_PERCENTAGES)
    ][
        [
            "dataset",
            "source_dataset",
            "seed",
            "train_data_percent",
            "main_macro_nrmse",
        ]
    ].rename(columns={"main_macro_nrmse": "transfer_nrmse"})
    merged = single.merge(
        baseline,
        on=["dataset", "seed", "train_data_percent"],
        how="left",
        validate="many_to_one",
    )
    if merged["baseline_nrmse"].isna().any():
        raise ValueError("Missing target-only STraTS baselines for single-source comparisons.")
    merged["relative_gain_percent"] = 100.0 * (
        merged["baseline_nrmse"] - merged["transfer_nrmse"]
    ) / merged["baseline_nrmse"]
    aggregated = (
        merged.groupby(
            ["train_data_percent", "dataset", "source_dataset"], as_index=False
        )
        .agg(
            baseline_nrmse_mean=("baseline_nrmse", "mean"),
            transfer_nrmse_mean=("transfer_nrmse", "mean"),
            relative_gain_percent_mean=("relative_gain_percent", "mean"),
            relative_gain_percent_sd=("relative_gain_percent", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    aggregated["target_label"] = aggregated["dataset"].map(DATASET_LABELS)
    aggregated["source_label"] = aggregated["source_dataset"].map(DATASET_LABELS)
    return aggregated


def compute_top6_dataset_gain(data: StudyData) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for base_id, strategy in [
        ("strats_main", "Pretraining"),
        ("strats_pooled_all_sources", "Pooled training"),
    ]:
        paired = pair_dataset_transfer(data.dataset_metrics, base_id)
        aggregated = (
            paired.groupby(["train_data_percent", "dataset"], as_index=False)
            .agg(
                baseline_nrmse_mean=("baseline_nrmse", "mean"),
                transfer_nrmse_mean=("transfer_nrmse", "mean"),
                relative_gain_percent_mean=("relative_gain_percent", "mean"),
                relative_gain_percent_sd=("relative_gain_percent", "std"),
                n_seeds=("seed", "nunique"),
            )
        )
        aggregated.insert(0, "strategy", strategy)
        aggregated.insert(0, "base_configuration_id", base_id)
        frames.append(aggregated)
    result = pd.concat(frames, ignore_index=True)
    result["dataset_label"] = result["dataset"].map(DATASET_LABELS)
    return result


def compute_attribute_gain(
    data: StudyData, reps: int, rng: np.random.Generator
) -> pd.DataFrame:
    metrics = data.attribute_metrics
    base = metrics[
        metrics["base_configuration_id"].eq("strats_no_pretrain")
        & metrics["horizon_group"].eq(ALL_HORIZONS)
        & metrics["in_primary_9"].fillna(False)
        & metrics["train_data_percent"].isin(SINGLE_SOURCE_PERCENTAGES)
    ][
        [
            "dataset",
            "attribute",
            "seed",
            "train_data_percent",
            "person_balanced_nrmse",
        ]
    ].rename(columns={"person_balanced_nrmse": "baseline_nrmse"})
    transfer = metrics[
        metrics["base_configuration_id"].eq("strats_main")
        & metrics["horizon_group"].eq(ALL_HORIZONS)
        & metrics["in_primary_9"].fillna(False)
        & metrics["train_data_percent"].isin(SINGLE_SOURCE_PERCENTAGES)
    ][
        [
            "dataset",
            "attribute",
            "seed",
            "train_data_percent",
            "person_balanced_nrmse",
        ]
    ].rename(columns={"person_balanced_nrmse": "transfer_nrmse"})
    merged = transfer.merge(
        base,
        on=["dataset", "attribute", "seed", "train_data_percent"],
        how="inner",
        validate="one_to_one",
    )
    merged["relative_gain_percent"] = 100.0 * (
        merged["baseline_nrmse"] - merged["transfer_nrmse"]
    ) / merged["baseline_nrmse"]
    dataset_means = (
        merged.groupby(["train_data_percent", "attribute", "dataset"], as_index=False)
        .agg(relative_gain_percent=("relative_gain_percent", "mean"))
    )

    rows: list[dict[str, Any]] = []
    for (percent, attribute), group in dataset_means.groupby(
        ["train_data_percent", "attribute"], sort=False
    ):
        values = group["relative_gain_percent"].to_numpy(dtype=float)
        low, high = bootstrap_mean_interval(values, reps, rng)
        rows.append(
            {
                "train_data_percent": int(percent),
                "attribute": attribute,
                "attribute_label": ATTRIBUTE_LABELS.get(attribute, attribute),
                "relative_gain_percent_mean": float(np.mean(values)),
                "relative_gain_percent_median": float(np.median(values)),
                "ci95_low": low,
                "ci95_high": high,
                "n_datasets": int(group["dataset"].nunique()),
            }
        )
    order = {attribute: index for index, attribute in enumerate(data.primary_attributes)}
    result = pd.DataFrame(rows)
    result["attribute_order"] = result["attribute"].map(order)
    return result.sort_values(["attribute_order", "train_data_percent"]).reset_index(drop=True)


def compute_horizon_gain(
    data: StudyData, reps: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in ["short", "medium", "long"]:
        paired = pair_dataset_transfer(
            data.dataset_metrics,
            "strats_main",
            percentages=SINGLE_SOURCE_PERCENTAGES,
            panel=PRIMARY_PANEL,
            horizon_group=horizon,
        )
        dataset_means = (
            paired.groupby(["train_data_percent", "dataset"], as_index=False)
            .agg(relative_gain_percent=("relative_gain_percent", "mean"))
        )
        for percent, group in dataset_means.groupby("train_data_percent"):
            values = group["relative_gain_percent"].to_numpy(dtype=float)
            low, high = bootstrap_mean_interval(values, reps, rng)
            rows.append(
                {
                    "horizon_group": horizon,
                    "train_data_percent": int(percent),
                    "relative_gain_percent_mean": float(np.mean(values)),
                    "relative_gain_percent_median": float(np.median(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_datasets": int(group["dataset"].nunique()),
                }
            )
    result = pd.DataFrame(rows)
    horizon_order = {"short": 0, "medium": 1, "long": 2}
    result["horizon_order"] = result["horizon_group"].map(horizon_order)
    return result.sort_values(["horizon_order", "train_data_percent"]).reset_index(drop=True)


def compute_suitability_validation(
    rankings: pd.DataFrame,
    single_source_gain: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = rankings.merge(
        single_source_gain[
            [
                "train_data_percent",
                "dataset",
                "source_dataset",
                "relative_gain_percent_mean",
                "relative_gain_percent_sd",
            ]
        ],
        left_on=["target_dataset", "source_dataset"],
        right_on=["dataset", "source_dataset"],
        how="left",
        validate="one_to_many",
    )
    merged = merged.drop(columns=["dataset"])
    if merged["relative_gain_percent_mean"].isna().any():
        missing = merged.loc[
            merged["relative_gain_percent_mean"].isna(),
            ["target_dataset", "source_dataset"],
        ].drop_duplicates()
        raise ValueError(f"Missing observed gains for suitability validation:\n{missing}")

    per_target_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for percent in SINGLE_SOURCE_PERCENTAGES:
        block = merged[merged["train_data_percent"].eq(percent)].copy()
        for target, group in block.groupby("target_dataset", sort=False):
            group = group.sort_values("source_suitability_rank")
            rho = spearman_correlation(
                group["source_suitability_score"].to_numpy(), group["relative_gain_percent_mean"].to_numpy()
            )
            top_heuristic = group.loc[group["source_suitability_rank"].idxmin(), "source_dataset"]
            best_index = group["relative_gain_percent_mean"].idxmax()
            best_source = group.loc[best_index, "source_dataset"]
            best_gain = float(group.loc[best_index, "relative_gain_percent_mean"])
            top_gain = float(
                group.loc[group["source_dataset"].eq(top_heuristic), "relative_gain_percent_mean"].iloc[0]
            )
            per_target_rows.append(
                {
                    "train_data_percent": percent,
                    "target_dataset": target,
                    "target_label": DATASET_LABELS[target],
                    "spearman_rho": rho,
                    "heuristic_top_source": top_heuristic,
                    "heuristic_top_source_label": DATASET_LABELS[top_heuristic],
                    "empirical_best_source": best_source,
                    "empirical_best_source_label": DATASET_LABELS[best_source],
                    "heuristic_top_gain_percent": top_gain,
                    "empirical_best_gain_percent": best_gain,
                    "regret_percentage_points": best_gain - top_gain,
                    "top1_hit": bool(top_heuristic == best_source),
                }
            )

        target_block = pd.DataFrame(per_target_rows)
        target_block = target_block[target_block["train_data_percent"].eq(percent)]
        rhos = target_block["spearman_rho"].to_numpy(dtype=float)
        rho_low, rho_high = bootstrap_mean_interval(rhos, reps, rng)
        pooled_rank_rho = spearman_correlation(
            -block["source_suitability_rank"].to_numpy(dtype=float),
            block["relative_gain_percent_mean"].to_numpy(dtype=float),
        )
        summary_rows.append(
            {
                "train_data_percent": percent,
                "mean_target_spearman_rho": float(np.nanmean(rhos)),
                "median_target_spearman_rho": float(np.nanmedian(rhos)),
                "mean_target_spearman_ci95_low": rho_low,
                "mean_target_spearman_ci95_high": rho_high,
                "pooled_rank_spearman_rho": pooled_rank_rho,
                "top1_hits": int(target_block["top1_hit"].sum()),
                "n_targets": int(len(target_block)),
                "mean_regret_percentage_points": float(
                    target_block["regret_percentage_points"].mean()
                ),
                "median_regret_percentage_points": float(
                    target_block["regret_percentage_points"].median()
                ),
            }
        )

    per_target = pd.DataFrame(per_target_rows).sort_values(
        ["train_data_percent", "target_dataset"]
    )
    summary = pd.DataFrame(summary_rows).sort_values("train_data_percent")
    merged["target_label"] = merged["target_dataset"].map(DATASET_LABELS)
    merged["source_label"] = merged["source_dataset"].map(DATASET_LABELS)
    return merged, summary, per_target


def get_model_seed_row(
    model_seed: pd.DataFrame,
    base_configuration_id: str,
    percent: int,
    panel: str = PRIMARY_PANEL,
    horizon_group: str = ALL_HORIZONS,
) -> pd.Series:
    rows = model_seed[
        model_seed["base_configuration_id"].eq(base_configuration_id)
        & model_seed["train_data_percent"].eq(percent)
        & model_seed["panel"].eq(panel)
        & model_seed["horizon_group"].eq(horizon_group)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one model-seed row for {base_configuration_id}, percent={percent}, "
            f"panel={panel}, horizon={horizon_group}; found {len(rows)}."
        )
    return rows.iloc[0]


def compute_model_comparison(
    data: StudyData, reps: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    transfer_models = {"strats_pooled_all_sources", "strats_main"}
    for base_id in PRIMARY_MODEL_ORDER:
        row: dict[str, Any] = {
            "base_configuration_id": base_id,
            "model_label": MODEL_LABELS[base_id],
            "transfer_strategy": (
                "Pooled training"
                if base_id == "strats_pooled_all_sources"
                else "Pretraining"
                if base_id == "strats_main"
                else "None"
            ),
        }
        for percent in SINGLE_SOURCE_PERCENTAGES:
            model_row = get_model_seed_row(data.model_seed, base_id, percent)
            row[f"nrmse_{percent}_mean"] = float(model_row["main_macro_nrmse_mean"])
            row[f"nrmse_{percent}_sd"] = (
                float(model_row["main_macro_nrmse_std"])
                if pd.notna(model_row["main_macro_nrmse_std"])
                else float("nan")
            )
            row[f"n_seeds_{percent}"] = int(model_row["n_seeds"])
            if base_id in transfer_models:
                paired = pair_dataset_transfer(
                    data.dataset_metrics,
                    base_id,
                    percentages=[percent],
                    panel=PRIMARY_PANEL,
                    horizon_group=ALL_HORIZONS,
                )
                point, low, high = cluster_bootstrap_relative_gain(paired, reps, rng)
                row[f"gain_{percent}_percent"] = point
                row[f"gain_{percent}_ci95_low"] = low
                row[f"gain_{percent}_ci95_high"] = high
            else:
                row[f"gain_{percent}_percent"] = float("nan")
                row[f"gain_{percent}_ci95_low"] = float("nan")
                row[f"gain_{percent}_ci95_high"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_panel_sensitivity(data: StudyData) -> pd.DataFrame:
    key_models = [
        "gru_d_target_only",
        "strats_no_pretrain",
        "strats_pooled_all_sources",
        "strats_main",
    ]
    rows: list[dict[str, Any]] = []
    for panel in [PRIMARY_PANEL, COMMON_PANEL, EXTENDED_PANEL]:
        for percent in SINGLE_SOURCE_PERCENTAGES:
            baseline_row = get_model_seed_row(
                data.model_seed, "strats_no_pretrain", percent, panel=panel
            )
            baseline_value = float(baseline_row["main_macro_nrmse_mean"])
            for base_id in key_models:
                model_row = get_model_seed_row(data.model_seed, base_id, percent, panel=panel)
                value = float(model_row["main_macro_nrmse_mean"])
                rows.append(
                    {
                        "panel": panel,
                        "panel_label": PANEL_LABELS[panel],
                        "train_data_percent": percent,
                        "base_configuration_id": base_id,
                        "model_label": MODEL_LABELS[base_id],
                        "main_macro_nrmse_mean": value,
                        "main_macro_nrmse_sd": (
                            float(model_row["main_macro_nrmse_std"])
                            if pd.notna(model_row["main_macro_nrmse_std"])
                            else float("nan")
                        ),
                        "relative_gain_vs_target_only_strats_percent": (
                            100.0 * (baseline_value - value) / baseline_value
                            if base_id in {"strats_pooled_all_sources", "strats_main"}
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compute_maxobs_ablation(data: StudyData) -> pd.DataFrame:
    comparisons = [
        ("Target-only STraTS", "strats_no_pretrain", "strats_no_pretrain_maxobs_none"),
        ("All-source pretrained STraTS", "strats_main", "strats_main_maxobs_none"),
    ]
    rows: list[dict[str, Any]] = []
    for label, default_id, none_id in comparisons:
        default_row = get_model_seed_row(data.model_seed, default_id, 100)
        none_row = get_model_seed_row(data.model_seed, none_id, 100)
        default_mean = float(default_row["main_macro_nrmse_mean"])
        none_mean = float(none_row["main_macro_nrmse_mean"])
        rows.append(
            {
                "model": label,
                "default_base_configuration_id": default_id,
                "no_truncation_base_configuration_id": none_id,
                "max_obs_256_nrmse_mean": default_mean,
                "max_obs_256_nrmse_sd": float(default_row["main_macro_nrmse_std"]),
                "max_obs_none_nrmse_mean": none_mean,
                "max_obs_none_nrmse_sd": float(none_row["main_macro_nrmse_std"]),
                "nrmse_difference_none_minus_256": none_mean - default_mean,
                "relative_difference_percent": 100.0 * (none_mean - default_mean) / default_mean,
            }
        )
    return pd.DataFrame(rows)


def derive_results(
    data: StudyData, bootstrap_reps: int, random_seed: int
) -> tuple[DerivedResults, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    characteristics, availability, horizon_distribution = compute_dataset_characteristics(data)
    rankings = compute_source_suitability_rankings(data.summary_per_dataset)
    validate_ranking_against_origin_map(rankings, data.origin_map)
    single_source_gain = compute_single_source_gain(data)
    suitability_plot_data, suitability_summary, suitability_by_target = compute_suitability_validation(
        rankings, single_source_gain, bootstrap_reps, rng
    )
    top6_dataset_gain = compute_top6_dataset_gain(data)
    attribute_gain = compute_attribute_gain(data, bootstrap_reps, rng)
    horizon_gain = compute_horizon_gain(data, bootstrap_reps, rng)
    model_comparison = compute_model_comparison(data, bootstrap_reps, rng)
    panel_sensitivity = compute_panel_sensitivity(data)
    maxobs_ablation = compute_maxobs_ablation(data)

    derived = DerivedResults(
        dataset_characteristics=characteristics,
        attribute_availability=availability,
        horizon_distribution=horizon_distribution,
        source_suitability_rankings=rankings,
        single_source_gain=single_source_gain,
        top6_dataset_gain=top6_dataset_gain,
        attribute_gain=attribute_gain,
        horizon_gain=horizon_gain,
        suitability_summary=suitability_summary,
        suitability_by_target=suitability_by_target,
        model_comparison=model_comparison,
        panel_sensitivity=panel_sensitivity,
        maxobs_ablation=maxobs_ablation,
    )
    return derived, suitability_plot_data


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def topk_base_id(strategy: str, k: int) -> str:
    if k == 0:
        return "strats_no_pretrain"
    if strategy == "Pretraining":
        return "strats_main" if k == 6 else f"strats_dataset_pretrain_{k}"
    if strategy == "Pooled training":
        return "strats_pooled_all_sources" if k == 6 else f"strats_pooled_top_{k}"
    raise ValueError(f"Unknown strategy: {strategy}")


def write_exports(paths: ProjectPaths, data: StudyData, results: DerivedResults,
                  suitability_plot_data: pd.DataFrame) -> list[Path]:
    mapping = {
        "source_suitability_rankings": results.source_suitability_rankings,
        "single_source_transfer_gain_long": results.single_source_gain,
        "top6_dataset_transfer_gain_long": results.top6_dataset_gain,
        "attribute_transfer_gain_long": results.attribute_gain,
        "horizon_transfer_gain_long": results.horizon_gain,
        "suitability_validation_summary": results.suitability_summary,
        "suitability_validation_by_target": results.suitability_by_target,
        "suitability_validation_plot_data": suitability_plot_data,
        "principal_model_comparison": results.model_comparison,
        "target_panel_sensitivity": results.panel_sensitivity,
        "maxobs_ablation": results.maxobs_ablation,
        "dataset_characteristics": results.dataset_characteristics,
        "forecast_horizon_distribution": results.horizon_distribution,
        "all_primary_panel_model_results": data.model_seed.query("panel == @PRIMARY_PANEL and horizon_group == @ALL_HORIZONS"),
    }
    out = []
    for name, frame in mapping.items():
        path = paths.data_exports_dir / (name + ".csv")
        save_csv(frame, path)
        out.append(path)
    return out


def supplementary_rng(*key: Any) -> np.random.Generator:
    """A separate reproducible stream per comparison, independent of main figures.

    Adding a supplementary row does not change any existing bootstrap draw.
    The seed is a calculation setting; no result values are stored here.
    """
    payload = json.dumps([DEFAULT_RANDOM_SEED, *key], ensure_ascii=True).encode("utf-8")
    entropy = int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")
    return np.random.default_rng(entropy)


def check_unique(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    duplicates = frame.duplicated(list(keys), keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, list(keys)].head(8).to_string(index=False)
        raise ValueError(f"Duplicate result keys in {name}:\n{sample}")


def check_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {name}: {column}")


def pair_supplementary_results(
    frame: pd.DataFrame, model: str, metric: str, keys: Sequence[str]
) -> pd.DataFrame:
    """Require identical evaluation keys rather than silently dropping missing rows."""
    baseline = frame.loc[frame.base_configuration_id.eq("strats_no_pretrain"),
                         [*keys, metric]].rename(columns={metric: "baseline_nrmse"})
    transfer = frame.loc[frame.base_configuration_id.eq(model),
                         [*keys, metric]].rename(columns={metric: "transfer_nrmse"})
    paired = baseline.merge(transfer, on=list(keys), how="outer",
                            validate="one_to_one", indicator=True)
    if paired.empty or not paired["_merge"].eq("both").all():
        missing = paired.loc[paired["_merge"].ne("both"), [*keys, "_merge"]].head(10)
        raise ValueError(f"Unmatched supplementary comparisons for {model}:\n{missing}")
    check_finite(paired, ["baseline_nrmse", "transfer_nrmse"], model)
    if (paired.baseline_nrmse <= 0).any():
        raise ValueError(f"Nonpositive baseline prevents relative-gain calculation: {model}")
    paired = paired.drop(columns="_merge")
    paired["relative_gain_percent"] = (
        100.0 * (paired.baseline_nrmse - paired.transfer_nrmse) / paired.baseline_nrmse
    )
    return paired


def collect_seed_summary(data: StudyData, panels: Sequence[str]) -> pd.DataFrame:
    block = data.model_seed[
        data.model_seed.base_configuration_id.isin(NEURAL_MODELS)
        & data.model_seed.train_data_percent.isin(TRAIN_PERCENTAGES)
        & data.model_seed.panel.isin(panels)
        & data.model_seed.horizon_group.eq(ALL_HORIZONS)
    ].copy()
    keys = ["panel", "base_configuration_id", "train_data_percent"]
    check_unique(block, keys, "neural seed summaries")
    expected = len(panels) * len(NEURAL_MODELS) * len(TRAIN_PERCENTAGES)
    if len(block) != expected or not block.n_seeds.eq(3).all():
        raise ValueError("Incomplete neural-model/panel/fraction/seed coverage in supplementary results.")
    if not block.all_runs_is_complete_evaluation_coverage.fillna(False).all():
        raise ValueError("An evaluation panel lacks complete evaluation-cell coverage.")
    block = block.rename(columns={"main_macro_nrmse_mean": "nrmse_mean",
                                  "main_macro_nrmse_std": "nrmse_sd"})
    check_finite(block, ["nrmse_mean", "nrmse_sd"], "neural seed summaries")
    return block[[*keys, "nrmse_mean", "nrmse_sd", "n_seeds", "seeds"]].reset_index(drop=True)


def compute_supplementary_gains(
    data: StudyData, results: DerivedResults, kind: str
) -> pd.DataFrame:
    """Extend the existing outcome/horizon summaries without changing their estimand."""
    if kind == "outcome":
        frame = data.attribute_metrics[
            data.attribute_metrics.attribute.isin(data.primary_attributes)
            & data.attribute_metrics.horizon_group.eq(ALL_HORIZONS)
            & data.attribute_metrics.train_data_percent.isin(TRAIN_PERCENTAGES)
        ]
        dimension, metric = "attribute", "person_balanced_nrmse"
        existing = results.attribute_gain
    elif kind == "horizon":
        frame = data.dataset_metrics[
            data.dataset_metrics.panel.eq(PRIMARY_PANEL)
            & data.dataset_metrics.horizon_group.isin(["short", "medium", "long"])
            & data.dataset_metrics.train_data_percent.isin(TRAIN_PERCENTAGES)
        ]
        dimension, metric = "horizon_group", "main_macro_nrmse"
        existing = results.horizon_gain
    else:
        raise ValueError(f"Unknown supplementary gain summary: {kind}")
    keys = ["dataset", "seed", "train_data_percent", dimension]
    rows = []
    for model in TRANSFER_MODELS:
        paired = pair_supplementary_results(frame, model, metric, keys)
        by_dataset = paired.groupby(["train_data_percent", dimension, "dataset"], as_index=False).agg(
            relative_gain_percent=("relative_gain_percent", "mean"),
            n_seeds=("seed", "nunique"),
        )
        if not by_dataset.n_seeds.eq(3).all():
            raise ValueError(f"Incomplete paired seed coverage for {kind}, {model}.")
        for (pct, value), group in by_dataset.groupby(["train_data_percent", dimension], sort=True):
            values = group.relative_gain_percent.to_numpy(dtype=float)
            point = float(values.mean())
            low, high = bootstrap_mean_interval(
                values, DEFAULT_BOOTSTRAP_REPS, supplementary_rng(kind, model, int(pct), value)
            )
            # Reuse the interval just recomputed for the main-paper derivation at
            # shared pretraining endpoints. This is in-memory live data, not an
            # archived result, and preserves exact agreement with Figure 6.
            shared = existing[(existing.train_data_percent == pct) & (existing[dimension] == value)]
            if model == "strats_main" and len(shared):
                if len(shared) != 1 or not np.isclose(
                    point, float(shared.iloc[0].relative_gain_percent_mean), rtol=1e-12, atol=1e-12
                ):
                    raise ValueError(f"Supplement/main gain mismatch: {kind}, {pct}, {value}")
                low, high = float(shared.iloc[0].ci95_low), float(shared.iloc[0].ci95_high)
            datasets = [d for d in DATASET_ORDER if d in set(group.dataset)]
            rows.append({"base_configuration_id": model, "train_data_percent": int(pct),
                         dimension: value, "relative_gain_percent_mean": point,
                         "ci95_low": low, "ci95_high": high, "n_datasets": len(datasets),
                         "datasets": ";".join(datasets)})
    return pd.DataFrame(rows)


def compute_supplementary_results(data: StudyData, results: DerivedResults) -> dict[str, pd.DataFrame]:
    all_panels = collect_seed_summary(data, [PRIMARY_PANEL, COMMON_PANEL, EXTENDED_PANEL])
    neural = all_panels[all_panels.panel.eq(PRIMARY_PANEL)].copy()
    for column in ["relative_gain_percent", "ci95_low", "ci95_high"]:
        neural[column] = np.nan
    for index, row in neural.iterrows():
        model, pct = row.base_configuration_id, int(row.train_data_percent)
        if model not in TRANSFER_MODELS:
            continue
        paired = pair_dataset_transfer(data.dataset_metrics, model, [pct])
        point, low, high = cluster_bootstrap_relative_gain(
            paired, DEFAULT_BOOTSTRAP_REPS, supplementary_rng("macro", model, pct)
        )
        if pct in SINGLE_SOURCE_PERCENTAGES:
            shared = results.model_comparison.set_index("base_configuration_id").loc[model]
            if not np.isclose(point, shared[f"gain_{pct}_percent"], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Supplement/main macro-gain mismatch: {model}, {pct}")
            low, high = shared[f"gain_{pct}_ci95_low"], shared[f"gain_{pct}_ci95_high"]
        neural.loc[index, ["relative_gain_percent", "ci95_low", "ci95_high"]] = [point, low, high]

    dframe = data.dataset_metrics[
        data.dataset_metrics.panel.eq(PRIMARY_PANEL)
        & data.dataset_metrics.horizon_group.eq(ALL_HORIZONS)
        & data.dataset_metrics.base_configuration_id.isin(NEURAL_MODELS)
        & data.dataset_metrics.train_data_percent.isin(TRAIN_PERCENTAGES)
    ].copy()
    check_unique(dframe, ["dataset", "base_configuration_id", "train_data_percent", "seed"],
                 "dataset-specific neural results")
    check_finite(dframe, ["main_macro_nrmse"], "dataset-specific neural results")
    if not dframe.evaluation_cell_coverage.eq(1).all():
        raise ValueError("Dataset-specific neural results lack complete evaluation-cell coverage.")
    dataset_absolute = dframe.groupby(
        ["dataset", "base_configuration_id", "train_data_percent"], as_index=False
    ).agg(nrmse_mean=("main_macro_nrmse", "mean"), nrmse_sd=("main_macro_nrmse", "std"),
          n_seeds=("seed", "nunique"))
    expected = len(DATASET_ORDER) * len(NEURAL_MODELS) * len(TRAIN_PERCENTAGES)
    if len(dataset_absolute) != expected or not dataset_absolute.n_seeds.eq(3).all():
        raise ValueError("Incomplete dataset-specific neural-model learning curves.")
    # Verify that dataset means reproduce the macro numbers plotted in the main paper.
    regrouped = dataset_absolute.groupby(["base_configuration_id", "train_data_percent"]).nrmse_mean.mean()
    for _, row in neural.iterrows():
        if not np.isclose(row.nrmse_mean, regrouped.loc[(row.base_configuration_id, row.train_data_percent)],
                          rtol=1e-10, atol=1e-12):
            raise ValueError("Dataset-specific and macro NRMSE inputs are inconsistent; regenerate calculated errors.")

    source_count_rows = []
    for pct in TOPK_PERCENTAGES:
        for strategy in ["Pretraining", "Pooled training"]:
            for k in range(7):
                model = topk_base_id(strategy, k)
                r = get_model_seed_row(data.model_seed, model, pct)
                source_count_rows.append({"train_data_percent": pct, "strategy": strategy,
                                          "n_sources": k, "base_configuration_id": model,
                                          "nrmse_mean": float(r.main_macro_nrmse_mean),
                                          "nrmse_sd": float(r.main_macro_nrmse_std),
                                          "n_seeds": int(r.n_seeds)})
    source_count = pd.DataFrame(source_count_rows)

    single = data.source_target[
        data.source_target.panel.eq(PRIMARY_PANEL)
        & data.source_target.horizon_group.eq(ALL_HORIZONS)
        & data.source_target.source_config.eq("single_source")
        & data.source_target.train_data_percent.isin(SINGLE_SOURCE_PERCENTAGES)
    ]
    single_keys = ["dataset", "source_dataset", "train_data_percent"]
    check_unique(single, [*single_keys, "seed"], "single-source results")
    if not single.evaluation_cell_coverage.eq(1).all():
        raise ValueError("A single-source target result lacks complete evaluation-cell coverage.")
    single_sd = single.groupby(single_keys, as_index=False).agg(
        transfer_nrmse_sd=("main_macro_nrmse", "std")
    )
    single_source = results.single_source_gain.merge(single_sd, on=single_keys, how="left", validate="one_to_one")
    check_finite(single_source, ["transfer_nrmse_sd", "relative_gain_percent_sd"], "single-source SDs")

    cap_rows = []
    for label, limited, unlimited in [
        ("Target-only STraTS", "strats_no_pretrain", "strats_no_pretrain_maxobs_none"),
        ("All-source pretrained STraTS", "strats_main", "strats_main_maxobs_none"),
    ]:
        for dataset in ["macro", *DATASET_ORDER]:
            errors = []
            for model in [limited, unlimited]:
                if dataset == "macro":
                    r = get_model_seed_row(data.model_seed, model, 100)
                    errors.append((float(r.main_macro_nrmse_mean), float(r.main_macro_nrmse_std), int(r.n_seeds)))
                else:
                    block = data.dataset_metrics[
                        data.dataset_metrics.panel.eq(PRIMARY_PANEL)
                        & data.dataset_metrics.horizon_group.eq(ALL_HORIZONS)
                        & data.dataset_metrics.train_data_percent.eq(100)
                        & data.dataset_metrics.base_configuration_id.eq(model)
                        & data.dataset_metrics.dataset.eq(dataset)
                    ]
                    check_unique(block, ["seed"], "observation-limit results")
                    errors.append((float(block.main_macro_nrmse.mean()), float(block.main_macro_nrmse.std()),
                                   int(block.seed.nunique())))
            (cap, cap_sd, cap_n), (uncap, uncap_sd, uncap_n) = errors
            if cap_n != 3 or uncap_n != 3 or not np.isfinite([cap, cap_sd, uncap, uncap_sd]).all():
                raise ValueError(f"Incomplete observation-limit results: {label}, {dataset}")
            cap_rows.append({"model": label, "dataset": dataset, "train_data_percent": 100,
                             "max_obs_256_nrmse_mean": cap, "max_obs_256_nrmse_sd": cap_sd,
                             "max_obs_none_nrmse_mean": uncap, "max_obs_none_nrmse_sd": uncap_sd,
                             "nrmse_difference_none_minus_256": uncap - cap,
                             "relative_difference_percent": 100.0 * (uncap - cap) / cap,
                             "n_seeds": cap_n})

    return {
        "neural_global": neural.reset_index(drop=True),
        "dataset_absolute": dataset_absolute,
        "panels_absolute": all_panels,
        "outcome_gain": compute_supplementary_gains(data, results, "outcome"),
        "horizon_gain": compute_supplementary_gains(data, results, "horizon"),
        "observation_cap": pd.DataFrame(cap_rows),
        "source_count": source_count,
        "single_source": single_source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-only", action="store_true", help="Compute the frozen metadata-based source ranking only.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "results" / "metrics")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = ProjectPaths(args.metrics_dir, ROOT / "metadata", args.data_dir, ROOT, args.output_dir)
    if args.rank_only:
        summary = pd.read_csv(paths.preprocessing_dir / "summary_per_dataset.csv", dtype=str, keep_default_na=False)
        rankings = compute_source_suitability_rankings(summary)
        validate_ranking_against_origin_map(rankings, parse_origin_map(ROOT / "strats.py"))
        paths.data_exports_dir.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(paths.data_exports_dir / "source_suitability_rankings.csv", index=False)
        print(rankings[["target_dataset", "source_suitability_rank", "source_dataset", "source_suitability_score"]].to_string(index=False))
        return
    data = load_study_data(paths)
    for message in validate_study_data(data):
        logging.info(message)
    results, ranking_validation = derive_results(data, DEFAULT_BOOTSTRAP_REPS, DEFAULT_RANDOM_SEED)
    supplement = compute_supplementary_results(data, results)
    paths.data_exports_dir.mkdir(parents=True, exist_ok=True)
    write_exports(paths, data, results, ranking_validation)
    for name, frame in supplement.items():
        frame.to_csv(paths.data_exports_dir / f"supplementary_{name}.csv", index=False)
    logging.info("Saved transfer-analysis CSVs to %s", paths.data_exports_dir)


if __name__ == "__main__":
    main()
