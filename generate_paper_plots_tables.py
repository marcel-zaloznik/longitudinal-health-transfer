#!/usr/bin/env python3
"""Generate manuscript figures and tables from local data and completed runs."""
from __future__ import annotations

import argparse

import hashlib
import ast
import json
import logging
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd


SCRIPT_VERSION = "6.1.0"
PRIMARY_PANEL = "primary_9"
COMMON_PANEL = "common_all_datasets"
EXTENDED_PANEL = "extended_20"
ALL_HORIZONS = "all"
TRAIN_PERCENTAGES = [10, 30, 50, 70, 90, 100]
TOPK_PERCENTAGES = [10, 50, 100]
SINGLE_SOURCE_PERCENTAGES = [10, 100]
MM_TO_INCH = 1.0 / 25.4
FULL_WIDTH_MM = 170.0
HALF_WIDTH_MM = 85.0
DEFAULT_DPI = 300
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

DATASET_SHORT_LABELS = {
    "ABCD": "ABCD",
    "BE_KUL_Wave_1A": "LGS boys",
    "BE_KUL_Wave_1B": "LGS girls",
    "BE_KUL_Wave_23456": "Leuven adults",
    "SLOfit_lifelong": "SLOfit life",
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
}

LEARNING_CURVE_MODELS = [
    "persistence_age_sex_mean_target_only",
    "linear_regression_fixed_target_only",
    "gru_d_target_only",
    "strats_no_pretrain",
    "strats_pooled_all_sources",
    "strats_main",
]

MODEL_LINE_STYLES = {
    "persistence_age_sex_mean_target_only": (":", "v"),
    "linear_regression_fixed_target_only": ("--", "x"),
    "gru_d_target_only": ("-", "s"),
    "strats_no_pretrain": ("-", "o"),
    "strats_pooled_all_sources": ("-", "D"),
    "strats_main": ("-", "^"),
}

# The same three names for the evaluation panels in every table.
PANEL_LABELS = {
    PRIMARY_PANEL: "Primary nine-outcome panel",
    COMMON_PANEL: "Common three-outcome panel",
    EXTENDED_PANEL: "Extended 20-attribute panel",
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

# Revision experiments. Each folder holds a results_to_share.zip written by
# that experiment's own summarize step; only its aggregate tables are read.
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENT_FOLDERS = {
    "source_exclusion": "source_exclusion",
    "learning_rate": "learning_rate",
    "normalization": "scale_consistent",
    "grud_transfer": "grud_transfer",
}
RESULTS_ARCHIVE = "results_to_share.zip"

# Evaluation eligibility rule of preprocess_cohorts.py / prepare_data.py.
MIN_OBSERVED_TARGETS = 30
# Preprocessing settings of the same notebook, needed to rebuild the cohort
# from which SLOfit students was downsampled.
DOWNSAMPLED_DATASET = "SLOfit_students"
NEAR_DUPLICATE_YEARS = 0.5

REQUIRED_PROJECT_FILES = [
    Path("results/metrics/model_run_metrics.csv"),
    Path("results/metrics/model_seed_summary.csv"),
    Path("results/metrics/dataset_metrics.csv"),
    Path("results/metrics/dataset_attribute_metrics.csv"),
    Path("results/metrics/source_target_metrics.csv"),
    Path("results/metrics/analysis_manifest.json"),
    Path("metadata/summary_per_dataset.csv"),
    Path("data/dataset_full.csv"),
    Path("data/dataset.csv"),
    Path("data/input_output_pairs.csv"),
    Path("data/primary_target_panel.json"),
    Path("data/extended_target_panel.json"),
    Path("data/eligible_dataset_attribute_pairs.json"),
    Path("data/target_attribute_stds_by_dataset_fold.json"),
    Path("strats.py"),
    *[EXPERIMENTS_DIR / folder / RESULTS_ARCHIVE for folder in EXPERIMENT_FOLDERS.values()],
]

GENERATED_SUBDIRS = ["figures/main", "figures/supplementary", "tables/main", "tables/supplementary"]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    metrics_dir: Path
    preprocessing_dir: Path
    data_dir: Path
    src_dir: Path
    output_dir: Path
    main_figures_dir: Path
    supplementary_figures_dir: Path
    main_tables_dir: Path
    supplementary_tables_dir: Path
    experiments_dir: Path


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
    fold_stds: dict[str, Any]
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


def discover_project_root(project_root: Path) -> Path:
    root = project_root.expanduser().resolve()
    missing = [rel.as_posix() for rel in REQUIRED_PROJECT_FILES if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required local inputs are missing beneath " + str(root) + ":\n  - " + "\n  - ".join(missing)
        )
    return root


def build_paths(project_root: Path, script_dir: Path) -> ProjectPaths:
    out = script_dir.resolve()
    return ProjectPaths(
        root=project_root, metrics_dir=project_root / "results" / "metrics",
        preprocessing_dir=project_root / "metadata",
        data_dir=project_root / "data", src_dir=project_root, output_dir=out,
        main_figures_dir=out / "figures" / "main",
        supplementary_figures_dir=out / "figures" / "supplementary",
        main_tables_dir=out / "tables" / "main",
        supplementary_tables_dir=out / "tables" / "supplementary",
        experiments_dir=project_root / EXPERIMENTS_DIR,
    )

def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def configure_matplotlib() -> None:
    """One style for every figure; no figure text is smaller than 8 pt."""
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "lines.linewidth": 1.5,
            "lines.markersize": 5.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DEFAULT_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def make_output_dirs(paths: ProjectPaths) -> None:
    for directory in [
        paths.output_dir,
        paths.main_figures_dir,
        paths.supplementary_figures_dir,
        paths.main_tables_dir,
        paths.supplementary_tables_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


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
        fold_stds=read_json(paths.data_dir / "target_attribute_stds_by_dataset_fold.json"),
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


# ---------------------------------------------------------------------------
# Revision experiments. Only the aggregate summaries inside each experiment's
# results_to_share.zip are read; participant-level errors are not read.
# ---------------------------------------------------------------------------

@dataclass
class RevisionData:
    exclusion_by_target: pd.DataFrame      # Experiment 1, five affected targets
    exclusion_macro: pd.DataFrame
    exclusions: dict[str, list[str]]
    rate_by_dataset: pd.DataFrame          # Experiment 2
    rate_macro: pd.DataFrame
    rate_selection: pd.DataFrame
    rate_gains: pd.DataFrame
    scale_by_dataset: pd.DataFrame         # Experiment 3
    scale_macro: pd.DataFrame
    grud_paired: pd.DataFrame              # Experiment 4
    grud_jobs: pd.DataFrame
    grud_units: pd.DataFrame
    grud_attribute_scores: pd.DataFrame


def read_archive_table(archive: Path, member: str, columns: Sequence[str]) -> pd.DataFrame:
    name = f"{archive.parent.name}/{archive.name}:{member}"
    with zipfile.ZipFile(archive) as bundle:
        if member not in bundle.namelist():
            raise ValueError(f"{name} is missing; run that experiment's summarize step again.")
        with bundle.open(member) as handle:
            frame = pd.read_csv(handle)
    require_columns(frame, name, columns)
    return frame


def load_revision_data(paths: ProjectPaths) -> RevisionData:
    logging.info("Loading the aggregate results of the four revision experiments")
    archive = {key: paths.experiments_dir / folder / RESULTS_ARCHIVE
               for key, folder in EXPERIMENT_FOLDERS.items()}
    exclusion_columns = [
        "percent", "mode", "n_seeds", "target_only_nrmse_mean", "target_only_nrmse_sd",
        "matched_all_source_nrmse_mean", "matched_all_source_nrmse_sd",
        "excluded_nrmse_mean", "excluded_nrmse_sd",
        "excluded_minus_matched_nrmse_mean", "excluded_minus_matched_nrmse_sd",
        "matched_all_source_gain_pct", "excluded_gain_pct",
    ]
    rate_columns = ["target", "percent", "policy", "n_seeds", "nrmse_mean", "nrmse_sd"]
    scale_columns = [
        "target", "percent", "n_seeds", "reset_nrmse_mean", "reset_nrmse_sd",
        "scale_consistent_nrmse_mean", "scale_consistent_nrmse_sd", "target_only_nrmse_mean",
        "scale_minus_reset_nrmse_mean", "scale_minus_reset_nrmse_sd",
        "reset_gain_vs_target_only_pct", "scale_gain_vs_target_only_pct",
    ]
    with zipfile.ZipFile(archive["source_exclusion"]) as bundle:
        exclusions = json.loads(bundle.read("exclusions.json").decode("utf-8"))
    return RevisionData(
        exclusion_by_target=read_archive_table(
            archive["source_exclusion"], "summary/comparison_by_target.csv", ["target", *exclusion_columns]),
        exclusion_macro=read_archive_table(
            archive["source_exclusion"], "summary/affected_targets_summary.csv", exclusion_columns),
        exclusions={str(target): [str(source) for source in sources] for target, sources in exclusions.items()},
        rate_by_dataset=read_archive_table(archive["learning_rate"], "summary/dataset_summary.csv", rate_columns),
        rate_macro=read_archive_table(archive["learning_rate"], "summary/overall_summary.csv", rate_columns),
        rate_selection=read_archive_table(
            archive["learning_rate"], "summary/selection_counts.csv",
            ["target", "percent", "selected_lr_key", "n_folds"]),
        rate_gains=read_archive_table(
            archive["learning_rate"], "summary/historical_gains.csv",
            ["target", "percent", "transfer_reference", "n_seeds", "historical_target_only_nrmse_mean",
             "selected_nrmse_mean", "historical_transfer_nrmse_mean", "historical_transfer_nrmse_sd",
             "gain_vs_selected_pct"]),
        scale_by_dataset=read_archive_table(archive["normalization"], "summary/dataset_summary.csv", scale_columns),
        scale_macro=read_archive_table(archive["normalization"], "summary/seven_target_summary.csv", scale_columns),
        grud_paired=read_archive_table(
            archive["grud_transfer"], "summary/paired_summary.csv",
            ["target", "percent", "target_only_mean", "target_only_sd", "transfer_mean", "transfer_sd",
             "delta_mean", "delta_sd", "gain_percent", "n_seeds"]),
        grud_jobs=read_archive_table(
            archive["grud_transfer"], "audit/jobs.csv", ["target", "seed", "excluded_sources"]),
        grud_units=read_archive_table(
            archive["grud_transfer"], "summary/raw_units_and_calibration_by_seed.csv",
            ["target", "percent", "seed", "arm", "attribute", "person_balanced_rmse",
             "calibration_slope", "n_test_people", "in_primary_panel"]),
        grud_attribute_scores=read_archive_table(
            archive["grud_transfer"], "summary/attribute_scores_by_seed.csv",
            ["target", "percent", "seed", "arm", "attribute", "n_test_people"]),
    )


def check_experiment_grid(frame: pd.DataFrame, name: str, targets: Sequence[str],
                          extra_keys: Sequence[str] = ()) -> None:
    """Every target at 10% and 100%, once per extra key, from three runs."""
    keys = ["target", "percent", *extra_keys]
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: duplicated result rows.")
    if set(frame["target"]) != set(targets) or set(frame["percent"]) != set(SINGLE_SOURCE_PERCENTAGES):
        raise ValueError(f"{name}: unexpected targets or training fractions.")
    combinations = frame[list(extra_keys)].drop_duplicates().shape[0] if extra_keys else 1
    if len(frame) != len(targets) * len(SINGLE_SOURCE_PERCENTAGES) * combinations:
        raise ValueError(f"{name}: incomplete results.")
    if not frame["n_seeds"].eq(3).all():
        raise ValueError(f"{name}: every result must come from three runs.")


def dataset_run_means(dataset_metrics: pd.DataFrame, base_id: str, percent: int) -> pd.Series:
    """Primary-panel NRMSE of one configuration, averaged over runs within each target dataset."""
    block = dataset_metrics[
        dataset_metrics["base_configuration_id"].eq(base_id)
        & dataset_metrics["panel"].eq(PRIMARY_PANEL)
        & dataset_metrics["horizon_group"].eq(ALL_HORIZONS)
        & dataset_metrics["train_data_percent"].eq(percent)
    ]
    means = block.groupby("dataset")["main_macro_nrmse"].mean().reindex(DATASET_ORDER)
    if means.isna().any():
        raise ValueError(f"Missing dataset-specific results for {base_id} at {percent}%.")
    return means


def grud_excluded_sources(revision: RevisionData) -> dict[str, list[str]]:
    """Source cohorts left out of GRU-D pretraining for each target (the same in every run)."""
    excluded: dict[str, list[str]] = {}
    for target, group in revision.grud_jobs.groupby("target"):
        values = set(group["excluded_sources"].fillna("").astype(str))
        if len(values) != 1:
            raise ValueError(f"GRU-D source exclusions differ between runs for {target}.")
        excluded[str(target)] = [source for source in values.pop().split("|") if source]
    return excluded


def validate_revision_data(data: StudyData, revision: RevisionData) -> list[str]:
    """The experiment summaries must be complete and agree with the submitted results."""
    affected = [dataset for dataset in DATASET_ORDER if dataset in revision.exclusions]
    if set(affected) != set(revision.exclusions):
        raise ValueError("Experiment 1 names a target outside the seven datasets.")
    for target, sources in revision.exclusions.items():
        if not sources or target in sources or not set(sources) <= set(DATASET_ORDER):
            raise ValueError(f"Invalid source exclusion for {target}: {sources}")
    check_experiment_grid(revision.exclusion_by_target, "Experiment 1", affected, ["mode"])
    if set(revision.exclusion_by_target["mode"]) != {"pretrain", "pooled"}:
        raise ValueError("Experiment 1 must contain pretraining and pooled results.")
    if len(revision.exclusion_macro) != 4 or not revision.exclusion_macro["n_seeds"].eq(3).all():
        raise ValueError("Experiment 1: incomplete five-target summary.")

    policies = {"fixed_lr1e-4", "fixed_lr5e-4", "validation_selected"}
    if set(revision.rate_by_dataset["policy"]) != policies:
        raise ValueError("Experiment 2: unexpected learning-rate policies.")
    check_experiment_grid(revision.rate_by_dataset, "Experiment 2", DATASET_ORDER, ["policy"])
    check_experiment_grid(revision.rate_macro, "Experiment 2 (overall)", ["Overall"], ["policy"])
    check_experiment_grid(revision.rate_gains, "Experiment 2 (gains)", [*DATASET_ORDER, "Overall"],
                          ["transfer_reference"])
    check_experiment_grid(revision.scale_by_dataset, "Experiment 3", DATASET_ORDER)
    check_experiment_grid(revision.scale_macro, "Experiment 3 (macro)", ["SEVEN_TARGET_MACRO"])
    check_experiment_grid(revision.grud_paired, "Experiment 4", [*DATASET_ORDER, "Overall"])

    def agree(left: float, right: float, what: str) -> None:
        if not np.isclose(left, right, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{what}: the experiment summary ({left:.9f}) differs from the current "
                             f"results ({right:.9f}). Regenerate both from the same error files.")

    # The submitted STraTS results used as references inside the experiments
    # must be the ones in results/metrics.
    references = {"historical_pooled": "strats_pooled_all_sources", "historical_pretrained": "strats_main"}
    for percent in SINGLE_SOURCE_PERCENTAGES:
        target_only = dataset_run_means(data.dataset_metrics, "strats_no_pretrain", percent)
        for target in affected:
            rows = revision.exclusion_by_target[
                revision.exclusion_by_target["target"].eq(target)
                & revision.exclusion_by_target["percent"].eq(percent)
            ]
            for value in rows["target_only_nrmse_mean"]:
                agree(float(value), float(target_only[target]), f"Experiment 1 target-only STraTS, {target}")
        for reference, base_id in references.items():
            submitted = dataset_run_means(data.dataset_metrics, base_id, percent)
            for dataset in DATASET_ORDER:
                row = one_row(revision.rate_gains, target=dataset, percent=percent,
                              transfer_reference=reference)
                agree(float(row["historical_transfer_nrmse_mean"]), float(submitted[dataset]),
                      f"Experiment 2 {reference}, {dataset}")
                agree(float(row["historical_target_only_nrmse_mean"]), float(target_only[dataset]),
                      f"Experiment 2 target-only STraTS, {dataset}")
        # Experiment 3 reuses the validation-selected target-only STraTS of Experiment 2.
        for dataset in DATASET_ORDER:
            selected = one_row(revision.rate_by_dataset, target=dataset, percent=percent,
                               policy="validation_selected")
            reused = one_row(revision.scale_by_dataset, target=dataset, percent=percent)
            agree(float(reused["target_only_nrmse_mean"]), float(selected["nrmse_mean"]),
                  f"Experiment 3 target-only reference, {dataset}")

    # Experiment 4 applies the source exclusions of Experiment 1.
    excluded = grud_excluded_sources(revision)
    for dataset in DATASET_ORDER:
        if sorted(excluded.get(dataset, [])) != sorted(revision.exclusions.get(dataset, [])):
            raise ValueError(f"GRU-D source exclusions for {dataset} differ from Experiment 1.")

    return [
        "Revision experiments are complete: every target at 10% and 100% from three runs.",
        "Submitted STraTS scores inside Experiments 1 and 2 equal the current calculated errors.",
        "Experiment 3 uses the validation-selected target-only STraTS of Experiment 2; "
        "Experiment 4 applies the source exclusions of Experiment 1.",
    ]


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


def derive_results(
    data: StudyData, bootstrap_reps: int, random_seed: int
) -> DerivedResults:
    rng = np.random.default_rng(random_seed)
    characteristics, availability, horizon_distribution = compute_dataset_characteristics(data)
    rankings = compute_source_suitability_rankings(data.summary_per_dataset)
    validate_ranking_against_origin_map(rankings, data.origin_map)
    single_source_gain = compute_single_source_gain(data)
    _, suitability_summary, suitability_by_target = compute_suitability_validation(
        rankings, single_source_gain, bootstrap_reps, rng
    )
    top6_dataset_gain = compute_top6_dataset_gain(data)
    attribute_gain = compute_attribute_gain(data, bootstrap_reps, rng)
    horizon_gain = compute_horizon_gain(data, bootstrap_reps, rng)
    model_comparison = compute_model_comparison(data, bootstrap_reps, rng)
    panel_sensitivity = compute_panel_sensitivity(data)

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
    )
    return derived


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile(r"[\\&%$#_{}~^]")
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def format_number(value: Any, digits: int = 3, missing: str = "--") -> str:
    if value is None or pd.isna(value):
        return missing
    return f"{float(value):.{digits}f}"


def format_integer(value: Any, missing: str = "--") -> str:
    if value is None or pd.isna(value):
        return missing
    return f"{int(round(float(value))):,}".replace(",", r"\,")


def format_mean_sd(mean: Any, sd: Any, n_seeds: int | None = None, digits: int = 3) -> str:
    if mean is None or pd.isna(mean):
        return "--"
    mean_text = f"{float(mean):.{digits}f}"
    if sd is None or pd.isna(sd) or (n_seeds is not None and n_seeds < 2):
        return mean_text
    return f"{mean_text} ({float(sd):.{digits}f})"


def format_gain_ci(point: Any, low: Any, high: Any, digits: int = 1) -> str:
    if point is None or pd.isna(point):
        return "--"
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return f"{float(point):.{digits}f}"
    return f"{float(point):.{digits}f} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


# ---------------------------------------------------------------------------
# Revised presentation. No model or evaluation changes are made below.
# ---------------------------------------------------------------------------

ATTRIBUTE_UNITS = {
    "Height": "cm", "Weight": "kg", "BMI": r"kg/m$^2$",
    "Waist_circumference": "cm", "Systolic_blood_pressure": "mmHg",
    "Diastolic_blood_pressure": "mmHg", "Resting_heart_rate": "beats/min",
    "Glucose": "mmol/L", "Total_cholesterol": "mmol/L", "HDL": "mmol/L",
    "LDL": "mmol/L", "Triglycerides": "mmol/L", "Smoking": "Category",
    "Fitness_speed": "km/h", "PA_h_per_w": "h/week", "Alcohol": "g/day",
    "Diabetes": "Binary", "Sleep": "h/night", "Diet_HHS": "Score",
    "Diet_T2P": "Score",
}
# Units are listed in metadata/codebook.csv. Fitness km/h and the meaning
# of the dietary scores were confirmed by the author. No diet formula is assumed.

FIGURES = [('main', 'figure_1_model_input_and_prediction', 'fig:strats_pipeline'),
 ('main', 'figure_2_learning_curves', 'fig:figure_1_learning_curves'),
 ('main', 'figure_2_number_of_sources', 'fig:figure_2_number_of_sources'),
 ('main',
  'figure_S1_dataset_specific_top6_transfer',
  'fig:figure_S1_dataset_specific_top6_transfer'),
 ('main', 'figure_3_single_source_transfer', 'fig:figure_3_single_source_transfer'),
 ('main', 'figure_6_outcome_specific_transfer', 'fig:figure_S2_attribute_specific_transfer'),
 ('supplementary', 'figure_S1_dataset_learning_curves', 'fig:supp_dataset_learning_curves')]

# Captions synchronized with Paper/current paper; no manuscript is read at runtime.
FIGURE_CAPTIONS = {
    'fig:strats_pipeline': (
        r"STraTS input representation and multi-target prediction.",
        r"Observed measurements are represented as variable--value--time triplets. Time is "
        r"measured relative to the requested future age and divided by a training-derived "
        r"horizon SD. Solid outlines identify STraTS representation components; dashed outlines "
        r"identify task-specific adaptations. Contextual inputs are combined with the "
        r"time-series representation before the 20-output regression head. Variable-level "
        r"dropout is applied during training only, and only observed future targets contribute "
        r"to the weighted loss. Contextual inputs and scaling are defined in "
        r"Section~\ref{STraTS_2_3_1}.",
    ),
    'fig:figure_1_learning_curves': (
        r"Prediction error across target training fractions.",
        r"Points show mean primary macro NRMSE across three runs; error bars show run-to-run "
        r"SD. NRMSE balances participants, eligible outcomes and target datasets; lower values "
        r"indicate better performance. Pooled and pretrained STraTS use all six non-target "
        r"datasets. Percentages refer to participants in the three target training folds; "
        r"validation and test participants are unchanged.",
    ),
    'fig:figure_2_number_of_sources': (
        r"Prediction error as source datasets are added.",
        r"Sources follow the fixed target-specific heuristic order in Additional file 1, Table "
        r"S3. Zero sources denotes target-only STraTS. Points show mean primary macro NRMSE "
        r"across three runs, with run-to-run SD bars; lower values indicate better performance. "
        r"Panels show 10\%, 50\% and 100\% of participants in the three target training folds. "
        r"Numerical results are provided in Additional file 1, Table S12.",
    ),
    'fig:figure_S1_dataset_specific_top6_transfer': (
        r"Dataset-specific gains from all-source STraTS pretraining and pooled training.",
        r"Cells show the relative reduction in primary-panel NRMSE (\%) compared with "
        r"target-only STraTS, calculated within each target dataset and run and averaged across "
        r"three runs. Blue indicates improvement and red deterioration. Percentages refer to "
        r"participants in the three target training folds. Run-to-run SDs are provided in "
        r"Additional file 1, Table S4.",
    ),
    'fig:figure_3_single_source_transfer': (
        r"Single-source STraTS pretraining gains at 10\% and 100\% of target training "
        r"participants.",
        r"Rows identify target datasets and columns identify sources. Cells show the relative "
        r"reduction in primary-panel NRMSE (\%) compared with target-only STraTS, calculated "
        r"within each target dataset and run and averaged across three runs. Blue indicates "
        r"improvement and red deterioration; grey diagonal cells are not applicable. "
        r"Percentages refer to participants in the three target training folds. Errors and "
        r"run-to-run SDs are provided in Additional file 1, Table S13.",
    ),
    'fig:figure_S2_attribute_specific_transfer': (
        r"Outcome-specific gains from all-source STraTS pretraining.",
        r"Points show the mean relative reduction in outcome-specific NRMSE compared with "
        r"target-only STraTS. Gains are calculated within each target dataset and run, averaged "
        r"across three runs, and then averaged equally across eligible datasets. Labels give "
        r"the number of contributing datasets. Bars show descriptive 95\% percentile intervals "
        r"from 10\,000 dataset-bootstrap resamples. Filled circles and open squares denote 10\% "
        r"and 100\% of target training participants, respectively. Positive values indicate "
        r"improvement.",
    ),
    'fig:supp_dataset_learning_curves': (
        r"Absolute neural-model learning curves within each target dataset.",
        r"Points show mean primary-panel NRMSE across three runs; error bars show their "
        r"standard deviations. Lower values are better. Pooled and pretrained STraTS use all "
        r"six sources. Percentages refer to participants in the three target training folds. "
        r"Vertical scales differ between panels. Numerical values are in Table S11.",
    ),
}

FIGURE_PLACEMENT = {'fig:strats_pipeline': {'width': '\\textwidth',
                         'overleaf_path': 'figures/main/figure_1_model_input_and_prediction.pdf',
                         'placement': 'htbp',
                         'generator': 'new'},
 'fig:figure_1_learning_curves': {'width': '\\textwidth',
                                  'overleaf_path': 'figures/main/figure_2_learning_curves.pdf',
                                  'placement': 'htbp',
                                  'generator': 'new'},
 'fig:figure_2_number_of_sources': {'width': '\\textwidth',
                                    'overleaf_path': 'figures/main/figure_2_number_of_sources.pdf',
                                    'placement': 'htbp',
                                    'generator': 'revised'},
 'fig:figure_S1_dataset_specific_top6_transfer': {'width': '\\textwidth',
                                                  'overleaf_path': 'figures/main/figure_S1_dataset_specific_top6_transfer.pdf',
                                                  'placement': 'htbp',
                                                  'generator': 'revised'},
 'fig:figure_3_single_source_transfer': {'width': '\\textwidth',
                                         'overleaf_path': 'figures/main/figure_3_single_source_transfer.pdf',
                                         'placement': 'htbp',
                                         'generator': 'revised'},
 'fig:figure_S2_attribute_specific_transfer': {'width': '\\textwidth',
                                               'overleaf_path': 'figures/main/figure_6_outcome_specific_transfer.pdf',
                                               'placement': 'htbp',
                                               'generator': 'new'},
 'fig:supp_dataset_learning_curves': {'width': '\\textwidth',
                                 'overleaf_path': 'figures/supplementary/figure_S1_dataset_learning_curves.pdf',
                                 'placement': 'H',
                                 'generator': 'supplementary'}}


def write_table(path: Path, caption: str, label: str, alignment: str,
                header: Sequence[str], rows: Sequence[Sequence[str] | str],
                note: str = "", *, tabcolsep: str = "3pt") -> Path:
    """Editable table with manuscript-specific spacing; never shrink its font."""
    lines = [r"\begin{table*}[htbp]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
             r"\renewcommand{\arraystretch}{1.14}",
             rf"\begin{{tabular}}{{{alignment}}}", r"\toprule"]
    lines.extend(header)
    lines.append(r"\midrule")
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\vspace{2mm}", r"\begin{minipage}{\textwidth}",
                  r"\footnotesize\textit{Note:} " + note, r"\end{minipage}"]
    lines += [r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\r\n")
    return path


def fraction_text(percent: int) -> str:
    """One wording for a target training fraction in every table, as in the figures."""
    return rf"{int(percent)}\% of target training participants"


def fraction_columns(span: int, stacked: bool = False) -> list[str]:
    """Header lines placing the 10% and 100% training fractions over `span` columns each.

    The main tables stack the wording on two lines to stay within the text width.
    """
    def cell(percent: int) -> str:
        text = (rf"\shortstack{{{percent}\% of target\\training participants}}" if stacked
                else fraction_text(percent))
        return rf"\multicolumn{{{span}}}{{c}}{{{text}}}"
    low, high = SINGLE_SOURCE_PERCENTAGES
    return [rf"& {cell(low)} & {cell(high)} \\",
            rf"\cmidrule(lr){{2-{span + 1}}}\cmidrule(lr){{{span + 2}-{2 * span + 1}}}"]


def compute_eligibility_cells(data: StudyData, availability: pd.DataFrame) -> pd.DataFrame:
    """Eligible folds and observed-value counts of every structurally available dataset-attribute cell.

    Also confirms that the saved eligibility flags follow the stated rule.
    """
    available = availability.set_index("attribute")
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        folds = data.eligibility[dataset]
        for attribute in data.extended_attributes:
            cells = [folds[fold][attribute] for fold in sorted(folds, key=int)]
            for cell in cells:
                expected = (cell["std"] is not None
                            and cell["train_observed_unique_output_rows"] >= MIN_OBSERVED_TARGETS
                            and cell["test_observed_unique_output_rows"] >= MIN_OBSERVED_TARGETS)
                if bool(cell["eligible"]) != expected:
                    raise ValueError(f"Eligibility of {dataset}/{attribute} does not follow the "
                                     f"{MIN_OBSERVED_TARGETS}-value rule.")
            n_eligible = sum(bool(cell["eligible"]) for cell in cells)
            if not bool(available.loc[attribute, dataset]):
                if n_eligible:
                    raise ValueError(f"{dataset}/{attribute} is evaluated but marked structurally absent.")
                continue
            development = [int(cell["train_observed_unique_output_rows"]) for cell in cells]
            test = [int(cell["test_observed_unique_output_rows"]) for cell in cells]
            rows.append({"dataset": dataset, "attribute": attribute, "n_folds": len(cells),
                         "n_eligible": n_eligible, "development_min": min(development),
                         "development_max": max(development), "test_min": min(test), "test_max": max(test)})
    return pd.DataFrame(rows)


def macro_gain_interval(reference: pd.Series, transfer: pd.Series, *key: Any) -> tuple[float, float, float]:
    """Gain (%) of the dataset-averaged `transfer` error over `reference`, with its dataset-bootstrap interval.

    Both series hold run-averaged NRMSEs indexed by target dataset, as in the
    principal comparison. Every comparison has its own reproducible stream.
    """
    paired = pd.DataFrame({"baseline_nrmse": reference, "transfer_nrmse": transfer})
    if len(paired) < 2 or paired.isna().any().any():
        raise ValueError(f"Incomplete dataset-specific errors for the comparison {key}.")
    paired = paired.rename_axis("dataset").reset_index()
    return cluster_bootstrap_relative_gain(paired, DEFAULT_BOOTSTRAP_REPS, supplementary_rng(*key))


def compute_best_target_only_gains(data: StudyData, results: DerivedResults) -> pd.DataFrame:
    """Gains of all-source pooled and pretrained STraTS over the best target-only model at each fraction."""
    comparison = results.model_comparison.set_index("base_configuration_id")
    target_only = [model for model in PRIMARY_MODEL_ORDER if model not in TRANSFER_MODELS]
    rows: list[dict[str, Any]] = []
    for percent in SINGLE_SOURCE_PERCENTAGES:
        errors = comparison.loc[target_only, f"nrmse_{percent}_mean"].astype(float)
        best = str(errors.idxmin())
        reference = dataset_run_means(data.dataset_metrics, best, percent)
        for model in TRANSFER_MODELS:
            transfer = dataset_run_means(data.dataset_metrics, model, percent)
            point, low, high = macro_gain_interval(reference, transfer, "best_target_only", model, percent)
            expected = 100.0 * (errors[best] - comparison.loc[model, f"nrmse_{percent}_mean"]) / errors[best]
            if not np.isclose(point, expected, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Inconsistent gain over the best target-only model: {model}, {percent}%.")
            rows.append({"base_configuration_id": model, "train_data_percent": percent,
                         "best_target_only": best, "gain_percent": point,
                         "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


# (analysis, heading in Table 6). The order is that of the revision experiments.
ADDITIONAL_ANALYSES = [
    ("source_exclusion", "A. Excluding potentially overlapping sources: STraTS, five targets"),
    ("learning_rate", "B. Target-only learning-rate sensitivity: STraTS, seven targets"),
    ("normalization", "C. Normalization at fine-tuning: all-source STraTS, seven targets"),
    ("grud_transfer", "D. Transfer with GRU-D: seven targets"),
]

# These describe the same references used by compute_additional_analyses.
ADDITIONAL_GAIN_REFERENCES = {
    "source_exclusion": "main-analysis target-only STraTS",
    "learning_rate": "validation-selected target-only STraTS",
    "normalization": "validation-selected target-only STraTS",
    "grud_transfer": "fixed-layout target-only GRU-D",
}


def compute_additional_analyses(revision: RevisionData) -> pd.DataFrame:
    """Aggregate results of the four revision experiments (main Table 6).

    Errors and their run SDs are the experiments' own macro summaries. Each
    gain is recomputed from the dataset-specific run means, checked against
    the gain reported by the experiment, and given a dataset-bootstrap interval.
    """
    rows: list[dict[str, Any]] = []

    def add(analysis: str, model: str, label: str, percent: int, mean: Any, sd: Any,
            reference: pd.Series | None = None, transfer: pd.Series | None = None,
            reported_gain: Any = None) -> None:
        """`model` identifies the row (and its bootstrap stream); `label` is only what is printed."""
        row = {"analysis": analysis, "model": model, "label": label, "train_data_percent": percent,
               "nrmse_mean": float(mean), "nrmse_sd": float(sd),
               "gain_percent": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
        if reference is not None:
            point, low, high = macro_gain_interval(reference, transfer, "additional", analysis, model, percent)
            if not np.isclose(point, float(reported_gain), rtol=1e-6, atol=1e-6):
                raise ValueError(f"{analysis}, {label}, {percent}%: the gain recomputed from dataset-specific "
                                 f"errors ({point:.6f}) differs from the experiment summary ({float(reported_gain):.6f}).")
            row.update(gain_percent=point, ci95_low=low, ci95_high=high)
        rows.append(row)

    for percent in SINGLE_SOURCE_PERCENTAGES:
        # Experiment 1: all six sources (matched rerun) against exclusion of the suspected sources.
        macro = revision.exclusion_macro[revision.exclusion_macro["percent"].eq(percent)].set_index("mode")
        by_target = revision.exclusion_by_target[revision.exclusion_by_target["percent"].eq(percent)]
        if not np.isclose(macro.loc["pretrain", "target_only_nrmse_mean"],
                          macro.loc["pooled", "target_only_nrmse_mean"], rtol=1e-12, atol=1e-12):
            raise ValueError("Experiment 1 uses different target-only references for the two strategies.")
        add("source_exclusion", "target_only", "Target-only", percent,
            macro.loc["pretrain", "target_only_nrmse_mean"], macro.loc["pretrain", "target_only_nrmse_sd"])
        for mode, strategy in [("pretrain", "Pretrained"), ("pooled", "Pooled")]:
            targets = by_target[by_target["mode"].eq(mode)].set_index("target")
            for column, sources in [("matched_all_source", "all sources"), ("excluded", "sources excluded")]:
                add("source_exclusion", f"{mode}_{column}", f"{strategy}, {sources}", percent,
                    macro.loc[mode, column + "_nrmse_mean"], macro.loc[mode, column + "_nrmse_sd"],
                    targets["target_only_nrmse_mean"], targets[column + "_nrmse_mean"],
                    macro.loc[mode, column + "_gain_pct"])

        # Experiment 2: target-only STraTS at two learning rates; submitted transfer models as references.
        macro = revision.rate_macro[revision.rate_macro["percent"].eq(percent)].set_index("policy")
        for policy, label in [("fixed_lr1e-4", r"Rate $10^{-4}$"),
                              ("fixed_lr5e-4", r"Rate $5\times10^{-4}$"),
                              ("validation_selected", "Selected rate")]:
            add("learning_rate", policy, label, percent,
                macro.loc[policy, "nrmse_mean"], macro.loc[policy, "nrmse_sd"])
        gains = revision.rate_gains[revision.rate_gains["percent"].eq(percent)]
        for reference, label in [("historical_pooled", "Pooled, main-analysis"),
                                 ("historical_pretrained", "Pretrained, main-analysis")]:
            block = gains[gains["transfer_reference"].eq(reference)].set_index("target")
            datasets = block.loc[DATASET_ORDER]
            add("learning_rate", reference, label, percent,
                block.loc["Overall", "historical_transfer_nrmse_mean"],
                block.loc["Overall", "historical_transfer_nrmse_sd"],
                datasets["selected_nrmse_mean"], datasets["historical_transfer_nrmse_mean"],
                block.loc["Overall", "gain_vs_selected_pct"])

        # Experiment 3: the submitted reset of normalization (matched rerun) against scale-consistent transfer.
        macro_row = one_row(revision.scale_macro, percent=percent)
        datasets = (revision.scale_by_dataset[revision.scale_by_dataset["percent"].eq(percent)]
                    .set_index("target").loc[DATASET_ORDER])
        for column, label, gain in [("reset", "Reset standardization", "reset_gain_vs_target_only_pct"),
                                    ("scale_consistent", "Scale-consistent", "scale_gain_vs_target_only_pct")]:
            add("normalization", column, label, percent,
                macro_row[column + "_nrmse_mean"], macro_row[column + "_nrmse_sd"],
                datasets["target_only_nrmse_mean"], datasets[column + "_nrmse_mean"], macro_row[gain])

        # Experiment 4: GRU-D target-only against source-pretrained, fine-tuned GRU-D.
        paired = revision.grud_paired[revision.grud_paired["percent"].eq(percent)].set_index("target")
        datasets = paired.loc[DATASET_ORDER]
        add("grud_transfer", "target_only", "Target-only", percent,
            paired.loc["Overall", "target_only_mean"], paired.loc["Overall", "target_only_sd"])
        add("grud_transfer", "transfer", "Pretrained", percent,
            paired.loc["Overall", "transfer_mean"], paired.loc["Overall", "transfer_sd"],
            datasets["target_only_mean"], datasets["transfer_mean"], paired.loc["Overall", "gain_percent"])
    return pd.DataFrame(rows)


# Names of the target-only models inside a sentence.
MODEL_NAMES_IN_TEXT = {
    "persistence_age_sex_mean_target_only": "the age--sex mean",
    "linear_regression_fixed_target_only": "linear regression",
    "random_forest_fixed_target_only": "random forest",
    "gru_d_target_only": "GRU-D",
    "strats_no_pretrain": "target-only STraTS",
}


def write_main_tables(paths: ProjectPaths, data: StudyData, results: DerivedResults,
                      extra: Mapping[str, pd.DataFrame]) -> list[Path]:
    main: list[Path] = []
    chars = results.dataset_characteristics.set_index("dataset")
    rows = []
    for d in DATASET_ORDER:
        r = chars.loc[d]
        rows.append([latex_escape(DATASET_LABELS[d]), format_integer(r.n_people),
                     format_integer(r.n_visits),
                     f"{r.mean_visits:.1f} ({r.sd_visits:.1f})",
                     f"{r.min_age:.1f}--{r.max_age:.1f}",
                     f"{int(r.n_available_attributes)}/20",
                     f"{r.completeness_available_percent:.0f}"])
    main.append(write_table(
        paths.main_tables_dir / "table_1_dataset_characteristics.tex",
        "Characteristics of the seven analysis datasets", "tab:dataset_characteristics",
        r"@{}lrrrrrr@{}",
        [r"Dataset & People & Visits & \shortstack{Visits/person\\mean (SD)} & \shortstack{Age range\\(years)} & \shortstack{Available\\attributes} & \shortstack{Completeness\\(\%)} \\"],
        rows,
        r"ABCD is from the Netherlands; LGS boys, LGS girls and Leuven adults are from Belgium; "
        r"the SLOfit datasets are from Slovenia; YFS is from Finland. Completeness is the "
        r"percentage of observed values among attributes structurally available in each "
        r"dataset, not among all 20 attributes. Additional age, eligibility and "
        rf"forecast-horizon summaries are provided in Additional file 1, {supp_ref('horizons')}. LGS, Leuven "
        r"Growth Study; SD, standard deviation."
    ))

    # Reviewer 1, comment 4: an available attribute is not always evaluated.
    eligibility = extra["eligibility"]
    marks = {(r.dataset, r.attribute): (r"\rlap{$^{\dagger}$}" if r.n_eligible == 0 else r"\rlap{$^{\ddagger}$}")
             for r in eligibility.itertuples() if r.n_eligible < r.n_folds}
    explanations = []
    if any(r.n_eligible == 0 for r in eligibility.itertuples()):
        explanations.append(r"$^{\dagger}$No eligible evaluation folds")
    partial_counts = sorted({(int(r.n_eligible), int(r.n_folds))
                             for r in eligibility.itertuples()
                             if 0 < r.n_eligible < r.n_folds})
    if partial_counts:
        # The supplied data give two of five folds. Keep the legend correct
        # rather than hard-coding that count if the eligibility inputs change.
        if len(partial_counts) == 1:
            eligible, folds = partial_counts[0]
            words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
            description = (f"eligible in {words.get(eligible, str(eligible))} of "
                           f"{words.get(folds, str(folds))} folds")
        else:
            description = "eligible in only some folds"
        explanations.append(r"$^{\ddagger}$" + description)
    eligibility_note = ""
    if explanations:
        eligibility_note = ("; ".join(explanations)
                            + rf" (Additional file 1, {supp_ref('excluded_cells')}). ")
    availability = results.attribute_availability.set_index("attribute")
    rows = []
    for a in data.extended_attributes:
        name = latex_escape(ATTRIBUTE_LABELS[a])
        if a in data.primary_attributes:
            name += r"$^{*}$"
        rows.append([name, ATTRIBUTE_UNITS[a]] +
                    ["+" + marks.get((d, a), "") if bool(availability.loc[a, d]) else "--"
                     for d in DATASET_ORDER])
    main.append(write_table(
        paths.main_tables_dir / "table_2_attribute_availability.tex",
        "Attribute units and availability across datasets", "tab:supp_attribute_availability",
        r"@{}llccccccc@{}",
        [r"Attribute & Unit & ABCD & \shortstack{LGS\\boys} & \shortstack{LGS\\girls} & \shortstack{Leuven\\adults} & \shortstack{SLOfit\\lifelong} & \shortstack{SLOfit\\students} & YFS \\"],
        rows,
        r"+, structurally available, although individual visits may have missing values; "
        r"--, structurally absent. $^{*}$Primary evaluation outcome "
        r"(Section~\ref{Longitudinal_multi_target_prediction_task}). "
        + eligibility_note +
        r"Fitness is the final speed in the 20-m shuttle-run test. HHS and T2P denote dietary "
        r"components associated with Healthy Heart Score and Test2Prevent. Smoking and diabetes "
        r"are numerically encoded categories and are excluded from the primary panel."
    ))

    rows = [r"\multicolumn{5}{@{}l}{\textit{Gain reference: target-only STraTS}} \\"]
    labels = {
        "persistence_age_sex_mean_target_only": "Age--sex mean",
        "linear_regression_fixed_target_only": "Linear regression",
        "random_forest_fixed_target_only": "Random forest",
        "gru_d_target_only": "GRU-D, target-only",
        "strats_no_pretrain": "STraTS, target-only",
        "strats_pooled_all_sources": "STraTS, pooled",
        "strats_main": "STraTS, pretrained",
    }

    for _, r in results.model_comparison.iterrows():
        cells = [labels[r.base_configuration_id]]
        for pct in SINGLE_SOURCE_PERCENTAGES:
            cells += [format_mean_sd(r[f"nrmse_{pct}_mean"], r[f"nrmse_{pct}_sd"],
                                    int(r[f"n_seeds_{pct}"])),
                    format_gain_ci(r[f"gain_{pct}_percent"], r[f"gain_{pct}_ci95_low"],
                                    r[f"gain_{pct}_ci95_high"])]
        rows.append(cells)

    # Reviewer 1, comment 1: the same gains measured against the best target-only model.
    best = extra["best_target_only"]
    best_names = [MODEL_NAMES_IN_TEXT[one_row(best, base_configuration_id=TRANSFER_MODELS[0],
                                              train_data_percent=pct).best_target_only]
                  for pct in SINGLE_SOURCE_PERCENTAGES]
    if len(set(best_names)) == 1:
        best_heading = f"Gain over the best target-only model ({best_names[0]})"
    else:
        best_heading = "Gain over the best target-only model"
    rows += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{best_heading}}}}} \\"]
    for model in TRANSFER_MODELS:
        cells = [labels[model]]
        for pct in SINGLE_SOURCE_PERCENTAGES:
            r = one_row(best, base_configuration_id=model, train_data_percent=pct)
            cells += ["", format_gain_ci(r.gain_percent, r.ci95_low, r.ci95_high)]
        rows.append(cells)

    gain_header = (r"Method & NRMSE & \shortstack{Transfer gain (\%)\\{[95\% interval]}} "
                   r"& NRMSE & \shortstack{Transfer gain (\%)\\{[95\% interval]}} \\")
    main.append(write_table(
        paths.main_tables_dir / "table_3_principal_model_comparison.tex",
        "Principal model comparison at limited and full target training data",
        "tab:principal_model_comparison", r"@{}lcccc@{}",
        [*fraction_columns(2, stacked=True), gain_header],
        rows,
        r"NRMSE is the primary person-, outcome- and dataset-balanced error; lower values "
        r"indicate better performance. Neural-model values are mean (SD) across three runs; "
        r"conventional baselines were run once. Pooled and pretrained STraTS use all six "
        r"sources. Relative gains compare macro NRMSE with the reference identified in each "
        r"block. Brackets show descriptive 95\% percentile intervals from 10\,000 bootstrap "
        r"resamples of the seven target datasets, after averaging errors across runs within "
        r"each dataset."
    ))

    rows = []
    for _, r in results.suitability_summary.iterrows():
        rows.append([f"{int(r.train_data_percent)}\\%",
                     f"{r.mean_target_spearman_rho:.2f} [{r.mean_target_spearman_ci95_low:.2f}, {r.mean_target_spearman_ci95_high:.2f}]",
                     f"{int(r.top1_hits)}/{int(r.n_targets)}",
                     f"{r.mean_regret_percentage_points:.1f}"])
    main.append(write_table(
        paths.main_tables_dir / "table_4_suitability_ranking.tex",
        "Evaluation of the fixed source-suitability ranking", "tab:supp_suitability_validation",
        r"@{}crrr@{}",
        [r"\shortstack{Target training\\participants} & \shortstack{Mean within-target\\Spearman $\rho$ [95\% interval]} & \shortstack{Top-ranked source\\was empirically best} & \shortstack{Mean regret\\(percentage points)} \\"],
        rows,
        r"Each correlation compares suitability scores with observed single-source gains across "
        r"six sources within one target. Brackets show a descriptive 95\% interval obtained by "
        r"bootstrapping the seven targets. Regret is the best observed single-source gain minus "
        r"the gain of the heuristic-first source, in percentage points. The empirical best is "
        r"identified retrospectively from test results; this is not a validated selection "
        r"procedure, and no-transfer is not a candidate."
    ))

    ps = results.panel_sensitivity
    rows = []
    for panel in [PRIMARY_PANEL, COMMON_PANEL, EXTENDED_PANEL]:
        cells = [PANEL_LABELS[panel]]
        for pct in SINGLE_SOURCE_PERCENTAGES:
            for model in ["strats_pooled_all_sources", "strats_main"]:
                row = ps[(ps.panel == panel) & (ps.train_data_percent == pct) &
                         (ps.base_configuration_id == model)].iloc[0]
                cells.append(f"{row.relative_gain_vs_target_only_strats_percent:.1f}")
        rows.append(cells)
    main.append(write_table(
        paths.main_tables_dir / "table_5_outcome_panel_sensitivity.tex",
        "Transfer gains under alternative outcome-panel definitions", "tab:supp_panel_sensitivity",
        r"@{}lrrrr@{}",
        [*fraction_columns(2, stacked=True),
         r"Evaluation panel & Pooled (\%) & Pretrained (\%) & Pooled (\%) & Pretrained (\%) \\"],
        rows,
        r"Gains (\%) are relative to target-only STraTS within the same panel; both transfer "
        r"strategies use all six sources. Panels re-evaluate the same fitted 20-output models "
        r"without retraining. The common panel contains height, weight and BMI. The extended "
        r"panel includes numerically encoded smoking and diabetes where eligible and is not a "
        r"validated clinical composite. Macro NRMSEs, run-to-run SDs and GRU-D results are "
        rf"provided in Additional file 1, {supp_ref('panel_errors')}."
    ))

    # Editor comment 2 and the reviewers' experiments: one aggregate summary in the main paper.
    summary = extra["additional_analyses"]
    rows = []
    for analysis, heading in ADDITIONAL_ANALYSES:
        if rows:
            rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{heading}}}}} \\")
        reference = ADDITIONAL_GAIN_REFERENCES[analysis]
        rows.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\footnotesize Gain reference: {reference}.}} \\")
        block = summary[summary["analysis"].eq(analysis)]
        for model in block["model"].drop_duplicates():
            cells = [one_row(block, model=model, train_data_percent=SINGLE_SOURCE_PERCENTAGES[0]).label]
            for pct in SINGLE_SOURCE_PERCENTAGES:
                r = one_row(block, model=model, train_data_percent=pct)
                cells += [format_mean_sd(r.nrmse_mean, r.nrmse_sd),
                          format_gain_ci(r.gain_percent, r.ci95_low, r.ci95_high)]
            rows.append(cells)
    main.append(write_table(
        paths.main_tables_dir / "table_6_additional_analyses.tex",
        "Sensitivity analyses and transfer with a second architecture",
        "tab:additional_analyses", r"@{}lcccc@{}",
        [*fraction_columns(2, stacked=True),
         gain_header.replace("Method", "Model").replace("Transfer gain", "Relative gain")],
        rows,
        r"Primary macro NRMSE is reported as mean (SD) across three runs. Gains are relative to "
        r"the reference identified in each block. Brackets show descriptive 95\% percentile "
        r"intervals from 10\,000 bootstrap resamples of the contributing targets. Reset "
        r"re-estimates standardization on target training data. Source-exclusion and "
        r"normalization controls are matched reruns of the main-analysis procedures; the "
        r"fixed-rate target-only models are also reruns. GRU-D uses a fixed 20-attribute "
        r"layout, source-cohort exclusions and scale-consistent transfer; its target-only "
        r"reference therefore differs from Table~\ref{tab:principal_model_comparison}. "
        r"Procedures and dataset-specific results are provided in "
        r"Section~\ref{Additional_sensitivity_analyses} and Additional file 1, Tables S15--S18.",
        tabcolsep="1.1pt"
    ))

    return main

# ---------------------------------------------------------------------------
# Figures (revision). One style, one vocabulary and one colour per model in
# every figure. Font sizes are points at the 170 mm design width. The
# manuscript class prints figures at about 139 mm, so no figure text is drawn
# below FIGURE_MIN_FONT_PT (8 pt here, about 6.5 pt in the manuscript).
# ---------------------------------------------------------------------------

FIGURE_MIN_FONT_PT = 8.0
MANUSCRIPT_TEXT_WIDTH_MM = 139.0
CELL_FONT_PT = 8.0
PANEL_LABEL_PT = 11.0

# The same words for the same things in every figure.
TRAINING_AXIS_LABEL = "Target training participants (%)"
MACRO_NRMSE_LABEL = "Primary macro NRMSE"
DATASET_NRMSE_LABEL = "Primary-panel NRMSE"
GAIN_LABEL = "Relative gain over target-only STraTS (%)"
GAIN_TICK_STEP = 10.0
NOT_APPLICABLE_GREY = "0.88"
MINUS_SIGN = chr(0x2212)   # typographic minus, as on the Matplotlib axes
EN_DASH = chr(0x2013)      # marks a cell that does not apply

# Colour follows the model in every figure. These Okabe-Ito hues stay
# distinguishable for every pair under protanopia and deuteranopia; the
# Matplotlib default green/orange pair used before did not. Each model also
# keeps its own marker (MODEL_LINE_STYLES), so identity never rests on colour
# alone.
MODEL_COLORS = {
    "gru_d_target_only": "#0072B2",
    "strats_no_pretrain": "#E69F00",
    "strats_pooled_all_sources": "#009E73",
    "strats_main": "#D55E00",
}

# (strategy name in the derived data, model identity for colour/marker, label)
SOURCE_COUNT_SERIES = [
    ("Pretraining", "strats_main", "STraTS pretrained"),
    ("Pooled training", "strats_pooled_all_sources", "STraTS pooled"),
]
DATASET_GAIN_PANELS = [
    ("Pretraining", "All-source pretraining"),
    ("Pooled training", "All-source pooled training"),
]


def training_fraction_label(percent: int, two_lines: bool = False) -> str:
    joiner = "\n" if two_lines else " "
    return f"{int(percent)}% of target{joiner}training participants"


def two_line_label(label: str, longest_line: int = 10) -> str:
    """Break a long dataset name at its first space (heatmap row labels)."""
    return label.replace(" ", "\n", 1) if len(label) > longest_line else label


def figure_size(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm * MM_TO_INCH, height_mm * MM_TO_INCH


def figure_canvas(height_mm: float = 108.0) -> tuple[plt.Figure, plt.Axes]:
    """One plot per figure; publication width and no external style packages."""
    fig = plt.figure(figsize=figure_size(FULL_WIDTH_MM, height_mm))
    ax = fig.add_subplot(111)
    return fig, ax


def axes_mm(fig: plt.Figure, left_mm: float, bottom_mm: float,
            width_mm: float, height_mm: float) -> plt.Axes:
    """Place axes by millimetres so that cell and margin sizes are explicit."""
    width_in, height_in = fig.get_size_inches()
    return fig.add_axes([
        left_mm * MM_TO_INCH / width_in, bottom_mm * MM_TO_INCH / height_in,
        width_mm * MM_TO_INCH / width_in, height_mm * MM_TO_INCH / height_in,
    ])


def finish_chart(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)


def add_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Bold panel letter followed by a left-aligned title above the axes.

    Both start at the left edge of the axes, so a long title can never run
    into the letter of the neighbouring panel.
    """
    title_size = float(plt.rcParams["axes.titlesize"])
    line_spacing = 1.15
    n_lines = title.count("\n") + 1
    top = 5.0 + n_lines * title_size * 1.2 * line_spacing
    common = {"xy": (0.0, 1.0), "xycoords": "axes fraction", "textcoords": "offset points",
              "ha": "left", "va": "top"}
    ax.annotate(letter, xytext=(0.0, top + 1.0), fontsize=PANEL_LABEL_PT, fontweight="bold", **common)
    ax.annotate(title, xytext=(1.25 * PANEL_LABEL_PT, top), fontsize=title_size,
                linespacing=line_spacing, **common)


def plot_model_series(ax: plt.Axes, x: Any, mean: Any, sd: Any, model: str, label: str,
                      connect: bool = True) -> None:
    """Mean line and markers over SD error bars, encoded identically everywhere."""
    color = MODEL_COLORS[model]
    linestyle, marker = MODEL_LINE_STYLES[model]
    if not connect:
        linestyle = "none"
    x_values = np.asarray(x, dtype=float)
    means = np.asarray(mean, dtype=float)
    ax.errorbar(x_values, means, yerr=np.asarray(sd, dtype=float), fmt="none", ecolor=color,
                elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=2)
    ax.plot(x_values, means, linestyle=linestyle, marker=marker, color=color,
            markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)


def smallest_text_pt(fig: plt.Figure) -> float:
    sizes = [text.get_fontsize() for text in fig.findobj(Text)
             if text.get_visible() and text.get_text().strip()]
    return float(min(sizes)) if sizes else float("nan")


def save_figure(fig: plt.Figure, base: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    fig.canvas.draw()
    smallest = smallest_text_pt(fig)
    printed = smallest * MANUSCRIPT_TEXT_WIDTH_MM / FULL_WIDTH_MM
    if smallest < FIGURE_MIN_FONT_PT - 1e-6:
        logging.warning("%s: text of %.2f pt is below the %.1f pt minimum.",
                        base.name, smallest, FIGURE_MIN_FONT_PT)
    logging.info("%s: smallest text %.1f pt at %.0f mm (%.1f pt at the %.0f mm manuscript width)",
                 base.name, smallest, FULL_WIDTH_MM, printed, MANUSCRIPT_TEXT_WIDTH_MM)
    out = []
    for fmt in formats:
        path = base.with_suffix("." + fmt)
        # Physical width stays exactly 170 mm; margins are set in each figure.
        fig.savefig(path, dpi=dpi, facecolor="white",
                    metadata={"Creator": "generate_paper_plots_tables.py"})
        out.append(path)
    plt.close(fig)
    return out


def draw_pipeline(paths: ProjectPaths, formats: Sequence[str], dpi: int) -> list[Path]:
    fig, ax = figure_canvas(104)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    def box(x: float, y: float, w: float, h: float, text: str, adapted: bool = False) -> None:
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                    boxstyle="round,pad=0.006,rounding_size=0.012", fill=False,
                    linewidth=1.15, linestyle="--" if adapted else "-"))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2, linespacing=1.35)

    def arrow(a: tuple[float, float], b: tuple[float, float]) -> None:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                    linewidth=1.0, shrinkA=0, shrinkB=0, color="black"))

    xs = [0.10, 0.30, 0.50, 0.70, 0.90]
    labels = ["Target-relative\nmeasurement\ntriplets", "Variable +\nvalue + time\nembeddings",
              "Two\nTransformer\nblocks", "Attention\npooling", "20-output\nregression\nhead"]
    for k, (x, text) in enumerate(zip(xs, labels)):
        box(x, 0.60, 0.165, 0.22, text, adapted=(k in {0, 4}))
        if k: arrow((xs[k-1]+0.091, 0.60), (x-0.091, 0.60))

    ax.text(0.10, 0.89, "Earlier visits\n+ requested future age", ha="center", va="center", fontsize=8.4)
    arrow((0.10, 0.80), (0.10, 0.72))
    ax.text(0.105, 0.385, "Observed values only\nTime: " + r"$(a_i-a_t)/s_h$" +
            "\nVariable dropout\nduring training", ha="center", va="top", fontsize=8.0)

    box(0.43, 0.27, 0.34, 0.19, "Seven contextual inputs\nSex, ages, horizon and history size\nTwo-layer feed-forward network", True)
    # Two representations are concatenated before the head.
    arrow((0.607, 0.27), (0.90, 0.27))
    arrow((0.90, 0.27), (0.90, 0.478))
    ax.text(0.743, 0.30, "Concatenate", ha="center", va="bottom", fontsize=8.0)
    arrow((0.90, 0.72), (0.90, 0.81))
    ax.text(0.90, 0.91, "Future values at " + r"$a_t$" + "\nObserved targets\ncontribute to loss", ha="center", va="center", fontsize=8.2)
    ax.text(0.45, 0.045, "Solid outline: STraTS representation    Dashed outline: task-specific adaptation",
            ha="center", va="center", fontsize=8.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
    return save_figure(fig, paths.main_figures_dir / FIGURES[0][1], formats, dpi)


def draw_learning_curves(paths: ProjectPaths, data: StudyData,
                         formats: Sequence[str], dpi: int) -> list[Path]:
    logging.info("Creating main Figure 2: learning curves")
    fig, ax = figure_canvas(116)
    table = data.model_seed.query("panel == @PRIMARY_PANEL and horizon_group == @ALL_HORIZONS")
    # Keep the transfer comparison readable; conventional comparators remain
    # in Tables 3 and S2.
    neural_curves = ["gru_d_target_only", "strats_no_pretrain",
                     "strats_pooled_all_sources", "strats_main"]
    for base in neural_curves:
        block = table.query("base_configuration_id == @base").sort_values("train_data_percent")
        plot_model_series(ax, block.train_data_percent, block.main_macro_nrmse_mean,
                          block.main_macro_nrmse_std.fillna(0), base, MODEL_LABELS[base])
    ax.set_xlabel(TRAINING_AXIS_LABEL)
    ax.set_ylabel(MACRO_NRMSE_LABEL)
    ax.set_xticks(TRAIN_PERCENTAGES)
    ax.set_xlim(7, 103)
    finish_chart(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.49, -0.22), ncol=2, frameon=False,
              columnspacing=1.1, handlelength=2.5)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.30)
    return save_figure(fig, paths.main_figures_dir / FIGURES[1][1], formats, dpi)


def topk_base_id(strategy: str, k: int) -> str:
    if k == 0:
        return "strats_no_pretrain"
    if strategy == "Pretraining":
        return "strats_main" if k == 6 else f"strats_dataset_pretrain_{k}"
    if strategy == "Pooled training":
        return "strats_pooled_all_sources" if k == 6 else f"strats_pooled_top_{k}"
    raise ValueError(f"Unknown strategy: {strategy}")


def draw_source_count(
    paths: ProjectPaths,
    data: StudyData,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    logging.info("Creating main Figure 3: number of source datasets")
    rows: list[dict[str, Any]] = []
    for percent in TOPK_PERCENTAGES:
        for strategy in ["Pretraining", "Pooled training"]:
            for k in range(0, 7):
                base_id = topk_base_id(strategy, k)
                model_row = get_model_seed_row(data.model_seed, base_id, percent)
                rows.append(
                    {
                        "train_data_percent": percent,
                        "strategy": strategy,
                        "n_source_datasets": k,
                        "base_configuration_id": base_id,
                        "main_macro_nrmse_mean": float(model_row["main_macro_nrmse_mean"]),
                        "main_macro_nrmse_sd": float(model_row["main_macro_nrmse_std"]),
                        "n_seeds": int(model_row["n_seeds"]),
                    }
                )
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=figure_size(FULL_WIDTH_MM, 88.0), sharey=True)
    for panel_index, (ax, percent) in enumerate(zip(axes, TOPK_PERCENTAGES)):
        zero = None
        for strategy, model, label in SOURCE_COUNT_SERIES:
            block = frame[
                frame["train_data_percent"].eq(percent)
                & frame["strategy"].eq(strategy)
            ].sort_values("n_source_datasets")
            zero = block[block["n_source_datasets"].eq(0)]
            transfer = block[block["n_source_datasets"].gt(0)]
            # The line starts at the shared zero-source point (target-only STraTS).
            ax.plot(block["n_source_datasets"].to_numpy(dtype=float)[:2],
                    block["main_macro_nrmse_mean"].to_numpy(dtype=float)[:2],
                    linestyle=MODEL_LINE_STYLES[model][0], color=MODEL_COLORS[model], zorder=1.9)
            plot_model_series(ax, transfer["n_source_datasets"], transfer["main_macro_nrmse_mean"],
                              transfer["main_macro_nrmse_sd"].fillna(0), model, label)
        # Zero sources is target-only STraTS: same colour and marker as in Figure 2.
        plot_model_series(ax, zero["n_source_datasets"], zero["main_macro_nrmse_mean"],
                          zero["main_macro_nrmse_sd"].fillna(0), "strats_no_pretrain",
                          MODEL_LABELS["strats_no_pretrain"], connect=False)
        ax.set_xticks(range(0, 7), labels=["0", "1", "2", "3", "4", "5", "6\n(all)"])
        ax.set_xlim(-0.4, 6.4)
        finish_chart(ax)
        add_panel_title(ax, chr(ord("A") + panel_index), training_fraction_label(percent, two_lines=True))
    # Same quantity and the same axis label as Figure 2.
    axes[0].set_ylabel(MACRO_NRMSE_LABEL)
    fig.supxlabel("Number of source datasets, added in the fixed heuristic order",
                  fontsize=plt.rcParams["axes.labelsize"], y=0.03)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [len(handles) - 1, *range(len(handles) - 1)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.5, columnspacing=2.0)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.175, top=0.76, wspace=0.10)
    return save_figure(fig, paths.main_figures_dir / "figure_2_number_of_sources", formats, dpi)


def format_gain(value: float) -> str:
    """One decimal, a typographic minus and no plus sign; never '-0.0'."""
    rounded = round(float(value), 1)
    if rounded == 0.0:
        return "0.0"
    return f"{rounded:.1f}".replace("-", MINUS_SIGN)


def gain_color_scale(results: DerivedResults) -> tuple[TwoSlopeNorm, Any, float]:
    """One symmetric diverging scale shared by Figures 4 and 5.

    The limit is the largest absolute gain in either figure, rounded up to the
    colour-bar tick step, so every value lies inside the labelled range.
    """
    largest = max(
        float(np.nanmax(np.abs(frame["relative_gain_percent_mean"].to_numpy(dtype=float))))
        for frame in (results.top6_dataset_gain, results.single_source_gain)
    )
    limit = max(GAIN_TICK_STEP, math.ceil(largest / GAIN_TICK_STEP - 1e-9) * GAIN_TICK_STEP)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = matplotlib.colormaps["RdBu"].copy()
    cmap.set_bad(NOT_APPLICABLE_GREY)
    return norm, cmap, limit


def draw_gain_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    norm: TwoSlopeNorm,
    cmap: Any,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    rotate_x: bool,
) -> tuple[Any, list[Text]]:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                      aspect="auto", interpolation="nearest")
    if rotate_x:
        ax.set_xticks(np.arange(len(x_labels)), labels=x_labels, rotation=45,
                      ha="right", rotation_mode="anchor")
    else:
        ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels, linespacing=1.1)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="0.72", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=2.5)
    texts: list[Text] = []
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                label, text_color = EN_DASH, "0.40"
            else:
                label = format_gain(value)
                rgba = cmap(norm(float(value)))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "black" if luminance > 0.55 else "white"
            texts.append(ax.text(col_index, row_index, label, ha="center", va="center",
                                 fontsize=CELL_FONT_PT, color=text_color))
    return image, texts


def fit_cell_text(fig: plt.Figure, ax: plt.Axes, texts: Sequence[Text],
                  n_columns: int, max_fraction: float = 0.90) -> float:
    """Shrink cell numbers only if the widest one would come close to its cell border."""
    renderer = fig.canvas.get_renderer()
    cell_width = ax.get_window_extent(renderer).width / n_columns
    size = CELL_FONT_PT
    while size > 6.0 and max(t.get_window_extent(renderer).width for t in texts) > max_fraction * cell_width:
        size -= 0.25
        for text in texts:
            text.set_fontsize(size)
    return size


def draw_gain_heatmap_pair(
    matrices: Sequence[pd.DataFrame],
    titles: Sequence[str],
    x_labels: Sequence[str],
    x_label: str,
    scale: tuple[TwoSlopeNorm, Any, float],
    rotate_x: bool,
    height_mm: float,
    bottom_mm: float,
) -> plt.Figure:
    """Two heatmaps with target datasets as rows and one shared colour bar."""
    norm, cmap, limit = scale
    left_mm, top_mm, gap_mm, bar_gap_mm, bar_mm, right_mm = 22.0, 9.5, 3.5, 3.0, 3.5, 14.5
    panel_mm = (FULL_WIDTH_MM - left_mm - gap_mm - bar_gap_mm - bar_mm - right_mm) / 2.0
    plot_mm = height_mm - top_mm - bottom_mm
    # Long dataset names take two lines so that the cells can be wider.
    y_labels = [two_line_label(DATASET_LABELS[dataset]) for dataset in DATASET_ORDER]
    fig = plt.figure(figsize=figure_size(FULL_WIDTH_MM, height_mm))
    image = None
    texts: list[Text] = []
    first_ax = None
    for panel_index, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axes_mm(fig, left_mm + panel_index * (panel_mm + gap_mm), bottom_mm, panel_mm, plot_mm)
        image, cell_texts = draw_gain_heatmap(ax, matrix, norm, cmap, x_labels, y_labels, rotate_x)
        texts += cell_texts
        ax.set_xlabel(x_label)
        if panel_index == 0:
            ax.set_ylabel("Target dataset")
            first_ax = ax
        else:
            ax.tick_params(labelleft=False)
        add_panel_title(ax, chr(ord("A") + panel_index), title)
    cax = axes_mm(fig, left_mm + 2.0 * panel_mm + gap_mm + bar_gap_mm, bottom_mm, bar_mm, plot_mm)
    # Ticks at every step up to the limit: the largest value is inside the labelled range.
    colorbar = fig.colorbar(image, cax=cax,
                            ticks=np.arange(-limit, limit + 0.5 * GAIN_TICK_STEP, GAIN_TICK_STEP))
    colorbar.set_label(GAIN_LABEL)
    colorbar.outline.set_linewidth(0.7)
    fit_cell_text(fig, first_ax, texts, len(x_labels))
    return fig


def draw_dataset_gains(
    paths: ProjectPaths,
    results: DerivedResults,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    logging.info("Creating main Figure 4: dataset-specific all-source transfer gains")
    frame = results.top6_dataset_gain.copy()
    matrices = [
        frame[frame["strategy"].eq(strategy)]
        .pivot(index="dataset", columns="train_data_percent", values="relative_gain_percent_mean")
        .reindex(index=DATASET_ORDER, columns=TRAIN_PERCENTAGES)
        for strategy, _ in DATASET_GAIN_PANELS
    ]
    fig = draw_gain_heatmap_pair(
        matrices, [title for _, title in DATASET_GAIN_PANELS],
        [str(x) for x in TRAIN_PERCENTAGES], TRAINING_AXIS_LABEL,
        gain_color_scale(results), rotate_x=False, height_mm=90.0, bottom_mm=12.5,
    )
    return save_figure(fig, paths.main_figures_dir / "figure_S1_dataset_specific_top6_transfer", formats, dpi)


def draw_single_source_pair(
    paths: ProjectPaths,
    results: DerivedResults,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    logging.info("Creating main Figure 5: single-source transfer gains")
    frame = results.single_source_gain
    matrices = [
        frame[frame["train_data_percent"].eq(percent)]
        .pivot(index="dataset", columns="source_dataset", values="relative_gain_percent_mean")
        .reindex(index=DATASET_ORDER, columns=DATASET_ORDER)
        for percent in SINGLE_SOURCE_PERCENTAGES
    ]
    fig = draw_gain_heatmap_pair(
        matrices, [training_fraction_label(percent) for percent in SINGLE_SOURCE_PERCENTAGES],
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER], "Source dataset",
        gain_color_scale(results), rotate_x=True, height_mm=104.0, bottom_mm=27.5,
    )
    return save_figure(fig, paths.main_figures_dir / "figure_3_single_source_transfer", formats, dpi)


def draw_outcome_gains(paths: ProjectPaths, results: DerivedResults,
                       attributes: Sequence[str], formats: Sequence[str], dpi: int) -> list[Path]:
    logging.info("Creating main Figure 6: outcome-specific gains")
    fig, ax = figure_canvas(126)
    y = np.arange(len(attributes))
    count_labels = []
    # One model, so one colour; the training fraction is carried by the marker.
    color = MODEL_COLORS["strats_main"]
    ax.axvline(0, color="0.35", linewidth=0.8, zorder=1)
    for pct, offset, marker, face in [(10, -0.13, "o", color), (100, 0.13, "s", "white")]:
        block = results.attribute_gain.query("train_data_percent == @pct").set_index("attribute").reindex(attributes)
        values = block.relative_gain_percent_mean.to_numpy(float)
        errors = np.vstack([values-block.ci95_low.to_numpy(float), block.ci95_high.to_numpy(float)-values])
        ax.errorbar(values, y+offset, xerr=errors, fmt="none", ecolor=color,
                    elinewidth=1.1, capsize=2.5, capthick=1.0, zorder=2)
        ax.plot(values, y+offset, linestyle="none", marker=marker, markersize=5.5,
                markerfacecolor=face, markeredgecolor=color, markeredgewidth=1.2,
                label=training_fraction_label(pct), zorder=3)
        if pct == 10:
            count_labels = [f"{ATTRIBUTE_LABELS[a]} (n={int(block.loc[a, 'n_datasets'])})" for a in attributes]
    ax.set_yticks(y, count_labels)
    ax.set_ylim(len(attributes)-0.5, -0.5)
    ax.set_xlabel("Mean " + GAIN_LABEL[0].lower() + GAIN_LABEL[1:])
    finish_chart(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.14), ncol=1, frameon=False)
    fig.subplots_adjust(left=0.355, right=0.98, top=0.97, bottom=0.23)
    return save_figure(fig, paths.main_figures_dir / FIGURES[5][1], formats, dpi)


def write_figure_legends(paths: ProjectPaths, formats: Sequence[str]) -> Path:
    """Revised captions and widths; labels and Overleaf paths are unchanged."""
    fmt = "pdf" if "pdf" in formats else formats[0]
    lines = [
        "% Figure blocks synchronized with the supplied Paper/current paper LaTeX.",
        "% All captions, labels, inclusion widths and placements match that version.",
        "% Figure artwork is unchanged; all seven figures use width=\\textwidth.",
        "% The older filenames are intentionally retained; manuscript numbering comes from LaTeX.",
        "% Upload paths below match the current Overleaf project, using lowercase figures/main/.",
        "% See file_placement.txt for the mapping from this folder's lowercase figures/ outputs.",
        "",
    ]
    for scope, stem, label in FIGURES:
        title, legend = FIGURE_CAPTIONS[label]
        caption = " ".join(part for part in (title, legend) if part)
        placement = FIGURE_PLACEMENT[label]
        overleaf_path = Path(placement["overleaf_path"]).with_suffix("." + fmt).as_posix()
        if scope == "supplementary":
            lines += ["% Supplementary figure: already embedded in supplementary_information.tex."]
        lines += [rf"\begin{{figure}}[{placement['placement']}]", r"\centering",
                  rf"\includegraphics[width={placement['width']}]{{{overleaf_path}}}",
                  rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{figure}", ""]
    path = paths.output_dir / "figure_legends.tex"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\r\n")
    return path


def ensure_safe_output_path(paths: ProjectPaths, target: Path) -> None:
    """Never follow an output-tree symlink into unrelated files."""
    output = paths.output_dir.resolve()
    if not target.resolve().is_relative_to(output):
        raise ValueError(f"Output path resolves outside the output directory: {target}")
    for item in [target, *target.parents]:
        if item == paths.output_dir:
            break
        if item.is_symlink():
            raise ValueError(f"Refusing to write through a symlink: {item}")

# ---------------------------------------------------------------------------
# Supplementary results. These use the current metric files, not prior exports.
# The main-paper derivation above is intentionally kept in its original order.
# ---------------------------------------------------------------------------

NEURAL_MODELS = [
    "gru_d_target_only", "strats_no_pretrain",
    "strats_pooled_all_sources", "strats_main",
]
TRANSFER_MODELS = ["strats_pooled_all_sources", "strats_main"]
SOURCE_CODES = {
    "ABCD": "A", "BE_KUL_Wave_1A": "B", "BE_KUL_Wave_1B": "G",
    "BE_KUL_Wave_23456": "L", "SLOfit_lifelong": "F",
    "SLOfit_students": "S", "YFS": "Y",
}
SOURCE_CODE_NOTE = (
    "A: ABCD; B: LGS boys; G: LGS girls; L: Leuven adults; "
    "F: SLOfit lifelong; S: SLOfit students; Y: YFS."
)
GAIN_NOTE = (
    r"Gains (\%) are paired with target-only STraTS within run and dataset, "
    r"then averaged over runs and equally over contributing datasets. "
    r"Brackets: descriptive 95\% dataset-bootstrap intervals "
    r"(10\,000 resamples). Positive gains indicate improvement."
)

# (key, file stem, change against the submitted supplement). The position is
# the table number, and every reference to a supplementary table is built from
# this list. Tables S1-S7 and S9-S13 keep their submitted numbers: the removed
# observation-limit table (formerly S8) is replaced by a new table, and the
# other new tables follow S13.
SUPPLEMENTARY_TABLES = [
    ("horizons", "table_S1_dataset_and_horizon_details",
     "changed: share of all pairs and horizon-group shares added (Reviewer 1.3)"),
    ("baselines", "table_S2_conventional_baselines", "unchanged"),
    ("source_order", "table_S3_fixed_source_order", "unchanged"),
    ("dataset_gains", "table_S4_dataset_specific_transfer", "unchanged"),
    ("ranking_by_target", "table_S5_ranking_by_target", "training-fraction wording only"),
    ("outcome_gains", "table_S6_outcome_specific_transfer", "unchanged"),
    ("horizon_gains", "table_S7_horizon_specific_transfer", "unchanged"),
    ("excluded_cells", "table_S8_excluded_evaluation_cells",
     "NEW, replaces the observation-limit table: cells that failed the eligibility rule (Reviewer 1.4)"),
    ("panel_errors", "table_S9_panel_absolute_errors", "evaluation-panel wording only"),
    ("learning_curves", "table_S10_neural_learning_curves", "training-fraction wording only"),
    ("dataset_errors", "table_S11_dataset_absolute_errors", "unchanged"),
    ("source_count", "table_S12_source_count_results", "training-fraction wording only"),
    ("single_source", "table_S13_single_source_results", "training-fraction wording only"),
    ("downsampling", "table_S14_slofit_students_downsampling",
     "NEW: SLOfit students before and after downsampling (Reviewers 1.3 and 2.4)"),
    ("source_exclusion", "table_S15_source_exclusion_sensitivity",
     "current manuscript wording: main-analysis references; numerical content unchanged"),
    ("learning_rate", "table_S16_learning_rate_sensitivity",
     "current manuscript wording: main-analysis references; numerical content unchanged"),
    ("normalization", "table_S17_scale_consistent_transfer",
     "current manuscript wording: main-analysis procedure; numerical content unchanged"),
    ("grud_transfer", "table_S18_grud_transfer",
     "NEW: Experiment 4, GRU-D transfer (Reviewers 2.1 and 3.2)"),
    ("original_units", "table_S19_original_units_calibration",
     "NEW: errors in original units and calibration (Reviewers 2.5 and 3.5)"),
]


def supp_number(key: str) -> int:
    for number, (name, _, _) in enumerate(SUPPLEMENTARY_TABLES, start=1):
        if name == key:
            return number
    raise KeyError(f"Unknown supplementary table: {key}")


def supp_ref(key: str) -> str:
    return f"Table S{supp_number(key)}"


def supp_stem(key: str) -> str:
    return SUPPLEMENTARY_TABLES[supp_number(key) - 1][1]


def check_table_numbering() -> None:
    """File names and figure captions must agree with the numbers in SUPPLEMENTARY_TABLES."""
    for number, (key, stem, _) in enumerate(SUPPLEMENTARY_TABLES, start=1):
        if not stem.startswith(f"table_S{number}_"):
            raise ValueError(f"Supplementary table '{key}' is number {number} but its file is {stem}.")
    if supp_number("grud_transfer") - supp_number("source_exclusion") != 3:
        raise ValueError("The four experiment tables must be consecutive (they are cited as a range).")
    cited = {
        "fig:figure_2_number_of_sources": "source_order",
        "fig:figure_S1_dataset_specific_top6_transfer": "dataset_gains",
        "fig:figure_3_single_source_transfer": "single_source",
        "fig:supp_dataset_learning_curves": "dataset_errors",
    }
    for label, key in cited.items():
        if supp_ref(key) + "." not in " ".join(FIGURE_CAPTIONS[label]):
            raise ValueError(f"The caption of {label} no longer cites {supp_ref(key)}.")


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


def compute_supplementary_results(data: StudyData, results: DerivedResults) -> dict[str, Any]:
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

    return {
        "neural_global": neural.reset_index(drop=True),
        "dataset_absolute": dataset_absolute,
        "panels_absolute": all_panels,
        "outcome_gain": compute_supplementary_gains(data, results, "outcome"),
        "horizon_gain": compute_supplementary_gains(data, results, "horizon"),
        "source_count": source_count,
        "single_source": single_source,
    }


# ---------------------------------------------------------------------------
# SLOfit students before and after downsampling (Reviewers 1.3 and 2.4).
# ---------------------------------------------------------------------------

@dataclass
class DownsamplingSummary:
    attributes: list[str]         # structurally available in the downsampled dataset
    eligible: dict[str, float]    # cohort after preprocessing, before downsampling
    retained: dict[str, float]    # the analysis subset
    reference_dataset: str        # next-largest dataset, which set the retained size
    reference_values: int
    max_values: int               # most observed values of any eligible participant
    n_at_max: int
    retained_all_at_max: bool


def cohort_summary(visits: pd.DataFrame, attributes: Sequence[str]) -> dict[str, float]:
    people = visits.groupby("person", sort=False).agg(
        n_visits=("Age", "size"), first_age=("Age", "min"), last_age=("Age", "max"), female=("female", "first"))
    follow_up = people.last_age - people.first_age
    summary = {
        "n_people": float(len(people)), "n_visits": float(len(visits)),
        "visits_mean": people.n_visits.mean(), "visits_sd": people.n_visits.std(),
        "n_values": float(visits["observed"].sum()),
        "completeness": 100.0 * visits[list(attributes)].notna().to_numpy().mean(),
        "female": 100.0 * people.female.mean(),
        "first_age_mean": people.first_age.mean(), "first_age_sd": people.first_age.std(),
        "last_age_mean": people.last_age.mean(), "last_age_sd": people.last_age.std(),
        "follow_up_mean": follow_up.mean(), "follow_up_sd": follow_up.std(),
    }
    for attribute in attributes:
        summary[attribute + "_mean"] = visits[attribute].mean()
        summary[attribute + "_sd"] = visits[attribute].std()
    return {key: float(value) for key, value in summary.items()}


def compute_downsampling_summary(paths: ProjectPaths, data: StudyData,
                                 results: DerivedResults) -> DownsamplingSummary:
    """Rebuild the cohort from which SLOfit students was downsampled and describe both.

    The two preprocessing steps and the selection rule are those of
    preprocess_cohorts.py / prepare_data.py. The rebuilt selection
    must equal the participants in dataset.csv.
    """
    dataset = DOWNSAMPLED_DATASET
    logging.info("Rebuilding the %s cohort before downsampling (reads data/dataset_full.csv)",
                 DATASET_LABELS[dataset])
    attributes = list(data.extended_attributes)
    availability = results.attribute_availability.set_index("attribute")
    available = [a for a in attributes if bool(availability.loc[a, dataset])]

    # Read in pieces and keep only this dataset's visits, with participants as
    # integer codes in their order of first appearance in the file.
    codes: dict[str, int] = {}
    parts = []
    reader = pd.read_csv(paths.data_dir / "dataset_full.csv",
                         usecols=["Origin", "PersonID", "Sex", "Age", *attributes],
                         chunksize=500_000, low_memory=False)
    for chunk in reader:
        block = chunk[chunk["Origin"].eq(dataset)]
        if block.empty:
            continue
        if not block["Sex"].isin(["Female", "Male"]).all():
            raise ValueError("Unexpected Sex values in dataset_full.csv.")
        local, identifiers = pd.factorize(block["PersonID"].astype(str))
        person = np.array([codes.setdefault(identifier, len(codes)) for identifier in identifiers])[local]
        part = block[["Age", *available]].astype(float)
        part.insert(0, "person", person)
        part["female"] = block["Sex"].eq("Female").to_numpy()
        part["observed"] = block[attributes].notna().sum(axis=1).to_numpy()
        parts.append(part)
    if not parts:
        raise ValueError(f"dataset_full.csv contains no {dataset} visits.")
    visits = pd.concat(parts, ignore_index=True)
    del parts
    identifiers = np.array(list(codes))

    # Step 1: of two visits of one person at most 0.5 years apart, drop the one
    # with more missing values (the earlier one on a tie); repeat along the visits.
    person = visits["person"].to_numpy()
    age = visits["Age"].to_numpy()
    missing = len(attributes) - visits["observed"].to_numpy()
    order = np.lexsort((np.arange(len(visits)), age, person))   # person, then age, then file order
    person_sorted, age_sorted, missing_sorted = person[order], age[order], missing[order]
    same_person = person_sorted[1:] == person_sorted[:-1]
    near = same_person & (np.abs(age_sorted[1:] - age_sorted[:-1]) <= NEAR_DUPLICATE_YEARS)
    starts = np.flatnonzero(np.r_[True, ~same_person])
    ends = np.r_[starts[1:], len(order)]
    dropped: list[int] = []
    for k in np.unique(np.searchsorted(starts, np.flatnonzero(near) + 1, side="right") - 1):
        rows = order[starts[k]:ends[k]].tolist()
        ages = age_sorted[starts[k]:ends[k]].tolist()
        gaps = missing_sorted[starts[k]:ends[k]].tolist()
        i = 0
        while i < len(rows) - 1:
            if abs(ages[i + 1] - ages[i]) <= NEAR_DUPLICATE_YEARS:
                j = i + 1 if gaps[i + 1] > gaps[i] else i
                dropped.append(rows[j])
                del rows[j], ages[j], gaps[j]
            else:
                i += 1
    keep = np.ones(len(visits), dtype=bool)
    keep[dropped] = False
    # Step 2: people left with a single visit cannot form a prediction pair.
    keep &= np.bincount(person[keep], minlength=len(codes))[person] > 1
    visits = visits[keep]

    # Selection: participants in descending order of their number of observed
    # values (file order on ties) until the next-largest dataset is matched.
    analysis = pd.read_csv(paths.data_dir / "dataset.csv",
                           usecols=["Origin", "PersonID", *attributes], low_memory=False)
    values_by_dataset = analysis[attributes].notna().sum(axis=1).groupby(analysis["Origin"]).sum()
    retained_ids = set(analysis.loc[analysis["Origin"].eq(dataset), "PersonID"].astype(str))
    others = values_by_dataset.drop(dataset)
    reference_dataset, reference_values = str(others.idxmax()), int(others.max())
    appearance, people = pd.factorize(visits["person"])   # first appearance among the remaining visits
    values = np.bincount(appearance, weights=visits["observed"].to_numpy()).astype(np.int64)
    ranking = np.argsort(-values, kind="stable")
    n_keep = int(np.searchsorted(np.cumsum(values[ranking]), reference_values, side="left")) + 1
    selected = np.asarray(people)[ranking[:n_keep]]
    if n_keep > len(ranking) or set(identifiers[selected]) != retained_ids:
        raise ValueError(
            f"The {dataset} participants in dataset.csv are not those selected "
            "by the preprocess_cohorts.py rule from dataset_full.csv; the two files do not match."
        )
    retained = visits[visits["person"].isin(selected)]
    characteristics = results.dataset_characteristics.set_index("dataset").loc[dataset]
    if (len(retained) != int(characteristics.n_visits)
            or retained["person"].nunique() != int(characteristics.n_people)):
        raise ValueError(f"Rebuilt {dataset} subset does not have the visits of the analysis dataset.")
    return DownsamplingSummary(
        attributes=available,
        eligible=cohort_summary(visits, available),
        retained=cohort_summary(retained, available),
        reference_dataset=reference_dataset,
        reference_values=reference_values,
        max_values=int(values.max()),
        n_at_max=int((values == values.max()).sum()),
        retained_all_at_max=bool((values[ranking[:n_keep]] == values.max()).all()),
    )


def one_row(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    block = frame
    for key, value in filters.items():
        block = block[block[key].eq(value)]
    if len(block) != 1:
        raise ValueError(f"Expected one supplementary result for {filters}; found {len(block)}.")
    return block.iloc[0]


def group_heading(text: str, columns: int) -> str:
    # Starred row endings keep a heading with its following data block.
    return rf"\multicolumn{{{columns}}}{{@{{}}l}}{{\textit{{{text}}}}} \\*"


def keep_row(cells: Sequence[str], last: bool = False) -> str:
    return " & ".join(cells) + (r" \\" if last else r" \\*")


def format_plain(value: Any, digits: int = 1) -> str:
    """Fixed decimals without a negative zero (tables added in the revision)."""
    text = f"{float(value):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0.0 else text


def format_difference(mean: Any, sd: Any, digits: int = 3) -> str:
    """Signed paired difference with its run SD."""
    text = f"{float(mean):+.{digits}f}"
    if float(text) == 0.0:
        text = text[1:]
    return f"{text} ({float(sd):.{digits}f})"


def write_supplementary_table(
    path: Path, caption: str, label: str, alignment: str, columns: int,
    header: Sequence[str], rows: Sequence[Sequence[str] | str], note: str = ""
) -> Path:
    """Multipage, editable tables: repeated column headings and no font scaling."""
    lines = [r"\begingroup", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.14}",
             r"\setlength{\LTleft}{\fill}", r"\setlength{\LTright}{\fill}",
             rf"\begin{{longtable}}{{{alignment}}}",
             rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", *header, r"\midrule",
             r"\endfirsthead",
             rf"\multicolumn{{{columns}}}{{c}}{{\tablename~\thetable\ (continued)}}\\",
             r"\toprule", *header, r"\midrule", r"\endhead", r"\midrule",
             rf"\multicolumn{{{columns}}}{{r}}{{\footnotesize Continued on next page}}\\",
             r"\endfoot", r"\bottomrule"]
    if note:
        lines.append(r"\multicolumn{" + str(columns) + r"}{c}{\makebox[0pt][c]{\parbox[t]{\textwidth}{\footnotesize\textit{Note:} " + note + r"}}}\\")
    lines.append(r"\endlastfoot")
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + r" \\")
    lines += [r"\end{longtable}", r"\endgroup", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\r\n")
    return path


def write_supplementary_tables(
    paths: ProjectPaths, data: StudyData, results: DerivedResults,
    extra: Mapping[str, Any], revision: RevisionData
) -> list[Path]:
    written: dict[str, Path] = {}

    def emit(key: str, caption: str, label: str, alignment: str, columns: int,
             header: Sequence[str], rows: Sequence[Sequence[str] | str], note: str) -> None:
        path = paths.supplementary_tables_dir / (supp_stem(key) + ".tex")
        written[key] = write_supplementary_table(path, caption, label, alignment, columns, header, rows, note)

    training_column = r"\shortstack{Target training\\participants (\%)}"

    chars = results.dataset_characteristics.set_index("dataset")
    horizons = results.horizon_distribution.set_index("dataset")
    total_pairs = float(horizons.n_pairs.sum())
    rows = []
    for d in DATASET_ORDER:
        c, h = chars.loc[d], horizons.loc[d]
        rows.append([latex_escape(DATASET_LABELS[d]), f"{c.core_age_low:.1f}--{c.core_age_high:.1f}",
                     f"{int(c.n_eligible_primary_attributes)}/{len(data.primary_attributes)}",
                     f"{format_integer(h.n_pairs)} ({100.0 * h.n_pairs / total_pairs:.1f})",
                     f"{h.horizon_median:.1f} ({h.horizon_q1:.1f}--{h.horizon_q3:.1f})",
                     f"{h.horizon_min:.1f}--{h.horizon_max:.1f}",
                     "/".join(f"{h['pct_' + group]:.0f}" for group in ["short", "medium", "long"])])
    emit("horizons", "Dataset age coverage and forecast horizons", "tab:supp_horizon_distribution",
         r"@{}lrrrrrr@{}", 7,
         [r"Dataset & \shortstack{Core age\\range (years)} & \shortstack{Eligible primary\\outcomes} & \shortstack{Pairs\\(\% of all)} & \shortstack{Horizon median\\(IQR), years} & \shortstack{Horizon range\\(years)} & \shortstack{Short/medium/long\\horizon (\%)} \\"],
         rows, "Core ages are the reported mean age minus/plus one SD. Horizons summarize generated "
         "prefix-to-future pairs. The last column gives the rounded percentages of a dataset's pairs in the "
         "short, medium and long horizon groups, whose boundaries are tertiles of all pairs "
         f"({supp_ref('horizon_gains')}). IQR, interquartile range; SD, standard deviation.")

    baselines = [("Training-population mean", "persistence_train_mean"),
                 ("Age--sex mean", "persistence_age_sex_mean"),
                 ("LOCF with age--sex fallback", "persistence_locf_age_sex_mean"),
                 ("Linear regression", "linear_regression_fixed"),
                 ("Random forest", "random_forest_fixed")]
    rows = []
    for scope, title in [("target_only", "Target-only training"), ("pooled_all_datasets", "Pooled training")]:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(title, 7))
        for method_index, (name, base) in enumerate(baselines):
            cells = [name]
            for pct in TRAIN_PERCENTAGES:
                block = data.model_seed[
                    data.model_seed.base_configuration_id.eq(base + "_" + scope)
                    & data.model_seed.train_data_percent.eq(pct)
                    & data.model_seed.panel.eq(PRIMARY_PANEL)
                    & data.model_seed.horizon_group.eq(ALL_HORIZONS)
                ]
                if block.empty and base == "random_forest_fixed" and pct not in SINGLE_SOURCE_PERCENTAGES:
                    cells.append("--")
                    continue
                r = get_model_seed_row(data.model_seed, base + "_" + scope, pct)
                cells.append(format_mean_sd(r.main_macro_nrmse_mean, r.main_macro_nrmse_std, int(r.n_seeds)))
            rows.append(keep_row(cells, last=method_index == len(baselines) - 1))
    emit("baselines", "Conventional baselines across target training fractions", "tab:supp_baseline_comparison",
         r"@{}lrrrrrr@{}", 7,
         [r"& \multicolumn{6}{c}{Target training participants (\%)} \\", r"\cmidrule(lr){2-7}",
          "Method & " + " & ".join(str(p) for p in TRAIN_PERCENTAGES) + r" \\"], rows,
         "Values are primary macro NRMSEs; these baselines were run once. Pooled training uses all six "
         "source datasets. --: no experiment available. LOCF, last observation carried forward.")

    rows = []
    for d in DATASET_ORDER:
        ranking = results.source_suitability_rankings[
            results.source_suitability_rankings.target_dataset.eq(d)
        ].sort_values("source_suitability_rank")
        rows.append([latex_escape(DATASET_LABELS[d])] +
                    [f"{SOURCE_CODES[r.source_dataset]} ({r.source_suitability_score:.1f})" for _, r in ranking.iterrows()])
    emit("source_order", "Fixed source ordering and suitability scores", "tab:supp_fixed_source_order",
         r"@{}lrrrrrr@{}", 7, [r"Target & Rank 1 & Rank 2 & Rank 3 & Rank 4 & Rank 5 & Rank 6 \\"], rows,
         "Entries give source codes and pre-specified scores. " + SOURCE_CODE_NOTE +
         r" Top-$k$ experiments use the first $k$ sources in each row.")

    rows = []
    for d in DATASET_ORDER:
        for index, model in enumerate(TRANSFER_MODELS):
            cells = [latex_escape(DATASET_LABELS[d]) if index == 0 else "", "Pooled" if index == 0 else "Pretrained"]
            for pct in TRAIN_PERCENTAGES:
                r = one_row(results.top6_dataset_gain, dataset=d, base_configuration_id=model, train_data_percent=pct)
                cells.append(format_mean_sd(r.relative_gain_percent_mean, r.relative_gain_percent_sd, int(r.n_seeds), 1))
            rows.append(keep_row(cells, last=index == 1))
    emit("dataset_gains", "All-source transfer gains by target dataset and training fraction",
         "tab:supp_dataset_specific_transfer", r"@{}llrrrrrr@{}", 8,
         [r"& & \multicolumn{6}{c}{Target training participants (\%)} \\", r"\cmidrule(lr){3-8}",
          "Target & STraTS & " + " & ".join(str(p) for p in TRAIN_PERCENTAGES) + r" \\"], rows,
         r"Gains (\%): mean (SD) over three runs paired with target-only STraTS. Both strategies use "
         r"all six sources. Positive gains indicate improvement; negative gains indicate deterioration.")

    rows = []
    for pct in SINGLE_SOURCE_PERCENTAGES:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(fraction_text(pct), 6))
        for d in DATASET_ORDER:
            r = one_row(results.suitability_by_target, target_dataset=d, train_data_percent=pct)
            rows.append([latex_escape(DATASET_LABELS[d]), f"{r.spearman_rho:.2f}",
                         SOURCE_CODES[r.heuristic_top_source], SOURCE_CODES[r.empirical_best_source],
                         f"{r.heuristic_top_gain_percent:.1f}", f"{r.regret_percentage_points:.1f}"])
    emit("ranking_by_target", "Source-ranking results within each target dataset", "tab:supp_suitability_by_target",
         r"@{}lrrrrr@{}", 6,
         [r"Target & Spearman $\rho$ & \shortstack{Heuristic\\first} & \shortstack{Observed\\best} & \shortstack{Heuristic first\\gain (\%)} & \shortstack{Regret\\(points)} \\"], rows,
         f"Source codes: {supp_ref('source_order')}. Observed best is the largest test-set gain among six sources. "
         "Regret is observed-best gain minus heuristic-first gain, in percentage points. "
         "This is a retrospective comparison; no-transfer is not a candidate.")

    gain_header = [r"& & & \multicolumn{2}{c}{Gain (\%) [95\% interval]} \\", r"\cmidrule(lr){4-5}",
                   r"Outcome & " + training_column + r" & Datasets & Pooled & Pretrained \\"]
    rows = []
    for attribute in data.primary_attributes:
        if rows:
            rows.append(r"\midrule")
        for i, pct in enumerate(TRAIN_PERCENTAGES):
            pair = [one_row(extra["outcome_gain"], attribute=attribute, base_configuration_id=model, train_data_percent=pct)
                    for model in TRANSFER_MODELS]
            if pair[0].datasets != pair[1].datasets:
                raise ValueError(f"Different outcome contributors for the two transfer strategies: {attribute}, {pct}")
            cells = [latex_escape(ATTRIBUTE_LABELS[attribute]) if i == 0 else "", str(pct), str(int(pair[0].n_datasets))]
            cells += [format_gain_ci(r.relative_gain_percent_mean, r.ci95_low, r.ci95_high) for r in pair]
            rows.append(keep_row(cells, last=i == len(TRAIN_PERCENTAGES) - 1))
    emit("outcome_gains", "Outcome-specific gains from all-source pooling and pretraining",
         "tab:supp_attribute_gain", r"@{}lrrrr@{}", 5, gain_header, rows, GAIN_NOTE)

    hdef = data.manifest.get("horizon_definition", {})
    try:
        short_max, medium_max = float(hdef["short_max"]), float(hdef["medium_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing numeric horizon cut points in analysis_manifest.json.") from exc
    if not np.isfinite([short_max, medium_max]).all() or not 0 < short_max < medium_max:
        raise ValueError("Invalid horizon cut points in analysis_manifest.json.")
    rows = []
    for horizon in ["short", "medium", "long"]:
        if rows:
            rows.append(r"\midrule")
        for i, pct in enumerate(TRAIN_PERCENTAGES):
            pair = [one_row(extra["horizon_gain"], horizon_group=horizon, base_configuration_id=model, train_data_percent=pct)
                    for model in TRANSFER_MODELS]
            if pair[0].datasets != pair[1].datasets:
                raise ValueError(f"Different horizon contributors for the two transfer strategies: {horizon}, {pct}")
            cells = [horizon.capitalize() if i == 0 else "", str(pct), str(int(pair[0].n_datasets))]
            cells += [format_gain_ci(r.relative_gain_percent_mean, r.ci95_low, r.ci95_high) for r in pair]
            rows.append(keep_row(cells, last=i == len(TRAIN_PERCENTAGES) - 1))
    # Labels are read from the current aggregation manifest, not frozen in this file.
    horizon_note = (rf"Horizon $h$ in years: short, $h\leq {short_max:.3f}$; medium, "
                    rf"${short_max:.3f}<h\leq {medium_max:.3f}$; long, $h>{medium_max:.3f}$. "
                    r"Gains average run-paired gains equally across contributing datasets; "
                    r"brackets are descriptive 95\% dataset-bootstrap intervals (10\,000 resamples).")
    emit("horizon_gains", "All-source pooling and pretraining gains by forecast horizon", "tab:supp_horizon_gain",
         r"@{}lrrrr@{}", 5, [h.replace("Outcome", "Horizon") for h in gain_header], rows, horizon_note)

    # Reviewer 1, comment 4: which available cells failed the eligibility rule.
    eligibility = extra["eligibility"]
    rows = []
    for r in eligibility[eligibility.n_eligible < eligibility.n_folds].itertuples():
        name = latex_escape(ATTRIBUTE_LABELS[r.attribute]) + (r"$^{*}$" if r.attribute in data.primary_attributes else "")
        rows.append([latex_escape(DATASET_LABELS[r.dataset]), name, f"{r.n_eligible}/{r.n_folds}",
                     f"{r.development_min}--{r.development_max}", f"{r.test_min}--{r.test_max}"])
    if not rows:
        rows.append(r"\multicolumn{5}{c}{None} \\")
    emit("excluded_cells", "Dataset--attribute combinations excluded by the evaluation eligibility rule",
         "tab:supp_excluded_cells", r"@{}llrrr@{}", 5,
         [r"& & & \multicolumn{2}{c}{Observed future values per fold} \\", r"\cmidrule(lr){4-5}",
          r"Dataset & Attribute & \shortstack{Eligible\\folds} & \shortstack{Development\\participants} & \shortstack{Test\\participants} \\"],
         rows,
         "All structurally available dataset--attribute combinations (Table 2 of the article) that were not "
         "eligible for evaluation in all five folds. A fold was eligible when the reference SD was finite and "
         f"positive and at least {MIN_OBSERVED_TARGETS} observed values at unique future visits were available "
         "both among the non-test development participants and among the test participants. The last two "
         "columns give the range of these counts over the five folds. Combinations without an eligible fold "
         "were not evaluated; the others were evaluated in their eligible folds only. Every other available "
         "combination was eligible in all five folds. An asterisk identifies primary outcomes.")

    neural_headers = [training_column + r" & GRU-D & \shortstack{STraTS\\target-only} & \shortstack{STraTS\\pooled} & \shortstack{STraTS\\pretrained} \\"]
    rows = []
    for panel in [PRIMARY_PANEL, COMMON_PANEL, EXTENDED_PANEL]:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(latex_escape(PANEL_LABELS[panel]), 5))
        for i, pct in enumerate(TRAIN_PERCENTAGES):
            cells = [str(pct)]
            for model in NEURAL_MODELS:
                r = one_row(extra["panels_absolute"], panel=panel, base_configuration_id=model, train_data_percent=pct)
                cells.append(format_mean_sd(r.nrmse_mean, r.nrmse_sd, int(r.n_seeds)))
            rows.append(keep_row(cells, last=i == len(TRAIN_PERCENTAGES) - 1))
    emit("panel_errors", "Macro NRMSE across the three evaluation panels",
         "tab:supp_panel_absolute_errors", r"@{}rrrrr@{}", 5, neural_headers, rows,
         "Macro NRMSE: mean (SD) over three runs. Pooled and pretrained STraTS use all six sources. "
         "Panels evaluate the same fitted models; eligible outcomes differ by dataset. "
         "The common panel contains height, weight and BMI.")

    rows = []
    for pct in TRAIN_PERCENTAGES:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(fraction_text(pct), 3))
        for index, model in enumerate(NEURAL_MODELS):
            r = one_row(extra["neural_global"], base_configuration_id=model, train_data_percent=pct)
            gain = format_gain_ci(r.relative_gain_percent, r.ci95_low, r.ci95_high) if model in TRANSFER_MODELS else "--"
            rows.append(keep_row([latex_escape(MODEL_LABELS[model]),
                                  format_mean_sd(r.nrmse_mean, r.nrmse_sd, int(r.n_seeds)), gain],
                                 last=index == len(NEURAL_MODELS) - 1))
    emit("learning_curves", "GRU-D and STraTS performance across target training fractions",
         "tab:supp_neural_learning_curves", r"@{}lrr@{}", 3,
         [r"Method & Primary macro NRMSE & \shortstack{Transfer gain (\%)\\{[95\% interval]}} \\"], rows,
         r"Errors: mean (SD) over three runs. All-source transfer gains compare overall macro NRMSE "
         r"with target-only STraTS. Brackets: descriptive 95\% target-dataset bootstrap intervals "
         r"(10\,000 resamples). --: not an all-source STraTS transfer comparison.")

    rows = []
    for d in DATASET_ORDER:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(latex_escape(DATASET_LABELS[d]), 5))
        for i, pct in enumerate(TRAIN_PERCENTAGES):
            cells = [str(pct)]
            for model in NEURAL_MODELS:
                r = one_row(extra["dataset_absolute"], dataset=d, base_configuration_id=model, train_data_percent=pct)
                cells.append(format_mean_sd(r.nrmse_mean, r.nrmse_sd, int(r.n_seeds)))
            rows.append(keep_row(cells, last=i == len(TRAIN_PERCENTAGES) - 1))
    emit("dataset_errors", "Dataset-specific primary-panel NRMSE for GRU-D and STraTS",
         "tab:supp_dataset_absolute_errors", r"@{}rrrrr@{}", 5, neural_headers, rows,
         "Primary-panel NRMSE within each target dataset: mean (SD) over three runs. "
         "Pooled and pretrained STraTS use all six sources. These values are plotted in Figure S1.")

    rows = []
    for pct in TOPK_PERCENTAGES:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(fraction_text(pct), 3))
        for k in range(7):
            cells = [str(k)]
            for strategy in ["Pooled training", "Pretraining"]:
                r = one_row(extra["source_count"], train_data_percent=pct, n_sources=k, strategy=strategy)
                cells.append(format_mean_sd(r.nrmse_mean, r.nrmse_sd, int(r.n_seeds)))
            rows.append(keep_row(cells, last=k == 6))
    emit("source_count", "Prediction error as the number of source datasets increases",
         "tab:supp_source_count_results", r"@{}rrr@{}", 3,
         [r"Number of sources & Pooled STraTS & Pretrained STraTS \\"], rows,
         "Primary macro NRMSE: mean (SD) over three runs. Sources follow the fixed target-specific "
         f"order in {supp_ref('source_order')}. Zero sources denotes the same target-only STraTS reference in both columns.")

    rows = []
    for d in DATASET_ORDER:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading("Target: " + latex_escape(DATASET_LABELS[d]), 5))
        sources = [s for s in DATASET_ORDER if s != d]
        for index, source in enumerate(sources):
            cells = [latex_escape(DATASET_LABELS[source])]
            for pct in SINGLE_SOURCE_PERCENTAGES:
                r = one_row(extra["single_source"], dataset=d, source_dataset=source, train_data_percent=pct)
                cells += [format_mean_sd(r.transfer_nrmse_mean, r.transfer_nrmse_sd, int(r.n_seeds)),
                          format_mean_sd(r.relative_gain_percent_mean, r.relative_gain_percent_sd, int(r.n_seeds), 1)]
            rows.append(keep_row(cells, last=index == len(sources) - 1))
    emit("single_source", "Single-source pretraining errors and gains for every source--target pair",
         "tab:supp_single_source_results", r"@{}lrrrr@{}", 5,
         [*fraction_columns(2), r"Source & NRMSE & Gain (\%) & NRMSE & Gain (\%) \\"], rows,
         r"Primary-panel NRMSE and gains: mean (SD) over three runs. Gains are paired with target-only "
         r"STraTS within target dataset and run; positive values indicate improvement. "
         r"Source and target are always different datasets.")

    # Reviewers 1.3 and 2.4: how SLOfit students was downsampled and what was retained.
    sample: DownsamplingSummary = extra["downsampling"]
    sampled_label = DATASET_LABELS[DOWNSAMPLED_DATASET]

    def both(text: Any) -> list[str]:
        return [text(sample.eligible), text(sample.retained)]

    def mean_sd(name: str, digits: int = 1) -> Any:
        return lambda cohort: f"{cohort[name + '_mean']:.{digits}f} ({cohort[name + '_sd']:.{digits}f})"

    rows = [
        ["Participants", *both(lambda cohort: format_integer(cohort["n_people"]))],
        ["Visits", *both(lambda cohort: format_integer(cohort["n_visits"]))],
        ["Visits per participant, mean (SD)", *both(mean_sd("visits"))],
        ["Observed attribute values", *both(lambda cohort: format_integer(cohort["n_values"]))],
        [r"Completeness of available attributes (\%)", *both(lambda cohort: f"{cohort['completeness']:.1f}")],
        [r"Female participants (\%)", *both(lambda cohort: f"{cohort['female']:.1f}")],
        ["Age at first visit, years, mean (SD)", *both(mean_sd("first_age"))],
        ["Age at last visit, years, mean (SD)", *both(mean_sd("last_age"))],
        ["Follow-up, years, mean (SD)", *both(mean_sd("follow_up"))],
    ]
    for attribute in sample.attributes:
        rows.append([f"{latex_escape(ATTRIBUTE_LABELS[attribute])}, {ATTRIBUTE_UNITS[attribute]}, mean (SD)",
                     *both(mean_sd(attribute))])
    retained_text = (
        f"Every retained participant has the maximum of {sample.max_values} observed values, that is, "
        f"complete records at {sample.retained['visits_mean']:.0f} visits; {format_integer(sample.n_at_max)} "
        "eligible participants had this maximum, and those appearing first in the data file were retained. "
        if sample.retained_all_at_max and sample.retained["visits_sd"] == 0.0 else
        "Participants with equal numbers of observed values were taken in the order of the data file. "
    )
    emit("downsampling", f"{sampled_label} before and after downsampling", "tab:supp_downsampling",
         r"@{}lrr@{}", 3, [r"Characteristic & Eligible cohort & Retained subset \\"], rows,
         f"The eligible cohort is {sampled_label} after the preprocessing applied to all datasets: removal of "
         "near-duplicate visits and of participants with a single visit. Participants were ranked by their "
         "number of observed attribute values and retained from the top until the subset held at least as many "
         f"observed values as {DATASET_LABELS[sample.reference_dataset]}, the next-largest dataset "
         f"({format_integer(sample.reference_values)} values). " + retained_text +
         "The selection was deterministic and used no random seed. Completeness and attribute means (SD) "
         "are calculated over all visits.")

    # Editor comment 2 (Experiment 1).
    affected = [d for d in DATASET_ORDER if d in revision.exclusions]
    rows = []
    for mode, title in [("pretrain", "Pretraining"), ("pooled", "Pooled training")]:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(title, 8))
        for target in [*affected, None]:
            for i, pct in enumerate(SINGLE_SOURCE_PERCENTAGES):
                if target is None:
                    r = one_row(revision.exclusion_macro, mode=mode, percent=pct)
                    name, codes = "Mean of the five targets", ""
                else:
                    r = one_row(revision.exclusion_by_target, target=target, mode=mode, percent=pct)
                    name = latex_escape(DATASET_LABELS[target])
                    codes = ", ".join(SOURCE_CODES[source] for source in revision.exclusions[target])
                cells = [name if i == 0 else "", codes if i == 0 else "", str(pct),
                         format_mean_sd(r.matched_all_source_nrmse_mean, r.matched_all_source_nrmse_sd),
                         format_plain(r.matched_all_source_gain_pct),
                         format_mean_sd(r.excluded_nrmse_mean, r.excluded_nrmse_sd),
                         format_plain(r.excluded_gain_pct),
                         format_difference(r.excluded_minus_matched_nrmse_mean, r.excluded_minus_matched_nrmse_sd)]
                rows.append(keep_row(cells, last=i == len(SINGLE_SOURCE_PERCENTAGES) - 1))
    emit("source_exclusion", "Sensitivity of all-source transfer to excluding potentially overlapping source cohorts",
         "tab:supp_source_exclusion", r"@{}llrrrrrr@{}", 8,
         [r"& & & \multicolumn{2}{c}{All six sources} & \multicolumn{2}{c}{Sources excluded} & \\",
          r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
          r"Target & \shortstack{Excluded\\sources} & " + training_column +
          r" & NRMSE & Gain (\%) & NRMSE & Gain (\%) & \shortstack{Difference\\in NRMSE} \\"], rows,
         "Primary-panel NRMSE of STraTS: mean (SD) over three runs. Excluded sources are the cohorts that may "
         f"share participants with the target (codes as in {supp_ref('source_order')}); ABCD and YFS have none "
         "and are not included. The all-source models are reruns with the same seeding as the exclusion models, "
         r"not the runs used in the main analysis. Gain (\%) is relative to target-only STraTS from the main analysis. Difference is "
         "exclusion minus all sources, mean (SD) of the paired run differences; negative values indicate lower "
         "error after exclusion. Exclusion removes whole source cohorts, which also changes the amount and "
         "composition of the source data; it does not identify individual participants.")

    # Reviewer 1, comment 2 (Experiment 2).
    def rows_by_fraction(columns: int, dataset_cells: Any) -> list[Any]:
        """Seven datasets and their equal-weight mean, once per training fraction."""
        block: list[Any] = []
        for pct in SINGLE_SOURCE_PERCENTAGES:
            if block:
                block.append(r"\midrule")
            block.append(group_heading(fraction_text(pct), columns))
            for d in [*DATASET_ORDER, None]:
                name = "Mean of the seven datasets" if d is None else latex_escape(DATASET_LABELS[d])
                block.append(keep_row([name, *dataset_cells(d, pct)], last=d is None))
        return block

    def rate_cells(d: str | None, pct: int) -> list[str]:
        target = "Overall" if d is None else d
        summary = revision.rate_macro if d is None else revision.rate_by_dataset
        cells = []
        for policy in ["fixed_lr1e-4", "fixed_lr5e-4", "validation_selected"]:
            r = one_row(summary, target=target, percent=pct, policy=policy)
            cells.append(format_mean_sd(r.nrmse_mean, r.nrmse_sd))
        folds = revision.rate_selection[revision.rate_selection["percent"].eq(pct)]
        if d is not None:
            folds = folds[folds["target"].eq(d)]
        higher = int(folds.loc[folds["selected_lr_key"].eq("lr5e-4"), "n_folds"].sum())
        cells.append(f"{higher}/{int(folds['n_folds'].sum())}")
        for reference in ["historical_pooled", "historical_pretrained"]:
            r = one_row(revision.rate_gains, target=target, percent=pct, transfer_reference=reference)
            cells.append(format_plain(r.gain_vs_selected_pct))
        return cells

    emit("learning_rate", "Learning-rate sensitivity of target-only STraTS", "tab:supp_learning_rate",
         r"@{}lrrrrrr@{}", 7,
         [r"& \multicolumn{3}{c}{NRMSE of target-only STraTS} & & \multicolumn{2}{c}{Gain over selected rate (\%)} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
          r"Dataset & Rate $10^{-4}$ & Rate $5\times10^{-4}$ & Selected rate & \shortstack{Folds selecting\\$5\times10^{-4}$} & Pooled & Pretrained \\"],
         rows_by_fraction(7, rate_cells),
         "Primary-panel NRMSE: mean (SD) over three runs; the two learning rates started from the same initial "
         "weights. For every dataset, fold and training fraction, the rate with the lower validation loss "
         "averaged over the three runs was selected; test results were not used. Gains compare the main-analysis "
         "all-source pooled and pretrained STraTS with target-only STraTS at the selected rate. The main-analysis "
         r"target-only STraTS used $10^{-4}$; the values in this table are a rerun with different seeding.")

    # Reviewers 2.2 and 3.3 (Experiment 3).
    def scale_cells(d: str | None, pct: int) -> list[str]:
        r = (one_row(revision.scale_macro, percent=pct) if d is None
             else one_row(revision.scale_by_dataset, target=d, percent=pct))
        return [format_mean_sd(r.reset_nrmse_mean, r.reset_nrmse_sd),
                format_mean_sd(r.scale_consistent_nrmse_mean, r.scale_consistent_nrmse_sd),
                format_difference(r.scale_minus_reset_nrmse_mean, r.scale_minus_reset_nrmse_sd),
                format_plain(r.reset_gain_vs_target_only_pct), format_plain(r.scale_gain_vs_target_only_pct)]

    emit("normalization", "Scale-consistent transfer compared with re-estimated normalization",
         "tab:supp_scale_consistent", r"@{}lrrrrr@{}", 6,
         [r"& \multicolumn{2}{c}{NRMSE of pretrained STraTS} & & \multicolumn{2}{c}{Gain over target-only (\%)} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
          r"Dataset & Reset & Scale-consistent & \shortstack{Difference\\in NRMSE} & Reset & Scale-consistent \\"],
         rows_by_fraction(6, scale_cells),
         "Primary-panel NRMSE of all-source pretrained STraTS: mean (SD) over three runs. Reset: input, time, "
         "context and output standardization were re-estimated on target training data and the pretrained "
         "weights were copied unchanged, as in the main-analysis procedure (rerun with the seeding of the "
         "scale-consistent models). Scale-consistent: the source-training input, time and context "
         "standardization was kept and the output layer was converted to the target output scale, so that "
         "predictions were unchanged before fine-tuning. Difference is scale-consistent minus reset, mean (SD) "
         "of the paired run differences; negative values favour scale consistency. Gains are relative to "
         f"target-only STraTS at the validation-selected learning rate ({supp_ref('learning_rate')}).")

    # Reviewers 2.1 and 3.2 (Experiment 4).
    grud_excluded = grud_excluded_sources(revision)

    def grud_cells(d: str | None, pct: int) -> list[str]:
        r = one_row(revision.grud_paired, target="Overall" if d is None else d, percent=pct)
        codes = "" if d is None else (", ".join(SOURCE_CODES[source] for source in grud_excluded[d]) or "--")
        return [codes, format_mean_sd(r.target_only_mean, r.target_only_sd),
                format_mean_sd(r.transfer_mean, r.transfer_sd),
                format_difference(r.delta_mean, r.delta_sd), format_plain(r.gain_percent)]

    emit("grud_transfer", "GRU-D transfer by target dataset", "tab:supp_grud_transfer",
         r"@{}llrrrr@{}", 6,
         [r"& & \multicolumn{2}{c}{NRMSE of GRU-D} & & \\", r"\cmidrule(lr){3-4}",
          r"Dataset & \shortstack{Excluded\\sources} & Target-only & Pretrained & \shortstack{Difference\\in NRMSE} & Gain (\%) \\"],
         rows_by_fraction(6, grud_cells),
         "Primary-panel NRMSE: mean (SD) over three runs. Pretrained: GRU-D trained on the other datasets, "
         "without the sources that may share participants with the target (codes as in "
         f"{supp_ref('source_order')}; --: none excluded), and then fine-tuned on the target with "
         "scale-consistent transfer. Both models keep input positions for attributes absent from the target, "
         "which transfer requires; the target-only model therefore differs from the GRU-D of "
         f"{supp_ref('learning_curves')}. Difference is pretrained minus target-only, mean (SD) of the paired "
         r"run differences; negative values favour transfer. Gain (\%) is the relative reduction of the mean NRMSE.")

    # Reviewers 2.5 and 3.5: errors in original units and calibration of the GRU-D models.
    units = revision.grud_units[revision.grud_units["in_primary_panel"].astype(bool)]
    unit_keys = ["attribute", "target", "percent", "arm"]
    unit_means = units.groupby(unit_keys, as_index=False).agg(
        rmse=("person_balanced_rmse", "mean"), slope=("calibration_slope", "mean"),
        n_runs=("seed", "nunique"), n_people=("n_test_people", "max"))
    check_finite(unit_means, ["rmse", "slope"], "original-unit results")
    evaluated = {(r.dataset, r.attribute) for r in eligibility.itertuples()
                 if r.n_eligible > 0 and r.attribute in data.primary_attributes}
    if (set(zip(unit_means["target"], unit_means["attribute"])) != evaluated
            or len(unit_means) != 4 * len(evaluated) or not unit_means["n_runs"].eq(3).all()):
        raise ValueError("Original-unit results do not cover the evaluated primary outcomes of every dataset.")
    scored_people = revision.grud_attribute_scores.groupby(unit_keys)["n_test_people"].max()
    all_folds_only = sorted({(target, attribute) for attribute, target, percent, arm, people in zip(
        unit_means["attribute"], unit_means["target"], unit_means["percent"], unit_means["arm"],
        unit_means["n_people"]) if people != scored_people.loc[(attribute, target, percent, arm)]})
    rows = []
    for attribute in data.primary_attributes:
        if rows:
            rows.append(r"\midrule")
        rows.append(group_heading(f"{latex_escape(ATTRIBUTE_LABELS[attribute])} ({ATTRIBUTE_UNITS[attribute]})", 7))
        for d in [dataset for dataset in DATASET_ORDER if (dataset, attribute) in evaluated]:
            fold_sds = [value[attribute] for value in data.fold_stds[d].values() if value[attribute] is not None]
            for i, pct in enumerate(SINGLE_SOURCE_PERCENTAGES):
                pair = [one_row(unit_means, attribute=attribute, target=d, percent=pct, arm=arm)
                        for arm in ["target_only", "transfer"]]
                cells = [latex_escape(DATASET_LABELS[d]) if i == 0 else "", str(pct),
                         f"{float(np.mean(fold_sds)):.2f}" if i == 0 else ""]
                cells += [f"{r.rmse:.2f}" for r in pair] + [f"{r.slope:.2f}" for r in pair]
                rows.append(keep_row(cells, last=i == len(SINGLE_SOURCE_PERCENTAGES) - 1))
    fold_note = ""
    if all_folds_only:
        fold_note = (" For " + "; ".join(f"{ATTRIBUTE_LABELS[attribute].lower()} in {DATASET_LABELS[target]}"
                                         for target, attribute in all_folds_only)
                     + f", this includes folds that were not eligible for the NRMSE ({supp_ref('excluded_cells')}).")
    emit("original_units", "GRU-D prediction errors in original units and calibration", "tab:supp_original_units",
         r"@{}lrrrrrr@{}", 7,
         [r"& & & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{Calibration slope} \\",
          r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
          r"Dataset & " + training_column + r" & \shortstack{Reference\\SD} & Target-only & Pretrained & Target-only & Pretrained \\"],
         rows,
         f"GRU-D models of {supp_ref('grud_transfer')}; mean over three runs. RMSE is person-balanced and in "
         "the unit given after the outcome. The reference SD is the mean of the five fold-specific SDs that "
         "normalize the errors, so that NRMSE is approximately RMSE divided by this SD. The calibration slope "
         "is the slope of the regression of observed on predicted values with equal weight per participant; "
         "1 is ideal. All five test folds are used." + fold_note)

    missing = [key for key, _, _ in SUPPLEMENTARY_TABLES if key not in written]
    if missing or len(written) != len(SUPPLEMENTARY_TABLES):
        raise ValueError(f"Supplementary tables not written: {missing}")
    return [written[key] for key, _, _ in SUPPLEMENTARY_TABLES]


def draw_supplementary_learning_curves(
    paths: ProjectPaths, extra: Mapping[str, Any], formats: Sequence[str], dpi: int
) -> list[Path]:
    """Use exactly the neural ordering, colours, markers and labels of main Figure 2."""
    logging.info("Creating supplementary Figure S1: dataset-specific learning curves")
    frame = extra["dataset_absolute"]
    fig, axes = plt.subplots(4, 2, figsize=(FULL_WIDTH_MM * MM_TO_INCH, 210 * MM_TO_INCH))
    handles = None
    labels = None
    for index, d in enumerate(DATASET_ORDER):
        ax = axes.flat[index]
        for model in NEURAL_MODELS:
            block = frame[frame.dataset.eq(d) & frame.base_configuration_id.eq(model)].sort_values("train_data_percent")
            plot_model_series(ax, block.train_data_percent, block.nrmse_mean,
                              block.nrmse_sd.fillna(0), model, MODEL_LABELS[model])
        ax.set_title(f"{chr(97 + index)}) {DATASET_LABELS[d]}", loc="left")
        ax.set_xticks(TRAIN_PERCENTAGES)
        ax.set_xlim(7, 103)
        ax.set_ylabel(DATASET_NRMSE_LABEL)
        if index >= 5:
            ax.set_xlabel(TRAINING_AXIS_LABEL)
        finish_chart(ax)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center left", frameon=False,
                     handlelength=2.5, labelspacing=1.25)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.07, hspace=0.66, wspace=0.40)
    return save_figure(fig, paths.supplementary_figures_dir / "figure_S1_dataset_learning_curves", formats, dpi)


MAIN_TABLES = [   # (file stem, synchronization with the supplied current manuscript)
    ("table_1_dataset_characteristics", "current manuscript note; numerical content unchanged"),
    ("table_2_attribute_availability", "SLOfit lifelong header and current eligibility legend"),
    ("table_3_principal_model_comparison", "explicit STraTS gain-reference heading and current note"),
    ("table_4_suitability_ranking", "current manuscript note; numerical content unchanged"),
    ("table_5_outcome_panel_sensitivity", "current manuscript note; numerical content unchanged"),
    ("table_6_additional_analyses", "blocks A-D, gain references, model labels, current note and 1.1 pt column spacing"),
]


def write_file_placement(paths: ProjectPaths, main_tables: Sequence[Path]) -> Path:
    """Describe the current outputs without repeating obsolete revision instructions."""
    if [path.stem for path in main_tables] != [stem for stem, _ in MAIN_TABLES]:
        raise ValueError("MAIN_TABLES does not list the main tables that were written.")
    lines = ["GENERATED FILES (generator version " + SCRIPT_VERSION + ")",
             "=====================================", "",
             "Run from the repository root: python generate_paper_plots_tables.py",
             "Outputs go to results/paper/ unless --output-dir is supplied.",
             "Use --project-root to read inputs from another local repository copy.",
             "No existing figures, tables or manuscript files are required as generator inputs.", "",
             "MATCHING THE CURRENT PAPER", "--------------------------",
             "Tables and figure blocks match the revised sn-article.tex and",
             "supplementary_information.tex. Calculations, result values, table numbering and",
             "figure artwork are unchanged. No manuscript file is overwritten.", "",
             "MAIN TABLES (tables/main/)", "-------------------------",
             "Paste each complete file over the table with the same label in sn-article.tex."]
    lines += [f"  {stem}.tex: {change}" for stem, change in MAIN_TABLES]
    lines += ["", "SUPPLEMENTARY TABLES (tables/supplementary/)",
              "------------------------------------------",
              "Paste each complete file over the group with the same table label in",
              "supplementary_information.tex. Retain the numerical order S1-S19."]
    lines += [f"  {stem}.tex" for _, stem, _ in SUPPLEMENTARY_TABLES]
    lines += ["  S15-S17 use the current main-analysis wording; all other supplementary tables",
              "  are unchanged from generator 6.0.0.", "", "FIGURES", "-------",
              "Keep the lowercase figures/main/ and figures/supplementary/ paths when uploading",
              "the regenerated PDFs to Overleaf. The retained filenames are not the displayed",
              "figure numbers; LaTeX determines the numbers from manuscript order."]
    lines += [f"  figures/{scope}/{stem}.pdf" for scope, stem, _ in FIGURES]
    lines += ["Copy the main and supplementary figure blocks from figure_legends.tex into their",
              "respective documents. Captions, labels, widths and placement options match the",
              "supplied current manuscript, including the unchanged supplementary caption.", "",
              "INPUTS", "------",
              "  results/metrics/   aggregated main-analysis metrics",
              "  data/, metadata/summary_per_dataset.csv, strats.py",
              "  " + EXPERIMENTS_DIR.as_posix() + "/<experiment>/" + RESULTS_ARCHIVE,
              "     aggregate revision summaries; experiment output/ folders are not needed",
              "Run python evaluate.py --strict after changing raw errors and each experiment's summarize",
              "step after changing its results. generation_manifest.json records input fingerprints",
              "and library versions. Different library versions can cause minor rendering or",
              "floating-point differences; the generator retains the original calculation order.", ""]
    path = paths.output_dir / "file_placement.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def fingerprint_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    configure_logging(False)
    configure_matplotlib()
    root = discover_project_root(args.project_root)
    output = args.output_dir.expanduser().resolve() if args.output_dir else root / "results/paper"
    if output == root or output in root.parents or output in {root / "data", root / "metadata", root / "results/metrics"}:
        raise ValueError("Choose a separate output directory, not an input directory.")
    paths = build_paths(root, output)
    formats = ["pdf"]
    for rel in [*GENERATED_SUBDIRS, "generation.log", "generation_manifest.json",
                "figure_legends.tex", "file_placement.txt"]:
        ensure_safe_output_path(paths, paths.output_dir / rel)
    make_output_dirs(paths)
    log = logging.FileHandler(paths.output_dir / "generation.log", mode="w", encoding="utf-8")
    log.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(log)
    try:
        logging.info("Generator %s; reading current project results from %s", SCRIPT_VERSION, paths.metrics_dir)
        check_table_numbering()
        data = load_study_data(paths)
        messages = validate_study_data(data)
        revision = load_revision_data(paths)
        messages += validate_revision_data(data, revision)
        # Do not reorder this derivation: its original bootstrap stream is also
        # used by the main-paper figures and tables.
        results = derive_results(data, DEFAULT_BOOTSTRAP_REPS, DEFAULT_RANDOM_SEED)
        extra = compute_supplementary_results(data, results)
        # Added in the revision. Each has its own bootstrap stream, so the
        # intervals of the submitted tables do not change.
        extra["eligibility"] = compute_eligibility_cells(data, results.attribute_availability)
        extra["best_target_only"] = compute_best_target_only_gains(data, results)
        extra["additional_analyses"] = compute_additional_analyses(revision)
        extra["downsampling"] = compute_downsampling_summary(paths, data, results)
        messages += ["All supplementary neural learning curves have complete model/fraction/seed coverage.",
                     "Supplementary outcome/horizon gains have matched dataset/seed keys for both strategies.",
                     "Dataset-specific NRMSEs reproduce the corresponding main-paper macro means.",
                     "Shared main/supplementary bootstrap intervals agree exactly.",
                     f"Saved eligibility flags follow the {MIN_OBSERVED_TARGETS}-value rule.",
                     "Gains recomputed from dataset-specific errors equal those reported by the revision experiments.",
                     "The downsampling rule in preprocess_cohorts.py reproduces the analysis dataset exactly."]
        for message in messages:
            logging.info(message)
        main_tables = write_main_tables(paths, data, results, extra)
        supp_tables = write_supplementary_tables(paths, data, results, extra, revision)
        outputs: list[Path] = [*main_tables, *supp_tables]
        # Figures are those of generator 5.0.0; only the PNG copies are no longer written.
        outputs += draw_pipeline(paths, formats, DEFAULT_DPI)
        outputs += draw_learning_curves(paths, data, formats, DEFAULT_DPI)
        outputs += draw_source_count(paths, data, formats, DEFAULT_DPI)
        outputs += draw_dataset_gains(paths, results, formats, DEFAULT_DPI)
        outputs += draw_single_source_pair(paths, results, formats, DEFAULT_DPI)
        outputs += draw_outcome_gains(paths, results, data.primary_attributes, formats, DEFAULT_DPI)
        outputs += draw_supplementary_learning_curves(paths, extra, formats, DEFAULT_DPI)

        # supplementary_information.tex is maintained by hand and is not written.
        outputs += [write_figure_legends(paths, formats),
                    write_file_placement(paths, main_tables)]

        manifest_path = paths.output_dir / "generation_manifest.json"
        outputs += [paths.output_dir / "generation.log", manifest_path]
        manifest = {
            "script_version": SCRIPT_VERSION,
            "project_root": str(root), "output_directory": str(paths.output_dir),
            "source_data": "Current results/metrics/*.csv, original data/summary/source-order files "
                           "and the results_to_share.zip of each revision experiment",
            "bootstrap_reps": DEFAULT_BOOTSTRAP_REPS, "bootstrap_seed": DEFAULT_RANDOM_SEED,
            "supplementary_bootstrap_stream": "SHA-256 keyed streams, independent of the unchanged main derivation",
            "bootstrap_recomputed": True, "formats": formats,
            "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
            "pandas_version": pd.__version__, "changed_model_fitting": False,
            "changed_error_aggregation": False, "reads_old_plot_outputs_at_runtime": False,
            "reads_manuscript_at_runtime": False,
            "inputs": {rel.as_posix(): fingerprint_file(root / rel) for rel in REQUIRED_PROJECT_FILES},
            "figure_design": {
                "design_width_mm": FULL_WIDTH_MM,
                "manuscript_text_width_mm": MANUSCRIPT_TEXT_WIDTH_MM,
                "minimum_font_pt": FIGURE_MIN_FONT_PT,
                "model_colors": MODEL_COLORS,
                "gain_color_scale_limit_percent": gain_color_scale(results)[2],
            },
            "figures": [{"scope": scope, "stem": stem, "label": label,
                         "output_stem": f"figures/{scope}/{stem}"} for scope, stem, label in FIGURES],
            "main_tables": {stem: change for stem, change in MAIN_TABLES},
            "supplementary_tables": {stem: change for _, stem, change in SUPPLEMENTARY_TABLES},
            "outputs": sorted({p.relative_to(paths.output_dir).as_posix() for p in outputs}),
            "validation": messages,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logging.info("Wrote %d main and %d supplementary tables (.tex) and seven figures (.pdf).",
                     len(main_tables), len(supp_tables))
        logging.info("See file_placement.txt for what changed in each table and where it goes.")
        logging.info("Output directory: %s", paths.output_dir)
    finally:
        logging.getLogger().removeHandler(log)
        log.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as exc:
        logging.error("Generation stopped: %s", exc)
        raise SystemExit(1) from exc
