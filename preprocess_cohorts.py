#!/usr/bin/env python3
"""Remove near-duplicate visits and downsample SLOfit students."""
from __future__ import annotations

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NEAR_DUPLICATE_YEARS = 0.5
STUDENT_VALUE_TARGET = 273566


def clean_visits(frame: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """Keep the more complete visit; on a tie, drop the earlier visit."""
    person, identifiers = pd.factorize(frame["PersonID"], sort=False)
    age = pd.to_numeric(frame["Age"], errors="raise").to_numpy(dtype=float)
    if (person < 0).any() or not np.isfinite(age).all():
        raise ValueError("PersonID and Age must be present and Age must be finite.")
    missing = frame[attributes].isna().sum(axis=1).to_numpy()
    order = np.lexsort((np.arange(len(frame)), age, person))
    sorted_person, sorted_age, sorted_missing = person[order], age[order], missing[order]
    same_person = sorted_person[1:] == sorted_person[:-1]
    near = same_person & (np.abs(np.diff(sorted_age)) <= NEAR_DUPLICATE_YEARS)
    starts = np.flatnonzero(np.r_[True, ~same_person])
    ends = np.r_[starts[1:], len(order)]
    keep = np.ones(len(frame), dtype=bool)
    for group in np.unique(np.searchsorted(starts, np.flatnonzero(near) + 1, side="right") - 1):
        start, end = starts[group], ends[group]
        rows = order[start:end].tolist()
        ages = sorted_age[start:end].tolist()
        gaps = sorted_missing[start:end].tolist()
        i = 0
        while i < len(rows) - 1:
            if abs(ages[i + 1] - ages[i]) <= NEAR_DUPLICATE_YEARS:
                j = i + 1 if gaps[i + 1] > gaps[i] else i
                keep[rows[j]] = False
                del rows[j], ages[j], gaps[j]
            else:
                i += 1
    counts = np.bincount(person[keep], minlength=len(identifiers))
    keep &= counts[person] > 1
    return frame.loc[keep].copy()


def downsample_students(frame: pd.DataFrame, attributes: list[str],
                        value_target: int = STUDENT_VALUE_TARGET) -> pd.DataFrame:
    students = frame.loc[frame["Origin"].eq("SLOfit_students")]
    if students.empty:
        raise ValueError("No SLOfit_students records remain after visit cleaning.")
    person, identifiers = pd.factorize(students["PersonID"], sort=False)
    observed = students[attributes].notna().sum(axis=1).to_numpy()
    values = np.bincount(person, weights=observed).astype(np.int64)
    # Stable sorting preserves the notebook's first-appearance tie break.
    ranking = np.argsort(-values, kind="stable")
    cumulative = np.cumsum(values[ranking])
    if value_target <= 0 or cumulative[-1] < value_target:
        raise ValueError("The requested observed-value total cannot be reached.")
    n_keep = int(np.searchsorted(cumulative, value_target, side="left")) + 1
    selected = set(identifiers[ranking[:n_keep]])
    keep = ~frame["Origin"].eq("SLOfit_students") | frame["PersonID"].isin(selected)
    return frame.loc[keep].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/dataset_full.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/dataset.csv")
    parser.add_argument("--student-value-target", type=int, default=STUDENT_VALUE_TARGET)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if source == output:
        raise ValueError("Input and output must differ.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite deliberately.")
    frame = pd.read_csv(source, dtype={"PersonID": str}, low_memory=False)
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("The input must be nonempty and have unique column names.")
    for name in ("Origin", "PersonID", "Age", "Height", "Diet_T2P"):
        if name not in frame:
            raise ValueError(f"Missing column: {name}")
    attributes = frame.loc[:, "Height":"Diet_T2P"].columns.tolist()
    if len(attributes) != 20:
        raise ValueError("Expected the 20 ordered attributes from Height through Diet_T2P.")
    if frame["Origin"].isna().any() or frame.groupby("PersonID")["Origin"].nunique().gt(1).any():
        raise ValueError("Origin must be present and PersonID must be unique across cohorts.")
    cleaned = clean_visits(frame, attributes)
    selected = downsample_students(cleaned, attributes, args.student_value_target)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        selected.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Visits: {len(frame)} input, {len(cleaned)} after cleaning, {len(selected)} retained.")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
