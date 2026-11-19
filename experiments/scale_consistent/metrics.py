"""Frozen Experiment 1 person-balanced primary-panel metric."""
from __future__ import annotations
import hashlib, math
from pathlib import Path
from exp3_utils import ROOT, read_json

def identity_digest(errors) -> str:
    import numpy as np
    import pandas as pd
    columns = ["dataset", "fold", "person_id", "attribute", "input_age", "output_age", "horizon", "n_input_visits"]
    identity = errors[columns].copy()
    for col in ("dataset", "person_id", "attribute"):
        identity[col] = identity[col].astype(str)
    for col in ("fold", "n_input_visits"):
        identity[col] = identity[col].astype("int64")
    for col in ("input_age", "output_age", "horizon"):
        identity[col] = identity[col].astype("float64")
    values = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(np.sort(values).tobytes()).hexdigest()

def fold_metrics(errors, project: Path, target: str, fold: int, strict: bool = True) -> dict:
    """Sufficient statistics for the submitted person-balanced NRMSE.

    Crucially, combine PERSON mean-squared errors across folds before taking
    the attribute square root; never average fold RMSEs.
    """
    import numpy as np
    import pandas as pd
    if errors.empty or not errors.dataset.eq(target).all() or not errors.fold.eq(fold).all():
        raise ValueError("Raw errors contain unexpected/empty target or fold metadata.")
    if not np.isfinite(errors.error.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite raw errors.")
    if errors.person_id.isna().any():
        raise ValueError("Missing participant identifier in errors.")
    stds = read_json(project / "data/target_attribute_stds_by_dataset_fold.json")[target][str(fold)]
    eligibility = read_json(project / "data/eligible_dataset_attribute_pairs.json")[target][str(fold)]
    primary = read_json(project / "data/primary_target_panel.json")
    selected = [attribute for attribute in primary if eligibility[attribute]["eligible"]]
    selected_errors = errors[errors.attribute.isin(selected)].copy()
    if selected_errors.empty:
        raise ValueError("No evaluable primary-panel errors.")
    attribute_stds = selected_errors.attribute.map(stds).to_numpy(dtype=float)
    if not (np.isfinite(attribute_stds) & (attribute_stds > 0)).all():
        raise ValueError("An eligible evaluation cell has no positive finite reference standard deviation.")
    normalized = selected_errors.error.to_numpy(dtype=float) / attribute_stds
    selected_errors["squared"] = normalized ** 2
    selected_errors["absolute"] = np.abs(normalized)
    if not np.isfinite(selected_errors["squared"]).all():
        raise FloatingPointError("Normalized squared errors overflowed.")
    if strict:
        expected = read_json(ROOT / "setup/expected_test_counts.json")[target][str(fold)]
        observed = {str(k): int(v) for k, v in selected_errors.groupby("attribute").size().items()}
        if observed != expected:
            raise ValueError(f"Incomplete/changed evaluation observations for {target}, fold {fold}: expected {expected}; observed {observed}")
    people = selected_errors.groupby(["person_id", "attribute"], sort=False).agg(
        person_mse=("squared", "mean"), person_mae=("absolute", "mean"), n_errors=("error", "size"))
    attributes = {}
    for attribute, group in people.groupby(level="attribute", sort=False):
        attributes[str(attribute)] = {
            "sum_person_normalized_mse": float(group.person_mse.sum()),
            "sum_person_normalized_mae": float(group.person_mae.sum()),
            "n_people": len(group), "n_errors": int(group.n_errors.sum()),
        }
    return {"target": target, "fold": fold, "panel": "primary_9", "horizon_group": "all",
            "identity_digest": identity_digest(errors), "n_raw_errors": len(errors),
            "n_primary_errors": len(selected_errors), "n_test_people": int(selected_errors.person_id.nunique()),
            "attributes": attributes}

def combine_folds(folds: list[dict]) -> tuple[dict, list[dict]]:
    import numpy as np
    attributes = sorted({attribute for item in folds for attribute in item["attributes"]})
    rows = []
    for attribute in attributes:
        values = [item["attributes"][attribute] for item in folds if attribute in item["attributes"]]
        n_people = sum(value["n_people"] for value in values)
        mse = sum(value["sum_person_normalized_mse"] for value in values) / n_people
        mae = sum(value["sum_person_normalized_mae"] for value in values) / n_people
        rows.append({"attribute": attribute, "nrmse": math.sqrt(mse), "nmae": mae,
                     "n_test_people": n_people, "n_errors": sum(value["n_errors"] for value in values),
                     "n_folds_evaluated": len(values)})
    if not rows:
        raise ValueError("No attribute scores to combine.")
    result = {"nrmse": float(np.mean([row["nrmse"] for row in rows])),
              "nmae": float(np.mean([row["nmae"] for row in rows])),
              "n_attributes": len(rows), "n_primary_errors": sum(item["n_primary_errors"] for item in folds),
              "n_test_people": sum(item["n_test_people"] for item in folds), "n_folds": len(folds)}
    return result, rows
