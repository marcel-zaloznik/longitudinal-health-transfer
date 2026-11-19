"""Train-only reconstruction and function-preserving source-to-target output mapping.

Input value/time/context normalizers remain source-trained. Only the final
linear head changes coordinates; optimization labels/weights remain target-based.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
import engine
from exp3_utils import assert_statistics_equal


class PairMetadata:
    """Reproduce record metadata without allocating every source token sequence.

    No model fitting and no statistics from validation/test partitions occur here.
    Per-partition statistics below select only the explicitly supplied training IDs.
    Float32 casting of raw context/outputs matches engine.build_records exactly.
    """
    def __init__(self, frame: pd.DataFrame, pairs: pd.DataFrame):
        if not frame.index.is_unique:
            raise ValueError("Original row indices must be unique.")
        self.frame, self.pairs = frame, pairs
        self.people = pairs.PersonID.astype(str).to_numpy()
        self.outputs = pairs.output_row.to_numpy(dtype=np.int64)
        self.lengths = pairs.input_rows.map(len).to_numpy(dtype=np.int64)
        if not len(pairs) or (self.lengths <= 0).any():
            raise ValueError("Empty prediction-pair table or input prefix.")
        flat = np.concatenate(pairs.input_rows.to_numpy()).astype(np.int64)
        in_pos = frame.index.get_indexer(flat)
        out_pos = frame.index.get_indexer(self.outputs)
        if (in_pos < 0).any() or (out_pos < 0).any():
            raise ValueError("Prediction pairs refer to rows outside their input frame.")
        owners = frame.PersonID.astype(str).to_numpy()
        if not np.array_equal(owners[out_pos], self.people):
            raise ValueError("Output row belongs to the wrong participant.")
        if not np.array_equal(owners[in_pos], np.repeat(self.people, self.lengths)):
            raise ValueError("Input prefix crosses participant boundaries.")
        ages = frame.Age.to_numpy(dtype=np.float64)
        output_ages = ages[out_pos]
        if not np.isfinite(ages[in_pos]).all() or not np.isfinite(output_ages).all():
            raise ValueError("Nonfinite ages in prediction pairs.")
        if (ages[in_pos] >= np.repeat(output_ages, self.lengths)).any():
            raise ValueError("Prediction prefix includes the output or a later visit.")
        ends = np.cumsum(self.lengths) - 1
        starts = np.r_[0, ends[:-1] + 1]
        input_ages = ages[in_pos[ends]]
        self.horizons = output_ages - input_ages
        observed = np.isfinite(frame[engine.TARGET_COLS].to_numpy(dtype=float)).sum(axis=1)
        n_tokens = np.add.reduceat(observed[in_pos], starts)
        n_tokens = np.maximum(n_tokens, 1)  # The original NO_OBS fallback.
        if engine.MAX_OBS is not None:
            n_tokens = np.minimum(n_tokens, engine.MAX_OBS)
        first = frame.sort_values("Age", kind="mergesort").drop_duplicates("PersonID")
        sex = pd.Series(self.people).map(first.set_index("PersonID").Sex).astype(str).to_numpy()
        self.raw_demo = np.column_stack([
            sex == "Male", sex == "Female", output_ages, input_ages,
            self.horizons, self.lengths, n_tokens,
        ]).astype(np.float32)

    def mask(self, training_frame: pd.DataFrame) -> np.ndarray:
        return self.pairs.PersonID.isin(set(training_frame.PersonID.astype(str))).to_numpy()

    def target_statistics(self, training_frame: pd.DataFrame) -> dict:
        outputs = pd.unique(self.outputs[self.mask(training_frame)])
        # Original compute_target_stats deduplicates output rows in pair order.
        raw = self.frame.loc[outputs, engine.TARGET_COLS].to_numpy(dtype=np.float32)
        result = {}
        for j, col in enumerate(engine.TARGET_COLS):
            values = raw[:, j].astype(np.float64)
            values = values[np.isfinite(values)]
            if not values.size:
                result[col] = {"mean": 0.0, "std": 1.0, "n": 0}
            else:
                std = float(values.std(ddof=0))
                result[col] = {"mean": float(values.mean()),
                               "std": std if np.isfinite(std) and std >= engine.EPS else 1.0,
                               "n": int(values.size)}
        return result

    def normalizers(self, training_frame: pd.DataFrame) -> dict:
        mask = self.mask(training_frame)
        if not mask.any():
            raise ValueError("No source-training prediction records.")
        horizons = self.horizons[mask]
        finite = horizons[np.isfinite(horizons)]
        scale = float(np.nanstd(finite)) if finite.size else 1.0
        if not np.isfinite(scale) or scale < engine.EPS:
            scale = 1.0
        raw = self.raw_demo[mask].astype(float)
        context = {}
        for idx in engine.DEMO_CONT_IDXS:
            values = raw[:, idx]
            values = values[np.isfinite(values)]
            std = float(values.std(ddof=0)) if values.size else 1.0
            context[str(idx)] = {"mean": float(values.mean()) if values.size else 0.0,
                                "std": std if np.isfinite(std) and std >= engine.EPS else 1.0}
        return {"value": engine.compute_value_stats(training_frame),
                "time": {"horizon_scale": scale}, "context": context,
                "target": self.target_statistics(training_frame),
                "n_records": int(mask.sum())}


def validate_reconstruction(frame: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Small deterministic equivalence test against the unmodified record builder."""
    sample_indices = []
    origin_of_person = frame.drop_duplicates("PersonID").set_index("PersonID").Origin
    origins = pairs.PersonID.map(origin_of_person)
    for origin in sorted(origins.unique()):
        group = pairs.loc[origins.eq(origin)]
        # Include first, last and longest prefixes, not just short early records.
        choices = list(group.index[:4]) + list(group.index[-4:])
        choices += list(group.input_rows.map(len).nlargest(4).index)
        sample_indices.extend(choices)
    sample = pairs.loc[sorted(set(sample_indices))].copy()
    people = set(sample.PersonID)
    selected_frame = frame[frame.PersonID.isin(people)].copy()
    fast = PairMetadata(selected_frame, sample).normalizers(selected_frame)
    records = engine.build_records(selected_frame, sample, fast["value"], selected_frame, engine.MAX_OBS)
    slow = {"value": engine.compute_value_stats(selected_frame),
            "time": engine.compute_time_stats(records),
            "context": engine.compute_demo_stats(records),
            "target": engine.compute_target_stats(records), "n_records": len(records)}
    assert_statistics_equal(fast, slow, "fast_vs_original_record_builder")
    return {"passed": True, "records_checked": len(records),
            "origin_count": int(origins.nunique()),
            "components": ["value", "time", "context", "target", "n_records"]}


def prepare_scale_consistent(train, val, test, pairs, source_norm):
    """Keep source input coordinates; retain target training labels and loss weights."""
    records = [engine.build_records(part, pairs, source_norm["value"], part, engine.MAX_OBS)
               for part in (train, val, test)]
    if any(not part for part in records):
        raise ValueError("A target partition has no usable prediction records.")
    target_stats = engine.compute_target_stats(records[0])
    context = {int(k): v for k, v in source_norm["context"].items()}
    for part in records:
        engine.apply_time_stats(part, source_norm["time"])
        engine.apply_demo_stats(part, context)
    engine.standardize_targets(records[0], target_stats)
    engine.standardize_targets(records[1], target_stats)
    engine.set_eval_targets(records[2])
    return (*records, target_stats)


def map_head_in_place(model, source_stats: dict, target_stats: dict) -> dict:
    """For y_raw=s_s*(W_s h+b_s)+m_s, preserve y_raw with target coordinates.

    W_t[j,:]=(s_s[j]/s_t[j])*W_s[j,:]
    b_t[j]=(s_s[j]/s_t[j])*b_s[j]+(m_s[j]-m_t[j])/s_t[j]

    All original hidden weights are unchanged. Absent outcomes use the original
    mean=0/std=1 convention and remain masked if absent from target training.
    """
    scale, shift = [], []
    for col in engine.TARGET_COLS:
        s, t = source_stats[col], target_stats[col]
        vals = [s["std"], t["std"], s["mean"], t["mean"]]
        if not all(math.isfinite(float(x)) for x in vals) or min(s["std"], t["std"]) <= 0:
            raise ValueError(f"Invalid output scaling: {col}")
        scale.append(float(s["std"]) / float(t["std"]))
        shift.append((float(s["mean"]) - float(t["mean"])) / float(t["std"]))
    device = model.head.weight.device
    a = torch.tensor(scale, dtype=torch.float64, device=device)
    b = torch.tensor(shift, dtype=torch.float64, device=device)
    with torch.no_grad():
        model.head.weight.copy_((model.head.weight.double() * a[:, None]).to(model.head.weight.dtype))
        model.head.bias.copy_((model.head.bias.double() * a + b).to(model.head.bias.dtype))
    return {"attributes": list(engine.TARGET_COLS), "scale": scale, "shift": shift,
            "source_unobserved_target_observed": [c for c in engine.TARGET_COLS
                 if source_stats[c]["n"] == 0 and target_stats[c]["n"] > 0],
            "target_inactive": [c for c in engine.TARGET_COLS if target_stats[c]["n"] == 0]}


def map_and_check(model, source_stats, target_stats, train_records, device) -> dict:
    """Check the *whole model's* initial raw predictions before any optimizer step.

    Evaluation mode disables dropout. Explicitly restoring RNG states ensures the
    check does not perturb the matched training random stream.
    """
    if not train_records:
        raise ValueError("Need target-training records for the invariance check.")
    indices = np.unique(np.linspace(0, len(train_records) - 1, min(32, len(train_records)), dtype=int))
    dataset = engine.STRATSDataset([train_records[i] for i in indices])
    batch = engine.collate_fn([dataset[i] for i in range(len(dataset))])
    inputs = [x.to(device) for x in batch[:5]]
    def tensors(stats):
        return (torch.tensor([stats[c]["mean"] for c in engine.TARGET_COLS], device=device),
                torch.tensor([stats[c]["std"] for c in engine.TARGET_COLS], device=device))
    sm, ss = tensors(source_stats)
    tm, ts = tensors(target_stats)
    previous_mode = model.training
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            raw_before = model(*inputs) * ss + sm
            conversion = map_head_in_place(model, source_stats, target_stats)
            raw_after = model(*inputs) * ts + tm
            delta = (raw_before - raw_after).abs()
            tolerance = 2e-5 * (raw_before.abs() + ss + ts + 1.0)
            if not torch.isfinite(raw_before).all() or not torch.isfinite(raw_after).all() or (delta > tolerance).any():
                raise FloatingPointError("Output-head mapping did not preserve initial raw predictions within float32 tolerance.")
            scaled = delta / (ss + ts + 1.0)
            result = {"passed": True, "n_training_records_checked": len(indices),
                      "max_absolute_raw_difference": float(delta.max().cpu()),
                      "max_scale_adjusted_difference": float(scaled.max().cpu()),
                      "absolute_difference_by_attribute": dict(zip(engine.TARGET_COLS, delta.max(dim=0).values.cpu().tolist())),
                      "tolerance": "abs(diff) <= 2e-5*(abs(raw_before)+source_std+target_std+1)",
                      "compares": "Same source-normalized inputs; original source-output head vs target-coordinate head; eval mode; no optimizer steps.",
                      "conversion": conversion}
    finally:
        model.train(previous_mode)
    return result
