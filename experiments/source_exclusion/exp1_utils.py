"""Paths, provenance, input validation and shared definitions for experiment 1."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
VERSION = "1.1.0"
N_GROUPS = 6
JOBS_PER_GROUP = 5
TARGET_LABELS = {
    "BE_KUL_Wave_1A": "LGS boys",
    "BE_KUL_Wave_1B": "LGS girls",
    "BE_KUL_Wave_23456": "Leuven adults",
    "SLOfit_lifelong": "SLOfit lifelong",
    "SLOfit_students": "SLOfit students",
}
SEEDS = (1234, 2345, 3456)
PERCENTS = (10, 100)
ARMS = ("all_source", "excluded")
MODES = ("pretrain", "pooled")


def inside(path: Path | str) -> Path:
    """Refuse writes through symlinks or paths outside the experiment folder."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Output would escape the experiment folder: {path}")
    return path


def configure_local_runtime() -> None:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    locations = {
        "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
        "XDG_CACHE_HOME": "cache", "TORCH_HOME": "torch",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "MPLCONFIGDIR": "matplotlib",
    }
    for name, dirname in locations.items():
        path = inside(ROOT / "_runtime" / dirname)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]
    os.chdir(ROOT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(frame, path: Path) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def package_fingerprint() -> str:
    names = ["engine.py", "experiment.py", "exp1_utils.py", "analyse.py", "exclusions.json",
             "reference/original_metrics.csv", "reference/provenance.json"]
    return object_hash({name: sha256(ROOT / name) for name in names})


def find_project_root(explicit: str | None = None) -> Path:
    candidates = [Path(explicit).expanduser().resolve()] if explicit else list(ROOT.parents)
    for candidate in candidates:
        if (candidate / "data/dataset.csv").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate data/. Use: python experiment.py prepare --project-root /path/to/main")


def verify_inputs(project: Path) -> dict:
    provenance = read_json(ROOT / "reference/provenance.json")
    verified = {}
    for name, expected in provenance["input_sha256"].items():
        path = project / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"Required read-only input is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Input differs from the reviewed dataset: {name}. Do not mix changed data with historical results.")
        verified[name] = actual
    if sha256(ROOT / "reference/original_metrics.csv") != provenance["reference_subset_sha256"]:
        raise RuntimeError("The frozen historical reference table has changed.")
    return verified


def jobs() -> list[dict]:
    rows = []
    for target in TARGET_LABELS:
        for mode in MODES:
            for seed in SEEDS:
                rows.append({"job_id": len(rows), "target": target,
                             "target_label": TARGET_LABELS[target], "mode": mode, "seed": seed})
    return rows


def group_jobs(group_id: int) -> list[dict]:
    """Six disjoint, fixed groups; running all six covers each job exactly once."""
    if group_id not in range(N_GROUPS):
        raise ValueError(f"group must be between 0 and {N_GROUPS - 1}.")
    start = group_id * JOBS_PER_GROUP
    selected = jobs()[start:start + JOBS_PER_GROUP]
    if len(selected) != JOBS_PER_GROUP:
        raise RuntimeError("Unexpected experiment matrix: group is not five jobs.")
    return selected


def load_setup() -> tuple[dict, Path]:
    path = ROOT / "setup/manifest.json"
    if not path.exists():
        raise RuntimeError("Run: python experiment.py prepare")
    setup = read_json(path)
    if setup["package_fingerprint"] != package_fingerprint():
        raise RuntimeError("Experiment code changed after preparation. Do not mix code versions; use a fresh experiment folder.")
    project = Path(setup["project_root"])
    if not project.is_dir():
        raise RuntimeError("Project moved. Run prepare --project-root /new/path/to/main; unchanged inputs are required.")
    verify_inputs(project)
    return setup, project


def id_digest(frame) -> str:
    return object_hash(sorted(frame["PersonID"].astype(str).unique().tolist()))


def frame_counts(frame, pair_counts) -> dict:
    ids = frame["PersonID"].astype(str).unique()
    return {"n_people": len(ids), "n_visits": len(frame),
            "n_pairs": int(pair_counts.reindex(ids, fill_value=0).sum()),
            "people_digest": object_hash(sorted(ids.tolist()))}


def source_names(engine, target: str, arm: str) -> list[str]:
    full = list(engine.ORIGIN_MAP[target])
    exclusions = read_json(ROOT / "exclusions.json")
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    kept = full if arm == "all_source" else [x for x in full if x not in exclusions[target]]
    assert target not in kept and len(set(kept)) == len(kept)
    if arm == "excluded":
        assert set(kept).isdisjoint(exclusions[target])
        assert set(full) - set(kept) == set(exclusions[target])
    return kept


def stage_seed(seed: int, origin_index: int, fold: int | None = None) -> int:
    # Arm, training fraction and method deliberately do not enter the seed.
    return seed + (100_000 if fold is None else 200_000) + 1_000 * origin_index + (fold or 0)


def load_data(project: Path, engine, smoke: bool = False):
    import numpy as np
    import pandas as pd
    frame = engine.preprocess_dataframe(pd.read_csv(project / "data/dataset.csv", low_memory=False))
    split = engine.load_split5k(str(project / "data/split5k.json"))
    subsets = read_json(project / "data/target_train_subsets_by_dataset_fold_percent.json")
    pairs = pd.read_csv(project / "data/input_output_pairs.csv", dtype={"PersonID": str})
    if smoke:
        # Retain a few people per origin/fold, plus people from each 10% subset.
        keep = set()
        for origin, group in frame.groupby("Origin", sort=False):
            ids = set(group["PersonID"])
            for fold, members in enumerate(split):
                keep.update(sorted(ids & members)[:3])
                keep.update(subsets[origin][str(fold)]["10"][:3])
        frame = frame[frame.PersonID.isin(keep)].copy()  # Never reset original row indices.
        pairs = pairs[pairs.PersonID.isin(keep)].copy()
        pairs = pairs.groupby("PersonID", sort=False).head(3).copy()
    pairs["input_rows"] = pairs["input_rows"].apply(engine.parse_index_tuple)
    outputs = pd.to_numeric(pairs["output_row"], errors="raise")
    if not np.isfinite(outputs).all() or not np.equal(outputs, np.floor(outputs)).all():
        raise ValueError("Invalid output-row indices in input_output_pairs.csv")
    pairs["output_row"] = outputs.astype(int)
    if pairs["input_rows"].map(len).eq(0).any():
        raise ValueError("Empty input prefixes found.")
    return frame, pairs, split, subsets


def target_frames(frame, target: str, fold: int, percent: int, split, subsets, smoke: bool = False):
    target_frame = frame[frame.Origin.eq(target)].copy()
    test_ids, val_ids = split[fold], split[fold - 1]
    test = target_frame[target_frame.PersonID.isin(test_ids)].copy()
    val = target_frame[target_frame.PersonID.isin(val_ids)].copy()
    train = target_frame[~target_frame.PersonID.isin(test_ids | val_ids)].copy()
    if percent != 100:
        keep = set(map(str, subsets[target][str(fold)][str(percent)]))
        if not smoke and not keep.issubset(set(train.PersonID)):
            raise ValueError("The fixed target-training subset crosses a validation/test boundary.")
        train = train[train.PersonID.isin(keep)].copy()
    if smoke:
        train = train[train.PersonID.isin(sorted(train.PersonID.unique())[:3])].copy()
        val = val[val.PersonID.isin(sorted(val.PersonID.unique())[:3])].copy()
        test = test[test.PersonID.isin(sorted(test.PersonID.unique())[:3])].copy()
    ids = [set(x.PersonID) for x in (train, val, test)]
    if any(not x for x in ids) or any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Empty or overlapping target train/validation/test partitions.")
    return train, val, test


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """OS lock is released even if a process is killed; no stale PID guessing."""
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Job is already running: {path.parent.name}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Job is already running: {path.parent.name}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)
    def flush(self):
        self.stream.flush()
        self.log.flush()
    def isatty(self):
        return False


def environment(torch, device) -> dict:
    import numpy as np
    import pandas as pd
    import platform
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
            "cuda_build": torch.version.cuda, "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "process_id": os.getpid(), "cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
