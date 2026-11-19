"""Experiment 3 paths, local runtime, fixed matrix and provenance."""
from __future__ import annotations
import contextlib, hashlib, json, os, sys, tempfile
from pathlib import Path
from typing import Any, Iterator
ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"
SEEDS = (1234, 2345, 3456)
PERCENTS = (10, 100)
TARGET_LABELS = {
    "BE_KUL_Wave_1A": "LGS boys", "BE_KUL_Wave_1B": "LGS girls",
    "BE_KUL_Wave_23456": "Leuven adults", "SLOfit_lifelong": "SLOfit lifelong",
    "SLOfit_students": "SLOfit students",
}


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
        "NUMBA_CACHE_DIR": "numba", "PYTHONPYCACHEPREFIX": "pycache",
        "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data", "XDG_STATE_HOME": "state",
        "PIP_CACHE_DIR": "pip", "HF_HOME": "huggingface",
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

def find_project_root(explicit: str | None = None) -> Path:
    candidates = [Path(explicit).expanduser().resolve()] if explicit else list(ROOT.parents)
    for candidate in candidates:
        if (candidate / "data/dataset.csv").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate data/. Use: python experiment.py prepare --project-root /path/to/main")

def id_digest(frame) -> str:
    return object_hash(sorted(frame["PersonID"].astype(str).unique().tolist()))

def frame_counts(frame, pair_counts) -> dict:
    ids = frame["PersonID"].astype(str).unique()
    return {"n_people": len(ids), "n_visits": len(frame),
            "n_pairs": int(pair_counts.reindex(ids, fill_value=0).sum()),
            "people_digest": object_hash(sorted(ids.tolist()))}

def stage_seed(seed: int, origin_index: int, fold: int | None = None) -> int:
    # Arm, training fraction and method deliberately do not enter the seed.
    return seed + (100_000 if fold is None else 200_000) + 1_000 * origin_index + (fold or 0)

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

CODE_FILES = ("experiment.py", "exp3_utils.py", "engine.py", "normalization.py",
              "metrics.py", "training.py", "analyse.py", "tests.py")


def package_fingerprint() -> str:
    paths = [ROOT / n for n in CODE_FILES] + sorted((ROOT / "reference").glob("*"))
    return object_hash({str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()})


def verify_references() -> dict:
    ref = read_json(ROOT / "reference/provenance.json")
    for name, expected in ref["reference_sha256"].items():
        if sha256(ROOT / "reference" / name) != expected:
            raise RuntimeError(f"Frozen reference has changed: {name}")
    if sha256(ROOT / "engine.py") != ref["exp1_engine_sha256"]:
        raise RuntimeError("The frozen Experiment 1 model engine has changed.")
    return ref


def verify_inputs(project: Path) -> dict:
    ref = verify_references()
    for name, expected in ref["input_sha256"].items():
        path = project / "data" / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing/changed read-only input: {path}. Original data and splits are required.")
    return ref["input_sha256"]


def jobs() -> list[dict]:
    rows = []
    for target_index, (target, label) in enumerate(TARGET_LABELS.items()):
        for percent in PERCENTS:
            for seed_index, seed in enumerate(SEEDS):
                rows.append({"job_id": len(rows), "target": target, "target_label": label,
                             "percent": percent, "seed": seed,
                             "exp1_job_id": target_index * 6 + seed_index})
    return rows


def grouped_jobs(n_groups: int) -> list[list[dict]]:
    """Deterministic approximate workload balancing using old training times only."""
    import csv
    if not 1 <= n_groups <= len(jobs()):
        raise ValueError("groups must be between 1 and 30.")
    costs = {}
    with (ROOT / "reference/exp1_stage_audit.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "pretrain" and row["arm"] == "all_source":
                key = (row["target"], int(row["percent"]), int(row["seed"]))
                # Record preparation is also substantial. This is scheduling, not tuning.
                cost = float(row["elapsed_seconds"]) + 0.001 * float(row["train_records"])
                costs[key] = costs.get(key, 0.0) + cost
    groups, loads = [[] for _ in range(n_groups)], [0.0] * n_groups
    ordered = sorted(jobs(), key=lambda j: (-costs[(j["target"], j["percent"], j["seed"])], j["job_id"]))
    for job in ordered:
        index = min(range(n_groups), key=lambda k: (loads[k], len(groups[k]), k))
        groups[index].append(job)
        loads[index] += costs[(job["target"], job["percent"], job["seed"])]
    return groups


def load_data(project: Path, engine, target: str | None = None):
    import numpy as np
    import pandas as pd
    frame = engine.preprocess_dataframe(pd.read_csv(project / "data/dataset.csv", low_memory=False))
    split = engine.load_split5k(str(project / "data/split5k.json"))
    subsets = read_json(project / "data/target_train_subsets_by_dataset_fold_percent.json")
    pairs = pd.read_csv(project / "data/input_output_pairs.csv", dtype={"PersonID": str})
    if target is not None:
        frame = frame[frame.Origin.eq(target)].copy()  # Original row indices are preserved.
        pairs = pairs[pairs.PersonID.isin(set(frame.PersonID))].copy()
    pairs["input_rows"] = pairs["input_rows"].apply(engine.parse_index_tuple)
    outputs = pd.to_numeric(pairs["output_row"], errors="raise")
    if not np.isfinite(outputs).all() or not np.equal(outputs, np.floor(outputs)).all():
        raise ValueError("Nonintegral/nonfinite output-row indices.")
    pairs["output_row"] = outputs.astype(int)
    if pairs["input_rows"].map(len).eq(0).any():
        raise ValueError("Empty prediction prefixes.")
    return frame, pairs, split, subsets


def assert_statistics_equal(actual: dict, expected: dict, name: str, *, rtol=1e-12, atol=1e-12) -> None:
    """Recursive comparison, ignoring only JSON's conversion of integer dictionary keys."""
    import math
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise TypeError(f"Expected dictionaries at {name}")
    a = {str(k): v for k, v in actual.items()}
    b = {str(k): v for k, v in expected.items()}
    if set(a) != set(b):
        raise ValueError(f"Statistics keys differ: {name}")
    for key in a:
        if isinstance(a[key], dict):
            assert_statistics_equal(a[key], b[key], name + "." + key, rtol=rtol, atol=atol)
        elif isinstance(a[key], (int, float)) and not isinstance(a[key], bool):
            exact_count = key in {"n", "n_people", "n_visits", "n_pairs", "n_records"}
            good = a[key] == b[key] if exact_count else math.isclose(float(a[key]), float(b[key]), rel_tol=rtol, abs_tol=atol)
            if not good:
                raise ValueError(f"Statistics mismatch at {name}.{key}: {a[key]} != {b[key]}")
        elif a[key] != b[key]:
            raise ValueError(f"Metadata mismatch at {name}.{key}")


def load_setup(check_data: bool = True) -> tuple[dict, Path]:
    path = ROOT / "setup/manifest.json"
    if not path.exists():
        raise RuntimeError("Run: python experiment.py prepare")
    setup = read_json(path)
    if setup["package_fingerprint"] != package_fingerprint():
        raise RuntimeError("Code/references changed after preparation. Do not mix versions; use a fresh experiment folder.")
    project = Path(setup["project_root"])
    if check_data:
        verify_inputs(project)
    for name, expected in setup["setup_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Prepared audit/normalization file changed: {name}")
    return setup, project


def check_external_files(setup: dict, exp1_job_id: int | None = None, *, include_errors: bool = True) -> None:
    exp1 = Path(setup["experiment1_root"])
    for item in setup["external_inventory"]:
        if exp1_job_id is not None and item["exp1_job_id"] != exp1_job_id:
            continue
        if not include_errors and item["relative_path"].endswith("errors.csv.gz"):
            continue
        path = exp1 / item["relative_path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Experiment 1 input changed/missing: {path}. No training result will be silently reused.")


def receipt_folder(job: dict) -> Path:
    return ROOT / "setup/receipts" / f"job_{job['exp1_job_id']:03d}"


def control_folder(job: dict, fold: int) -> Path:
    return receipt_folder(job) / f"pct{job['percent']}" / f"fold{fold}"


def validate_source_config(config: dict, exp1_job_id: int, target: str, seed: int, ref: dict) -> None:
    job = config.get("job", {})
    if config.get("smoke") is not False:
        raise RuntimeError("A smoke-test checkpoint cannot be used as a research checkpoint.")
    if config.get("experiment_fingerprint") != ref["exp1_experiment_fingerprint"]:
        raise RuntimeError("Checkpoint belongs to a different Experiment 1 run/protocol.")
    if job.get("mode") != "pretrain" or job.get("job_id") != exp1_job_id or job.get("target") != target or job.get("seed") != seed:
        raise RuntimeError("Checkpoint target/seed/mode does not match the specified full all-source job.")
    if config.get("num_workers") != 0:
        raise RuntimeError("Unexpected Experiment 1 DataLoader setting.")
