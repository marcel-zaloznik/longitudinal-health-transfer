"""Read-only project inputs; all writes are confined to this experiment directory."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"
SEEDS = (1234, 2345, 3456)
PERCENTS = (10, 100)
LRS = {"lr1e-4": 1e-4, "lr5e-4": 5e-4}
DEFAULT_GROUPS = 6
TARGET_LABELS = {
    "ABCD": "ABCD",
    "BE_KUL_Wave_1A": "LGS boys",
    "BE_KUL_Wave_1B": "LGS girls",
    "BE_KUL_Wave_23456": "Leuven adults",
    "SLOfit_lifelong": "SLOfit lifelong",
    "SLOfit_students": "SLOfit students",
    "YFS": "YFS",
}
CORE_FILES = ("engine.py", "experiment.py", "exp2_utils.py", "analyse.py",
              "reference/original_metrics.csv", "reference/provenance.json",
              "reference/original_train_model.py.txt")


def inside(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Output would escape this experiment folder: {path}")
    return path


def configure_local_runtime() -> None:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    locations = {"TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
                 "XDG_CACHE_HOME": "cache", "TORCH_HOME": "torch",
                 "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton",
                 "CUDA_CACHE_PATH": "cuda", "MPLCONFIGDIR": "matplotlib",
                 "NUMBA_CACHE_DIR": "numba", "PYTHONPYCACHEPREFIX": "pycache"}
    for name, folder in locations.items():
        path = inside(ROOT / "_runtime" / folder)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]
    # Do not let interpreter/extension crashes produce a core dump elsewhere.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass
    os.chdir(ROOT)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path: Path, value: Any) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = inside(path.with_name(path.name + f".{os.getpid()}.tmp"))
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(temp, path)


def atomic_csv(frame, path: Path) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = inside(path.with_name(path.name + f".{os.getpid()}.tmp"))
    frame.to_csv(temp, index=False, compression="gzip" if path.name.endswith(".gz") else None)
    os.replace(temp, path)


def atomic_text(path: Path, text: str) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = inside(path.with_name(path.name + f".{os.getpid()}.tmp"))
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def package_fingerprint() -> str:
    return object_hash({name: sha256(ROOT / name) for name in CORE_FILES})


def find_project_root(explicit: str | None = None) -> Path:
    candidates = [Path(explicit).expanduser().resolve()] if explicit else list(ROOT.parents)
    for candidate in candidates:
        if (candidate / "data/dataset.csv").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate data/. Run prepare --project-root /path/to/main")


def verify_inputs(project: Path) -> dict:
    provenance = read_json(ROOT / "reference/provenance.json")
    verified = {}
    for name, expected in provenance["input_sha256"].items():
        path = project / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing read-only input: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Input changed: {name}. Do not compare changed data with the frozen historical results.")
        verified[name] = actual
    if sha256(ROOT / "reference/original_metrics.csv") != provenance["reference_subset_sha256"]:
        raise RuntimeError("The historical reference table changed.")
    return verified


def jobs() -> list[dict]:
    # This ordering distributes the six large SLOfit-students jobs across six groups.
    rows = []
    for percent in PERCENTS:
        for seed in SEEDS:
            for target, label in TARGET_LABELS.items():
                rows.append({"job_id": len(rows), "target": target, "target_label": label,
                             "percent": percent, "seed": seed})
    return rows


def group_jobs(group: int, n_groups: int = DEFAULT_GROUPS) -> list[dict]:
    if not 1 <= n_groups <= len(jobs()) or not 0 <= group < n_groups:
        raise ValueError("Require 1 <= groups <= 42 and 0 <= group < groups.")
    return [j for j in jobs() if j["job_id"] % n_groups == group]


def load_setup() -> tuple[dict, Path]:
    path = ROOT / "setup/manifest.json"
    if not path.is_file():
        raise RuntimeError("Run: python -u experiment.py prepare")
    setup = read_json(path)
    if setup["package_fingerprint"] != package_fingerprint():
        raise RuntimeError("Code changed after preparation. Use an untouched package in a fresh experiment folder.")
    project = Path(setup["project_root"])
    verify_inputs(project)
    for name, expected in setup["setup_sha256"].items():
        if sha256(ROOT / "setup" / name) != expected:
            raise RuntimeError(f"Prepared audit changed: {name}")
    return setup, project


def load_data(project: Path, engine, target: str | None = None, smoke: bool = False):
    import numpy as np
    import pandas as pd
    frame = engine.preprocess_dataframe(pd.read_csv(project / "data/dataset.csv", low_memory=False))
    split = engine.load_split5k(str(project / "data/split5k.json"))
    subsets = read_json(project / "data/target_train_subsets_by_dataset_fold_percent.json")
    pairs = pd.read_csv(project / "data/input_output_pairs.csv", dtype={"PersonID": str})
    if target is not None:
        frame = frame[frame.Origin.eq(target)].copy()  # Keep original row indices.
        pairs = pairs[pairs.PersonID.isin(set(frame.PersonID))].copy()
    if smoke:
        keep = set()
        for origin, group in frame.groupby("Origin", sort=False):
            ids = set(group.PersonID)
            for fold, members in enumerate(split):
                keep.update(sorted(ids & members)[:3])
                keep.update(map(str, subsets[origin][str(fold)]["10"][:3]))
        frame = frame[frame.PersonID.isin(keep)].copy()
        pairs = pairs[pairs.PersonID.isin(keep)].groupby("PersonID", sort=False).head(3).copy()
    pairs["input_rows"] = pairs.input_rows.apply(engine.parse_index_tuple)
    outputs = pd.to_numeric(pairs.output_row, errors="raise")
    if not np.isfinite(outputs).all() or not np.equal(outputs, np.floor(outputs)).all():
        raise ValueError("Invalid output-row indices.")
    pairs["output_row"] = outputs.astype(int)
    if pairs.input_rows.map(len).eq(0).any():
        raise ValueError("Empty input prefixes.")
    return frame, pairs, split, subsets


def target_frames(frame, target, fold, percent, split, subsets, smoke=False):
    target_frame = frame[frame.Origin.eq(target)].copy()
    test_ids, val_ids = split[fold], split[fold - 1]
    test = target_frame[target_frame.PersonID.isin(test_ids)].copy()
    val = target_frame[target_frame.PersonID.isin(val_ids)].copy()
    full_train = target_frame[~target_frame.PersonID.isin(test_ids | val_ids)].copy()
    keep = set(map(str, subsets[target][str(fold)][str(percent)]))
    if not smoke and not keep.issubset(set(full_train.PersonID)):
        raise ValueError("Fixed training subset crosses a validation/test boundary.")
    if not smoke and percent == 100 and keep != set(full_train.PersonID):
        raise ValueError("The fixed 100% subset is not the original full training partition.")
    train = full_train[full_train.PersonID.isin(keep)].copy()
    if smoke:
        train = train[train.PersonID.isin(sorted(train.PersonID.unique())[:3])].copy()
        val = val[val.PersonID.isin(sorted(val.PersonID.unique())[:3])].copy()
        test = test[test.PersonID.isin(sorted(test.PersonID.unique())[:3])].copy()
    ids = [set(p.PersonID) for p in (train, val, test)]
    if any(not p for p in ids) or any(ids[i] & ids[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError("Empty or overlapping train/validation/test partitions.")
    return train, val, test


def frame_counts(frame, pair_counts) -> dict:
    ids = frame.PersonID.astype(str).unique()
    return {"n_people": len(ids), "n_visits": len(frame),
            "n_pairs": int(pair_counts.reindex(ids, fill_value=0).sum()),
            "people_digest": object_hash(sorted(ids.tolist()))}


def stage_seed(seed: int, origin_index: int, fold: int) -> int:
    return seed + 200_000 + 1_000 * origin_index + fold


def state_digest(state) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def verified_done(folder: Path, config_hash: str, required=("errors.csv.gz", "metrics.json", "history.csv")) -> dict | None:
    folder = inside(folder)
    path = inside(folder / "done.json")
    if not path.exists():
        return None
    done = read_json(path)
    if done.get("config_hash") != config_hash:
        raise RuntimeError(f"Stored configuration mismatch: {folder}")
    if not set(required).issubset(done.get("file_sha256", {})):
        raise RuntimeError(f"Incomplete completion record: {folder}")
    for name, expected in done["file_sha256"].items():
        path = inside(folder / name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing/corrupt completed output: {path}. Do not silently overwrite a completed fit.")
    return done


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                f.write(b"0"); f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise RuntimeError(f"Already running: {path.parent}") from e
            try:
                yield
            finally:
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise RuntimeError(f"Already running: {path.parent}") from e
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.stream.flush(); self.log.flush()
    def isatty(self):
        return False


def environment(torch, device) -> dict:
    import numpy as np
    import pandas as pd
    import platform
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pandas": pd.__version__, "cuda_build": torch.version.cuda,
            "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
