"""Filesystem guards, frozen protocol and reproducibility utilities."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import contextlib, hashlib, json, os, platform, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
VERSION = '1.0.0'
SEEDS = (1234, 2345, 3456)
PERCENTS = (10, 100)
LABELS = {'ABCD':'ABCD', 'BE_KUL_Wave_1A':'LGS boys', 'BE_KUL_Wave_1B':'LGS girls',
          'BE_KUL_Wave_23456':'Leuven adults', 'SLOfit_lifelong':'SLOfit lifelong',
          'SLOfit_students':'SLOfit students', 'YFS':'YFS'}
ORIGINS = tuple(LABELS)
ATTRS = ('Height','Weight','Waist_circumference','BMI','Systolic_blood_pressure',
         'Diastolic_blood_pressure','Resting_heart_rate','Glucose','Total_cholesterol',
         'HDL','LDL','Triglycerides','Smoking','Fitness_speed','PA_h_per_w','Alcohol',
         'Diabetes','Sleep','Diet_HHS','Diet_T2P')
EXCLUSIONS = {'BE_KUL_Wave_1A':['BE_KUL_Wave_23456'], 'BE_KUL_Wave_1B':['BE_KUL_Wave_23456'],
              'BE_KUL_Wave_23456':['BE_KUL_Wave_1A','BE_KUL_Wave_1B'],
              'SLOfit_lifelong':['SLOfit_students'], 'SLOfit_students':['SLOfit_lifelong']}
ARMS = ('target_only', 'transfer')
HYPER = dict(hidden_size=64, batch_size=512, learning_rate=0.001, optimizer='Adam',
             max_epochs=1000, patience=10, min_delta=1e-4, dtype='float32',
             num_workers=0, gradient_clipping=None, weight_decay=0.0,
             mixed_precision=False, fixed_dynamic_slots=20)
INPUT_NAMES = ('dataset.csv','input_output_pairs.csv','split5k.json',
               'target_train_subsets_by_dataset_fold_percent.json',
               'target_attribute_stds_by_dataset_fold.json','eligible_dataset_attribute_pairs.json',
               'primary_target_panel.json')
CODE_FILES = ('runtime.py','data.py','model.py','training.py','metrics.py','analyse.py','experiment.py','tests.py')

def inside(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute(): path = ROOT/path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f'Refusing a write outside this experiment folder: {path}')
    return path

def configure() -> None:
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key, folder in {'TMPDIR':'tmp','TEMP':'tmp','TMP':'tmp','XDG_CACHE_HOME':'cache',
        'TORCH_HOME':'torch','TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton',
        'CUDA_CACHE_PATH':'cuda','MPLCONFIGDIR':'matplotlib','PIP_CACHE_DIR':'pip',
        'NUMBA_CACHE_DIR':'numba','JOBLIB_TEMP_FOLDER':'joblib'}.items():
        p = inside(ROOT/'_runtime'/folder); p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
    tempfile.tempdir = str(inside(ROOT/'_runtime/tmp'))
    os.chdir(ROOT)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()

def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read_json(path: Path):
    with Path(path).open(encoding='utf-8') as f: return json.load(f)

def atomic_json(path: Path, obj) -> None:
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=inside(path.with_name(path.name+f'.{os.getpid()}.tmp'))
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    os.replace(tmp,path)

def atomic_csv(path: Path, frame) -> None:
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=inside(path.with_name(path.name+f'.{os.getpid()}.tmp'))
    frame.to_csv(tmp,index=False,compression='gzip' if path.name.endswith('.gz') else None,
                 float_format='%.17g')
    os.replace(tmp,path)

def atomic_text(path: Path, text: str) -> None:
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=inside(path.with_name(path.name+f'.{os.getpid()}.tmp'))
    tmp.write_text(text,encoding='utf-8'); os.replace(tmp,path)

def atomic_npz(path: Path, **arrays) -> None:
    import numpy as np
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=inside(path.with_name(path.name+f'.{os.getpid()}.tmp'))
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def atomic_torch(path: Path, state) -> None:
    import torch
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=inside(path.with_name(path.name+f'.{os.getpid()}.tmp'))
    torch.save(state,tmp); os.replace(tmp,path)

@contextlib.contextmanager
def lock(path: Path):
    """OS-held lock: crashes release it; do not delete lock files."""
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
            try: msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e: raise RuntimeError(f'Another process holds {path}') from e
        else:
            import fcntl
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as e: raise RuntimeError(f'Another process holds {path}') from e
        yield
    finally:
        f.close()

def package_fingerprint() -> str:
    names=list(CODE_FILES)+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'reference').glob('*')) if p.is_file()]
    return digest({n:sha256(ROOT/n) for n in names})

def jobs():
    return [dict(job_id=i,target=t,seed=s) for i,(s,t) in enumerate((s,t) for s in SEEDS for t in ORIGINS)]

def sources(target: str) -> list[str]:
    return [t for t in ORIGINS if t!=target and t not in EXCLUSIONS.get(target,[])]

def stage_seed(seed: int, target: str, fold: int | None=None) -> int:
    return seed+(100000 if fold is None else 200000)+1000*ORIGINS.index(target)+(0 if fold is None else fold)

def loader_seed(seed: int, target: str, fold: int | None=None) -> int:
    return seed+(1000 if fold is None else 2000)*(ORIGINS.index(target)+1)+(0 if fold is None else fold)+1

def set_seeds(seed: int) -> None:
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def state_digest(state) -> str:
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def environment(device, cpu_threads: int) -> dict:
    import numpy as np, pandas as pd, torch
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                torch=str(torch.__version__),cuda_build=torch.version.cuda,device_type=device.type,
                device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                cuda_capability=list(torch.cuda.get_device_capability(device)) if device.type=='cuda' else None,
                cpu_threads=cpu_threads,matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                float32_matmul_precision=torch.get_float32_matmul_precision())

def environment_signature(env: dict) -> dict:
    return {k:v for k,v in env.items() if k not in ('device_name','cuda_capability')}

class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for s in self.streams: s.write(text); s.flush()
        return len(text)
    def flush(self):
        for s in self.streams: s.flush()
