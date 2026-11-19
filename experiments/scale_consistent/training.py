"""Original Experiment 1 training/logging wrapper; optimization is unchanged."""
from __future__ import annotations
import csv, math, os, time
from pathlib import Path
from exp3_utils import inside, atomic_json, sha256, read_json

def choose_device(name: str):
    import torch
    device = torch.device(name)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Use cuda:0 (GPU) or cpu.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Activate your existing CUDA-enabled PyTorch environment; no silent CPU fallback is used.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device

def save_weights(torch, state, path: Path) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)

def training(engine, model, train, val, stats, *, device, batch, lr, epochs, offset, name, folder):
    folder = inside(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not train or not val:
        raise ValueError(f"{name}: empty training or validation records; refusing an untrained result.")
    history_path = folder / "history.csv"
    start = time.monotonic()
    history = []
    last_report = [0.0, None]
    def batch_progress(phase, epoch, completed, total):
        now = time.monotonic()
        key = (phase, epoch)
        if key != last_report[1] or completed == total or now - last_report[0] >= 30:
            atomic_json(folder / "progress.json", {
                "stage": name, "phase": phase, "epoch": epoch,
                "batches_completed": completed, "batches_total": total,
                "elapsed_seconds": round(now - start, 1), "updated_unix_time": time.time(),
            })
            print(f"{name} epoch {epoch} {phase}: {completed}/{total} batches "
                  f"(stage elapsed {now - start:.0f}s)", flush=True)
            last_report[:] = [now, key]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse", "validation_mse", "elapsed_seconds"])
        def callback(epoch, train_loss, val_loss):
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError(f"{name}: nonfinite loss at epoch {epoch}; no successful checkpoint recorded.")
            history.append((int(epoch), float(train_loss), float(val_loss)))
            writer.writerow([epoch, train_loss, val_loss, round(time.monotonic() - start, 3)])
            handle.flush()
        engine.TRAIN_TRACE_CALLBACK = callback
        engine.BATCH_TRACE_CALLBACK = batch_progress
        try:
            state = engine.train_model(model=model, train_records=train, val_records=val,
                target_stats=stats, batch_size=batch, lr=lr, max_epochs=epochs,
                patience=engine.PATIENCE, device=device, seed_offset=offset, name=name)
        finally:
            engine.TRAIN_TRACE_CALLBACK = None
            engine.BATCH_TRACE_CALLBACK = None
    if not history:
        raise RuntimeError(f"{name}: training completed without any epochs.")
    # Apply the original min-delta rule when identifying the selected epoch.
    best_epoch, best_val = 0, float("inf")
    for epoch, _, loss in history:
        if loss + engine.MIN_DELTA < best_val:
            best_epoch, best_val = epoch, loss
    return state, {"epochs_run": len(history), "best_epoch": best_epoch,
                   "selected_validation_mse": best_val, "elapsed_seconds": time.monotonic() - start,
                   "history_sha256": sha256(history_path)}

def verified_done(folder: Path, config_hash: str, required: tuple[str, ...]) -> dict | None:
    done_path = folder / "done.json"
    if not done_path.exists():
        return None
    done = read_json(done_path)
    if done.get("config_hash") != config_hash:
        raise RuntimeError(f"Stored configuration mismatch: {folder}")
    for name in sorted(set(required) | set(done.get("file_sha256", {}))):
        path = folder / name
        if not path.exists() or sha256(path) != done["file_sha256"].get(name):
            raise RuntimeError(f"Completed output is missing or corrupted: {path}. Restore it or remove this stage directory before rerunning.")
    return done
