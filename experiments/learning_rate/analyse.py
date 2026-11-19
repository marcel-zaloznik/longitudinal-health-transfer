"""Original person-balanced primary-panel metric; explicit validation-only LR selection."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
from exp2_utils import (ROOT, SEEDS, PERCENTS, LRS, TARGET_LABELS, inside, read_json,
    atomic_json, atomic_csv, sha256, object_hash, load_setup, jobs, file_lock, verified_done)

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



def select_rates(validation):
    """This function cannot inspect test results: its input schema excludes them."""
    import numpy as np
    import pandas as pd
    required = {"target", "percent", "fold", "seed", "lr_key", "validation_mse"}
    if set(validation.columns) != required:
        raise ValueError("LR selection accepts validation scores and identifiers ONLY.")
    if not np.isfinite(validation.validation_mse.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite validation score.")
    if (validation.validation_mse < 0).any():
        raise ValueError("Negative validation MSE.")
    if not set(validation.target).issubset(TARGET_LABELS) or not set(validation.percent).issubset(PERCENTS):
        raise ValueError("Unknown dataset or training fraction.")
    if not set(validation.fold).issubset(range(5)):
        raise ValueError("Unknown fold.")
    if validation.duplicated(["target", "percent", "fold", "seed", "lr_key"]).any():
        raise ValueError("Duplicated validation fit.")
    records = []
    for (target, percent, fold), group in validation.groupby(["target", "percent", "fold"], sort=True):
        if len(group) != 6 or set(zip(group.lr_key, group.seed)) != {(key, seed) for key in LRS for seed in SEEDS}:
            raise ValueError("Every selection requires two rates and the same three seeds.")
        means = {key: float(group.loc[group.lr_key.eq(key), "validation_mse"].mean()) for key in LRS}
        chosen = min(LRS, key=lambda key: (means[key], LRS[key]))
        records.append({"target": target, "target_label": TARGET_LABELS[target],
            "percent": int(percent), "fold": int(fold), "selected_lr_key": chosen,
            "selected_learning_rate": LRS[chosen], "n_seeds": 3,
            "mean_validation_mse_lr1e4": means["lr1e-4"],
            "mean_validation_mse_lr5e4": means["lr5e-4"],
            "exact_tie": means["lr1e-4"] == means["lr5e-4"]})
    return pd.DataFrame(records)


def verify_history(stage, done):
    import numpy as np
    import pandas as pd
    h = pd.read_csv(stage / "history.csv", float_precision="round_trip")
    details = done["training"]
    if len(h) != details["epochs_run"] or not 1 <= len(h) <= 1000:
        raise ValueError("Training-history length disagrees with completion record.")
    if h.epoch.tolist() != list(range(1, len(h)+1)):
        raise ValueError("Nonconsecutive history epochs.")
    if not np.isfinite(h[["train_mse", "validation_mse"]].to_numpy()).all():
        raise ValueError("Nonfinite loss in completed training history.")
    best_epoch, best = 0, float("inf")
    for row in h.itertuples(index=False):
        if row.validation_mse + 1e-5 < best:
            best_epoch, best = int(row.epoch), float(row.validation_mse)
    if best_epoch != details["best_epoch"] or not math.isclose(best, details["selected_validation_mse"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Saved checkpoint score does not follow the original min-delta rule.")
    if bool(details["hit_epoch_cap"]) != (len(h)==1000):
        raise ValueError("Inconsistent epoch-cap flag.")
    if len(h) < 1000 and len(h)-best_epoch != 10:
        raise ValueError("Completed research fit did not stop using the original patience rule.")
    return h


def seed_summary(scores, keys):
    import pandas as pd
    rows = []
    for key, group in scores.groupby(keys, sort=True):
        if len(group)!=3 or set(group.seed)!=set(SEEDS):
            raise ValueError("Expected all three seeds for every summary row.")
        values = dict(zip(keys, key if isinstance(key,tuple) else (key,)))
        values.update(n_seeds=3, nrmse_mean=float(group.nrmse.mean()),
                      nrmse_sd=float(group.nrmse.std(ddof=1)),
                      nmae_mean=float(group.nmae.mean()), nmae_sd=float(group.nmae.std(ddof=1)))
        rows.append(values)
    return pd.DataFrame(rows)


def make_overall(scores):
    import pandas as pd
    rows = []
    for (percent, policy, seed), group in scores.groupby(["percent", "policy", "seed"], sort=True):
        if len(group)!=7 or set(group.target)!=set(TARGET_LABELS):
            raise ValueError("Overall scores require all seven datasets, equally weighted.")
        rows.append({"target":"Overall", "percent":int(percent), "policy":policy, "seed":int(seed),
                     "nrmse":float(group.nrmse.mean()), "nmae":float(group.nmae.mean()), "n_datasets":7})
    return pd.DataFrame(rows)


def historical_comparisons(scores, overall, reference):
    import pandas as pd
    expected_models = {"strats_no_pretrain", "strats_main", "strats_pooled_all_sources"}
    for (target, percent, seed), group in scores.groupby(["target", "percent", "seed"]):
        old = reference[(reference.dataset==target) & (reference.train_data_percent==percent) & (reference.seed==seed)]
        if len(old)!=3 or set(old.base_configuration_id)!=expected_models:
            raise ValueError("Missing historical comparison.")
        for new_col, old_col in (("n_primary_errors","n_error_rows"),("n_test_people","n_test_people"),
                                 ("n_attributes","n_attributes_evaluated"),("n_folds","n_folds_evaluated")):
            if len(set(group[new_col]))!=1 or len(set(old[old_col]))!=1 or int(group[new_col].iloc[0])!=int(old[old_col].iloc[0]):
                raise ValueError(f"Fresh/historical metric coverage differs: {target}, {new_col}")
    ref = reference.rename(columns={"dataset":"target", "train_data_percent":"percent"}).copy()
    global_ref = []
    for (percent, seed, model), group in ref.groupby(["percent","seed","base_configuration_id"]):
        if len(group)!=7:
            raise ValueError("Historical overall score lacks datasets.")
        global_ref.append({"target":"Overall","percent":percent,"seed":seed,
            "base_configuration_id":model,"main_macro_nrmse":float(group.main_macro_nrmse.mean())})
    ref = pd.concat([ref,pd.DataFrame(global_ref)],ignore_index=True).set_index(["target","percent","seed","base_configuration_id"])
    fresh = pd.concat([scores,overall],ignore_index=True)
    rows = []
    for (target, percent, seed), group in fresh.groupby(["target","percent","seed"],sort=True):
        values = group.set_index("policy").nrmse
        old_baseline = float(ref.loc[(target,percent,seed,"strats_no_pretrain"),"main_macro_nrmse"])
        baseline1 = float(values["fixed_lr1e-4"])
        baseline5 = float(values["fixed_lr5e-4"])
        selected = float(values["validation_selected"])
        if min(old_baseline,baseline1,baseline5,selected)<=0:
            raise ValueError("Cannot compute gains from nonpositive baseline NRMSE.")
        for model, label in (("strats_main","historical_pretrained"),("strats_pooled_all_sources","historical_pooled")):
            transfer = float(ref.loc[(target,percent,seed,model),"main_macro_nrmse"])
            rows.append({"target":target,"percent":int(percent),"seed":int(seed),"transfer_reference":label,
                "historical_target_only_nrmse":old_baseline,"fresh_lr1e4_nrmse":baseline1,
                "fresh_lr5e4_nrmse":baseline5,"selected_nrmse":selected,"historical_transfer_nrmse":transfer,
                "original_gain_pct":100*(old_baseline-transfer)/old_baseline,
                "gain_vs_fresh_lr1e4_pct":100*(baseline1-transfer)/baseline1,
                "gain_vs_selected_pct":100*(selected-transfer)/selected})
    per_seed = pd.DataFrame(rows)
    result = []
    quantities=("historical_target_only_nrmse","fresh_lr1e4_nrmse","fresh_lr5e4_nrmse",
                "selected_nrmse","historical_transfer_nrmse")
    for (target, percent, method),group in per_seed.groupby(["target","percent","transfer_reference"],sort=True):
        if set(group.seed)!=set(SEEDS) or len(group)!=3:
            raise ValueError("Missing historical-comparison seed.")
        row={"target":target,"percent":int(percent),"transfer_reference":method,"n_seeds":3}
        for quantity in quantities:
            row[quantity+"_mean"]=float(group[quantity].mean())
            row[quantity+"_sd"]=float(group[quantity].std(ddof=1))
        old=row["historical_target_only_nrmse_mean"]
        baseline=row["fresh_lr1e4_nrmse_mean"]
        selected=row["selected_nrmse_mean"]
        transfer=row["historical_transfer_nrmse_mean"]
        row.update(original_gain_pct=100*(old-transfer)/old,
                   gain_vs_fresh_lr1e4_pct=100*(baseline-transfer)/baseline,
                   gain_vs_selected_pct=100*(selected-transfer)/selected)
        row["gain_change_vs_fresh_lr1e4_pp"]=row["gain_vs_selected_pct"]-row["gain_vs_fresh_lr1e4_pct"]
        row["gain_change_vs_original_pp"]=row["gain_vs_selected_pct"]-row["original_gain_pct"]
        result.append(row)
    return per_seed,pd.DataFrame(result)


def baseline_changes(scores, overall, reference):
    import pandas as pd
    fresh=pd.concat([scores,overall],ignore_index=True)
    old=reference[reference.base_configuration_id.eq("strats_no_pretrain")].rename(
        columns={"dataset":"target","train_data_percent":"percent"})
    globals=[]
    for (percent,seed),group in old.groupby(["percent","seed"]):
        globals.append({"target":"Overall","percent":percent,"seed":seed,
                        "main_macro_nrmse":float(group.main_macro_nrmse.mean())})
    old=pd.concat([old,pd.DataFrame(globals)],ignore_index=True).set_index(["target","percent","seed"])
    rows=[]
    for (target,percent,seed),group in fresh.groupby(["target","percent","seed"],sort=True):
        v=group.set_index("policy").nrmse
        rows.append({"target":target,"percent":int(percent),"seed":int(seed),
            "lr5e4_minus_lr1e4":float(v["fixed_lr5e-4"]-v["fixed_lr1e-4"]),
            "selected_minus_lr1e4":float(v["validation_selected"]-v["fixed_lr1e-4"]),
            "fresh_lr1e4_minus_historical":float(v["fixed_lr1e-4"]-old.loc[(target,percent,seed),"main_macro_nrmse"])})
    per_seed=pd.DataFrame(rows)
    result=[]
    for (target,percent),group in per_seed.groupby(["target","percent"],sort=True):
        row={"target":target,"percent":int(percent),"n_seeds":3}
        for quantity in ("lr5e4_minus_lr1e4","selected_minus_lr1e4","fresh_lr1e4_minus_historical"):
            row[quantity+"_mean"]=float(group[quantity].mean())
            row[quantity+"_sd"]=float(group[quantity].std(ddof=1))
        result.append(row)
    return per_seed,pd.DataFrame(result)


def write_report(folder, dataset_summary, overall_summary, selections, gains, audits, hardware_warning):
    from exp2_utils import atomic_text
    summary = __import__("pandas").concat([dataset_summary,overall_summary],ignore_index=True)
    lines=["# Experiment 2 — target-only STraTS learning-rate sensitivity", "",
        "All 420 research fits passed completion, paired-initialization and evaluation-coverage checks.",
        "The learning rate was selected independently for each dataset/fold/training fraction using validation MSE averaged over three seeds. No test score entered selection.", "",
        "## Primary-panel results", "",
        "Mean ± sample SD over seeds 1234, 2345 and 3456. Lower NRMSE is better. Folds are combined at the person-MSE level before taking the attribute square root. Overall results weight all seven datasets equally.", "",
        "| Dataset | Training | Fixed 1e-4 | Fixed 5e-4 | Validation-selected | Folds choosing 5e-4 |",
        "|---|---:|---:|---:|---:|---:|"]
    for target in ["Overall"]+list(TARGET_LABELS):
        for percent in PERCENTS:
            group=summary[(summary.target==target)&(summary.percent==percent)].set_index("policy")
            fields=[f"{group.loc[p,'nrmse_mean']:.4f} ± {group.loc[p,'nrmse_sd']:.4f}" for p in ("fixed_lr1e-4","fixed_lr5e-4","validation_selected")]
            s=selections[selections.percent.eq(percent)]
            if target!="Overall":
                s=s[s.target.eq(target)]
            count=int(s.selected_lr_key.eq("lr5e-4").sum())
            label=TARGET_LABELS.get(target,target)
            lines.append(f"| {label} | {percent}% | {' | '.join(fields)} | {count}/{len(s)} |")
    lines += ["", "## Historical transfer comparisons (not newly trained transfer models)", "",
        "Positive gain means the historical transfer model has lower NRMSE than the indicated baseline. Gains are ratios of seed-mean NRMSEs, not averages of seed-wise percentages.", "",
        "| Training | Historical model | Gain vs historical target-only | Gain vs fresh 1e-4 | Gain vs validation-selected |",
        "|---|---|---:|---:|---:|"]
    for row in gains[gains.target.eq("Overall")].itertuples(index=False):
        lines.append(f"| {row.percent}% | {row.transfer_reference} | {row.original_gain_pct:.2f}% | {row.gain_vs_fresh_lr1e4_pct:.2f}% | {row.gain_vs_selected_pct:.2f}% |")
    lines += ["", "Dataset-level gains and baseline changes are in `historical_gains.csv` and `baseline_changes.csv`.",
        "", "## Training and scope checks", "",
        f"Fits reaching the 1,000-epoch cap: {int(audits.hit_epoch_cap.sum())}/420. Inspect `fit_audit.csv` for best/stopping epochs and `training_history.csv.gz` for full curves.",
        "Both rate arms used the same fresh initial weights and DataLoader seeds within each dataset/fold/percentage/seed pair; train/validation/test membership and normalization were unchanged.",
        "All original target attributes enter training and the original validation criterion; the frozen primary nine-outcome eligibility rules enter reported NRMSE.",
        "The validation-selected row is a selection procedure, not an ensemble and not the best of the two test results.",
        "", "## Interpretation limits", "",
        "This checks two prespecified constant learning rates at 10% and 100% target training. It does not establish complete hyperparameter optimization, address the intermediate-fraction ABCD curve, or retune pooled training.",
        "The primary paired sensitivity comparison is fresh 1e-4 versus fresh 5e-4/validation-selected. Historical target-only differences also reflect changed RNG orchestration; they are not attributable to learning rate alone.",
        "Pretrained/pooled reference models are the original all-source models. Their scores are not fresh matched experiments and do not incorporate Experiment 1 source exclusion. Do not treat this as resolving participant overlap.",
        "The 10% setting changes training availability only; validation/test sizes are unchanged. Seed SD is descriptive variability from three seeds after shared validation-based LR selection, not a confidence interval or an independent replication count of 15 folds.",
        "If transfer gains shrink, reverse or differ by cohort, report that directly. Persistent gains support robustness only to these tested rates."]
    if hardware_warning:
        lines += ["", "**Hardware audit:** multiple GPU model names or within-pair device changes were recorded. Inspect `environment_by_fit.csv`; matching seeds are not a cross-hardware bitwise guarantee."]
    lines += ["", "## Files to inspect/share", "",
        "Start with this report, `overall_summary.csv`, `dataset_summary.csv`, `selected_rates.csv`, `historical_gains.csv`, `baseline_changes.csv`, and `fit_audit.csv`.",
        "The generated `results_to_share.zip` contains aggregate scores, validation histories and provenance only. It excludes participant-level errors, identifiers and model weights. Keep the local raw errors for audit.", ""]
    atomic_text(folder/"REPORT.md","\n".join(lines))


def summarize(recompute=False):
    import numpy as np
    import pandas as pd
    import zipfile
    from exp2_utils import VERSION, CORE_FILES, atomic_text
    from experiment import pair_audit
    setup,project=load_setup()
    missing=[]
    for job in jobs():
        for fold in range(5):
            for key in LRS:
                if not (ROOT/"output/jobs"/f"job_{job['job_id']:03d}"/f"fold{fold}"/key/"done.json").is_file():
                    missing.append((job["job_id"],fold,key))
    if missing:
        raise RuntimeError(f"Cannot report an incomplete experiment: {len(missing)}/420 fits missing. First missing job/fold/rate: {missing[0]}. Use status.")
    folder=inside(ROOT/"summary")
    folder.mkdir(parents=True,exist_ok=True)
    with file_lock(folder/".summary.lock"):
        validation_rows, audit_rows, history_frames, env_rows, stages=[],[],[],[],[]
        signatures=set()
        print("Verifying all 420 fit files, paired controls and checkpoint-selection histories...",flush=True)
        for job in jobs():
            job_folder=ROOT/"output/jobs"/f"job_{job['job_id']:03d}"
            config=read_json(job_folder/"config.json")
            if config["smoke"] or config["job"]!=job or config["experiment_fingerprint"]!=setup["experiment_fingerprint"]:
                raise ValueError("Incompatible research job provenance.")
            signatures.add(object_hash(config["runtime_signature"]))
            ch=object_hash(config)
            for fold in range(5):
                pair=[]
                for key,lr in LRS.items():
                    stage=job_folder/f"fold{fold}"/key
                    done=verified_done(stage,ch)
                    if done is None or done["lr_key"]!=key or done["learning_rate"]!=lr:
                        raise ValueError("Incomplete or mismatched learning-rate fit.")
                    h=verify_history(stage,done)
                    details=done["training"]
                    ids={"target":job["target"],"percent":job["percent"],"fold":fold,"seed":job["seed"],"lr_key":key}
                    validation_rows.append({**ids,"validation_mse":details["selected_validation_mse"]})
                    audit_rows.append({"job_id":job["job_id"],**ids,"learning_rate":lr,**details,
                        "stage_seed":done["stage_seed"],"initial_state_sha256":done["initial_state_sha256"],
                        "n_train_records":done["n_train_records"],"n_validation_records":done["n_val_records"],
                        "n_test_records":done["n_test_records"],"n_train_people":done["train"]["n_people"],
                        "n_validation_people":done["validation"]["n_people"],"n_test_people":done["test"]["n_people"]})
                    for k,v in ids.items():
                        h[k]=v
                    history_frames.append(h)
                    env_rows.append({"job_id":job["job_id"],**ids,**done["environment"]})
                    stages.append((job,fold,key,stage,done))
                    pair.append(done)
                pair_audit(*pair)
        if len(signatures)!=1:
            raise ValueError("Research jobs used different software/numerical settings. Do not silently combine environments.")
        validation=pd.DataFrame(validation_rows)
        selections=select_rates(validation)  # No test metrics have been read.
        if len(selections)!=70:
            raise ValueError("Expected 70 dataset/fold/fraction LR selections.")
        selection_proof={"rule":setup["selection"],"experiment_fingerprint":setup["experiment_fingerprint"],
            "validation_input_sha256":object_hash(validation.to_dict(orient="records")),
            "selected_rates":selections.to_dict(orient="records"),"test_metrics_used":False}
        selection_path=folder/"selection_provenance.json"
        if selection_path.exists() and read_json(selection_path)!=selection_proof:
            raise ValueError("Previously frozen validation selection differs. Do not overwrite silently.")
        atomic_json(selection_path,selection_proof)
        atomic_csv(validation,folder/"validation_scores.csv")
        atomic_csv(selections,folder/"selected_rates.csv")
        print("Frozen 70 LR choices from validation only. Aggregating fixed-rate and selected test scores...",flush=True)
        choices=selections.set_index(["target","percent","fold"]).selected_lr_key.to_dict()
        metric_map={}
        for index,(job,fold,key,stage,done) in enumerate(stages,start=1):
            metrics=read_json(stage/"metrics.json")
            if metrics["target"]!=job["target"] or metrics["fold"]!=fold or metrics["identity_digest"]!=done["test_identity_digest"]:
                raise ValueError("Metric identity mismatch.")
            if recompute:
                errors=pd.read_csv(stage/"errors.csv.gz",dtype={"person_id":str},float_precision="round_trip")
                rebuilt=fold_metrics(errors,project,job["target"],fold)
                if rebuilt["identity_digest"]!=metrics["identity_digest"] or set(rebuilt["attributes"])!=set(metrics["attributes"]):
                    raise ValueError("Recomputed evaluation identities/panel changed.")
                for attr,values in rebuilt["attributes"].items():
                    for name,value in values.items():
                        if not np.isclose(value,metrics["attributes"][attr][name],rtol=1e-12,atol=1e-12):
                            raise ValueError("Recomputed sufficient statistics differ.")
                metrics=rebuilt
            metric_map[(job["target"],job["percent"],job["seed"],fold,key)]=metrics
            if index%20==0:
                print(f"Checked/loaded test metrics {index}/420",flush=True)
        # Verify each seed sees identical evaluation identities at a given rate/fold/fraction.
        for target in TARGET_LABELS:
            for percent in PERCENTS:
                for fold in range(5):
                    ids={metric_map[(target,percent,seed,fold,key)]["identity_digest"] for seed in SEEDS for key in LRS}
                    if len(ids)!=1:
                        raise ValueError("Evaluation observations differ across seeds/rates.")
        scores,attributes=[],[]
        for job in jobs():
            for policy in ("fixed_lr1e-4","fixed_lr5e-4","validation_selected"):
                folds=[]
                for fold in range(5):
                    key=choices[(job["target"],job["percent"],fold)] if policy=="validation_selected" else policy.removeprefix("fixed_")
                    folds.append(metric_map[(job["target"],job["percent"],job["seed"],fold,key)])
                values,attrs=combine_folds(folds)
                ids={"target":job["target"],"target_label":job["target_label"],"percent":job["percent"],"seed":job["seed"],"policy":policy}
                scores.append({**ids,**values})
                attributes.extend({**ids,**a} for a in attrs)
        scores=pd.DataFrame(scores)
        overall=make_overall(scores)
        dataset_summary=seed_summary(scores,["target","target_label","percent","policy"])
        overall_summary=seed_summary(overall,["target","percent","policy"])
        reference=pd.read_csv(ROOT/"reference/original_metrics.csv",float_precision="round_trip")
        gain_seed,gains=historical_comparisons(scores,overall,reference)
        change_seed,changes=baseline_changes(scores,overall,reference)
        counts=selections.groupby(["target","target_label","percent","selected_lr_key"],as_index=False).size().rename(columns={"size":"n_folds"})
        audits=pd.DataFrame(audit_rows)
        envs=pd.DataFrame(env_rows)
        outputs={"dataset_scores_by_seed.csv":scores,"overall_scores_by_seed.csv":overall,
            "dataset_summary.csv":dataset_summary,"overall_summary.csv":overall_summary,
            "attribute_scores_by_seed.csv":pd.DataFrame(attributes),"selection_counts.csv":counts,
            "historical_gains_by_seed.csv":gain_seed,"historical_gains.csv":gains,
            "baseline_changes_by_seed.csv":change_seed,"baseline_changes.csv":changes,
            "fit_audit.csv":audits,"environment_by_fit.csv":envs,
            "training_history.csv.gz":pd.concat(history_frames,ignore_index=True)}
        for name,table in outputs.items():
            atomic_csv(table,folder/name)
        hardware_warning=envs.device_name.nunique()>1
        write_report(folder,dataset_summary,overall_summary,selections,gains,audits,hardware_warning)
        # Aggregate-only audit and code are deliberately whitelisted for sharing.
        clean_partitions=pd.read_csv(ROOT/"setup/target_partitions.csv").drop(columns=["people_digest"])
        atomic_csv(clean_partitions,folder/"partition_counts.csv")
        atomic_json(folder/"provenance.json",{"version":VERSION,
            "experiment_fingerprint":setup["experiment_fingerprint"],"package_fingerprint":setup["package_fingerprint"],
            "input_sha256":setup["input_sha256"],"training_settings":setup["training_settings"],
            "seed_policy":setup["seed_policy"],"selection":setup["selection"],"limitations":setup["limitations"],
            "n_research_fits":420,"all_pairs_verified":True,"raw_metrics_recomputed":bool(recompute),
            "n_initialization_seeds":3,"n_selection_cells":70,"historical_transfer_not_rerun":True,
            "source":read_json(ROOT/"reference/provenance.json")})
        summary_names=list(outputs)+["validation_scores.csv","selected_rates.csv","selection_provenance.json",
            "REPORT.md","partition_counts.csv","provenance.json"]
        atomic_json(folder/"file_sha256.json",{n:sha256(folder/n) for n in summary_names})
        summary_names.append("file_sha256.json")
        out=inside(ROOT/"results_to_share.zip")
        temp=inside(ROOT/f"results_to_share.zip.{__import__('os').getpid()}.tmp")
        with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in summary_names:
                z.write(inside(folder/name),"summary/"+name)
            for name in CORE_FILES+("PROTOCOL.md","tests.py"):
                z.write(inside(ROOT/name),"package/"+name)
        __import__('os').replace(temp,out)
        print("Summary complete: summary/REPORT.md")
        print("Generated results_to_share.zip is local output, not a public release file.")
