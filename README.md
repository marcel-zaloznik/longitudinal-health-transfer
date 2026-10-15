# Transfer learning for longitudinal health prediction

Code for *Transfer learning for irregular longitudinal multi-target health prediction across heterogeneous datasets*, by Marcel Založnik and Mitja Luštrek.

Work carried out at the Jožef Stefan Institute (IJS) for the SmartCHANGE project.

## Input and installation

The private, analysis-ready input is `data/dataset.csv`; participant data are not distributed here. Required columns, order and units are in `metadata/codebook.csv`. Missing values are empty cells. Participant identifiers must be unique across cohorts. Smoking uses `No`, `Quit/strike`, `Yes`; Diabetes uses `No`, `Yes`. The scripts encode these as 0/1/2 and 0/1, respectively.

Original environment: Python 3.14.2. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Experiments

Run from the repository directory:

```bash
python prepare_data.py
python baselines.py --model persistence
python grud.py
python strats.py --mode target_only
python strats.py --mode topk_pretrain --n-origins 6
python strats.py --mode pooled --n-origins 6
```

`--model persistence` produces the train-mean, age/sex-mean and LOCF-with-age/sex-fallback baselines. Use each script's `--help` for paths and settings. Existing error files are not overwritten; use a new output path or results directory for a different run.

Paper settings use five participant-level folds. Neural seeds are 1234, 2345 and 3456; baseline seed is 1234. Set fractions with `--train-data-percent` and seeds with `--seed`.

| Experiment | Target-training percentages | Source settings |
|---|---|---|
| GRU-D | 10, 30, 50, 70, 90, 100 | Target-only |
| STraTS | 10, 30, 50, 70, 90, 100 | Target-only; six-source pretraining and pooling |
| STraTS top-k | 10, 50, 100 | Pretraining and pooling with `--n-origins` 1–5 |
| STraTS single-source | 10, 100 | Each cohort as `--source-dataset`; self-transfer is skipped |
| STraTS observation-cap analysis | 100 | Target-only and six-source pretraining with `--max-obs -1` |
| Persistence and linear regression | 10, 30, 50, 70, 90, 100 | Target-only and pooled |
| Random forest | 10, 100 | Target-only and pooled |

Single-source names and the frozen top-k order are in `ORIGIN_MAP` in `strats.py`. Other model settings use the script defaults.

## Analysis

```bash
python evaluate.py --strict
python transfer_analysis.py --rank-only
```

Evaluation summarizes available runs and rejects invalid errors. Missing ground-truth targets are not evaluated. The main metric is `main_macro_nrmse`, with `panel=primary_9` and `horizon_group=all`. Coverage is checked at dataset/fold/attribute level, not for every individual forecast.

Ranking uses public cohort metadata and checks the frozen source order. Full transfer statistics (`python transfer_analysis.py`) require the complete experiment grid above. The public workflow produces analysis CSVs; final paper figures and LaTeX tables are not generated here.

Derived data go to `data/`; errors and analysis outputs go to `results/`.

## Licence

MIT, copyright 2026 Jožef Stefan Institute (IJS). See `LICENSE` and `THIRD_PARTY_NOTICES.md`. The software licence does not grant access to or redistribution rights for the private datasets.
