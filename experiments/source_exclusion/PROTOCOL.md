# Source exclusion

Five potentially overlapping target cohorts; 10% and 100% target training;
five folds; seeds 1234, 2345 and 3456. Compare all sources with the whole-cohort
exclusions in `exclusions.json`, for pretraining and pooling. This is a
sensitivity analysis, not participant linkage. There are 30 source fits and
600 target fits. Architecture and optimizer settings are in `engine.py`;
job and seed definitions are in `exp1_utils.py`.

Use `summarize_float32.py` for strict reconstruction of saved float32 errors.
Source checkpoints are required by the scale-consistent experiment.
