# GRU-D transfer

Seven targets; 10% and 100% target training; five folds; seeds 1234, 2345 and
3456. Compare target-only training with pretraining on eligible source cohorts
and target fine-tuning. The cohort exclusions and all settings are in
`runtime.py`. There are 21 source fits and 420 target fits.

Keep 20 fixed dynamic feature positions, source input normalization, and map
the pretrained output head to target normalization. `analyse.py` reports
normalized scores, original-unit errors and calibration. Historical GRU-D,
random-forest and STraTS results are contextual comparisons, not tuning data.
