# Scale-consistent STraTS transfer

Five-target stage: reuse the source-exclusion experiment's all-source
checkpoints and reset-normalization controls; fit 150 scale-consistent target
models. Seven-target extension: add ABCD and YFS, with six source fits and
120 target fits across reset and scale-consistent arms. Run both stages.

All comparisons use 10% and 100% target training, five folds and seeds 1234,
2345 and 3456. Source input/context/time normalization is retained in the
scale-consistent arm; `normalization.py` maps the output head to target units.
The target-only comparator is validation-selected in the learning-rate
experiment. Settings are in `engine.py`, `exp3_utils.py` and `seven_utils.py`.
