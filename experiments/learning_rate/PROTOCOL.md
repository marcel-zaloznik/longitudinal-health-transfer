# Target-only learning rate

Seven targets; 10% and 100% target training; five folds; seeds 1234, 2345 and
3456. Fit both 1e-4 and 5e-4 with paired initialization and loader seeds.
For each target/fold/fraction, choose the rate with lower mean validation loss
across seeds; an exact tie selects 1e-4. Test scores do not select the rate.
There are 420 fits. Other settings are in `engine.py` and `exp2_utils.py`.

Use `summarize_fixed.py` for strict reconstruction of saved float32 errors.
