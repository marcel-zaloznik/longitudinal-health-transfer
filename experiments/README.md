# Revision experiments

Run from the repository root, after the main runs and `python evaluate.py --strict`.
All inputs, references, splits, predictions, checkpoints and generated archives
stay local. Do not upload the generated experiment folders or `results_to_share.zip`.

## Run order

```bash
python experiments/prepare_references.py source_exclusion
python experiments/source_exclusion/experiment.py prepare
python experiments/source_exclusion/experiment.py run --all --device cuda:0
python experiments/source_exclusion/experiment.py summarize

python experiments/prepare_references.py learning_rate
python experiments/learning_rate/experiment.py prepare
python experiments/learning_rate/experiment.py run --all --device cuda:0
python experiments/learning_rate/experiment.py summarize

python experiments/prepare_references.py scale_consistent
python experiments/scale_consistent/experiment.py prepare
python experiments/scale_consistent/experiment.py smoke --device cuda:0
python experiments/scale_consistent/experiment.py run --all --device cuda:0
python experiments/scale_consistent/experiment7.py prepare
python experiments/scale_consistent/experiment7.py smoke --device cuda:0
python experiments/scale_consistent/experiment7.py run --all --device cuda:0
python experiments/scale_consistent/experiment7.py summarize

python experiments/prepare_references.py grud_transfer
python experiments/grud_transfer/experiment.py prepare
python experiments/grud_transfer/experiment.py run --all --device cuda:0
python experiments/grud_transfer/experiment.py summarize
```

Use the same device type and package environment throughout. For CPU runs,
replace every `--device cuda:0` with `--device cpu`; there is no automatic fallback.
`run --dry-run`, `status` and `--help` list jobs and options. Each `PROTOCOL.md`
records the comparison; model and seed settings remain in the experiment modules.
Learning-rate and source-exclusion summaries restore float32 error precision
before the original float64 metric calculation.

The scale-consistent stage needs completed source-exclusion checkpoints and
receipts, not just aggregate tables. Its seven-target extension is required for
the paper. Reference generation is intentionally non-overwriting; code and input
hashes are checked on preparation and resume. Do not mix these adapted scripts
with manifests or checkpoints from the private working-folder packages.

## Tests

Synthetic regression checks (input-dependent checks are skipped when absent):

```bash
python experiments/source_exclusion/test_summarize_float32.py
python experiments/learning_rate/test_summary_fix.py
python experiments/scale_consistent/tests.py
python experiments/scale_consistent/tests7.py
python experiments/grud_transfer/tests.py
```

`source_exclusion/tests.py` also requires prepared private inputs;
`learning_rate/tests.py` requires its locally generated references.
