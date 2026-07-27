# E4 prediction-residual baseline readiness

## Scope

E4 asks which part of the 18-dimensional mouth motion remains after causal audio
prediction. It starts with residual measurement only:

```text
residual[t, d] = oracle_motion[t, d] - audio_gru_prediction[t, d]
```

Sparse selection, AWGN, JSCC, channel-aware training, and model retraining are
outside the first E4 experiment.

## Entry gate

Do not start E4 until the ten-speaker E3 run has all of the following:

- 800/100/100 train/validation/test samples with speaker identity isolation;
- three complete GRU seeds and no prediction schema/path failures;
- mean validation L1 better than `train_mean`;
- 200/200 LivePortrait reconstruction samples, zero failures, and a completion
  marker;
- committed source code and an ignored experiment-output directory.

The ten-speaker gate has passed. The frozen reconstruction evaluation completed
all 200 validation/test samples, produced 1,200 rows with zero failures, and
selected seed 43 using validation L1 only.

## Frozen inputs

- Use the best seed selected only by validation L1.
- Use its existing validation/test predictions; do not select a seed on test.
- Keep the existing train-only motion normalization statistics frozen.
- Compute residuals in both normalized space and original 18-D motion space.
- Report both raw-magnitude and train-normalized-magnitude Top-K. The frozen
  train standard deviations span several orders of magnitude, so neither
  ranking is silently treated as the only oracle.
- Preserve the first-frame zero convention and valid masks.
- Report validation and test separately and retain speaker identifiers.

## First experiment

The first E4 implementation should produce:

1. residual L1/RMSE and velocity statistics per dimension;
2. residual energy concentration curves over dimensions and frames;
3. oracle top-k magnitude retention for fixed budgets as an upper bound;
4. random fixed-budget retention as a control;
5. reconstruction metrics for the unmodified prediction, full residual oracle,
   and retained-residual conditions.

This experiment establishes whether the residual is concentrated enough to make
sparse transmission plausible. It does not yet claim a deployable selector.

The primary rate axis is retained residual values per non-reference valid frame,
`K` out of 18. Float32 storage counts and fixed-width/adaptive index counts are
reported only as uncoded accounting proxies. They exclude audio, the reference
image, entropy coding, packet headers, channel coding, and quantization, and
must not be described as an achieved bitrate. Random masks use a shared
deterministic schedule and therefore do not require per-frame adaptive indices.

## Required artifacts

Write ignored outputs under `outputs/residual_baseline/<timestamp>/`, including
resolved configuration, E3 fingerprint, selected seed, input hashes, per-sample
JSONL, aggregate JSON/CSV, and concentration plots. Resume protection must bind
the run to the E3 experiment fingerprint and prediction files.

## Ten-speaker motion-space result

The first real run used 100 validation samples (`s3`) and 100 test samples
(`s7`). It selected E3 seed 43 from validation L1 and hashed all 200 prediction
files. The run produced 6,600 fixed-budget rows and resumed without rewriting
its completion marker.

On test, the uncorrected prediction has raw-motion L1 `0.001893`. Raw-magnitude
Top-K reduces it to `0.001464/0.001189/0.000785/0.000501/0.000223/0.000064` for
`K=1/2/4/6/9/12`; random retention at `K=4` gives `0.001471`. Raw Top-K at
`K=4` retains about 82.3% of raw residual energy while sending 4 of 18 values
per eligible frame. Train-normalized-magnitude Top-K at the same budget gives
L1 `0.001203` and retains about 47.7% of raw energy. This confirms that residual
energy is concentrated and that scale choice materially affects oracle
selection.

These are oracle-magnitude selection results because the sender observes the
true residual. They support proceeding to reconstruction and later fixed-rule
selector experiments, but do not yet establish a deployable importance
predictor or a channel bitrate.
