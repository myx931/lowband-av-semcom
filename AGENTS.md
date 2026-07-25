# AGENTS.md

## Project purpose

This repository supports a master's thesis prototype on low-bandwidth audiovisual semantic communication for video conferencing.

The current research hypothesis is that audio can predict part of mouth motion, so the communication system should transmit only important mouth-motion prediction residuals instead of all visual motion features.

## Research constraints

* This is a master's graduation project, not a publication-oriented project.
* Prefer feasibility, reproducibility, and clear ablations over novelty at any cost.
* Maximum expected compute is approximately two RTX 3090 GPUs.
* Every core experiment should preferably support a single-GPU configuration.
* Use public datasets only.
* Do not propose collecting or annotating a new dataset.
* Do not train large generative models from scratch.
* Do not introduce reinforcement learning unless explicitly requested.
* Do not introduce diffusion video models, large multimodal models, or complex protocol stacks without approval.
* Start with GRID and an AWGN channel.
* Prefer low-dimensional facial or mouth motion representations.
* Prefer pretrained reconstruction models through adapters.
* Keep third-party code isolated in third_party/.
* Do not commit datasets, checkpoints, credentials, caches, or experiment outputs.

## Development rules

* Use Python 3.10 or 3.11 and PyTorch.
* Use a src-layout package.
* Use YAML configuration files.
* Do not hard-code absolute paths.
* All experiments must use fixed random seeds.
* All new modules should have type hints.
* Add unit tests for reusable logic.
* Add a smoke test for every executable pipeline.
* Run lint and tests before claiming completion.
* Never report an experiment as successful without command output or saved results.
* Never invent paper results, dataset statistics, pretrained weights, or benchmark numbers.
* Mark unsupported assumptions as TODO.
* Keep implementations minimal until the preceding milestone passes.

## Research implementation order

1. Repository and documentation.
2. GRID subset preprocessing.
3. Motion extraction.
4. Reconstruction baseline.
5. Audio-to-mouth-motion prediction.
6. Prediction residual analysis.
7. Sparse residual selection.
8. AWGN and lightweight JSCC.
9. Channel-aware learnable residual selection.
10. Full comparison and ablation experiments.

Do not skip directly to later milestones.

## Git workflow

* Do not commit directly to main.
* Use a dedicated branch for each milestone.
* Keep commits small and descriptive.
* Do not mix formatting-only changes with research logic.
* Every pull request must state:

  * what changed;
  * why;
  * how it was tested;
  * known limitations;
  * next step.

## Data and artifact policy

The following must remain outside normal Git tracking:

* datasets;
* extracted frames;
* audio features;
* checkpoints;
* pretrained weights;
* reconstructed videos;
* logs;
* TensorBoard files;
* local environment files;
* API keys.

Only small demonstration assets explicitly approved by the user may be committed.

## Communication style

When reporting progress:

* distinguish completed work from planned work;
* show exact commands used;
* include test results;
* state unresolved assumptions;
* recommend only one immediate next task;
* do not expand the project scope without explicit approval.
