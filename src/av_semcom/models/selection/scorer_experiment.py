"""Training and frozen evaluation for the E6 channel-aware residual scorer."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.data import (
    ResidualDataset,
    ResidualExample,
    load_residual_example,
)
from av_semcom.models.jscc.experiment import _derived_noise_seed, _metric_row
from av_semcom.models.jscc.export import select_validation_model_seeds
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import (
    atomic_save_checkpoint,
    file_sha256,
    load_checkpoint,
)
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.selection.config import ResidualScorerSettings
from av_semcom.models.selection.gate import (
    GatePolicy,
    _atomic_write_jsonl,
    _environment,
    _git_commit,
    _load_model,
    _new_run_directory,
    _read_json,
    _read_jsonl,
    _resolve_device,
    _write_dict_csv,
)
from av_semcom.models.selection.scorer import (
    ChannelAwareResidualScorer,
    raw_position_velocity_loss,
    rule_selection_mask,
)
from av_semcom.utils.reproducibility import seed_everything

_METRICS = ("normalized_residual_mse", "l1", "rmse", "velocity_l1")
_DETERMINISTIC_METHODS = (
    "dense_jscc",
    "raw_magnitude",
    "normalized_magnitude",
    "fixed_train_magnitude",
)


def run_scorer_training(
    settings: ResidualScorerSettings,
    jscc: JSCCSettings,
    predictor: AudioMotionSettings,
    e5_run_dir: Path,
    gate_run_dir: Path,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
    formal: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Train scorer-only models while keeping E5 JSCC weights frozen."""

    if formal:
        jscc.require_formal_backend()
    e5_run_dir = e5_run_dir.resolve()
    gate_run_dir = gate_run_dir.resolve()
    source = _training_source_provenance(
        e5_run_dir,
        gate_run_dir,
        jscc,
        predictor,
    )
    fingerprint = config_fingerprint({"residual_scorer": settings.config, "source": source})
    run_dir = (
        run_directory.resolve()
        if run_directory is not None
        else _new_run_directory(settings.output_root)
    )
    _prepare_training_run(run_dir, settings, source, fingerprint, resume=resume)
    complete_path = run_dir / "training_complete.json"
    if complete_path.is_file():
        if not resume:
            raise FileExistsError("residual scorer training is complete; pass --resume")
        _require_fingerprint(_read_json(complete_path), fingerprint, complete_path)
        return run_dir, _read_json(run_dir / "training_summary.json")

    experiment_fingerprint = str(source["e5_experiment_fingerprint"])
    train_examples = _load_examples(e5_run_dir, experiment_fingerprint, "train")
    validation_examples = _load_examples(
        e5_run_dir,
        experiment_fingerprint,
        "validation",
    )
    if {example.speaker_id for example in train_examples} & {
        example.speaker_id for example in validation_examples
    }:
        raise ValueError("speaker leakage between scorer train and validation")
    normalizer = load_motion_normalizer(predictor.motion_stats_path)
    if normalizer.scope != "train_stats":
        raise ValueError("residual scorer requires train-only motion statistics")
    fixed_indices = _fixed_train_indices(
        train_examples,
        normalizer.std,
        settings.budgets,
    )
    atomic_write_json(
        run_dir / "fixed_train_indices.json",
        {
            "experiment_fingerprint": fingerprint,
            "selection_rule": "train_only_mean_absolute_raw_residual",
            "indices": {str(key): value for key, value in fixed_indices.items()},
        },
    )

    selected_e5_seeds = {
        int(key): int(value) for key, value in source["selected_e5_model_seeds"].items()
    }
    device = _resolve_device(settings.device)
    model_rows: list[dict[str, Any]] = []
    for channel_uses in jscc.channel_uses:
        frozen_jscc = _load_model(
            jscc,
            e5_run_dir,
            experiment_fingerprint,
            channel_uses,
            selected_e5_seeds[channel_uses],
            device,
        )
        frozen_jscc.requires_grad_(False)
        for seed in settings.seeds:
            model_dir = run_dir / "models" / f"c_{channel_uses}" / f"seed_{seed}"
            marker = model_dir / "complete.json"
            if marker.is_file():
                if not resume:
                    raise FileExistsError(f"scorer model exists: {model_dir}")
                payload = _read_json(marker)
                _require_fingerprint(payload, fingerprint, marker)
                model_rows.append(payload)
                continue
            model_rows.append(
                _train_one_scorer(
                    settings,
                    frozen_jscc,
                    train_examples,
                    validation_examples,
                    normalizer.std,
                    channel_uses,
                    seed,
                    selected_e5_seeds[channel_uses],
                    model_dir,
                    fingerprint,
                    device,
                )
            )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_fingerprint": fingerprint,
        "e5_experiment_fingerprint": experiment_fingerprint,
        "jscc_weights_frozen": True,
        "budget_definition": "K=2C selected semantic dimensions per eligible frame",
        "train_sample_count": len(train_examples),
        "validation_sample_count": len(validation_examples),
        "models": model_rows,
    }
    atomic_write_json(run_dir / "training_summary.json", summary)
    _write_dict_csv(run_dir / "training_summary.csv", model_rows)
    atomic_write_json(
        complete_path,
        {
            "experiment_fingerprint": fingerprint,
            "model_count": len(model_rows),
            "status": "complete",
        },
    )
    return run_dir, summary


def run_scorer_evaluation(
    settings: ResidualScorerSettings,
    jscc: JSCCSettings,
    predictor: AudioMotionSettings,
    e5_run_dir: Path,
    gate_run_dir: Path,
    run_dir: Path,
    *,
    resume: bool = False,
    formal: bool = True,
) -> dict[str, Any]:
    """Evaluate frozen scorer checkpoints and matched rules on test once."""

    if formal:
        jscc.require_formal_backend()
    e5_run_dir = e5_run_dir.resolve()
    gate_run_dir = gate_run_dir.resolve()
    run_dir = run_dir.resolve()
    source = _training_source_provenance(
        e5_run_dir,
        gate_run_dir,
        jscc,
        predictor,
    )
    fingerprint = config_fingerprint({"residual_scorer": settings.config, "source": source})
    _require_fingerprint(
        _read_json(run_dir / "training_complete.json"),
        fingerprint,
        run_dir / "training_complete.json",
    )
    test_source_path = e5_run_dir / "test_metrics.jsonl"
    test_source_hash = file_sha256(test_source_path)
    evaluation_fingerprint = config_fingerprint(
        {
            "experiment_fingerprint": fingerprint,
            "source_test_metrics_sha256": test_source_hash,
        }
    )
    complete_path = run_dir / "evaluation_complete.json"
    if complete_path.is_file():
        if not resume:
            raise FileExistsError("residual scorer evaluation is complete; pass --resume")
        complete = _read_json(complete_path)
        if complete.get("evaluation_fingerprint") != evaluation_fingerprint:
            raise ValueError("residual scorer evaluation fingerprint mismatch")
        return _read_json(run_dir / "evaluation_summary.json")

    experiment_fingerprint = str(source["e5_experiment_fingerprint"])
    examples = _load_examples(e5_run_dir, experiment_fingerprint, "test")
    normalizer = load_motion_normalizer(predictor.motion_stats_path)
    if normalizer.scope != "train_stats":
        raise ValueError("residual scorer evaluation requires train-only motion statistics")
    policy = GatePolicy.from_dict(_read_json(gate_run_dir / "policy.json"))
    if policy.experiment_fingerprint != experiment_fingerprint:
        raise ValueError("channel gate and E5 experiment fingerprints differ")
    fixed_payload = _read_json(run_dir / "fixed_train_indices.json")
    _require_fingerprint(
        fixed_payload,
        fingerprint,
        run_dir / "fixed_train_indices.json",
    )
    fixed_indices = {
        int(key): [int(value) for value in values]
        for key, values in fixed_payload["indices"].items()
    }
    source_rows = _read_jsonl(test_source_path)
    source_index = _source_metric_index(source_rows)
    selected_e5_seeds = {
        int(key): int(value) for key, value in source["selected_e5_model_seeds"].items()
    }
    device = _resolve_device(settings.device)
    rows: list[dict[str, Any]] = []
    maximum_dense_difference = 0.0
    for channel_uses in jscc.channel_uses:
        frozen_jscc = _load_model(
            jscc,
            e5_run_dir,
            experiment_fingerprint,
            channel_uses,
            selected_e5_seeds[channel_uses],
            device,
        )
        frozen_jscc.requires_grad_(False)
        scorers = {
            seed: _load_scorer(
                settings,
                normalizer.std,
                run_dir,
                fingerprint,
                channel_uses,
                seed,
                device,
            )
            for seed in settings.seeds
        }
        for example_index, example in enumerate(examples):
            print(
                f"[residual-scorer] C={channel_uses} "
                f"sample {example_index + 1}/{len(examples)}: {example.sample_id}",
                flush=True,
            )
            example_rows, difference = _evaluate_example(
                settings,
                jscc,
                frozen_jscc,
                scorers,
                example,
                example_index,
                channel_uses,
                selected_e5_seeds[channel_uses],
                normalizer.std,
                fixed_indices[channel_uses],
                policy,
                source_index,
                device,
            )
            rows.extend(example_rows)
            maximum_dense_difference = max(maximum_dense_difference, difference)
    expected_result_count = (
        len(examples)
        * len(jscc.channel_uses)
        * len(jscc.test_snr_db)
        * len(jscc.noise_seeds)
        * (len(_DETERMINISTIC_METHODS) + len(settings.random_seeds) + len(settings.seeds))
    )
    if len(rows) != expected_result_count:
        raise RuntimeError(f"expected {expected_result_count} scorer results, got {len(rows)}")
    _atomic_write_jsonl(run_dir / "test_metrics.jsonl", rows)
    summary = _summarize(rows, settings, jscc)
    summary.update(
        {
            "evaluation_fingerprint": evaluation_fingerprint,
            "experiment_fingerprint": fingerprint,
            "source_test_metrics_sha256": test_source_hash,
            "maximum_dense_metric_difference": maximum_dense_difference,
        }
    )
    atomic_write_json(run_dir / "evaluation_summary.json", summary)
    _write_dict_csv(run_dir / "evaluation_summary.csv", summary["aggregate"])
    _write_plots(run_dir / "plots", summary)
    atomic_write_json(
        complete_path,
        {
            "evaluation_fingerprint": evaluation_fingerprint,
            "source_test_metrics_sha256": test_source_hash,
            "result_count": len(rows),
            "expected_result_count": expected_result_count,
            "maximum_dense_metric_difference": maximum_dense_difference,
            "status": "complete",
        },
    )
    return summary


def _train_one_scorer(
    settings: ResidualScorerSettings,
    frozen_jscc: torch.nn.Module,
    train_examples: Sequence[ResidualExample],
    validation_examples: Sequence[ResidualExample],
    motion_std: np.ndarray,
    channel_uses: int,
    seed: int,
    e5_model_seed: int,
    model_dir: Path,
    fingerprint: str,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed, deterministic=settings.deterministic)
    scorer = ChannelAwareResidualScorer(
        motion_std=torch.from_numpy(np.asarray(motion_std, dtype=np.float32)),
        hidden_dim=settings.hidden_dim,
        temperature=settings.temperature,
        max_channel_uses=max(settings.budgets),
    ).to(device)
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ResidualDataset(train_examples),
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ResidualDataset(validation_examples),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    snr_generator = torch.Generator().manual_seed(seed + 20_000)
    k = settings.budgets[channel_uses]
    for epoch in range(1, settings.max_epochs + 1):
        train = _train_epoch(
            scorer,
            frozen_jscc,
            train_loader,
            optimizer,
            settings,
            channel_uses,
            k,
            device,
            snr_generator,
        )
        validation = _validation_epoch(
            scorer,
            frozen_jscc,
            validation_loader,
            settings,
            channel_uses,
            k,
            e5_model_seed,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_position_l1": train["position_l1"],
            "train_velocity_l1": train["velocity_l1"],
            "validation_loss": validation["loss"],
            "validation_position_l1": validation["position_l1"],
            "validation_velocity_l1": validation["velocity_l1"],
        }
        history.append(row)
        print(
            f"[residual-scorer] C={channel_uses} K={k} seed={seed} epoch={epoch} "
            f"train={train['loss']:.6f} validation={validation['loss']:.6f}",
            flush=True,
        )
        if validation["loss"] < best_loss - settings.early_stopping_min_delta:
            best_loss = validation["loss"]
            best_epoch = epoch
            stale = 0
            atomic_save_checkpoint(
                model_dir / "best.pt",
                {
                    "experiment_fingerprint": fingerprint,
                    "channel_uses": channel_uses,
                    "k": k,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "model_config": {
                        "hidden_dim": settings.hidden_dim,
                        "temperature": settings.temperature,
                        "max_channel_uses": max(settings.budgets),
                    },
                    "model_state": scorer.state_dict(),
                },
            )
        else:
            stale += 1
            if stale >= settings.early_stopping_patience:
                break
    _atomic_write_jsonl(model_dir / "history.jsonl", history)
    result = {
        "experiment_fingerprint": fingerprint,
        "channel_uses": channel_uses,
        "k": k,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epoch_count": len(history),
        "jscc_weights_frozen": True,
    }
    atomic_write_json(model_dir / "complete.json", result)
    return result


def _train_epoch(
    scorer: ChannelAwareResidualScorer,
    frozen_jscc: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    settings: ResidualScorerSettings,
    channel_uses: int,
    k: int,
    device: torch.device,
    snr_generator: torch.Generator,
) -> dict[str, float]:
    scorer.train()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for batch in loader:
        residual = batch["residual"].to(device)
        mask = batch["mask"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        snr = (
            torch.rand(
                (residual.shape[0], 1, 1),
                generator=snr_generator,
                dtype=torch.float32,
            )
            * (settings.train_snr_max_db - settings.train_snr_min_db)
            + settings.train_snr_min_db
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        selected = scorer(
            residual,
            mask,
            snr,
            k=k,
            channel_uses=channel_uses,
        ).selected_residual
        decoded = frozen_jscc(selected, mask, snr).decoded_residual
        loss = raw_position_velocity_loss(
            decoded,
            residual,
            valid_mask,
            scorer.motion_std,
            velocity_weight=settings.velocity_weight,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            scorer.parameters(),
            settings.gradient_clip_norm,
        )
        optimizer.step()
        batch_size = residual.shape[0]
        totals += (
            np.asarray(
                [
                    float(loss.total.detach().cpu()),
                    float(loss.position_l1.detach().cpu()),
                    float(loss.velocity_l1.detach().cpu()),
                ]
            )
            * batch_size
        )
        count += batch_size
    return _loss_dict(totals, count)


@torch.no_grad()
def _validation_epoch(
    scorer: ChannelAwareResidualScorer,
    frozen_jscc: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    settings: ResidualScorerSettings,
    channel_uses: int,
    k: int,
    e5_model_seed: int,
    device: torch.device,
) -> dict[str, float]:
    scorer.eval()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for snr_index, snr_db in enumerate(settings.validation_snr_db):
        for noise_seed in settings.noise_seeds:
            for batch_index, batch in enumerate(loader):
                residual = batch["residual"].to(device)
                mask = batch["mask"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                selected = scorer(
                    residual,
                    mask,
                    snr_db,
                    k=k,
                    channel_uses=channel_uses,
                ).selected_residual
                decoded = frozen_jscc(
                    selected,
                    mask,
                    snr_db,
                    noise_seed=_derived_noise_seed(
                        e5_model_seed,
                        noise_seed,
                        snr_index,
                        batch_index,
                    ),
                ).decoded_residual
                loss = raw_position_velocity_loss(
                    decoded,
                    residual,
                    valid_mask,
                    scorer.motion_std,
                    velocity_weight=settings.velocity_weight,
                )
                batch_size = residual.shape[0]
                totals += (
                    np.asarray(
                        [
                            float(loss.total.cpu()),
                            float(loss.position_l1.cpu()),
                            float(loss.velocity_l1.cpu()),
                        ]
                    )
                    * batch_size
                )
                count += batch_size
    return _loss_dict(totals, count)


@torch.no_grad()
def _evaluate_example(
    settings: ResidualScorerSettings,
    jscc: JSCCSettings,
    frozen_jscc: torch.nn.Module,
    scorers: Mapping[int, ChannelAwareResidualScorer],
    example: ResidualExample,
    example_index: int,
    channel_uses: int,
    e5_model_seed: int,
    motion_std: np.ndarray,
    fixed_indices: Sequence[int],
    policy: GatePolicy,
    source_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    maximum_dense_difference = 0.0
    residual = torch.from_numpy(example.normalized_residual).unsqueeze(0).to(device)
    mask = torch.from_numpy(example.transmission_mask).unsqueeze(0).to(device)
    std = torch.from_numpy(np.asarray(motion_std, dtype=np.float32)).to(device)
    k = settings.budgets[channel_uses]
    variants: list[
        tuple[
            str,
            int | None,
            torch.Tensor | ChannelAwareResidualScorer | None,
        ]
    ] = [
        ("dense_jscc", None, None),
        (
            "raw_magnitude",
            None,
            rule_selection_mask(
                residual,
                mask,
                k=k,
                method="raw_magnitude",
                motion_std=std,
            ),
        ),
        (
            "normalized_magnitude",
            None,
            rule_selection_mask(
                residual,
                mask,
                k=k,
                method="normalized_magnitude",
                motion_std=std,
            ),
        ),
        (
            "fixed_train_magnitude",
            None,
            rule_selection_mask(
                residual,
                mask,
                k=k,
                method="fixed_train_magnitude",
                motion_std=std,
                fixed_indices=torch.tensor(fixed_indices, dtype=torch.long),
            ),
        ),
    ]
    for random_seed in settings.random_seeds:
        variants.append(
            (
                "random",
                random_seed,
                rule_selection_mask(
                    residual,
                    mask,
                    k=k,
                    method="random",
                    motion_std=std,
                    random_seed=_sample_seed(random_seed, example.sample_id),
                ),
            )
        )
    for scorer_seed, scorer in scorers.items():
        variants.append(("learned_scorer", scorer_seed, scorer))

    for snr_index, snr_db in enumerate(jscc.test_snr_db):
        transmit = policy.should_transmit(channel_uses, snr_db)
        for noise_seed in jscc.noise_seeds:
            derived_noise = _derived_noise_seed(
                e5_model_seed,
                noise_seed,
                snr_index,
                example_index,
            )
            for method, method_seed, selection in variants:
                if not transmit:
                    decoded = np.zeros_like(example.normalized_residual)
                    frequency = [0.0] * 18
                else:
                    if method == "dense_jscc":
                        selected = residual
                        selection_mask = mask.unsqueeze(-1).expand_as(residual)
                    elif method == "learned_scorer":
                        scorer = selection
                        if not isinstance(scorer, ChannelAwareResidualScorer):
                            raise TypeError("learned scorer variant is invalid")
                        result = scorer(
                            residual,
                            mask,
                            snr_db,
                            k=k,
                            channel_uses=channel_uses,
                        )
                        selected = result.selected_residual
                        selection_mask = result.hard_mask
                    else:
                        if not isinstance(selection, torch.Tensor):
                            raise TypeError("rule selection mask is invalid")
                        selection_mask = selection
                        selected = residual * selection_mask
                    decoded = (
                        frozen_jscc(
                            selected,
                            mask,
                            snr_db,
                            noise_seed=derived_noise,
                        )
                        .decoded_residual[0]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    eligible = int(example.transmission_mask.sum())
                    if eligible <= 0:
                        raise ValueError(f"{example.sample_id} has no eligible frames")
                    frequency = (
                        selection_mask[0].float().sum(dim=0).cpu().numpy() / eligible
                    ).tolist()
                row = _metric_row(
                    example,
                    decoded,
                    np.asarray(motion_std, dtype=np.float32),
                    condition=method,
                    channel_uses=channel_uses,
                    model_seed=e5_model_seed,
                    snr_db=snr_db,
                    noise_seed=noise_seed,
                )
                row.update(
                    {
                        "method": method,
                        "method_seed": method_seed,
                        "k": 18 if method == "dense_jscc" else k,
                        "gate_threshold_db": policy.thresholds_db[channel_uses],
                        "gate_transmit": transmit,
                        "complex_channel_uses_used": channel_uses if transmit else 0,
                        "selected_dimension_frequency": frequency,
                    }
                )
                rows.append(row)
                if method == "dense_jscc" and transmit:
                    source = source_index[
                        (
                            example.sample_id,
                            channel_uses,
                            e5_model_seed,
                            snr_db,
                            noise_seed,
                        )
                    ]
                    maximum_dense_difference = max(
                        maximum_dense_difference,
                        *(abs(float(row[metric]) - float(source[metric])) for metric in _METRICS),
                    )
    return rows, maximum_dense_difference


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    settings: ResidualScorerSettings,
    jscc: JSCCSettings,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["method"],
                row.get("method_seed"),
                row["channel_uses"],
                row["snr_db"],
            )
        ].append(row)
    groups: list[dict[str, Any]] = []
    for (method, method_seed, channel_uses, snr_db), members in sorted(
        grouped.items(),
        key=lambda item: str(item[0]),
    ):
        groups.append(
            {
                "method": method,
                "method_seed": method_seed,
                "channel_uses": channel_uses,
                "k": members[0]["k"],
                "snr_db": snr_db,
                "gate_transmit": members[0]["gate_transmit"],
                "sample_noise_realization_count": len(members),
                **{
                    metric: float(np.mean([float(row[metric]) for row in members]))
                    for metric in _METRICS
                },
                "selected_dimension_frequency": np.mean(
                    [
                        np.asarray(row["selected_dimension_frequency"], dtype=np.float64)
                        for row in members
                    ],
                    axis=0,
                ).tolist(),
            }
        )
    aggregate: list[dict[str, Any]] = []
    for channel_uses in jscc.channel_uses:
        for snr_db in jscc.test_snr_db:
            for method in (*_DETERMINISTIC_METHODS, "random", "learned_scorer"):
                members = [
                    group
                    for group in groups
                    if group["method"] == method
                    and group["channel_uses"] == channel_uses
                    and group["snr_db"] == snr_db
                ]
                expected = (
                    len(settings.random_seeds)
                    if method == "random"
                    else len(settings.seeds)
                    if method == "learned_scorer"
                    else 1
                )
                if len(members) != expected:
                    raise ValueError(
                        f"expected {expected} seed groups for {method}, "
                        f"C={channel_uses}, SNR={snr_db}"
                    )
                row: dict[str, Any] = {
                    "method": method,
                    "channel_uses": channel_uses,
                    "k": members[0]["k"],
                    "snr_db": snr_db,
                    "gate_transmit": members[0]["gate_transmit"],
                    "seed_count": expected if expected > 1 else 0,
                }
                for metric in _METRICS:
                    values = [float(member[metric]) for member in members]
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_std"] = float(np.std(values))
                aggregate.append(row)
    return {
        "schema_version": 1,
        "status": "complete",
        "result_count": len(rows),
        "groups": groups,
        "aggregate": aggregate,
    }


def _fixed_train_indices(
    examples: Sequence[ResidualExample],
    motion_std: np.ndarray,
    budgets: Mapping[int, int],
) -> dict[int, list[int]]:
    total = np.zeros(18, dtype=np.float64)
    count = 0
    std = np.asarray(motion_std, dtype=np.float64)
    for example in examples:
        values = example.normalized_residual[example.transmission_mask] * std
        total += np.abs(values).sum(axis=0)
        count += values.shape[0]
    if count == 0:
        raise ValueError("training residuals contain no eligible frames")
    ranking = np.argsort(-(total / count), kind="stable")
    return {channel_uses: ranking[:k].astype(int).tolist() for channel_uses, k in budgets.items()}


def _load_scorer(
    settings: ResidualScorerSettings,
    motion_std: np.ndarray,
    run_dir: Path,
    fingerprint: str,
    channel_uses: int,
    seed: int,
    device: torch.device,
) -> ChannelAwareResidualScorer:
    checkpoint = load_checkpoint(
        run_dir / "models" / f"c_{channel_uses}" / f"seed_{seed}" / "best.pt",
        expected_fingerprint=fingerprint,
        map_location=device,
    )
    scorer = ChannelAwareResidualScorer(
        motion_std=torch.from_numpy(np.asarray(motion_std, dtype=np.float32)),
        hidden_dim=settings.hidden_dim,
        temperature=settings.temperature,
        max_channel_uses=max(settings.budgets),
    ).to(device)
    scorer.load_state_dict(checkpoint["model_state"])
    scorer.eval()
    return scorer


def _training_source_provenance(
    e5_run_dir: Path,
    gate_run_dir: Path,
    jscc: JSCCSettings,
    predictor: AudioMotionSettings,
) -> dict[str, Any]:
    metadata = _read_json(e5_run_dir / "run_metadata.json")
    experiment_fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if not experiment_fingerprint:
        raise ValueError("E5 source run has no experiment fingerprint")
    training_complete = _read_json(e5_run_dir / "training_complete.json")
    if (
        training_complete.get("status") != "complete"
        or training_complete.get("experiment_fingerprint") != experiment_fingerprint
    ):
        raise ValueError("E5 training completion fingerprint mismatch")
    training_summary_path = e5_run_dir / "training_summary.json"
    training_summary = _read_json(training_summary_path)
    selected = select_validation_model_seeds(training_summary, jscc.channel_uses)
    policy_path = gate_run_dir / "policy.json"
    policy = GatePolicy.from_dict(_read_json(policy_path))
    if policy.experiment_fingerprint != experiment_fingerprint:
        raise ValueError("gate policy does not belong to the E5 source run")
    if policy.selected_model_seeds != selected:
        raise ValueError("gate policy and E5 validation model selection differ")
    gate_complete = _read_json(gate_run_dir / "complete.json")
    if (
        gate_complete.get("status") != "complete"
        or gate_complete.get("gate_fingerprint") != policy.gate_fingerprint
    ):
        raise ValueError("validation-only channel gate is not complete")
    return {
        "e5_experiment_fingerprint": experiment_fingerprint,
        "e5_training_summary_sha256": file_sha256(training_summary_path),
        "e5_validation_cache_marker_sha256": file_sha256(
            e5_run_dir / "residual_data/train_validation_complete.json"
        ),
        "selected_e5_model_seeds": {str(key): value for key, value in selected.items()},
        "selected_e5_checkpoint_sha256": {
            str(channel_uses): file_sha256(
                e5_run_dir
                / "models"
                / f"c_{channel_uses}"
                / f"seed_{selected[channel_uses]}"
                / "best.pt"
            )
            for channel_uses in jscc.channel_uses
        },
        "validation_only_gate_policy_sha256": file_sha256(policy_path),
        "validation_only_gate_fingerprint": policy.gate_fingerprint,
        "motion_stats_sha256": file_sha256(predictor.motion_stats_path),
    }


def _prepare_training_run(
    run_dir: Path,
    settings: ResidualScorerSettings,
    source: Mapping[str, Any],
    fingerprint: str,
    *,
    resume: bool,
) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"residual scorer run exists: {run_dir}")
        _require_fingerprint(_read_json(metadata_path), fingerprint, metadata_path)
        return
    if resume:
        raise FileNotFoundError(f"cannot resume missing scorer run: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "resolved_config.json", dict(settings.config))
    atomic_write_json(run_dir / "source_provenance.json", dict(source))
    atomic_write_json(run_dir / "environment.json", _environment())
    atomic_write_json(
        metadata_path,
        {
            "experiment_fingerprint": fingerprint,
            "git_commit": _git_commit(),
        },
    )


def _load_examples(
    e5_run_dir: Path,
    experiment_fingerprint: str,
    split: str,
) -> list[ResidualExample]:
    paths = sorted((e5_run_dir / "residual_data" / split).glob("*.npz"))
    if not paths:
        raise ValueError(f"E5 run has no cached {split} residuals")
    examples = [
        load_residual_example(path, expected_fingerprint=experiment_fingerprint) for path in paths
    ]
    if any(example.split != split for example in examples):
        raise ValueError(f"E5 {split} cache contains another split")
    return examples


def _source_metric_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("condition") != "jscc_awgn":
            continue
        key = (
            row["sample_id"],
            row["channel_uses"],
            row["model_seed"],
            row["snr_db"],
            row["noise_seed"],
        )
        if key in index:
            raise ValueError(f"duplicate E5 test metric identity: {key}")
        index[key] = row
    return index


def _sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _loss_dict(total: np.ndarray, count: int) -> dict[str, float]:
    if count <= 0 or not np.isfinite(total).all():
        raise RuntimeError("non-finite or empty residual scorer epoch")
    values = total / count
    return {
        "loss": float(values[0]),
        "position_l1": float(values[1]),
        "velocity_l1": float(values[2]),
    }


def _require_fingerprint(
    payload: Mapping[str, Any],
    fingerprint: str,
    path: Path,
) -> None:
    if payload.get("experiment_fingerprint") != fingerprint:
        raise ValueError(f"residual scorer fingerprint mismatch: {path}")


def _write_plots(path: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    aggregate = summary["aggregate"]
    if not aggregate:
        return
    path.mkdir(parents=True, exist_ok=True)
    for channel_uses in sorted({int(row["channel_uses"]) for row in aggregate}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for method in (
            "dense_jscc",
            "raw_magnitude",
            "fixed_train_magnitude",
            "random",
            "learned_scorer",
        ):
            members = [
                row
                for row in aggregate
                if row["method"] == method and row["channel_uses"] == channel_uses
            ]
            axes[0].plot(
                [row["snr_db"] for row in members],
                [row["l1_mean"] for row in members],
                marker="o",
                label=method,
            )
            axes[1].plot(
                [row["snr_db"] for row in members],
                [row["velocity_l1_mean"] for row in members],
                marker="o",
                label=method,
            )
        axes[0].set_title(f"C={channel_uses}: motion L1")
        axes[1].set_title(f"C={channel_uses}: velocity L1")
        for axis in axes:
            axis.set_xlabel("SNR (dB)")
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(path / f"c_{channel_uses}_position_velocity.png", dpi=160)
        plt.close(figure)
