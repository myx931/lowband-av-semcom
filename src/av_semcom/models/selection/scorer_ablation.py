"""Validation-only 2x2 diagnosis of the residual scorer."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from av_semcom.data.preprocessing import atomic_write_json, config_fingerprint
from av_semcom.models.jscc.config import JSCCSettings
from av_semcom.models.jscc.data import ResidualDataset, ResidualExample
from av_semcom.models.jscc.experiment import _derived_noise_seed, _metric_row
from av_semcom.models.jscc.export import select_validation_model_seeds
from av_semcom.models.motion.perturbations import load_motion_normalizer
from av_semcom.models.predictor.artifacts import (
    atomic_save_checkpoint,
    file_sha256,
    load_checkpoint,
)
from av_semcom.models.predictor.config import AudioMotionSettings
from av_semcom.models.selection.config import (
    ResidualScorerAblationSettings,
    ResidualScorerSettings,
    ScorerAblationVariant,
)
from av_semcom.models.selection.gate import (
    _atomic_write_jsonl,
    _environment,
    _git_commit,
    _load_model,
    _new_run_directory,
    _read_json,
    _resolve_device,
    _write_dict_csv,
)
from av_semcom.models.selection.scorer import (
    ChannelAwareResidualScorer,
    rule_selection_mask,
)
from av_semcom.models.selection.scorer_experiment import (
    _load_examples,
    _train_epoch,
    _validation_epoch,
)
from av_semcom.utils.reproducibility import seed_everything

_METRICS = ("normalized_residual_mse", "l1", "rmse", "velocity_l1")


def partition_validation_examples(
    examples: Sequence[ResidualExample],
    *,
    calibration_sample_count: int,
    salt: str,
) -> tuple[list[ResidualExample], list[ResidualExample]]:
    """Hash-partition validation before checkpoint fitting or audit evaluation."""

    if not 0 < calibration_sample_count < len(examples):
        raise ValueError("calibration count must leave a non-empty audit partition")
    if not salt:
        raise ValueError("partition salt must be non-empty")
    identities = [example.sample_id for example in examples]
    if len(identities) != len(set(identities)):
        raise ValueError("validation examples contain duplicate sample IDs")
    ranked = sorted(
        examples,
        key=lambda example: (
            hashlib.sha256(f"{salt}:{example.sample_id}".encode()).hexdigest(),
            example.sample_id,
        ),
    )
    return (
        sorted(ranked[:calibration_sample_count], key=lambda item: item.sample_id),
        sorted(ranked[calibration_sample_count:], key=lambda item: item.sample_id),
    )


def run_scorer_ablation_training(
    ablation: ResidualScorerAblationSettings,
    scorer_settings: ResidualScorerSettings,
    jscc: JSCCSettings,
    predictor: AudioMotionSettings,
    e5_run_dir: Path,
    *,
    run_directory: Path | None = None,
    resume: bool = False,
    formal: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Train all 2x2 cells without opening test metrics or test residuals."""

    if formal:
        jscc.require_formal_backend()
    e5_run_dir = e5_run_dir.resolve()
    source = _source_provenance(e5_run_dir, jscc, predictor, ablation.channel_uses)
    fingerprint = config_fingerprint(
        {
            "ablation": ablation.config,
            "scorer": scorer_settings.config,
            "source": source,
        }
    )
    run_dir = (
        run_directory.resolve()
        if run_directory is not None
        else _new_run_directory(ablation.output_root)
    )
    _prepare_run(run_dir, ablation, scorer_settings, source, fingerprint, resume=resume)
    complete_path = run_dir / "training_complete.json"
    if complete_path.is_file():
        if not resume:
            raise FileExistsError("scorer ablation training is complete; pass --resume")
        _require_fingerprint(_read_json(complete_path), fingerprint, complete_path)
        return run_dir, _read_json(run_dir / "training_summary.json")

    e5_fingerprint = str(source["e5_experiment_fingerprint"])
    train_examples = _load_examples(e5_run_dir, e5_fingerprint, "train")
    validation_examples = _load_examples(e5_run_dir, e5_fingerprint, "validation")
    calibration, audit = partition_validation_examples(
        validation_examples,
        calibration_sample_count=ablation.calibration_sample_count,
        salt=ablation.partition_salt,
    )
    if {item.speaker_id for item in train_examples} & {
        item.speaker_id for item in validation_examples
    }:
        raise ValueError("speaker leakage between ablation train and validation")
    partition_payload = {
        "experiment_fingerprint": fingerprint,
        "method": "sha256(partition_salt:sample_id), ascending",
        "partition_salt": ablation.partition_salt,
        "calibration_sample_ids": [item.sample_id for item in calibration],
        "audit_sample_ids": [item.sample_id for item in audit],
        "calibration_speakers": sorted({item.speaker_id for item in calibration}),
        "audit_speakers": sorted({item.speaker_id for item in audit}),
        "test_data_accessed": False,
    }
    atomic_write_json(run_dir / "validation_partition.json", partition_payload)

    normalizer = load_motion_normalizer(predictor.motion_stats_path)
    if normalizer.scope != "train_stats":
        raise ValueError("scorer ablation requires train-only motion statistics")
    selected_e5_seeds = {
        int(key): int(value) for key, value in source["selected_e5_model_seeds"].items()
    }
    device = _resolve_device(scorer_settings.device)
    model_rows: list[dict[str, Any]] = []
    for channel_uses in ablation.channel_uses:
        frozen_jscc = _load_model(
            jscc,
            e5_run_dir,
            e5_fingerprint,
            channel_uses,
            selected_e5_seeds[channel_uses],
            device,
        )
        frozen_jscc.requires_grad_(False)
        for variant in ablation.variants:
            for seed in scorer_settings.seeds:
                model_dir = run_dir / "models" / f"c_{channel_uses}" / variant.name / f"seed_{seed}"
                marker = model_dir / "complete.json"
                if marker.is_file():
                    if not resume:
                        raise FileExistsError(f"ablation model exists: {model_dir}")
                    payload = _read_json(marker)
                    _require_fingerprint(payload, fingerprint, marker)
                    model_rows.append(payload)
                    continue
                model_rows.append(
                    _train_variant(
                        scorer_settings,
                        variant,
                        frozen_jscc,
                        train_examples,
                        calibration,
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
        "e5_experiment_fingerprint": e5_fingerprint,
        "jscc_weights_frozen": True,
        "test_data_accessed": False,
        "train_sample_count": len(train_examples),
        "calibration_sample_count": len(calibration),
        "reserved_audit_sample_count": len(audit),
        "train_speakers": sorted({item.speaker_id for item in train_examples}),
        "calibration_speakers": sorted({item.speaker_id for item in calibration}),
        "reserved_audit_speakers": sorted({item.speaker_id for item in audit}),
        "model_count": len(model_rows),
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
            "test_data_accessed": False,
        },
    )
    return run_dir, summary


def run_scorer_ablation_evaluation(
    ablation: ResidualScorerAblationSettings,
    scorer_settings: ResidualScorerSettings,
    jscc: JSCCSettings,
    predictor: AudioMotionSettings,
    e5_run_dir: Path,
    run_dir: Path,
    *,
    resume: bool = False,
    formal: bool = True,
) -> dict[str, Any]:
    """Evaluate frozen cells only on the reserved validation audit partition."""

    if formal:
        jscc.require_formal_backend()
    e5_run_dir = e5_run_dir.resolve()
    run_dir = run_dir.resolve()
    source = _source_provenance(e5_run_dir, jscc, predictor, ablation.channel_uses)
    fingerprint = config_fingerprint(
        {
            "ablation": ablation.config,
            "scorer": scorer_settings.config,
            "source": source,
        }
    )
    _require_fingerprint(
        _read_json(run_dir / "training_complete.json"),
        fingerprint,
        run_dir / "training_complete.json",
    )
    complete_path = run_dir / "audit_complete.json"
    if complete_path.is_file():
        if not resume:
            raise FileExistsError("scorer ablation audit is complete; pass --resume")
        _require_fingerprint(_read_json(complete_path), fingerprint, complete_path)
        return _read_json(run_dir / "audit_summary.json")

    e5_fingerprint = str(source["e5_experiment_fingerprint"])
    validation_examples = _load_examples(e5_run_dir, e5_fingerprint, "validation")
    calibration, audit = partition_validation_examples(
        validation_examples,
        calibration_sample_count=ablation.calibration_sample_count,
        salt=ablation.partition_salt,
    )
    _verify_partition(run_dir, fingerprint, calibration, audit)
    normalizer = load_motion_normalizer(predictor.motion_stats_path)
    if normalizer.scope != "train_stats":
        raise ValueError("scorer ablation audit requires train-only motion statistics")
    selected_e5_seeds = {
        int(key): int(value) for key, value in source["selected_e5_model_seeds"].items()
    }
    device = _resolve_device(scorer_settings.device)
    rows: list[dict[str, Any]] = []
    for channel_uses in ablation.channel_uses:
        frozen_jscc = _load_model(
            jscc,
            e5_run_dir,
            e5_fingerprint,
            channel_uses,
            selected_e5_seeds[channel_uses],
            device,
        )
        frozen_jscc.requires_grad_(False)
        scorers = {
            (variant.name, seed): _load_ablation_scorer(
                scorer_settings,
                normalizer.std,
                run_dir,
                fingerprint,
                channel_uses,
                variant,
                seed,
                device,
            )
            for variant in ablation.variants
            for seed in scorer_settings.seeds
        }
        for example_index, example in enumerate(audit):
            print(
                f"[scorer-ablation] C={channel_uses} "
                f"audit {example_index + 1}/{len(audit)}: {example.sample_id}",
                flush=True,
            )
            rows.extend(
                _evaluate_audit_example(
                    ablation,
                    scorer_settings,
                    frozen_jscc,
                    scorers,
                    example,
                    example_index,
                    channel_uses,
                    selected_e5_seeds[channel_uses],
                    normalizer.std,
                    device,
                )
            )
    conditions_per_channel_snr_noise = 2 + len(ablation.variants) * len(scorer_settings.seeds)
    expected_count = (
        len(audit)
        * len(ablation.channel_uses)
        * len(scorer_settings.validation_snr_db)
        * len(scorer_settings.noise_seeds)
        * conditions_per_channel_snr_noise
    )
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} audit rows, got {len(rows)}")
    _atomic_write_jsonl(run_dir / "audit_metrics.jsonl", rows)
    summary = _summarize(rows, ablation, scorer_settings)
    summary.update(
        {
            "experiment_fingerprint": fingerprint,
            "expected_result_count": expected_count,
            "audit_sample_count": len(audit),
            "audit_speakers": sorted({item.speaker_id for item in audit}),
            "test_data_accessed": False,
        }
    )
    atomic_write_json(run_dir / "audit_summary.json", summary)
    _write_dict_csv(run_dir / "audit_summary.csv", summary["aggregate"])
    _write_plots(run_dir / "plots", summary)
    atomic_write_json(
        complete_path,
        {
            "experiment_fingerprint": fingerprint,
            "result_count": len(rows),
            "status": "complete",
            "test_data_accessed": False,
        },
    )
    return summary


def _train_variant(
    settings: ResidualScorerSettings,
    variant: ScorerAblationVariant,
    frozen_jscc: torch.nn.Module,
    train_examples: Sequence[ResidualExample],
    calibration_examples: Sequence[ResidualExample],
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
        use_snr=variant.use_snr,
    ).to(device)
    variant_settings = replace(settings, velocity_weight=variant.velocity_weight)
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
    calibration_loader = DataLoader(
        ResidualDataset(calibration_examples),
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
            variant_settings,
            channel_uses,
            k,
            device,
            snr_generator,
            noise_seed_base=(seed * 1_000_003 + epoch * 10_007 + channel_uses * 101) % (2**31 - 1),
        )
        calibration = _validation_epoch(
            scorer,
            frozen_jscc,
            calibration_loader,
            variant_settings,
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
            "calibration_loss": calibration["loss"],
            "calibration_position_l1": calibration["position_l1"],
            "calibration_velocity_l1": calibration["velocity_l1"],
        }
        history.append(row)
        print(
            f"[scorer-ablation] C={channel_uses} K={k} variant={variant.name} "
            f"seed={seed} epoch={epoch} train={train['loss']:.6f} "
            f"calibration={calibration['loss']:.6f}",
            flush=True,
        )
        if calibration["loss"] < best_loss - settings.early_stopping_min_delta:
            best_loss = calibration["loss"]
            best_epoch = epoch
            stale = 0
            atomic_save_checkpoint(
                model_dir / "best.pt",
                {
                    "experiment_fingerprint": fingerprint,
                    "channel_uses": channel_uses,
                    "k": k,
                    "variant": variant.name,
                    "seed": seed,
                    "epoch": epoch,
                    "calibration_loss": best_loss,
                    "model_config": {
                        "hidden_dim": settings.hidden_dim,
                        "temperature": settings.temperature,
                        "max_channel_uses": max(settings.budgets),
                        "use_snr": variant.use_snr,
                    },
                    "velocity_weight": variant.velocity_weight,
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
        "variant": variant.name,
        "use_snr": variant.use_snr,
        "velocity_weight": variant.velocity_weight,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "epoch_count": len(history),
        "jscc_weights_frozen": True,
        "test_data_accessed": False,
    }
    atomic_write_json(model_dir / "complete.json", result)
    return result


@torch.no_grad()
def _evaluate_audit_example(
    ablation: ResidualScorerAblationSettings,
    settings: ResidualScorerSettings,
    frozen_jscc: torch.nn.Module,
    scorers: Mapping[tuple[str, int], ChannelAwareResidualScorer],
    example: ResidualExample,
    example_index: int,
    channel_uses: int,
    e5_model_seed: int,
    motion_std: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    residual = torch.from_numpy(example.normalized_residual).unsqueeze(0).to(device)
    mask = torch.from_numpy(example.transmission_mask).unsqueeze(0).to(device)
    std = torch.from_numpy(np.asarray(motion_std, dtype=np.float32)).to(device)
    k = settings.budgets[channel_uses]
    raw_mask = rule_selection_mask(
        residual,
        mask,
        k=k,
        method="raw_magnitude",
        motion_std=std,
    )
    for snr_index, snr_db in enumerate(settings.validation_snr_db):
        for noise_seed in settings.noise_seeds:
            derived_noise = _derived_noise_seed(
                e5_model_seed,
                noise_seed,
                snr_index,
                example_index,
            )
            candidates: list[
                tuple[
                    str,
                    str | None,
                    int | None,
                    torch.Tensor,
                    torch.Tensor,
                ]
            ] = [
                (
                    "dense_jscc",
                    None,
                    None,
                    residual,
                    mask.unsqueeze(-1).expand_as(residual),
                ),
                (
                    "raw_magnitude",
                    None,
                    None,
                    residual * raw_mask,
                    raw_mask,
                ),
            ]
            for variant in ablation.variants:
                for seed in settings.seeds:
                    result = scorers[(variant.name, seed)](
                        residual,
                        mask,
                        snr_db,
                        k=k,
                        channel_uses=channel_uses,
                    )
                    candidates.append(
                        (
                            "learned_scorer",
                            variant.name,
                            seed,
                            result.selected_residual,
                            result.hard_mask,
                        )
                    )
            eligible = int(example.transmission_mask.sum())
            if eligible <= 0:
                raise ValueError(f"{example.sample_id} has no eligible frames")
            for method, variant_name, method_seed, selected, selection_mask in candidates:
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
                frequency = (selection_mask[0].float().sum(dim=0).cpu().numpy() / eligible).tolist()
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
                        "variant": variant_name,
                        "method_seed": method_seed,
                        "k": 18 if method == "dense_jscc" else k,
                        "evaluation_scope": "validation_audit",
                        "test_data_accessed": False,
                        "selected_dimension_frequency": frequency,
                    }
                )
                rows.append(row)
    return rows


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    ablation: ResidualScorerAblationSettings,
    settings: ResidualScorerSettings,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["method"],
                row.get("variant"),
                row.get("method_seed"),
                row["channel_uses"],
                row["snr_db"],
            )
        ].append(row)
    groups: list[dict[str, Any]] = []
    for (method, variant, method_seed, channel_uses, snr_db), members in sorted(
        grouped.items(),
        key=lambda item: str(item[0]),
    ):
        groups.append(
            {
                "method": method,
                "variant": variant,
                "method_seed": method_seed,
                "channel_uses": channel_uses,
                "k": members[0]["k"],
                "snr_db": snr_db,
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
    for channel_uses in ablation.channel_uses:
        for snr_db in settings.validation_snr_db:
            baseline_rows: dict[str, Mapping[str, Any]] = {}
            for method in ("dense_jscc", "raw_magnitude"):
                member = next(
                    group
                    for group in groups
                    if group["method"] == method
                    and group["channel_uses"] == channel_uses
                    and group["snr_db"] == snr_db
                )
                baseline_rows[method] = member
                aggregate.append(
                    {
                        "method": method,
                        "variant": None,
                        "use_snr": None,
                        "velocity_weight": None,
                        "channel_uses": channel_uses,
                        "k": member["k"],
                        "snr_db": snr_db,
                        "seed_count": 0,
                        **{f"{metric}_mean": member[metric] for metric in _METRICS},
                        **{f"{metric}_std": 0.0 for metric in _METRICS},
                        "l1_gain_vs_raw_percent": (
                            (float(baseline_rows["raw_magnitude"]["l1"]) - float(member["l1"]))
                            / float(baseline_rows["raw_magnitude"]["l1"])
                            * 100.0
                            if method == "raw_magnitude"
                            else None
                        ),
                    }
                )
            raw_l1 = float(baseline_rows["raw_magnitude"]["l1"])
            for variant in ablation.variants:
                members = [
                    group
                    for group in groups
                    if group["method"] == "learned_scorer"
                    and group["variant"] == variant.name
                    and group["channel_uses"] == channel_uses
                    and group["snr_db"] == snr_db
                ]
                if len(members) != len(settings.seeds):
                    raise ValueError(
                        f"expected {len(settings.seeds)} seeds for {variant.name}, "
                        f"C={channel_uses}, SNR={snr_db}"
                    )
                row: dict[str, Any] = {
                    "method": "learned_scorer",
                    "variant": variant.name,
                    "use_snr": variant.use_snr,
                    "velocity_weight": variant.velocity_weight,
                    "channel_uses": channel_uses,
                    "k": members[0]["k"],
                    "snr_db": snr_db,
                    "seed_count": len(members),
                }
                for metric in _METRICS:
                    values = [float(member[metric]) for member in members]
                    row[f"{metric}_mean"] = float(np.mean(values))
                    row[f"{metric}_std"] = float(np.std(values))
                row["l1_gain_vs_raw_percent"] = (raw_l1 - float(row["l1_mean"])) / raw_l1 * 100.0
                aggregate.append(row)
    return {
        "schema_version": 1,
        "status": "complete",
        "evaluation_scope": "reserved_validation_audit",
        "result_count": len(rows),
        "groups": groups,
        "aggregate": aggregate,
    }


def _source_provenance(
    e5_run_dir: Path,
    jscc: JSCCSettings,
    predictor: AudioMotionSettings,
    channel_uses: Sequence[int],
) -> dict[str, Any]:
    metadata = _read_json(e5_run_dir / "run_metadata.json")
    e5_fingerprint = str(metadata.get("experiment_fingerprint", ""))
    if not e5_fingerprint:
        raise ValueError("E5 source run has no experiment fingerprint")
    complete = _read_json(e5_run_dir / "training_complete.json")
    if (
        complete.get("status") != "complete"
        or complete.get("experiment_fingerprint") != e5_fingerprint
    ):
        raise ValueError("E5 training source is not complete")
    summary_path = e5_run_dir / "training_summary.json"
    selected = select_validation_model_seeds(_read_json(summary_path), jscc.channel_uses)
    return {
        "e5_experiment_fingerprint": e5_fingerprint,
        "e5_training_summary_sha256": file_sha256(summary_path),
        "e5_train_validation_marker_sha256": file_sha256(
            e5_run_dir / "residual_data/train_validation_complete.json"
        ),
        "selected_e5_model_seeds": {str(value): selected[value] for value in channel_uses},
        "selected_e5_checkpoint_sha256": {
            str(value): file_sha256(
                e5_run_dir / "models" / f"c_{value}" / f"seed_{selected[value]}" / "best.pt"
            )
            for value in channel_uses
        },
        "motion_stats_sha256": file_sha256(predictor.motion_stats_path),
        "test_metrics_opened": False,
        "test_residual_cache_opened": False,
    }


def _prepare_run(
    run_dir: Path,
    ablation: ResidualScorerAblationSettings,
    scorer: ResidualScorerSettings,
    source: Mapping[str, Any],
    fingerprint: str,
    *,
    resume: bool,
) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"scorer ablation run exists: {run_dir}")
        _require_fingerprint(_read_json(metadata_path), fingerprint, metadata_path)
        return
    if resume:
        raise FileNotFoundError(f"cannot resume missing scorer ablation run: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "resolved_config.json",
        {
            "residual_scorer": dict(scorer.config),
            "residual_scorer_ablation": dict(ablation.config),
        },
    )
    atomic_write_json(run_dir / "source_provenance.json", dict(source))
    atomic_write_json(run_dir / "environment.json", _environment())
    atomic_write_json(
        metadata_path,
        {
            "experiment_fingerprint": fingerprint,
            "git_commit": _git_commit(),
            "evaluation_scope": "validation_only",
            "test_data_accessed": False,
        },
    )


def _verify_partition(
    run_dir: Path,
    fingerprint: str,
    calibration: Sequence[ResidualExample],
    audit: Sequence[ResidualExample],
) -> None:
    path = run_dir / "validation_partition.json"
    payload = _read_json(path)
    _require_fingerprint(payload, fingerprint, path)
    expected_calibration = [item.sample_id for item in calibration]
    expected_audit = [item.sample_id for item in audit]
    if payload.get("calibration_sample_ids") != expected_calibration:
        raise ValueError("validation calibration partition changed")
    if payload.get("audit_sample_ids") != expected_audit:
        raise ValueError("validation audit partition changed")
    if set(expected_calibration) & set(expected_audit):
        raise ValueError("validation calibration/audit partitions overlap")


def _load_ablation_scorer(
    settings: ResidualScorerSettings,
    motion_std: np.ndarray,
    run_dir: Path,
    fingerprint: str,
    channel_uses: int,
    variant: ScorerAblationVariant,
    seed: int,
    device: torch.device,
) -> ChannelAwareResidualScorer:
    checkpoint = load_checkpoint(
        run_dir / "models" / f"c_{channel_uses}" / variant.name / f"seed_{seed}" / "best.pt",
        expected_fingerprint=fingerprint,
        map_location=device,
    )
    if bool(checkpoint["model_config"]["use_snr"]) != variant.use_snr:
        raise ValueError("ablation checkpoint SNR factor mismatch")
    if float(checkpoint["velocity_weight"]) != variant.velocity_weight:
        raise ValueError("ablation checkpoint velocity factor mismatch")
    scorer = ChannelAwareResidualScorer(
        motion_std=torch.from_numpy(np.asarray(motion_std, dtype=np.float32)),
        hidden_dim=settings.hidden_dim,
        temperature=settings.temperature,
        max_channel_uses=max(settings.budgets),
        use_snr=variant.use_snr,
    ).to(device)
    scorer.load_state_dict(checkpoint["model_state"])
    scorer.eval()
    return scorer


def _require_fingerprint(
    payload: Mapping[str, Any],
    fingerprint: str,
    path: Path,
) -> None:
    if payload.get("experiment_fingerprint") != fingerprint:
        raise ValueError(f"scorer ablation fingerprint mismatch: {path}")


def _write_plots(path: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    aggregate = summary["aggregate"]
    path.mkdir(parents=True, exist_ok=True)
    for channel_uses in sorted({int(row["channel_uses"]) for row in aggregate}):
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for method, variant in (
            ("dense_jscc", None),
            ("raw_magnitude", None),
            ("learned_scorer", "full"),
            ("learned_scorer", "no_snr"),
            ("learned_scorer", "no_velocity"),
            ("learned_scorer", "no_snr_no_velocity"),
        ):
            members = [
                row
                for row in aggregate
                if row["method"] == method
                and row.get("variant") == variant
                and row["channel_uses"] == channel_uses
            ]
            if not members:
                continue
            label = method if variant is None else variant
            axis.plot(
                [row["snr_db"] for row in members],
                [row["l1_mean"] for row in members],
                marker="o",
                label=label,
            )
        axis.set_title(f"Validation audit C={channel_uses}")
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("raw motion L1")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(path / f"c_{channel_uses}_validation_ablation.png", dpi=160)
        plt.close(figure)
