#!/usr/bin/env python3
"""Constrained-selection HMM soft-state UAV trainer (v4).

Network architecture is unchanged. This version adds probability-aware model
selection, correct EMA checkpointing, and validation-only variance scaling.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

try:
    from latent_state_emamse import (
        LatentStateConfig, _sequence_order, build_offline_teacher,
        causal_filter_posteriors, extract_window_latents, pretrain_emamse,
        soft_state_kl_loss,
    )
except ModuleNotFoundError:
    from supplemental_experiments.latent_state_emamse import (
        LatentStateConfig, _sequence_order, build_offline_teacher,
        causal_filter_posteriors, extract_window_latents, pretrain_emamse,
        soft_state_kl_loss,
    )


def balanced_soft_state_distillation(
    state_logits, teacher_posterior, temperature=1.0,
    confidence_floor=0.25,
):
    """Posterior-mass-balanced soft distillation for causal HMM states."""
    if state_logits.ndim != 2 or teacher_posterior.ndim != 2:
        raise ValueError("state logits and teacher posterior must be [B, K]")
    if state_logits.shape != teacher_posterior.shape:
        raise ValueError(
            f"shape mismatch: logits={tuple(state_logits.shape)}, "
            f"teacher={tuple(teacher_posterior.shape)}"
        )
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")

    logits = state_logits.float()
    teacher = teacher_posterior.float().clamp_min(1e-8)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-8)
    if temperature != 1.0:
        teacher = teacher.pow(1.0 / float(temperature))
        teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-8)

    log_student = F.log_softmax(logits / float(temperature), dim=-1)
    log_teacher = teacher.clamp_min(1e-8).log()
    n_states = teacher.shape[-1]
    entropy = -(teacher * log_teacher).sum(-1)
    confidence = 1.0 - entropy / math.log(float(n_states))
    confidence = (
        confidence_floor
        + (1.0 - confidence_floor) * confidence.clamp(0.0, 1.0)
    )
    weighted_teacher = confidence[:, None] * teacher
    state_mass = weighted_teacher.sum(0).clamp_min(1e-6)
    state_cross_entropy = (
        -weighted_teacher * log_student
    ).sum(0) / state_mass
    return state_cross_entropy.mean() * float(temperature) ** 2


def _relative_upper_bound(value, tolerance):
    return float(value) + max(abs(float(value)), 1e-12) * float(tolerance)


def _select_constrained_checkpoint(records, crps_tolerance, brier_tolerance):
    """Apply CRPS constraint, Brier near-tie constraint, then CalGap."""
    if not records:
        raise RuntimeError("No validation checkpoint candidate was recorded.")
    minimum_crps = min(float(r["BalancedCRPS"]) for r in records)
    crps_limit = _relative_upper_bound(minimum_crps, crps_tolerance)
    crps_pool = [
        r for r in records if float(r["BalancedCRPS"]) <= crps_limit
    ]
    minimum_brier = min(float(r["StateBrier"]) for r in crps_pool)
    brier_limit = _relative_upper_bound(minimum_brier, brier_tolerance)
    brier_pool = [
        r for r in crps_pool if float(r["StateBrier"]) <= brier_limit
    ]
    selected = min(
        brier_pool,
        key=lambda r: (
            float(r["CalibrationGap95"]),
            float(r["BalancedCRPS"]),
            -int(r["epoch"]),
        ),
    )
    return selected, minimum_crps, crps_limit, minimum_brier, brier_limit


def load_base(path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = path.stem
    if not module_name.isidentifier():
        raise ValueError(
            f"Base filename must be importable, got {path.name}."
        )
    directory = str(path.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != path:
        raise ImportError(
            f"Imported {module_name} from {module.__file__}, expected {path}."
        )
    return module


class StateDataset(Dataset):
    def __init__(self, base, sequences, indices, history, horizon,
                 train=False, posterior=None):
        self.data = base.OnTheFlyUAVDataset(
            sequences, indices, history, horizon, is_training=train
        )
        self.posterior = posterior

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item["window_index"] = torch.tensor(index, dtype=torch.long)
        if self.posterior is not None:
            posterior = torch.as_tensor(
                self.posterior[index], dtype=torch.float32
            )
            item["state_posterior"] = posterior
            item["intent"] = posterior.argmax().long()  # grouping only
        return item


def make_loader(dataset, batch_size, workers, shuffle, seed):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def infer_split(latent, window_ids, indices, teacher_path):
    bundle = joblib.load(teacher_path)
    order, lengths = _sequence_order(indices, window_ids)
    observations = bundle["scaler"].transform(latent[order])
    posterior = causal_filter_posteriors(
        bundle["hmm"], observations, lengths
    )
    posterior = posterior[:, np.asarray(
        bundle["semantic_order"], dtype=np.int64
    )]
    aligned = np.zeros((len(indices), posterior.shape[1]), np.float32)
    aligned[window_ids[order]] = posterior.astype(np.float32)
    return aligned


def gaussian_crps(mean, logvar, target):
    """Closed-form Gaussian CRPS, evaluated element-wise."""
    logvar = logvar.float().clamp(-12.0, 8.0)
    sigma = torch.exp(0.5 * logvar).clamp_min(1e-6)
    z_score = (target.float() - mean.float()) / sigma
    density = (
        torch.exp(-0.5 * z_score.square()) / math.sqrt(2.0 * math.pi)
    )
    cdf = 0.5 * (1.0 + torch.erf(z_score / math.sqrt(2.0)))
    return sigma * (
        z_score * (2.0 * cdf - 1.0)
        + 2.0 * density
        - 1.0 / math.sqrt(math.pi)
    )


def soft_state_average(values, posterior):
    """Give every latent state equal weight without hardening its posterior."""
    mass = posterior.sum(0).clamp_min(1e-8)
    per_state = (posterior * values[:, None]).sum(0) / mass
    return float(per_state.mean()), per_state


@torch.no_grad()
def validation(base, model, loader, target_scaler, device):
    model.eval()
    predicted_positions, true_positions = [], []
    predicted_states, teacher_states = [], []
    crps_values, nll_values, coverage_values = [], [], []

    for batch in loader:
        inputs = {
            "seq": batch["seq"].to(device, non_blocking=True),
            "last_delta": batch["last_delta"].to(
                device, non_blocking=True
            ),
        }
        target = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(
            "cuda", dtype=base.AMP_DTYPE, enabled=device.type == "cuda"
        ):
            prediction, logvar, logits, _ = model(inputs, tau=1.0)

        prediction32 = prediction.float()
        target32 = target.float()
        logvar32 = logvar.float().clamp(-12.0, 8.0)
        standardised_squared_error = (
            (target32 - prediction32).square() * torch.exp(-logvar32)
        )
        crps_values.append(
            gaussian_crps(prediction32, logvar32, target32)
            .mean((1, 2)).cpu()
        )
        nll_values.append(
            (0.5 * (logvar32 + standardised_squared_error))
            .mean((1, 2)).cpu()
        )
        coverage_values.append(
            (standardised_squared_error.sqrt() <= 1.959963984540054)
            .float().mean((1, 2)).cpu()
        )
        predicted_positions.append(np.cumsum(
            base.to_real(prediction, target_scaler), axis=1
        ))
        true_positions.append(np.cumsum(
            base.to_real(target.cpu().numpy(), target_scaler), axis=1
        ))
        predicted_states.append(
            logits.float().softmax(-1).cpu().numpy()
        )
        teacher_states.append(batch["state_posterior"].numpy())

    predicted_position = np.concatenate(predicted_positions)
    true_position = np.concatenate(true_positions)
    state_prediction = np.concatenate(predicted_states)
    teacher_posterior = torch.from_numpy(
        np.concatenate(teacher_states)
    ).float()
    crps = torch.cat(crps_values)
    nll = torch.cat(nll_values)
    coverage = torch.cat(coverage_values)

    distance_error = np.linalg.norm(
        predicted_position - true_position, axis=-1
    )
    ade = float(distance_error.mean())
    fde = float(distance_error[:, -1].mean())
    balanced_crps, state_crps = soft_state_average(
        crps, teacher_posterior
    )
    balanced_nll, state_nll = soft_state_average(
        nll, teacher_posterior
    )
    balanced_pic, state_pic = soft_state_average(
        coverage, teacher_posterior
    )
    teacher_numpy = teacher_posterior.numpy()

    return {
        "ADE": ade,
        "FDE": fde,
        "score": balanced_crps,
        "BalancedCRPS": balanced_crps,
        "BalancedNLL": balanced_nll,
        "BalancedPIC95": 100.0 * balanced_pic,
        "CalibrationGap95": 100.0 * float(
            torch.abs(state_pic - 0.95).mean()
        ),
        "LowCRPS": float(state_crps[0]),
        "ModerateCRPS": float(state_crps[1]),
        "HighCRPS": float(state_crps[2]),
        "LowNLL": float(state_nll[0]),
        "ModerateNLL": float(state_nll[1]),
        "HighNLL": float(state_nll[2]),
        "LowPIC95": 100.0 * float(state_pic[0]),
        "ModeratePIC95": 100.0 * float(state_pic[1]),
        "HighPIC95": 100.0 * float(state_pic[2]),
        "StateAcc": float((
            state_prediction.argmax(1) == teacher_numpy.argmax(1)
        ).mean()),
        "StateBrier": float(np.mean(np.sum(
            (state_prediction - teacher_numpy) ** 2, axis=1
        ))),
    }


@torch.no_grad()
def fit_variance_temperature(model, loader, device, shrinkage=256.0):
    """Fit state/horizon variance temperatures on validation data only.

    The Gaussian-NLL-optimal standard-deviation scale is the RMS
    standardised residual. State estimates are shrunk towards a global
    horizon estimate to avoid unstable corrections.
    """
    model.eval()
    numerator = denominator = global_sum = None
    sample_count = 0

    for batch in loader:
        inputs = {
            "seq": batch["seq"].to(device, non_blocking=True),
            "last_delta": batch["last_delta"].to(
                device, non_blocking=True
            ),
        }
        target = batch["target"].to(
            device, non_blocking=True
        ).float()
        prediction, logvar, logits, _ = model(inputs, tau=1.0)
        logvar = logvar.float().clamp(-12.0, 8.0)
        z_squared = (
            (target - prediction.float()).square() * torch.exp(-logvar)
        ).mean(-1)
        # This posterior is available online; no test labels are used.
        posterior = logits.float().softmax(-1)

        if numerator is None:
            state_count = posterior.shape[-1]
            horizon = z_squared.shape[-1]
            numerator = torch.zeros(
                state_count, horizon, dtype=torch.float64, device=device
            )
            denominator = torch.zeros_like(numerator)
            global_sum = torch.zeros(
                horizon, dtype=torch.float64, device=device
            )

        numerator += posterior.double().transpose(0, 1) @ z_squared.double()
        denominator += posterior.double().sum(0)[:, None]
        global_sum += z_squared.double().sum(0)
        sample_count += z_squared.shape[0]

    if sample_count == 0:
        raise RuntimeError(
            "Cannot calibrate variance with an empty validation loader."
        )

    global_z_squared = global_sum / float(sample_count)
    scale_squared = (
        numerator + float(shrinkage) * global_z_squared[None, :]
    ) / (
        denominator + float(shrinkage)
    )
    return scale_squared.clamp(
        0.75 ** 2, 3.0 ** 2
    ).sqrt().float().cpu()


class VarianceTemperatureModel(nn.Module):
    """Apply fixed validation-fitted scaling to the existing log variance."""
    def __init__(self, model, state_horizon_scale):
        super().__init__()
        self.model = model
        self.register_buffer(
            "state_horizon_scale_squared",
            state_horizon_scale.square(),
        )

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if (
            not isinstance(output, (tuple, list))
            or len(output) < 3
            or output[1] is None
        ):
            return output
        items = list(output)
        posterior = items[2].float().softmax(-1)
        # Variances, rather than standard deviations, are softly mixed.
        scale_squared = (
            posterior @ self.state_horizon_scale_squared
        )
        items[1] = (
            items[1].float()
            + scale_squared.clamp_min(1e-8).log().unsqueeze(-1)
        )
        return tuple(items)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)


def run_seed(base, shared, protocol, args, seed):
    base.seed_everything(seed, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "checkpoint_candidates"
    candidate_records = []
    if not args.evaluate_only:
        candidate_dir.mkdir(parents=True, exist_ok=True)

    ablation = base.cfg_ours()
    ablation.update(
        intent_supervision=False,
        intent_weighted_loss=False,
        intent_conditioning=True,
    )
    model = base.IntentAwarePredictor(
        shared["feat_dim"], args.d_model, shared["horizon"],
        protocol["scales"], shared["scaler_target"].scale_, ablation,
    ).to(device)
    config = LatentStateConfig(
        n_states=3,
        latent_dim=args.latent_dim,
        mask_ratio=args.mask_ratio,
        ssl_epochs=args.ssl_epochs,
        ssl_lr=args.ssl_lr,
        hmm_n_init=args.hmm_restarts,
        state_loss_weight=args.state_weight,
        state_temperature=args.state_temperature,
        seed=seed,
    )
    history_length = shared["max_s"]
    horizon = shared["horizon"]

    ssl_dataset = StateDataset(
        base, shared["train_seqs"], shared["train_indices"],
        history_length, horizon, train=True,
    )
    ssl_loader = make_loader(
        ssl_dataset, args.batch_size, args.workers, True, seed + 1
    )
    ssl = pretrain_emamse(
        model, ssl_loader, shared["feat_dim"], device, config,
        output_dir / "emamse_ssl.pt",
    )

    latents, window_ids, histories = {}, {}, {}
    for split_index, split in enumerate(("train", "val", "test")):
        dataset = StateDataset(
            base, shared[f"{split}_seqs"], shared[f"{split}_indices"],
            history_length, horizon, train=False,
        )
        loader = make_loader(
            dataset, args.eval_batch_size, args.workers, False,
            seed + 10 + split_index,
        )
        latents[split], window_ids[split] = extract_window_latents(
            ssl, loader, device
        )
        # Only the training histories are needed to establish the semantic
        # ordering of HMM states. Avoid a second full pass over val and test.
        if (
            split == "train"
            and not globals().get(
                "skip_train_history_extraction", False
            )
        ):
            histories[split] = np.concatenate(
                [batch["seq"].numpy() for batch in loader]
            )

    teacher_dir = output_dir / "hmm_teacher"
    state_posterior = {
        "train": build_offline_teacher(
            latents["train"], window_ids["train"],
            shared["train_indices"], histories.get("train"),
            config, teacher_dir,
        )
    }
    for split in ("val", "test"):
        state_posterior[split] = infer_split(
            latents[split], window_ids[split], shared[f"{split}_indices"],
            teacher_dir / "hmm_teacher.joblib",
        )

    distribution_rows = []
    for split, posterior in state_posterior.items():
        hard_state = posterior.argmax(1)
        entropy = -np.sum(
            posterior * np.log(np.maximum(posterior, 1e-12)), axis=1
        )
        for state, name in enumerate(("low", "moderate", "high")):
            mask = hard_state == state
            distribution_rows.append({
                "split": split,
                "state": name,
                "count": int(mask.sum()),
                "proportion": float(mask.mean()),
                "entropy": (
                    float(entropy[mask].mean()) if mask.any() else np.nan
                ),
            })
    pd.DataFrame(distribution_rows).to_csv(
        output_dir / "state_distribution.csv", index=False
    )

    del ssl, ssl_loader, latents, histories
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = StateDataset(
        base, shared["train_seqs"], shared["train_indices"],
        history_length, horizon, True, state_posterior["train"],
    )
    validation_dataset = StateDataset(
        base, shared["val_seqs"], shared["val_indices"],
        history_length, horizon, False, state_posterior["val"],
    )
    test_dataset = StateDataset(
        base, shared["test_seqs"], shared["test_indices"],
        history_length, horizon, False, state_posterior["test"],
    )
    validation_loader = make_loader(
        validation_dataset, args.eval_batch_size, args.workers,
        False, seed + 20,
    )
    test_loader = make_loader(
        test_dataset, args.eval_batch_size, args.workers,
        False, seed + 21,
    )

    optimizer = base._make_optimizer(model, args.learning_rate)
    warmup = min(5, max(args.epochs - 1, 1))
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, 0.1, total_iters=warmup),
            CosineAnnealingLR(
                optimizer, max(args.epochs - warmup, 1), eta_min=1e-6
            ),
        ],
        [warmup],
    )
    amp_scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    ema = base.ModelEMA(model, 0.999, 2000)
    best_score, best_epoch = float("inf"), -1
    history = []
    checkpoint = output_dir / "best_checkpoint.pt"
    load_checkpoint = checkpoint
    if args.evaluate_only:
        checkpoint_root = Path(
            args.checkpoint_root or args.output_dir
        )
        load_checkpoint = (
            checkpoint_root / f"seed_{seed}" / "best_checkpoint.pt"
        )
        if not load_checkpoint.exists():
            raise FileNotFoundError(
                f"Evaluation checkpoint not found: {load_checkpoint}"
            )

    # Reuse one loader so persistent workers remain alive across epochs.
    # Its sampler/generator advances on every iteration, so ordering still
    # changes between epochs without paying worker start-up cost repeatedly.
    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True,
        seed + 1000,
    )

    epoch_iterator = () if args.evaluate_only else range(args.epochs)
    for epoch in epoch_iterator:
        model.train()
        sums, batch_count = np.zeros(4), 0
        progress = epoch / max(args.epochs - 1, 1)

        for batch in train_loader:
            inputs = {
                "seq": batch["seq"].to(device, non_blocking=True),
                "last_delta": batch["last_delta"].to(
                    device, non_blocking=True
                ),
            }
            target = batch["target"].to(device, non_blocking=True)
            teacher_posterior = batch["state_posterior"].to(
                device, non_blocking=True
            )
            last_state = inputs["seq"][:, -1, 3:6]
            forcing_probability = max(0.05, 0.55 - 0.45 * progress)
            forcing_mask = (
                torch.rand(inputs["seq"].size(0), horizon, device=device)
                < forcing_probability
            )
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda", dtype=base.AMP_DTYPE,
                enabled=device.type == "cuda",
            ):
                prediction, logvar, logits, turn_rate, auxiliary = model(
                    inputs,
                    tau=args.state_temperature,
                    target_deltas=target,
                    tf_mask=forcing_mask,
                    current_horizon=horizon,
                    return_aux=True,
                )
                trajectory_loss = base.uav_loss_v20(
                    prediction, logvar, target, last_state,
                    intent_logits=None, intent_targets=None,
                    epoch=epoch, total_epochs=args.epochs,
                    turn_rates_pred=turn_rate, is_training=True,
                    current_horizon=horizon, aux_info=auxiliary,
                    use_intent_supervision=False,
                    use_intent_weighted_loss=False,
                )
                state_loss = balanced_soft_state_distillation(
                    logits,
                    teacher_posterior,
                    args.state_temperature,
                    args.state_confidence_floor,
                )
                auxiliary_loss_function = globals().get(
                    "training_auxiliary_loss"
                )
                if callable(auxiliary_loss_function):
                    horizon_loss = auxiliary_loss_function(
                        prediction, logvar, target, teacher_posterior,
                        epoch, args.epochs,
                    )
                else:
                    horizon_loss = prediction.new_zeros(())
                loss = (
                    trajectory_loss
                    + args.state_weight * state_loss
                    + horizon_loss
                )

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            ema.update(model)
            sums += (
                loss.item(), trajectory_loss.item(), state_loss.item(),
                horizon_loss.item(),
            )
            batch_count += 1

        scheduler.step()

        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
            # Validation and checkpoint now refer to identical EMA weights.
            ema.apply_to(model)
            metrics = validation(
                base, model, validation_loader,
                shared["scaler_target"], device,
            )
            candidate_path = (
                candidate_dir / f"epoch_{epoch + 1:04d}.pt"
            )
            candidate_record = {
                "epoch": int(epoch + 1),
                "BalancedCRPS": float(metrics["BalancedCRPS"]),
                "StateBrier": float(metrics["StateBrier"]),
                "CalibrationGap95": float(metrics["CalibrationGap95"]),
                "ADE": float(metrics["ADE"]),
                "FDE": float(metrics["FDE"]),
                "StateAcc": float(metrics["StateAcc"]),
                "candidate_path": str(candidate_path),
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": int(epoch + 1),
                "score": float(metrics["BalancedCRPS"]),
                "metrics": dict(metrics),
                "selection_metric": "pending_constrained_selection",
            }, candidate_path)
            candidate_records.append(candidate_record)

            current_minimum = min(
                row["BalancedCRPS"] for row in candidate_records
            )
            current_limit = _relative_upper_bound(
                current_minimum, args.checkpoint_crps_tolerance
            )
            retained_records = []
            for row in candidate_records:
                if row["BalancedCRPS"] <= current_limit:
                    retained_records.append(row)
                else:
                    Path(row["candidate_path"]).unlink(missing_ok=True)
            candidate_records = retained_records
            pd.DataFrame(candidate_records).to_csv(
                output_dir / "checkpoint_candidate_pool.csv", index=False
            )
            ema.restore(model)

            record = {
                "epoch": epoch + 1,
                "loss": sums[0] / batch_count,
                "trajectory_loss": sums[1] / batch_count,
                "state_kl": sums[2] / batch_count,
                "long_horizon_loss": sums[3] / batch_count,
                **metrics,
            }
            history.append(record)
            pd.DataFrame(history).to_csv(
                output_dir / "training_history.csv", index=False
            )
            print(
                f"seed={seed} epoch={epoch + 1} "
                f"ADE={metrics['ADE']:.4f} FDE={metrics['FDE']:.4f} "
                f"CRPS={metrics['BalancedCRPS']:.5f} "
                f"PIC95={metrics['BalancedPIC95']:.2f} "
                f"highPIC95={metrics['HighPIC95']:.2f} "
                f"state={metrics['StateAcc']:.3f}"
            )

    if not args.evaluate_only:
        (
            selected,
            minimum_crps,
            crps_limit,
            minimum_brier,
            brier_limit,
        ) = _select_constrained_checkpoint(
            candidate_records,
            args.checkpoint_crps_tolerance,
            args.checkpoint_brier_tolerance,
        )
        selected_path = Path(selected["candidate_path"])
        selected_checkpoint = torch.load(
            selected_path, map_location="cpu", weights_only=False
        )
        selected_checkpoint.update({
            "score": float(selected["BalancedCRPS"]),
            "selection_metric": (
                "CRPS_constraint_then_StateBrier_"
                "then_CalibrationGap95"
            ),
            "selection_details": {
                "minimum_CRPS": float(minimum_crps),
                "CRPS_limit": float(crps_limit),
                "minimum_Brier_in_CRPS_pool": float(minimum_brier),
                "Brier_limit": float(brier_limit),
                "CRPS_tolerance": float(
                    args.checkpoint_crps_tolerance
                ),
                "Brier_tolerance": float(
                    args.checkpoint_brier_tolerance
                ),
            },
        })
        torch.save(selected_checkpoint, checkpoint)
        best_epoch = int(selected["epoch"])
        best_score = float(selected["BalancedCRPS"])
        load_checkpoint = checkpoint

        rows = []
        for row in candidate_records:
            item = dict(row)
            item["CRPS_eligible"] = bool(
                row["BalancedCRPS"] <= crps_limit
            )
            item["Brier_eligible"] = bool(
                item["CRPS_eligible"]
                and row["StateBrier"] <= brier_limit
            )
            item["selected"] = bool(int(row["epoch"]) == best_epoch)
            rows.append(item)
        pd.DataFrame(rows).to_csv(
            output_dir / "checkpoint_selection.csv", index=False
        )
        print(
            "[checkpoint-v4] "
            f"selected epoch={best_epoch}; "
            f"CRPS={selected['BalancedCRPS']:.6f} "
            f"(minimum={minimum_crps:.6f}, limit={crps_limit:.6f}); "
            f"StateBrier={selected['StateBrier']:.6f} "
            f"(pool minimum={minimum_brier:.6f}, "
            f"limit={brier_limit:.6f}); "
            f"CalGap95={selected['CalibrationGap95']:.3f}",
            flush=True,
        )

    saved = torch.load(
        load_checkpoint, map_location=device, weights_only=False
    )
    best_epoch = int(saved.get("epoch", best_epoch))
    best_score = float(saved.get("score", best_score))
    if load_checkpoint.resolve() != checkpoint.resolve():
        torch.save(saved, checkpoint)
    model.load_state_dict(saved["model_state_dict"])

    # Make validation calibration reproducible and independent of the number
    # of random draws consumed during training.
    base.seed_everything(seed + 40000, True)
    variance_scale = fit_variance_temperature(
        model, validation_loader, device, shrinkage=256.0
    )
    calibration_rows = []
    for state, name in enumerate(("low", "moderate", "high")):
        for step, scale in enumerate(
            variance_scale[state].tolist(), start=1
        ):
            calibration_rows.append({
                "state": name,
                "prediction_step": step,
                "standard_deviation_scale": scale,
            })
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "validation_variance_temperature.csv", index=False
    )
    calibrated_model = VarianceTemperatureModel(
        model, variance_scale
    ).to(device)

    test_groups = base.build_fixed_groups_from_labels([
        {**window, "label": int(label)}
        for window, label in zip(
            shared["test_indices"], state_posterior["test"].argmax(1)
        )
    ])
    evaluation_ablation = dict(ablation)
    if args.disable_eval_projection:
        evaluation_ablation["reachable_projection"] = False
        print(
            "[diagnostic] Reachable-domain projection is disabled only for "
            "the final probabilistic evaluation."
        )

    evaluation_config = {
        **protocol,
        "ablation": evaluation_ablation,
        "physical_limits": shared["physical_limits"],
        "final_mc_samples": args.mc_samples,
        "energy_score_pairs": 32,
        "save_prediction_npz": args.save_predictions,
    }
    # Use an independent, reproducible Monte Carlo stream for final testing.
    base.seed_everything(seed + 50000, True)
    groups, probability_metrics = base.run_maneuver_eval_on_best(
        calibrated_model, test_loader, shared["scaler_target"],
        test_groups, evaluation_config, device, str(output_dir), "test",
    )
    base.save_best_maneuver_results(
        groups, str(output_dir), "test"
    )

    summary = {
        "Seed": seed,
        "BestEpoch": best_epoch,
        "ValidationScore": best_score,
        "BaseScript": Path(args.base_script).name,
        "EvaluationProjection": not args.disable_eval_projection,
        "ProjectionInitialState": (
            "disabled"
            if args.disable_eval_projection
            else "last_observed_delta"
        ),
        "CheckpointMetric": (
            "CRPS_constraint_then_StateBrier_then_CalibrationGap95"
        ),
        "CalibrationScaleMean": float(variance_scale.mean()),
        "CalibrationScaleMax": float(variance_scale.max()),
        **probability_metrics,
    }
    for group_name, group_metrics in groups.items():
        for source, destination in (
            ("ade", "ADE"), ("fde", "FDE"),
            ("DeltaNLL", "NLL"), ("mPIC95", "PIC95"),
            ("MIW95", "MIW95"),
        ):
            summary[f"{group_name.upper()}_{destination}"] = (
                group_metrics[source]
            )
    pd.DataFrame([summary]).to_csv(
        output_dir / "seed_metrics_hmm.csv", index=False
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-script",
        default="./uav_formal_base_v4.py",
        help="Importable base implementation (default: uav_formal_base_v4.py).",
    )
    parser.add_argument("--data-dir", default="uavdataset/")
    parser.add_argument("--output-dir", default="HMM_SoftState_5Hz_ProbCal")
    parser.add_argument(
        "--cache-dir", default="Shared_UAV_Protocol_70_10_20"
    )
    parser.add_argument(
        "--mamba-dir",
        default="./mamba-main",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ssl-epochs", type=int, default=30)
    parser.add_argument("--ssl-lr", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--hmm-restarts", type=int, default=5)
    parser.add_argument("--state-weight", type=float, default=0.2)
    parser.add_argument("--state-temperature", type=float, default=1.0)
    parser.add_argument(
        "--state-confidence-floor", type=float, default=0.25
    )
    parser.add_argument(
        "--checkpoint-crps-tolerance", type=float, default=0.01,
        help="Retain checkpoints within this relative margin of minimum CRPS.",
    )
    parser.add_argument(
        "--checkpoint-brier-tolerance", type=float, default=0.02,
        help="StateBrier near-tie margin inside the CRPS candidate pool.",
    )
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip training and recalibrate/evaluate existing checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-root", default=None,
        help="Root containing seed_<seed>/best_checkpoint.pt for evaluate-only.",
    )
    parser.add_argument(
        "--disable-eval-projection", action="store_true",
        help=(
            "Diagnostic only: disable reachable-domain projection during the "
            "final probabilistic evaluation without changing training."
        ),
    )
    return parser.parse_args()


def validate_evaluate_only_checkpoints(args):
    """Fail before dataset preparation when requested checkpoints are absent."""
    if not args.evaluate_only:
        return

    requested_seeds = [int(seed) for seed in args.seeds]
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError(
            "--seeds contains duplicate values: "
            f"{requested_seeds}. Each independent run must be listed once."
        )

    checkpoint_root = Path(
        args.checkpoint_root or args.output_dir
    ).expanduser().resolve()
    missing = [
        seed for seed in requested_seeds
        if not (
            checkpoint_root / f"seed_{seed}" / "best_checkpoint.pt"
        ).is_file()
    ]
    if not missing:
        print(
            "[preflight] All evaluate-only checkpoints are available: "
            + ", ".join(map(str, requested_seeds)),
            flush=True,
        )
        return

    available = []
    if checkpoint_root.is_dir():
        for checkpoint in checkpoint_root.glob(
            "seed_*/best_checkpoint.pt"
        ):
            name = checkpoint.parent.name
            try:
                available.append(int(name.removeprefix("seed_")))
            except ValueError:
                continue
    available = sorted(set(available))

    raise FileNotFoundError(
        "Evaluate-only checkpoint preflight failed before dataset loading.\n"
        f"  checkpoint root: {checkpoint_root}\n"
        f"  requested seeds: {requested_seeds}\n"
        f"  missing seeds: {missing}\n"
        f"  available seeds: {available or 'none'}\n"
        "--evaluate-only does not train missing seeds. Use exactly the seeds "
        "listed as available, point --checkpoint-root to the correct run, "
        "or run training without --evaluate-only to create the missing "
        "checkpoints."
    )


def main():
    args = parse_args()
    if not 0.0 <= args.state_confidence_floor <= 1.0:
        raise SystemExit("--state-confidence-floor must be in [0, 1]")
    if args.checkpoint_crps_tolerance < 0.0:
        raise SystemExit("--checkpoint-crps-tolerance must be non-negative")
    if args.checkpoint_brier_tolerance < 0.0:
        raise SystemExit("--checkpoint-brier-tolerance must be non-negative")
    validate_evaluate_only_checkpoints(args)
    base = load_base(args.base_script)
    sys.path.append(args.mamba_dir)
    try:
        base.get_mamba2()
    except TypeError:
        base.get_mamba2(args.mamba_dir)

    protocol = {
        "data_dir": args.data_dir,
        "horizon": 20,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "scales": [5, 10, 20, 30, 45],
        "d_model": args.d_model,
        "train_num_workers": args.workers,
        "eval_num_workers": args.workers,
        "num_workers": args.workers,
        "split_seed": 42,
        "protocol_cache_dir": args.cache_dir,
        "force_rebuild_features": False,
        "force_rebuild_split": False,
        "force_rebuild_scalers": False,
        "deterministic": True,
        "train_stride": 5,
        "eval_stride": 2,
        "sample_interval_s": 0.2,
        "distance_accuracy_ratio": 0.15,
        "distance_accuracy_floor_m": 0.25,
        "distance_accuracy_cap_m": 1.5,
        "direction_threshold_deg": 25.0,
        "physical_limit_quantile": 99.5,
        "physical_limit_margin": 1.2,
        "reachable_projection": True,
    }
    base.seed_everything(42, True)
    shared = base.prepare_shared_dataset(protocol)
    results = [
        run_seed(base, shared, protocol, args, seed)
        for seed in args.seeds
    ]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        Path(args.output_dir) / "all_seed_metrics_hmm.csv", index=False
    )


if __name__ == "__main__":
    main()

