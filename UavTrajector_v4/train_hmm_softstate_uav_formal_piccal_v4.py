#!/usr/bin/env python3
"""Formal soft-state training with direct validation PIC95 calibration.

This launcher reuses the existing probability-calibrated trainer and frozen
SSL/HMM teacher.  It changes no network layer and adds no trainable parameter.
Only checkpoint selection and the validation-only variance scale are replaced.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# This import also installs the multiprocessing-safe base loader and the shared
# frozen SSL/HMM teacher. It consumes --shared-teacher-dir from sys.argv.
import train_hmm_softstate_uav_formal_probcal_v4 as formal

core = formal.core
_original_validation = core.validation
_original_run_seed = core.run_seed
_base_temperature_model = core.VarianceTemperatureModel


def _pop_int(name, default):
    if name not in sys.argv:
        return int(default)
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires an integer value")
    value = int(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    return value


CALIBRATION_SAMPLES = _pop_int("--calibration-samples", 100)
if CALIBRATION_SAMPLES < 20:
    raise SystemExit("--calibration-samples must be at least 20")

_last_sample_scale = None
_last_sample_rows = None
_last_posterior_map = None
_last_posterior_rows = None

POSTERIOR_MAP_PSEUDOCOUNT = 32.0
CONDITIONAL_CALIBRATION_ITERATIONS = 6
CONDITIONAL_CALIBRATION_DAMPING = 0.5


def validation_with_original_selection(base, model, loader, scaler, device):
    """Preserve metrics; v4 performs post-training constrained selection."""
    metrics = _original_validation(base, model, loader, scaler, device)
    metrics["score"] = metrics["BalancedCRPS"]
    return metrics


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if values.size == 0:
        return np.nan
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(quantile, cumulative, values))


@torch.no_grad()
def fit_pic95_variance_temperature(model, loader, device, shrinkage=256.0):
    """Fit soft-state, horizon-wise scales that target position PIC95.

    H-PGD predicts displacement variances, while the paper reports coverage of
    cumulative positions.  We therefore calibrate standardised cumulative
    position residuals and convert the desired cumulative variance back to
    increment-wise multipliers. State scales are fitted with the validation
    HMM soft posterior. At deployment, the model posterior is transformed to
    that state space by the validation-fitted transition matrix.
    """
    model.eval()
    ratio_chunks = []
    posterior_chunks = []
    variance_chunks = []
    normal_975 = 1.959963984540054

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
        prediction, logvar, _, _ = model(inputs, tau=1.0)
        increment_variance = torch.exp(
            logvar.float().clamp(-12.0, 8.0)
        )
        position_error = torch.cumsum(
            target - prediction.float(), dim=1
        )
        position_variance = torch.cumsum(increment_variance, dim=1)
        ratio_chunks.append((
            position_error.abs()
            / position_variance.clamp_min(1e-10).sqrt()
        ).cpu().numpy())
        if "state_posterior" not in batch:
            raise KeyError(
                "Validation batches must contain causal HMM state_posterior."
            )
        posterior_chunks.append(
            batch["state_posterior"].cpu().numpy()
        )
        variance_chunks.append(increment_variance.cpu().numpy())

    if not ratio_chunks:
        raise RuntimeError(
            "Cannot calibrate variance with an empty validation loader."
        )

    ratio = np.concatenate(ratio_chunks, axis=0)        # [N,H,D]
    posterior = np.concatenate(posterior_chunks, axis=0)  # [N,S]
    increment_variance = np.concatenate(
        variance_chunks, axis=0
    )                                                   # [N,H,D]
    sample_count, horizon, dimension = ratio.shape
    state_count = posterior.shape[1]

    global_weights = np.ones(
        sample_count * dimension, dtype=np.float64
    )
    global_position_scale = np.empty(horizon, np.float64)
    for step in range(horizon):
        quantile = weighted_quantile(
            ratio[:, step, :].reshape(-1), global_weights, 0.95
        )
        global_position_scale[step] = quantile / normal_975

    state_position_scale = np.empty(
        (state_count, horizon), dtype=np.float64
    )
    for state in range(state_count):
        state_weights = posterior[:, state].astype(np.float64)
        coordinate_weights = np.repeat(state_weights, dimension)
        effective_count = state_weights.sum()
        mixing = effective_count / (effective_count + float(shrinkage))
        for step in range(horizon):
            quantile = weighted_quantile(
                ratio[:, step, :].reshape(-1),
                coordinate_weights,
                0.95,
            )
            raw_scale = max(quantile / normal_975, 1e-6)
            global_scale = max(global_position_scale[step], 1e-6)
            state_position_scale[state, step] = np.exp(
                mixing * np.log(raw_scale)
                + (1.0 - mixing) * np.log(global_scale)
            )

    # Convert the target cumulative position variance to a displacement-scale
    # sequence. The cumulative maximum prevents negative variance increments.
    increment_scale_squared = np.empty_like(state_position_scale)
    mean_dimension_variance = increment_variance.mean(axis=2)  # [N,H]
    for state in range(state_count):
        weights = posterior[:, state].astype(np.float64)
        weight_sum = max(weights.sum(), 1e-12)
        mean_increment_variance = (
            mean_dimension_variance * weights[:, None]
        ).sum(axis=0) / weight_sum
        base_cumulative_variance = np.cumsum(mean_increment_variance)
        desired_cumulative_variance = (
            base_cumulative_variance
            * np.square(state_position_scale[state])
        )
        desired_cumulative_variance = np.maximum.accumulate(
            desired_cumulative_variance
        )
        desired_increment_variance = np.diff(np.concatenate([
            np.zeros(1, dtype=np.float64),
            desired_cumulative_variance,
        ]))
        increment_scale_squared[state] = (
            desired_increment_variance
            / np.maximum(mean_increment_variance, 1e-12)
        )

    scale = np.sqrt(np.maximum(increment_scale_squared, 1e-12))
    # Bounds prevent validation-specific extremes from dominating test output.
    return torch.from_numpy(
        np.clip(scale, 0.75, 3.0).astype(np.float32)
    )


def weighted_higher_quantile(values, weights, quantile):
    """Conservative weighted empirical quantile."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if values.size == 0:
        return np.nan
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(
        cumulative, float(quantile) * cumulative[-1], side="left"
    ))
    return float(values[min(index, values.size - 1)])


def fit_posterior_transition(
    predicted_posterior,
    teacher_posterior,
    pseudocount=POSTERIOR_MAP_PSEUDOCOUNT,
):
    """Align the deployable state head with the causal HMM posterior.

    The row-stochastic 3-by-3 map is fitted on validation histories only.
    An identity pseudocount preserves the known semantic ordering of the
    low, moderate and high states when a validation cross-moment is small.
    """
    predicted = np.asarray(predicted_posterior, dtype=np.float64)
    teacher = np.asarray(teacher_posterior, dtype=np.float64)
    if predicted.shape != teacher.shape or predicted.ndim != 2:
        raise ValueError(
            "Predicted and teacher posteriors must have matching [N,S] shape."
        )
    predicted = np.clip(predicted, 1e-8, None)
    teacher = np.clip(teacher, 1e-8, None)
    predicted /= predicted.sum(axis=1, keepdims=True)
    teacher /= teacher.sum(axis=1, keepdims=True)

    state_count = predicted.shape[1]
    cross_moment = predicted.T @ teacher
    cross_moment += float(pseudocount) * np.eye(state_count)
    transition = cross_moment / np.maximum(
        cross_moment.sum(axis=1, keepdims=True), 1e-12
    )
    aligned = predicted @ transition
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
    return transition, aligned


@torch.no_grad()
def fit_sample_interval_scale(model, loader, device, shrinkage=256.0):
    """Fit deployable, HMM-conditioned sampled-position interval scales.

    The model state head is first aligned to the causal HMM posterior with a
    validation-only row-stochastic transition. State-specific interval scales
    are fitted with the HMM soft posterior, while deployment mixes them with
    the aligned model posterior. A damped conditional correction targets
    nominal coverage within each HMM state instead of only in the pooled
    validation set.
    """
    model.eval()
    ratio_blocks = []
    predicted_posterior_blocks = []
    teacher_posterior_blocks = []

    for batch in loader:
        if "state_posterior" not in batch:
            raise KeyError(
                "Validation batches must contain causal HMM state_posterior."
            )
        inputs = {
            "seq": batch["seq"].to(device, non_blocking=True),
            "last_delta": batch["last_delta"].to(
                device, non_blocking=True
            ),
        }
        target = batch["target"].to(
            device, non_blocking=True
        ).float()
        output = model(inputs, tau=1.0)
        logits = output[2]
        sample_delta = model.generate_probabilistic_trajectories(
            inputs, num_samples=CALIBRATION_SAMPLES
        )

        sample_position = np.cumsum(
            sample_delta.float().cpu().numpy(), axis=2
        )
        target_position = np.cumsum(
            target.cpu().numpy(), axis=1
        )
        center = sample_position.mean(axis=0)
        lower = np.quantile(sample_position, 0.025, axis=0)
        upper = np.quantile(sample_position, 0.975, axis=0)
        side_radius = np.where(
            target_position >= center,
            upper - center,
            center - lower,
        )
        radius_floor = np.maximum(
            np.median(
                np.abs(sample_position - center[None, ...]),
                axis=(0, 1),
            ) * 1e-3,
            1e-6,
        )
        ratio_blocks.append(
            np.abs(target_position - center)
            / np.maximum(side_radius, radius_floor[None, :, :])
        )
        predicted_posterior_blocks.append(
            logits.float().softmax(-1).cpu().numpy()
        )
        teacher_posterior_blocks.append(
            batch["state_posterior"].cpu().numpy()
        )

    if not ratio_blocks:
        raise RuntimeError("Validation loader produced no calibration samples.")

    ratio = np.concatenate(ratio_blocks, axis=0)  # [N,H,D]
    predicted_posterior = np.concatenate(
        predicted_posterior_blocks, axis=0
    )
    teacher_posterior = np.concatenate(
        teacher_posterior_blocks, axis=0
    )
    posterior_map, deployment_posterior = fit_posterior_transition(
        predicted_posterior, teacher_posterior
    )
    sample_count, horizon, dimension = ratio.shape
    state_count = teacher_posterior.shape[1]
    quantile_level = min(
        1.0,
        math.ceil((sample_count + 1) * 0.95) / sample_count,
    )

    global_scale = np.empty((horizon, dimension), np.float64)
    for step in range(horizon):
        for axis in range(dimension):
            try:
                global_scale[step, axis] = np.quantile(
                    ratio[:, step, axis],
                    quantile_level,
                    method="higher",
                )
            except TypeError:
                global_scale[step, axis] = np.quantile(
                    ratio[:, step, axis],
                    quantile_level,
                    interpolation="higher",
                )

    state_scale = np.empty(
        (state_count, horizon, dimension), dtype=np.float64
    )
    names = ("low", "moderate", "high")
    axes = ("x", "y", "z")
    raw_state_scale = np.empty_like(state_scale)
    effective_masses = np.empty(state_count, dtype=np.float64)
    for state in range(state_count):
        weights = teacher_posterior[:, state].astype(np.float64)
        effective_count = float(weights.sum())
        effective_masses[state] = effective_count
        mixing = effective_count / (
            effective_count + float(shrinkage)
        )
        for step in range(horizon):
            for axis in range(dimension):
                raw = weighted_higher_quantile(
                    ratio[:, step, axis], weights, quantile_level
                )
                global_value = max(global_scale[step, axis], 1e-6)
                raw = max(raw, 1e-6)
                scale = np.exp(
                    mixing * np.log(raw)
                    + (1.0 - mixing) * np.log(global_value)
                )
                scale = float(np.clip(scale, 0.50, 6.00))
                state_scale[state, step, axis] = scale
                raw_state_scale[state, step, axis] = raw

    def conditional_calibration_error(candidate_scale):
        candidate_mixed = np.einsum(
            "ns,shd->nhd", deployment_posterior, candidate_scale
        )
        errors = []
        for state in range(state_count):
            weights = teacher_posterior[:, state].astype(np.float64)
            weight_sum = max(weights.sum(), 1e-12)
            for step in range(horizon):
                for axis in range(dimension):
                    covered = (
                        ratio[:, step, axis]
                        <= candidate_mixed[:, step, axis]
                    )
                    coverage = np.sum(weights * covered) / weight_sum
                    errors.append(abs(float(coverage) - 0.95))
        return float(np.mean(errors))

    # Correct the deployed mixture conditionally. The update is damped and
    # shrunk by effective state mass because changing one component also
    # affects samples assigned partially to the other soft states. Retaining
    # the best validation iterate prevents a late update from worsening the
    # average state-wise calibration error.
    best_state_scale = state_scale.copy()
    best_conditional_error = conditional_calibration_error(state_scale)
    for _ in range(CONDITIONAL_CALIBRATION_ITERATIONS):
        mixed_scale = np.einsum(
            "ns,shd->nhd", deployment_posterior, state_scale
        )
        updated_scale = state_scale.copy()
        for state in range(state_count):
            weights = teacher_posterior[:, state].astype(np.float64)
            effective_count = float(weights.sum())
            shrink = effective_count / (
                effective_count + float(shrinkage)
            )
            for step in range(horizon):
                for axis in range(dimension):
                    correction = weighted_higher_quantile(
                        ratio[:, step, axis]
                        / np.maximum(
                            mixed_scale[:, step, axis], 1e-8
                        ),
                        weights,
                        quantile_level,
                    )
                    correction = float(np.clip(correction, 0.5, 2.0))
                    log_update = (
                        CONDITIONAL_CALIBRATION_DAMPING
                        * shrink
                        * np.log(max(correction, 1e-8))
                    )
                    updated_scale[state, step, axis] = np.clip(
                        state_scale[state, step, axis]
                        * np.exp(log_update),
                        0.50,
                        6.00,
                    )
        state_scale = updated_scale
        current_error = conditional_calibration_error(state_scale)
        if current_error < best_conditional_error:
            best_conditional_error = current_error
            best_state_scale = state_scale.copy()

    state_scale = best_state_scale
    mixed_scale = np.einsum(
        "ns,shd->nhd", deployment_posterior, state_scale
    )

    def weighted_coverage(indicator, weights):
        weights = np.asarray(weights, dtype=np.float64)
        return float(
            100.0
            * np.sum(weights * np.asarray(indicator, dtype=np.float64))
            / max(weights.sum(), 1e-12)
        )

    rows = []
    for state in range(state_count):
        state_weights = teacher_posterior[:, state].astype(np.float64)
        for step in range(horizon):
            for axis in range(dimension):
                rows.append({
                    "record_type": "state_scale",
                    "state": names[state],
                    "prediction_step": step + 1,
                    "axis": axes[axis],
                    "sample_interval_scale": float(
                        state_scale[state, step, axis]
                    ),
                    "raw_state_scale": float(
                        raw_state_scale[state, step, axis]
                    ),
                    "global_scale": float(global_scale[step, axis]),
                    "effective_state_mass": float(
                        effective_masses[state]
                    ),
                    "calibration_samples": CALIBRATION_SAMPLES,
                    "validation_windows": sample_count,
                    "mean_conditional_calibration_error_pct": float(
                        100.0 * best_conditional_error
                    ),
                    "validation_PIC95_before": weighted_coverage(
                        ratio[:, step, axis] <= 1.0,
                        state_weights,
                    ),
                    "validation_PIC95_after": weighted_coverage(
                        ratio[:, step, axis]
                        <= mixed_scale[:, step, axis],
                        state_weights,
                    ),
                })
    for step in range(horizon):
        for axis in range(dimension):
            rows.append({
                "record_type": "mixture_diagnostic",
                "state": "soft_mixture",
                "prediction_step": step + 1,
                "axis": axes[axis],
                "sample_interval_scale": float(
                    mixed_scale[:, step, axis].mean()
                ),
                "raw_state_scale": np.nan,
                "global_scale": float(global_scale[step, axis]),
                "effective_state_mass": float(sample_count),
                "calibration_samples": CALIBRATION_SAMPLES,
                "validation_windows": sample_count,
                "validation_PIC95_before": float(
                    100.0 * np.mean(ratio[:, step, axis] <= 1.0)
                ),
                "validation_PIC95_after": float(
                    100.0 * np.mean(
                        ratio[:, step, axis]
                        <= mixed_scale[:, step, axis]
                    )
                ),
            })

    posterior_rows = []
    for source in range(state_count):
        for target in range(state_count):
            posterior_rows.append({
                "source_model_state": names[source],
                "target_hmm_state": names[target],
                "transition_probability": float(
                    posterior_map[source, target]
                ),
                "pseudocount": POSTERIOR_MAP_PSEUDOCOUNT,
                "validation_windows": sample_count,
            })

    return (
        torch.from_numpy(state_scale.astype(np.float32)),
        pd.DataFrame(rows),
        torch.from_numpy(posterior_map.astype(np.float32)),
        pd.DataFrame(posterior_rows),
    )


def fit_combined_calibration(model, loader, device, shrinkage=256.0):
    """Fit analytic log-variance and sampled-interval calibration."""
    global _last_sample_scale, _last_sample_rows
    global _last_posterior_map, _last_posterior_rows
    analytic_scale = fit_pic95_variance_temperature(
        model, loader, device, shrinkage=shrinkage
    )
    (
        _last_sample_scale,
        _last_sample_rows,
        _last_posterior_map,
        _last_posterior_rows,
    ) = fit_sample_interval_scale(
        model, loader, device, shrinkage=shrinkage
    )
    return analytic_scale


class SampleCalibratedVarianceModel(_base_temperature_model):
    """Calibrate both analytic log variance and sampled position intervals."""
    def __init__(self, model, state_horizon_scale):
        super().__init__(model, state_horizon_scale)
        if _last_sample_scale is None or _last_posterior_map is None:
            raise RuntimeError(
                "Sample interval scale and posterior map must be fitted."
            )
        self.register_buffer(
            "state_sample_interval_scale",
            _last_sample_scale.clone(),
        )
        self.register_buffer(
            "state_posterior_transition",
            _last_posterior_map.clone(),
        )
        self._cached_posterior = None

    def _align_posterior(self, posterior):
        posterior = posterior.to(
            self.state_posterior_transition.dtype
        )
        aligned = posterior @ self.state_posterior_transition
        return aligned / aligned.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if (
            not isinstance(output, (tuple, list))
            or len(output) < 3
        ):
            return output
        items = list(output)
        aligned_posterior = self._align_posterior(
            items[2].float().softmax(-1)
        )
        self._cached_posterior = aligned_posterior.detach()
        if items[1] is not None:
            scale_squared = (
                aligned_posterior
                @ self.state_horizon_scale_squared
            )
            items[1] = (
                items[1].float()
                + scale_squared.clamp_min(1e-8).log().unsqueeze(-1)
            )
        return tuple(items) if isinstance(output, tuple) else items

    def generate_probabilistic_trajectories(
        self, inputs, num_samples=100, **kwargs
    ):
        sample_delta = self.model.generate_probabilistic_trajectories(
            inputs, num_samples=num_samples, **kwargs
        )
        posterior = self._cached_posterior
        if (
            posterior is None
            or posterior.shape[0] != sample_delta.shape[1]
        ):
            output = self.model(inputs, tau=1.0)
            posterior = self._align_posterior(
                output[2].float().softmax(-1).detach()
            )

        # Mix validation-fitted state scales using the online soft posterior.
        scale = torch.einsum(
            "bs,shd->bhd",
            posterior.to(self.state_sample_interval_scale.dtype),
            self.state_sample_interval_scale,
        )
        sample_position = torch.cumsum(sample_delta.float(), dim=2)
        center = sample_position.mean(dim=0, keepdim=True)
        calibrated_position = center + (
            sample_position - center
        ) * scale.unsqueeze(0)
        first = calibrated_position[:, :, :1]
        remainder = (
            calibrated_position[:, :, 1:]
            - calibrated_position[:, :, :-1]
        )
        calibrated_delta = torch.cat([first, remainder], dim=2)
        return calibrated_delta.to(sample_delta.dtype)


def run_seed_with_correct_metadata(base, shared, protocol, args, seed):
    result = _original_run_seed(base, shared, protocol, args, seed)
    result["CheckpointMetric"] = (
        "CRPS_constraint_then_StateBrier_then_CalibrationGap95"
    )
    result["CalibrationObjective"] = (
        "validation_HMM_conditioned_sampled_position_PIC95"
    )
    result["CalibrationSamples"] = CALIBRATION_SAMPLES
    result["PosteriorAlignment"] = (
        "validation_row_stochastic_model_to_HMM"
    )
    result["ConditionalCalibrationIterations"] = (
        CONDITIONAL_CALIBRATION_ITERATIONS
    )
    if _last_sample_scale is not None:
        result["SampleCalibrationScaleMean"] = float(
            _last_sample_scale.mean()
        )
        result["SampleCalibrationScaleMax"] = float(
            _last_sample_scale.max()
        )
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    if _last_sample_rows is not None:
        _last_sample_rows.to_csv(
            seed_dir / "validation_sample_interval_scale.csv",
            index=False,
        )
    if _last_posterior_rows is not None:
        _last_posterior_rows.to_csv(
            seed_dir / "validation_state_posterior_map.csv",
            index=False,
        )
    if _last_sample_scale is not None:
        np.save(
            seed_dir / "validation_sample_interval_scale.npy",
            _last_sample_scale.cpu().numpy(),
        )
    if _last_posterior_map is not None:
        np.save(
            seed_dir / "validation_state_posterior_map.npy",
            _last_posterior_map.cpu().numpy(),
        )
    pd.DataFrame([result]).to_csv(
        seed_dir / "seed_metrics_hmm.csv", index=False
    )
    checkpoint_path = seed_dir / "best_checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint["selection_metric"] = (
        "CRPS_constraint_then_StateBrier_then_CalibrationGap95"
    )
    torch.save(checkpoint, checkpoint_path)
    return result


core.validation = validation_with_original_selection
core.fit_variance_temperature = fit_combined_calibration
core.VarianceTemperatureModel = SampleCalibratedVarianceModel
core.run_seed = run_seed_with_correct_metadata


if __name__ == "__main__":
    print(
        "[formal-piccal-v4] checkpoint=constrained-CRPS-Brier-CalGap; "
        "calibration=validation-HMM-conditioned sampled-position PIC95; "
        "posterior=validation model-to-HMM map; "
        f"K_cal={CALIBRATION_SAMPLES}",
        flush=True,
    )
    core.main()
