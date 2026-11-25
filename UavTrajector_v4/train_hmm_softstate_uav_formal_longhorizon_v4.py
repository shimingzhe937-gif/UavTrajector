#!/usr/bin/env python3
"""State-conditioned long-horizon training without a larger network.

The EMam-SE + MSA-DTP + H-PGD architecture and inference graph are unchanged.
This launcher adds only training-time objectives:

1. soft-state-balanced sampling;
2. a late-horizon cumulative-position Smooth-L1 objective; and
3. a late-horizon cumulative Gaussian CRPS objective.

The cumulative objectives are softly conditioned on the frozen three-state
HMM posterior. No hard manoeuvre threshold or test label is used.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Installs the causal split, shared frozen SSL/HMM teacher, baseline-compatible
# checkpoint selection, and validation-only cumulative-position PIC calibration.
import train_hmm_softstate_uav_formal_piccal_v4 as formal

core = formal.core
_base_make_loader = core.make_loader
_base_run_seed = core.run_seed


def _pop_float(name, default):
    if name not in sys.argv:
        return float(default)
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a numeric value")
    value = float(sys.argv[index + 1])
    del sys.argv[index:index + 2]
    return value


BALANCE_STRENGTH = _pop_float("--balance-strength", 0.35)
POSITION_WEIGHT = _pop_float("--position-weight", 0.05)
PROBABILITY_WEIGHT = _pop_float("--probability-weight", 0.01)
LATE_HORIZON_START = _pop_float("--late-horizon-start", 0.50)
LOW_STATE_GAIN = _pop_float("--low-state-gain", 0.80)
HIGH_STATE_GAIN = _pop_float("--high-state-gain", 1.35)
RAMP_FRACTION = _pop_float("--horizon-ramp-fraction", 0.20)

if not 0.0 <= BALANCE_STRENGTH <= 1.0:
    raise SystemExit("--balance-strength must be in [0, 1]")
if POSITION_WEIGHT < 0.0 or PROBABILITY_WEIGHT < 0.0:
    raise SystemExit("auxiliary loss weights must be non-negative")
if not 0.0 <= LATE_HORIZON_START < 1.0:
    raise SystemExit("--late-horizon-start must be in [0, 1)")
if LOW_STATE_GAIN <= 0.0 or HIGH_STATE_GAIN <= 0.0:
    raise SystemExit("state gains must be positive")
if not 0.0 < RAMP_FRACTION <= 1.0:
    raise SystemExit("--horizon-ramp-fraction must be in (0, 1]")


def soft_balanced_loader(dataset, batch_size, workers, shuffle, seed):
    """Balance posterior state mass without converting it to hard labels."""
    posterior = getattr(dataset, "posterior", None)
    if not shuffle or posterior is None or BALANCE_STRENGTH <= 0.0:
        return _base_make_loader(
            dataset, batch_size, workers, shuffle, seed
        )

    posterior = np.asarray(posterior, dtype=np.float64)
    if posterior.ndim != 2 or posterior.shape[0] != len(dataset):
        raise ValueError(
            f"posterior shape {posterior.shape} is incompatible with "
            f"dataset length {len(dataset)}"
        )
    state_mass = posterior.mean(axis=0)
    if np.any(state_mass <= 0.0):
        raise ValueError(f"invalid state mass: {state_mass}")

    inverse_frequency = (
        posterior / (posterior.shape[1] * state_mass[None, :])
    ).sum(axis=1)
    sample_weight = (
        1.0 - BALANCE_STRENGTH
        + BALANCE_STRENGTH * inverse_frequency
    )
    sample_weight = np.clip(sample_weight, 0.5, 2.0)
    sample_weight /= sample_weight.mean()

    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": False,
        "generator": generator,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def _soft_state_horizon_weights(posterior, horizon):
    """Return batch-wise late-horizon weights mixed by soft state posterior."""
    dtype, device = torch.float32, posterior.device
    fraction = torch.arange(
        1, horizon + 1, device=device, dtype=dtype
    ) / float(horizon)
    late_progress = (
        (fraction - LATE_HORIZON_START)
        / max(1.0 - LATE_HORIZON_START, 1e-6)
    ).clamp(0.0, 1.0)
    # Smoothly activate the auxiliary objective around the selected boundary.
    gate = late_progress.square() * (3.0 - 2.0 * late_progress)

    state_power = torch.tensor(
        [0.5, 1.0, 2.0], device=device, dtype=dtype
    )
    state_gain = torch.tensor(
        [LOW_STATE_GAIN, 1.0, HIGH_STATE_GAIN],
        device=device, dtype=dtype,
    )
    templates = state_gain[:, None] * gate[None, :] * (
        0.5 + 0.5 * late_progress[None, :].pow(state_power[:, None])
    )
    posterior = posterior.float()
    posterior = posterior / posterior.sum(-1, keepdim=True).clamp_min(1e-8)
    weights = posterior @ templates
    # Batch normalisation preserves relative state emphasis while keeping the
    # auxiliary-loss magnitude stable across batches.
    return weights / weights.mean().detach().clamp_min(1e-8)


def state_conditioned_long_horizon_loss(
    prediction, logvar, target, posterior, epoch, total_epochs
):
    """Cumulative-position point and probabilistic objectives."""
    prediction = prediction.float()
    target = target.float()
    logvar = logvar.float().clamp(-12.0, 8.0)
    horizon = prediction.shape[1]
    weights = _soft_state_horizon_weights(posterior, horizon)

    position_error = torch.cumsum(prediction - target, dim=1)
    point_per_step = F.smooth_l1_loss(
        position_error,
        torch.zeros_like(position_error),
        reduction="none",
        beta=0.5,
    ).mean(dim=-1)
    point_loss = (weights * point_per_step).mean()

    # Under the decoder's diagonal-increment Gaussian assumption, cumulative
    # position variance is the sum of per-step displacement variances.
    position_variance = torch.cumsum(torch.exp(logvar), dim=1).clamp_min(1e-8)
    sigma = position_variance.sqrt()
    z_score = (-position_error) / sigma
    density = (
        torch.exp(-0.5 * z_score.square()) / math.sqrt(2.0 * math.pi)
    )
    cdf = 0.5 * (1.0 + torch.erf(z_score / math.sqrt(2.0)))
    crps = sigma * (
        z_score * (2.0 * cdf - 1.0)
        + 2.0 * density
        - 1.0 / math.sqrt(math.pi)
    )
    probability_loss = (weights * crps.mean(dim=-1)).mean()

    ramp_epochs = max(RAMP_FRACTION * max(total_epochs, 1), 1.0)
    ramp = min(1.0, float(epoch + 1) / ramp_epochs)
    return ramp * (
        POSITION_WEIGHT * point_loss
        + PROBABILITY_WEIGHT * probability_loss
    )


def run_seed_with_metadata(base, shared, protocol, args, seed):
    result = _base_run_seed(base, shared, protocol, args, seed)
    result.update({
        "BalanceStrength": BALANCE_STRENGTH,
        "LongPositionWeight": POSITION_WEIGHT,
        "LongProbabilityWeight": PROBABILITY_WEIGHT,
        "LateHorizonStart": LATE_HORIZON_START,
        "LowStateGain": LOW_STATE_GAIN,
        "HighStateGain": HIGH_STATE_GAIN,
        "HorizonRampFraction": RAMP_FRACTION,
        "TrainingModification": (
            "soft_state_balancing_plus_state_conditioned_"
            "cumulative_position_and_crps"
        ),
    })
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    pd.DataFrame([result]).to_csv(
        seed_dir / "seed_metrics_hmm.csv", index=False
    )
    return result


core.make_loader = soft_balanced_loader
core.training_auxiliary_loss = state_conditioned_long_horizon_loss
core.run_seed = run_seed_with_metadata


if __name__ == "__main__":
    print(
        "[formal-longhorizon-v4] "
        f"balance={BALANCE_STRENGTH:.2f}, "
        f"position_weight={POSITION_WEIGHT:.3f}, "
        f"probability_weight={PROBABILITY_WEIGHT:.3f}, "
        f"late_start={LATE_HORIZON_START:.2f}, "
        f"state_gain=({LOW_STATE_GAIN:.2f},1.00,{HIGH_STATE_GAIN:.2f})",
        flush=True,
    )
    core.main()

