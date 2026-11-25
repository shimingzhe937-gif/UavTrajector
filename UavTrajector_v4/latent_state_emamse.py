#!/usr/bin/env python3
"""Offline SSL + causal Gaussian-HMM weak supervision for EMam-SE.

Designed to be imported by the formal UAV ablation/comparison scripts.
The HMM is an OFFLINE teacher only. Online inference uses model.i_head.
Requires: numpy, torch, scikit-learn, hmmlearn, joblib.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import json
import math

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler


@dataclass
class LatentStateConfig:
    n_states: int = 3
    latent_dim: int = 16
    mask_ratio: float = 0.20
    ssl_epochs: int = 30
    ssl_lr: float = 1e-4
    ssl_weight_decay: float = 1e-2
    hmm_covariance_type: str = "diag"
    hmm_n_iter: int = 300
    hmm_n_init: int = 5
    hmm_min_occupancy: float = 0.05
    state_loss_weight: float = 0.20
    state_temperature: float = 1.0
    seed: int = 2026


class EMamSESelfSupervisor(nn.Module):
    """Reuse the predictor's projection and EMam-SE encoder.

    No second online encoder is introduced. The projection/encoder parameters
    are the same objects as predictor.proj and predictor.encoder.
    """
    def __init__(self, predictor: nn.Module, feature_dim: int, latent_dim: int = 16):
        super().__init__()
        if getattr(predictor, "_encoder_kind", None) != "mamba":
            raise ValueError("SSL pretraining requires the EMam-SE variant.")
        self.proj = predictor.proj
        self.encoder = predictor.encoder
        d_model = int(predictor.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim),
        )

    def encode(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(self.proj(sequence))
        # Mean + last state is more robust than relying on the last state alone.
        pooled = 0.5 * encoded.mean(dim=1) + 0.5 * encoded[:, -1]
        latent = F.normalize(self.latent_head(pooled), dim=-1)
        return encoded, latent

    def masked_loss(self, sequence: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        batch, steps, _ = sequence.shape
        projected = self.proj(sequence)
        mask = torch.rand(batch, steps, device=sequence.device) < mask_ratio
        # Keep the latest point visible because it anchors the online state.
        mask[:, -1] = False
        # Guarantee at least one masked location per sequence.
        empty = ~mask.any(dim=1)
        if empty.any():
            mask[empty, 0] = True
        corrupted = torch.where(mask.unsqueeze(-1), self.mask_token, projected)
        reconstructed = self.reconstruction_head(self.encoder(corrupted))
        return F.smooth_l1_loss(reconstructed[mask], sequence[mask])


def pretrain_emamse(
    predictor: nn.Module,
    train_loader: Iterable[Dict[str, torch.Tensor]],
    feature_dim: int,
    device: torch.device,
    cfg: LatentStateConfig,
    checkpoint: str | Path,
) -> EMamSESelfSupervisor:
    """Pretrain predictor.proj + predictor.encoder on TRAIN histories only."""
    checkpoint = Path(checkpoint)
    ssl = EMamSESelfSupervisor(predictor, feature_dim, cfg.latent_dim).to(device)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        ssl.load_state_dict(state, strict=True)
        return ssl

    optimizer = torch.optim.AdamW(
        ssl.parameters(), lr=cfg.ssl_lr, weight_decay=cfg.ssl_weight_decay
    )
    ssl.train()
    for epoch in range(cfg.ssl_epochs):
        running, count = 0.0, 0
        for batch in train_loader:
            sequence = batch["seq"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = ssl.masked_loss(sequence, cfg.mask_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            count += 1
        print(f"SSL epoch {epoch + 1:03d}/{cfg.ssl_epochs}: loss={running/max(count,1):.6f}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ssl.state_dict(), checkpoint)
    return ssl


@torch.no_grad()
def extract_window_latents(
    ssl: EMamSESelfSupervisor,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return latent vectors and original dataset window indices."""
    ssl.eval()
    latent_chunks, index_chunks = [], []
    for batch in loader:
        _, latent = ssl.encode(batch["seq"].to(device, non_blocking=True))
        latent_chunks.append(latent.cpu().numpy())
        if "window_index" not in batch:
            raise KeyError("Dataset must return 'window_index'.")
        index_chunks.append(batch["window_index"].cpu().numpy())
    return np.concatenate(latent_chunks), np.concatenate(index_chunks).astype(np.int64)


def _sequence_order(
    indices: Sequence[dict], selected_window_indices: np.ndarray
) -> Tuple[np.ndarray, List[int]]:
    """Sort windows by (trajectory, start) and return HMM sequence lengths."""
    records = []
    for local_row, window_idx in enumerate(selected_window_indices.tolist()):
        item = indices[int(window_idx)]
        records.append((int(item["seq_idx"]), int(item["start"]), local_row))
    records.sort(key=lambda x: (x[0], x[1]))
    order = np.asarray([r[2] for r in records], dtype=np.int64)
    lengths, previous, count = [], None, 0
    for seq_idx, _, _ in records:
        if previous is None or seq_idx == previous:
            count += 1
        else:
            lengths.append(count)
            count = 1
        previous = seq_idx
    if count:
        lengths.append(count)
    return order, lengths


def _hmm_parameter_count(model: GaussianHMM, dim: int) -> int:
    k = model.n_components
    start = k - 1
    trans = k * (k - 1)
    means = k * dim
    covariance = k * dim if model.covariance_type == "diag" else k * dim * (dim + 1) // 2
    return start + trans + means + covariance


def fit_hmm_with_restarts(
    latent: np.ndarray,
    lengths: List[int],
    cfg: LatentStateConfig,
) -> Tuple[GaussianHMM, RobustScaler, List[dict]]:
    """Fit HMM only on training histories and retain diagnostics."""
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(latent)
    z = scaler.transform(latent)
    candidates, labels = [], []
    for restart in range(cfg.hmm_n_init):
        model = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.hmm_covariance_type,
            n_iter=cfg.hmm_n_iter,
            tol=1e-4,
            min_covar=1e-4,
            random_state=cfg.seed + restart,
            verbose=False,
        ).fit(z, lengths)
        log_likelihood = float(model.score(z, lengths))
        hard = model.predict(z, lengths)
        occupancy = np.bincount(hard, minlength=cfg.n_states) / len(hard)
        n_params = _hmm_parameter_count(model, z.shape[1])
        bic = -2.0 * log_likelihood + n_params * math.log(len(z))
        candidates.append((model, log_likelihood, bic, occupancy))
        labels.append(hard)
    valid = [x for x in candidates if float(x[3].min()) >= cfg.hmm_min_occupancy]
    pool = valid if valid else candidates
    best = min(pool, key=lambda x: x[2])
    diagnostics = []
    for i, (_, ll, bic, occ) in enumerate(candidates):
        ari = [adjusted_rand_score(labels[i], labels[j]) for j in range(len(labels)) if j != i]
        diagnostics.append({
            "restart": i, "log_likelihood": ll, "bic": bic,
            "min_occupancy": float(occ.min()), "occupancy": occ.tolist(),
            "mean_pairwise_ari": float(np.mean(ari)) if ari else 1.0,
        })
    return best[0], scaler, diagnostics


def causal_filter_posteriors(
    model: GaussianHMM,
    observations: np.ndarray,
    lengths: Sequence[int],
) -> np.ndarray:
    """Compute p(s_t | z_1:t); unlike predict_proba, no future smoothing."""
    log_emission = model._compute_log_likelihood(observations)
    log_start = np.log(np.maximum(model.startprob_, 1e-30))
    log_trans = np.log(np.maximum(model.transmat_, 1e-30))
    output = np.empty_like(log_emission)
    offset = 0
    for length in lengths:
        alpha = log_start + log_emission[offset]
        alpha -= np.logaddexp.reduce(alpha)
        output[offset] = np.exp(alpha)
        for local in range(1, int(length)):
            scores = alpha[:, None] + log_trans
            alpha = log_emission[offset + local] + np.logaddexp.reduce(scores, axis=0)
            alpha -= np.logaddexp.reduce(alpha)
            output[offset + local] = np.exp(alpha)
        offset += int(length)
    return output


def order_states_by_history_dynamics(
    posterior: np.ndarray,
    ordered_sequences: np.ndarray,
    displacement_dims: slice = slice(0, 3),
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-hoc state naming; does not impose label thresholds.

    Uses observed-history displacement variation only to order the learned
    states from low to high. It does not create or fit the states.
    """
    delta = ordered_sequences[..., displacement_dims]
    variation = np.linalg.norm(np.diff(delta, axis=1), axis=-1).mean(axis=1)
    state_score = []
    for k in range(posterior.shape[1]):
        weight = posterior[:, k]
        state_score.append(float(np.sum(weight * variation) / max(np.sum(weight), 1e-12)))
    old_order = np.argsort(state_score)  # low, moderate, high
    return posterior[:, old_order], old_order


def build_offline_teacher(
    latent: np.ndarray,
    window_indices: np.ndarray,
    train_indices: Sequence[dict],
    ordered_history_sequences: np.ndarray,
    cfg: LatentStateConfig,
    output_dir: str | Path,
) -> np.ndarray:
    """Fit HMM and return posterior aligned to original dataset indices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order, lengths = _sequence_order(train_indices, window_indices)
    z_ordered = latent[order]
    model, scaler, diagnostics = fit_hmm_with_restarts(z_ordered, lengths, cfg)
    scaled = scaler.transform(z_ordered)
    posterior = causal_filter_posteriors(model, scaled, lengths)
    posterior, semantic_order = order_states_by_history_dynamics(
        posterior, ordered_history_sequences[order]
    )
    aligned = np.zeros((len(train_indices), cfg.n_states), dtype=np.float32)
    aligned[window_indices[order]] = posterior.astype(np.float32)
    joblib.dump({"hmm": model, "scaler": scaler, "semantic_order": semantic_order}, output_dir / "hmm_teacher.joblib")
    np.save(output_dir / "train_causal_state_posterior.npy", aligned)
    (output_dir / "hmm_diagnostics.json").write_text(json.dumps({
        "config": asdict(cfg), "lengths": lengths, "restarts": diagnostics,
        "transition_matrix": model.transmat_.tolist(),
        "semantic_order_old_indices": semantic_order.tolist(),
    }, indent=2), encoding="utf-8")
    return aligned


def soft_state_kl_loss(
    state_logits: torch.Tensor,
    teacher_posterior: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    teacher = teacher_posterior.clamp_min(1e-6)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    return F.kl_div(
        F.log_softmax(state_logits.float() / temperature, dim=-1),
        teacher.float(), reduction="batchmean",
    )
