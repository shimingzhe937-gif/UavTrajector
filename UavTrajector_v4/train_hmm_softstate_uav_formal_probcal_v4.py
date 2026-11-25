#!/usr/bin/env python3
"""Formal launcher for the probability-calibrated soft-state trainer.

One SSL/HMM teacher is fitted on training histories and frozen for all seeds.
The base module is imported normally so multiprocessing workers share class
identities.  No model layers are added here.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

import train_hmm_softstate_uav_probcal_v4 as core


def _pop_option(name, default):
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a path")
    value = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    return value


def causal_window_indices(raw_sequences, max_s, horizon, stride):
    windows = []
    required = int(max_s) + int(horizon)
    for sequence_index, sequence in enumerate(raw_sequences):
        if len(sequence) < required:
            continue
        for start in range(0, len(sequence) - required + 1, int(stride)):
            windows.append({
                "seq_idx": int(sequence_index),
                "start": int(start),
                "intensity": 0.0,
                "label": 0,
            })
    if not windows:
        raise RuntimeError("No valid trajectory windows were constructed.")
    return windows


def safe_protocol_cache_dir(config):
    cache_dir = config.get("protocol_cache_dir")
    if not cache_dir:
        cache_dir = os.path.join(
            config.get("maneuver_save_dir", "HMM_SoftState_5Hz"),
            "shared_protocol_cache",
        )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def no_heuristic_label_assignment(indices, q25, q75):
    del q25, q75
    for item in indices:
        item["label"] = 0
    return indices


def normal_import_base(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = path.stem
    if not module_name.isidentifier():
        raise ValueError(
            f"Base filename must be importable, got {path.name}. "
            "Rename it to uav_formal_base_v4.py."
        )
    directory = str(path.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != path:
        raise ImportError(
            f"Imported {module_name} from {module.__file__}, expected {path}."
        )
    module._protocol_cache_dir = safe_protocol_cache_dir
    module.build_window_indices = causal_window_indices
    module.assign_intent_labels = no_heuristic_label_assignment
    backing = getattr(module, "_v2", None)
    if backing is not None:
        backing._protocol_cache_dir = safe_protocol_cache_dir
        backing.build_window_indices = causal_window_indices
        backing.assign_intent_labels = no_heuristic_label_assignment
    return module


core.load_base = normal_import_base

teacher_root = Path(
    _pop_option("--shared-teacher-dir", "HMM_ThreeState_FrozenTeacher_ProbCal")
).expanduser().resolve()
ssl_checkpoint = teacher_root / "emamse_ssl.pt"
hmm_dir = teacher_root / "hmm_teacher"
hmm_file = hmm_dir / "hmm_teacher.joblib"
train_posterior_file = hmm_dir / "train_causal_state_posterior.npy"
core.skip_train_history_extraction = (
    hmm_file.exists() and train_posterior_file.exists()
)

_original_pretrain = core.pretrain_emamse
_original_extract = core.extract_window_latents
_original_build = core.build_offline_teacher
_original_infer = core.infer_split
_extract_call_index = 0


def shared_pretrain(predictor, train_loader, feature_dim, device, config,
                    checkpoint):
    del checkpoint
    action = "Reusing" if ssl_checkpoint.exists() else "Fitting"
    print(f"[formal-probcal] {action} SSL encoder: {ssl_checkpoint}", flush=True)
    return _original_pretrain(
        predictor, train_loader, feature_dim, device, config, ssl_checkpoint
    )


def shared_extract(encoder, loader, device):
    """Cache deterministic SSL window embeddings for reuse across seeds."""
    global _extract_call_index
    split = ("train", "val", "test")[_extract_call_index % 3]
    _extract_call_index += 1
    cache_file = teacher_root / f"{split}_window_latents.npz"
    expected_count = len(loader.dataset)

    if cache_file.exists():
        with np.load(cache_file) as cached:
            latent = cached["latent"]
            window_ids = cached["window_ids"]
        if len(latent) != expected_count or len(window_ids) != expected_count:
            raise ValueError(
                f"Cached {split} latent count does not match the current "
                f"protocol: {len(latent)} vs {expected_count}. Use a new "
                "--shared-teacher-dir."
            )
        print(
            f"[formal-probcal] Reusing {split} SSL latents: {cache_file}",
            flush=True,
        )
        return latent, window_ids

    latent, window_ids = _original_extract(encoder, loader, device)
    np.savez(
        cache_file,
        latent=np.asarray(latent),
        window_ids=np.asarray(window_ids),
    )
    print(
        f"[formal-probcal] Cached {split} SSL latents: {cache_file}",
        flush=True,
    )
    return latent, window_ids


def shared_build(latent, window_indices, train_indices,
                 ordered_history_sequences, config, output_dir):
    del output_dir
    if hmm_file.exists() and train_posterior_file.exists():
        posterior = np.load(train_posterior_file).astype(np.float32)
        expected_shape = (len(train_indices), int(config.n_states))
        if posterior.shape != expected_shape:
            raise ValueError(
                f"Frozen posterior shape {posterior.shape}, expected "
                f"{expected_shape}. Use a new --shared-teacher-dir."
            )
        print(f"[formal-probcal] Reusing frozen K=3 HMM: {hmm_file}", flush=True)
        core.skip_train_history_extraction = True
        return posterior
    print(f"[formal-probcal] Fitting K=3 HMM: {hmm_dir}", flush=True)
    posterior = _original_build(
        latent, window_indices, train_indices,
        ordered_history_sequences, config, hmm_dir,
    )
    core.skip_train_history_extraction = True
    return posterior


def shared_infer(latent, window_ids, indices, teacher_path):
    del teacher_path
    if not hmm_file.exists():
        raise FileNotFoundError(f"Frozen teacher was not created: {hmm_file}")
    return _original_infer(latent, window_ids, indices, hmm_file)


core.pretrain_emamse = shared_pretrain
core.extract_window_latents = shared_extract
core.build_offline_teacher = shared_build
core.infer_split = shared_infer


if __name__ == "__main__":
    print(f"[formal-probcal] Shared teacher root: {teacher_root}", flush=True)
    core.main()

