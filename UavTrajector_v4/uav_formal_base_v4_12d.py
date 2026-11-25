#!/usr/bin/env python3
"""Twelve-dimensional, position-only data front end for the formal V4 model.

The EMam-SE + MSA-DTP + H-PGD network and the V4 HMM soft-state training
objective are unchanged.  This module only replaces the legacy 31-D data
front end with causal features that can be constructed from 3-D positions:

    [delta_xyz, acceleration_xyz, speed, horizontal_speed,
     sin(heading), cos(heading), sin(flight_path), cos(flight_path)]

Existing 31-D checkpoints are intentionally incompatible and must not be
loaded into this model.  Train new checkpoints with this file as --base-script.
"""
from __future__ import annotations

import concurrent.futures
import gc
import glob
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import uav_formal_base_v2 as _v2


FEATURE_DIM = 12
FEATURE_SCHEMA = (
    "delta_x", "delta_y", "delta_z",
    "accel_x", "accel_y", "accel_z",
    "speed", "horizontal_speed",
    "sin_heading", "cos_heading",
    "sin_flight_path", "cos_flight_path",
)
FEATURE_CACHE_NAME = "raw_flights_position12_causal_sg7_p2_v1.npz"


for _name in dir(_v2):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_v2, _name)


def _causal_savgol(values: np.ndarray, window: int = 7, degree: int = 2) -> np.ndarray:
    """Right-aligned Savitzky--Golay filter using current/past samples only."""
    values = np.asarray(values, dtype=np.float64)
    output = values.copy()
    if len(values) < window:
        return output
    offsets = np.arange(-(window - 1), 1, dtype=np.float64)
    design = np.vander(offsets, N=degree + 1, increasing=True)
    weights = np.linalg.pinv(design)[0]
    for index in range(window - 1, len(values)):
        output[index] = weights @ values[index - window + 1:index + 1]
    return output


def position_features_12(position: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Build the manuscript's 12 causal motion features from positions only."""
    position = np.asarray(position, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"position must be [T,3], got {position.shape}")
    if len(position) != len(timestamps):
        raise ValueError("position/timestamp length mismatch")
    if len(position) < 2:
        raise ValueError("at least two position samples are required")

    filtered = _causal_savgol(position, window=7, degree=2)
    dt = np.diff(timestamps, prepend=timestamps[0])
    fallback = float(np.median(dt[1:][dt[1:] > 1e-6])) if np.any(dt[1:] > 1e-6) else 0.2
    dt[0] = fallback
    dt = np.clip(dt, 1e-3, 10.0)[:, None]

    delta = np.diff(filtered, axis=0, prepend=filtered[[0]])
    velocity = delta / dt
    acceleration = np.diff(velocity, axis=0, prepend=velocity[[0]]) / dt
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1, keepdims=True)
    heading = np.arctan2(velocity[:, 1:2], velocity[:, 0:1])
    flight_path = np.arctan2(velocity[:, 2:3], horizontal_speed + 1e-12)

    features = np.concatenate(
        [
            delta,
            acceleration,
            speed,
            horizontal_speed,
            np.sin(heading),
            np.cos(heading),
            np.sin(flight_path),
            np.cos(flight_path),
        ],
        axis=1,
    ).astype(np.float32)
    features = np.clip(features, -1e5, 1e5)
    features[~np.isfinite(features)] = 0.0
    if features.shape[1] != FEATURE_DIM:
        raise AssertionError(f"expected 12 features, got {features.shape}")
    return features


def process_single_flight(args):
    """Multiprocessing-safe UAV-Flow record converter."""
    flight_id, log_str, min_length = args
    try:
        import orjson

        record = orjson.loads(log_str)
        raw = np.asarray(record["raw_logs"], dtype=np.float64)
        preprocessed = np.asarray(record["preprocessed_logs"], dtype=np.float64)
        length = min(len(raw), len(preprocessed))
        if length < min_length or raw.shape[1] <= 6 or preprocessed.shape[1] < 3:
            return None
        features = position_features_12(preprocessed[:length, :3], raw[:length, 6])
        if not np.isfinite(features).all():
            return None
        return str(flight_id), features
    except Exception:
        return None


def _load_or_extract_all_flights_12d(config):
    cache_dir = Path(_v2._protocol_cache_dir(config))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / FEATURE_CACHE_NAME
    metadata_file = cache_dir / "position12_feature_schema.json"

    if cache_file.exists() and not config.get("force_rebuild_features", False):
        cache = np.load(cache_file, allow_pickle=False)
        flights = {key: cache[key] for key in cache.files}
        if flights and all(value.ndim == 2 and value.shape[1] == FEATURE_DIM for value in flights.values()):
            print(f"[12D] Loaded {len(flights)} cached trajectories: {cache_file}")
            return flights
        raise RuntimeError(f"Invalid 12-D feature cache: {cache_file}")

    minimum_length = max(config["scales"]) + int(config["horizon"])
    tasks, seen = [], set()
    for parquet_file in sorted(glob.glob(os.path.join(config["data_dir"], "*.parquet"))):
        frame = pd.read_parquet(parquet_file)
        log_column = "logs" if "logs" in frame.columns else "log"
        for flight_id, log_string in zip(
            frame.drop_duplicates("id")["id"],
            frame.drop_duplicates("id")[log_column],
        ):
            flight_id = str(flight_id)
            if flight_id not in seen:
                tasks.append((flight_id, log_string, minimum_length))
                seen.add(flight_id)
        del frame
    if not tasks:
        raise RuntimeError(f"No UAV-Flow parquet records found in {config['data_dir']}")

    cpu_total = os.cpu_count() or 4
    workers = min(max(cpu_total - 2, 2), int(config.get("feature_workers", 32)))
    chunk_size = max(1, len(tasks) // max(workers * 8, 1))
    flights = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, max_tasks_per_child=100
    ) as executor:
        results = executor.map(process_single_flight, tasks, chunksize=chunk_size)
        for result in tqdm(results, total=len(tasks), desc="extracting 12-D causal features", ncols=90):
            if result is not None:
                flight_id, features = result
                flights[flight_id] = features
    if not flights:
        raise RuntimeError("No valid 12-D trajectories were extracted")

    np.savez_compressed(cache_file, **flights)
    metadata_file.write_text(
        json.dumps(
            {
                "feature_dim": FEATURE_DIM,
                "feature_schema": FEATURE_SCHEMA,
                "position_filter": "right-aligned causal Savitzky-Golay, window=7, degree=2",
                "requires_attitude": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[12D] Saved {len(flights)} trajectories: {cache_file}")
    return flights


def _feature_scale_mask_12d(feature_dim):
    if int(feature_dim) != FEATURE_DIM:
        raise ValueError(f"The position-only V4 expects 12 features, got {feature_dim}")
    # Deltas use d_scaler.  Heading/flight-path sin-cos terms are bounded and
    # remain unscaled so their geometry is preserved by heading alignment.
    mask = np.zeros(FEATURE_DIM, dtype=bool)
    mask[3:8] = True
    return mask


def _fit_or_load_scalers_12d(train_raw, feature_dim, train_ids, config):
    if int(feature_dim) != FEATURE_DIM:
        raise ValueError(f"Expected feature_dim=12, got {feature_dim}")
    cache_dir = Path(_v2._protocol_cache_dir(config))
    digest = hashlib.sha256("\n".join(sorted(map(str, train_ids))).encode()).hexdigest()[:16]
    scaler_path = cache_dir / f"scalers_position12_train_{digest}.pkl"
    mask = _feature_scale_mask_12d(feature_dim)
    if scaler_path.exists() and not config.get("force_rebuild_scalers", False):
        d_scaler, f_scaler = joblib.load(scaler_path)
        if len(d_scaler.scale_) != 3 or len(f_scaler.scale_) != int(mask.sum()):
            raise RuntimeError(f"Invalid 12-D scaler cache: {scaler_path}")
        return d_scaler, f_scaler, mask

    # Fit the training-only scalers from a bounded, deterministic frame sample.
    # Stacking every frame from 30k+ flights can require several GB of RAM and
    # is unnecessary for stable first/second-moment estimates.
    sample_limit = int(config.get("scaler_frame_samples", 50000))
    rng = np.random.default_rng(int(config.get("split_seed", 42)))
    per_flight = max(1, sample_limit // max(len(train_raw), 1))
    sampled_rows = []
    for flight in train_raw:
        flight = np.asarray(flight)
        take = min(len(flight), per_flight)
        if take == len(flight):
            sampled_rows.append(flight)
        elif take > 0:
            sampled_rows.append(
                flight[rng.choice(len(flight), size=take, replace=False)]
            )
    rows = np.vstack(sampled_rows)
    if len(rows) > sample_limit:
        rows = rows[
            rng.choice(len(rows), size=sample_limit, replace=False)
        ]
    deltas = rows[:, :3]
    nonzero = np.linalg.norm(deltas, axis=1) > 1e-3
    valid_delta = deltas[nonzero] if np.any(nonzero) else deltas
    xy_scale = max(float(np.std(valid_delta[:, :2])), 0.05)
    z_scale = max(float(np.std(valid_delta[:, 2])), 0.01)
    d_scaler = StandardScaler(with_mean=False).fit(valid_delta)
    d_scaler.scale_ = np.asarray([xy_scale, xy_scale, z_scale], dtype=np.float64)

    selected = np.clip(rows[:, mask], -500.0, 500.0)
    f_scaler = StandardScaler().fit(selected)
    f_scaler.scale_ = np.maximum(f_scaler.scale_, 1e-3)
    # Acceleration x/y must share a zero-centred scale so rotating a normalised
    # window is equivalent to rotating it in physical coordinates.
    accel_xy_scale = max(float(np.std(selected[:, :2])), 1e-3)
    accel_z_scale = max(float(np.std(selected[:, 2])), 1e-3)
    f_scaler.mean_[:3] = 0.0
    f_scaler.scale_[0:2] = accel_xy_scale
    f_scaler.scale_[2] = accel_z_scale
    joblib.dump((d_scaler, f_scaler), scaler_path)
    print(f"[12D] Saved training-only scalers: {scaler_path}")
    return d_scaler, f_scaler, mask


class OnTheFlyUAVDataset(_v2.Dataset):
    """Window dataset with correct rotation semantics for the 12-D schema."""
    def __init__(self, seq_list, indices, max_s, horizon, is_training=False):
        self.seq_list = seq_list
        self.indices = indices
        self.max_s = int(max_s)
        self.horizon = int(horizon)
        self.is_training = bool(is_training)

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _rotate_features(features, rotation):
        output = features.copy()
        output[..., 0:3] = features[..., 0:3] @ rotation.T
        output[..., 3:6] = features[..., 3:6] @ rotation.T
        heading_vector = np.concatenate(
            [features[..., 9:10], features[..., 8:9]], axis=-1
        )
        rotated_heading = heading_vector @ rotation[:2, :2].T
        output[..., 9:10] = rotated_heading[..., 0:1]
        output[..., 8:9] = rotated_heading[..., 1:2]
        return output

    def __getitem__(self, index):
        item = self.indices[index]
        sequence = self.seq_list[int(item["seq_idx"])]
        start = int(item["start"])
        window = sequence[start:start + self.max_s + self.horizon]
        history = window[:self.max_s]
        target = window[self.max_s:, :3]
        last_delta = history[-1, :3]
        angle = -np.arctan2(last_delta[1], last_delta[0] + 1e-8)
        if self.is_training:
            angle += np.random.uniform(0.0, 2.0 * np.pi)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return {
            "seq": _v2.torch.from_numpy(self._rotate_features(history, rotation)).float(),
            "last_delta": _v2.torch.from_numpy(last_delta @ rotation.T).float(),
            "target": _v2.torch.from_numpy(target @ rotation.T).float(),
            "intent": _v2.torch.tensor(int(item.get("label", 0)), dtype=_v2.torch.long),
        }


V4_12D_BASE_DESCRIPTION = (
    "V4 EMam-SE + MSA-DTP + H-PGD with 12 causal position-only features"
)


def _sync_to_backing_module():
    # Protocol launchers may patch these hooks on this module after import.
    for hook in ("_protocol_cache_dir", "build_window_indices", "assign_intent_labels"):
        value = globals().get(hook)
        if value is not None:
            setattr(_v2, hook, value)
    _v2.process_single_flight = process_single_flight
    _v2._load_or_extract_all_flights = _load_or_extract_all_flights_12d
    _v2._feature_scale_mask = _feature_scale_mask_12d
    _v2._fit_or_load_scalers = _fit_or_load_scalers_12d
    _v2.OnTheFlyUAVDataset = OnTheFlyUAVDataset


def prepare_shared_dataset(config):
    _sync_to_backing_module()
    shared = _v2.prepare_shared_dataset(config)
    if int(shared["feat_dim"]) != FEATURE_DIM:
        raise RuntimeError(f"12-D preflight failed: feat_dim={shared['feat_dim']}")
    shared["feature_schema"] = FEATURE_SCHEMA
    return shared
