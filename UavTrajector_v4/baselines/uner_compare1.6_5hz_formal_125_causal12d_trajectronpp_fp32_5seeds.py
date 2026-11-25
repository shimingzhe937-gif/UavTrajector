#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一、公平、可复现的低空航空器轨迹预测对比实验脚本
=========================================================

本脚本用于替换原始 uner_compare1_4.py，并实现：
1. 按轨迹 ID 固定划分 Train / Validation / Test = 70% / 10% / 20%；
2. 仅使用训练集拟合归一化器与意图阈值；验证集、测试集复用训练阈值；
3. 所有模型使用一致的历史输入长度与预测长度；
4. Ours 训练和推理均仅使用模型预测的意图概率，禁止未来真实意图标签进入解码器；
5. 统一采用验证集 ADE + 0.5*FDE 选择最佳 epoch；
6. 确定性、概率性和动力学指标分别统计；
7. 概率指标仅对当前实验中已启用且可采样的概率模型报告；
8. 初步筛选固定使用随机种子 2026，并保存逐模型结果；
9. 输入采用仅由历史三维位置构造的12维因果运动特征；
10. 每个模型×种子运行仅创建一次Train/Val/Test DataLoader，并通过persistent workers复用worker；
    每个epoch只更新Sampler状态，运行结束显式回收worker和文件描述符。

依赖：numpy, pandas, torch, scikit-learn, scipy, tqdm, joblib, pyarrow,
      可选：orjson、thop、mamba_ssm。

注意：UAV-Flow轨迹按5 Hz对齐，本实验使用 CONFIG["sample_interval_s"]=0.2 s。
"""

# 必须在 import torch 之前设置
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gc
import re
import json
import time
import math
import random
import glob
import sys
import importlib
import warnings
import hashlib
import concurrent.futures
import multiprocessing as mp

# 尽量提高本进程软文件描述符上限；无权限时静默回退，仍可通过num_workers=0运行。
try:
    import resource
    _soft_nofile, _hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    _target_nofile = min(_hard_nofile, max(_soft_nofile, 8192))
    if _soft_nofile < _target_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_target_nofile, _hard_nofile))
except Exception:
    pass
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterator

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.signal import savgol_coeffs, savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warnings.filterwarnings("ignore")

try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

try:
    import orjson
except Exception:
    orjson = None

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

# ============================================================================
# 全局超参数
# ============================================================================
DROPOUT_BASE = 0.15
DROPOUT_HEAVY = 0.30
AMP_DTYPE = torch.float16
USE_AMP = torch.cuda.is_available()
EPS = 1e-8
FEATURE_DIM = 12
FEATURE_LAYOUT_VERSION = "causal_position_motion_12d_v1"


def amp_enabled_for_model(model_type: str) -> bool:
    """Keep the recurrent CVAE baseline in FP32 to avoid latent-prior overflow."""
    return bool(USE_AMP and str(model_type).lower() != "trajectronpp")

# ============================================================================
# 随机性与通用工具
# ============================================================================
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """固定 Python、NumPy、PyTorch 与 DataLoader 随机性。"""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True


def seed_worker(worker_id: int) -> None:
    """为 DataLoader worker 设置可复现随机状态。"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", str(name)).strip("_")


def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")

# ============================================================================
# Mamba2 动态加载：未安装时自动使用 GRU 回退，保证其余实验仍可运行。
# ============================================================================
_MAMBA2 = None
_MAMBA2_TRIED = False


def get_mamba2(mamba_main_dir: Optional[str] = None):
    global _MAMBA2, _MAMBA2_TRIED
    if _MAMBA2_TRIED:
        return _MAMBA2

    _MAMBA2_TRIED = True
    candidates = []
    if mamba_main_dir:
        candidates.append(mamba_main_dir)
    candidates.append("./mamba-main")

    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)

    try:
        module = importlib.import_module("mamba_ssm.modules.mamba2")
        _MAMBA2 = getattr(module, "Mamba2")
        print("✅ Mamba2 已加载。")
    except Exception as exc:
        _MAMBA2 = None
        print(f"⚠️ Mamba2 不可用，将使用 GRU 回退编码器：{exc}")

    return _MAMBA2


def check_mamba_cuda() -> None:
    print("\n" + "=" * 64)
    print("Mamba2 / CUDA 加速状态检查")
    print("=" * 64)
    try:
        import causal_conv1d_cuda  # noqa: F401
        print("[✅] causal_conv1d_cuda")
    except Exception:
        print("[⚠️] causal_conv1d_cuda 未连接")
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
        print("[✅] mamba2 Triton kernel")
    except Exception:
        print("[⚠️] mamba2 Triton kernel 未连接")
    print("=" * 64 + "\n")

# ============================================================================
# 数据处理：12维因果位置运动特征（保留旧提取器仅供审计，不参与运行）
# ============================================================================
def _loads_json_payload(log_str: Any) -> Dict[str, Any]:
    if isinstance(log_str, dict):
        return log_str
    if isinstance(log_str, bytes):
        if orjson is not None:
            return orjson.loads(log_str)
        return json.loads(log_str.decode("utf-8"))
    if isinstance(log_str, str):
        if orjson is not None:
            return orjson.loads(log_str)
        return json.loads(log_str)
    raise TypeError(f"Unsupported log payload type: {type(log_str)}")


def _legacy_noncausal_31d_extractor(args: Tuple[str, Any, int]) -> Optional[Tuple[str, np.ndarray]]:
    """CPU worker：提取单条飞行轨迹的 31 维特征。"""
    flight_id, log_str, min_length = args

    def wrap_angle(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        out = np.zeros_like(num, dtype=np.float64)
        np.divide(num, den, out=out, where=np.abs(den) > eps)
        return out

    try:
        payload = _loads_json_payload(log_str)
        raw_logs = np.asarray(payload["raw_logs"], dtype=np.float64)
        prep_logs = np.asarray(payload["preprocessed_logs"], dtype=np.float32)
    except Exception:
        return None

    if len(prep_logs) < min_length or raw_logs.ndim != 2 or prep_logs.ndim != 2:
        return None
    if raw_logs.shape[0] != prep_logs.shape[0] or prep_logs.shape[1] < 6:
        return None

    try:
        if raw_logs.shape[1] > 6:
            dt = np.diff(raw_logs[:, 6])
        else:
            dt = np.ones(len(raw_logs) - 1, dtype=np.float64)
        dt = np.clip(dt, 0.01, 10.0)
        dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 1.0).reshape(-1, 1)

        pos = prep_logs[:, :3].astype(np.float64)
        angs = prep_logs[:, 3:6].astype(np.float64)
        if np.nanmax(np.abs(angs)) > 10.0:
            angs = np.deg2rad(angs)
        angs[:, 2] = np.unwrap(angs[:, 2])

        n = len(pos)
        if n >= 7:
            pos_s = savgol_filter(pos, 7, 3, axis=0)
        elif n >= 5:
            pos_s = savgol_filter(pos, 5, 3, axis=0)
        elif n >= 3:
            pos_s = savgol_filter(pos, 3, 2, axis=0)
        else:
            pos_s = pos

        deltas = np.vstack([np.zeros((1, 3)), np.diff(pos_s, axis=0)])
        velocity = deltas / dt
        acceleration = np.vstack([np.zeros((1, 3)), np.diff(velocity, axis=0)]) / dt
        jerk = np.vstack([np.zeros((1, 3)), np.diff(acceleration, axis=0)]) / dt

        d_angs = np.vstack([np.zeros((1, 3)), np.diff(angs, axis=0)]) / dt
        d_angs = wrap_angle(d_angs)
        phi, theta = angs[:, 0:1], angs[:, 1:2]
        p = d_angs[:, 0:1] - d_angs[:, 2:3] * np.sin(theta)
        q = d_angs[:, 1:2] * np.cos(phi) + d_angs[:, 2:3] * np.sin(phi) * np.cos(theta)
        r = -d_angs[:, 1:2] * np.sin(phi) + d_angs[:, 2:3] * np.cos(phi) * np.cos(theta)

        speed = np.linalg.norm(velocity, axis=1, keepdims=True) + EPS
        xy_speed = np.linalg.norm(velocity[:, :2], axis=1, keepdims=True) + EPS
        track_angle = np.arctan2(velocity[:, 1:2], velocity[:, 0:1] + EPS)
        flight_path_angle = np.arctan2(velocity[:, 2:3], xy_speed)

        v_unit = velocity / speed
        normal_acc = np.linalg.norm(
            acceleration - np.sum(acceleration * v_unit, axis=1, keepdims=True) * v_unit,
            axis=1,
            keepdims=True,
        )
        v_cross_a = np.cross(velocity, acceleration)
        curvature = safe_div(np.linalg.norm(v_cross_a, axis=1, keepdims=True), speed ** 3)

        slip_angle = wrap_angle(angs[:, 2:3] - track_angle)
        angle_of_attack = wrap_angle(angs[:, 1:2] - flight_path_angle)
        kinetic_energy = 0.5 * speed ** 2
        potential_energy = 9.81 * (pos[:, 2:3] - pos[0, 2])
        power_signal = np.concatenate([np.zeros((1, 1)), np.diff(kinetic_energy + potential_energy, axis=0)], axis=0) / dt
        torsion = safe_div(
            np.sum(v_cross_a * jerk, axis=1, keepdims=True),
            np.linalg.norm(v_cross_a, axis=1, keepdims=True) ** 2,
        )

        feature = np.concatenate(
            [
                deltas, velocity, acceleration, jerk,
                np.sin(angs), np.cos(angs),
                p, q, r,
                track_angle, flight_path_angle, curvature, normal_acc,
                slip_angle, angle_of_attack,
                kinetic_energy, potential_energy, power_signal, torsion,
            ],
            axis=-1,
        ).astype(np.float32)

        if feature.shape[1] != 31:
            return None

        feature = np.clip(feature, -1e5, 1e5)
        feature[~np.isfinite(feature)] = 0.0

        col_max = np.abs(feature).max(axis=0)
        if int((col_max > 1e4).sum()) > feature.shape[1] // 3:
            return None

        return str(flight_id), feature
    except Exception:
        return None


def causal_savgol_positions(
    positions: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Right-aligned Savitzky--Golay filter using current/past samples only."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [T, 3]")
    window_length = int(window_length)
    polyorder = int(polyorder)
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("causal Savitzky-Golay window must be an odd integer >= 3")
    if polyorder < 1 or polyorder >= window_length:
        raise ValueError("polyorder must satisfy 1 <= polyorder < window_length")

    output = positions.copy()
    if len(positions) < window_length:
        return output
    coefficients = savgol_coeffs(
        window_length=window_length,
        polyorder=polyorder,
        pos=window_length - 1,
        use="dot",
    )
    for index in range(window_length - 1, len(positions)):
        history = positions[index - window_length + 1 : index + 1]
        output[index] = coefficients @ history
    return output


# Override the legacy 31-D/non-causal extractor above.  Keeping the original
# definition makes the revision easy to audit, while all worker calls below
# resolve to this causal 12-D implementation.
def process_single_flight(
    args: Tuple[str, Any, int, str, int, int, float]
) -> Optional[Tuple[str, np.ndarray]]:
    """Extract 12 motion features derived exclusively from observed 3-D positions."""
    (
        flight_id,
        log_str,
        min_length,
        position_filter,
        filter_window,
        filter_polyorder,
        sample_interval_s,
    ) = args

    try:
        payload = _loads_json_payload(log_str)
        raw_logs = np.asarray(payload["raw_logs"], dtype=np.float64)
        prep_logs = np.asarray(payload["preprocessed_logs"], dtype=np.float32)
    except Exception:
        return None

    if len(prep_logs) < min_length or raw_logs.ndim != 2 or prep_logs.ndim != 2:
        return None
    if raw_logs.shape[0] != prep_logs.shape[0] or prep_logs.shape[1] < 3:
        return None

    try:
        positions = prep_logs[:, :3].astype(np.float64)
        if not np.all(np.isfinite(positions)):
            return None

        mode = str(position_filter).strip().lower()
        if mode == "none":
            positions_used = positions
        elif mode == "causal_savgol":
            positions_used = causal_savgol_positions(
                positions,
                window_length=filter_window,
                polyorder=filter_polyorder,
            )
        else:
            raise ValueError(
                f"Unsupported position_filter={mode!r}; use 'none' or 'causal_savgol'."
            )

        dt = max(float(sample_interval_s), EPS)
        deltas = np.vstack([np.zeros((1, 3)), np.diff(positions_used, axis=0)])
        velocity = deltas / dt
        acceleration = np.vstack(
            [np.zeros((1, 3)), np.diff(velocity, axis=0)]
        ) / dt

        speed = np.linalg.norm(velocity, axis=1, keepdims=True) + EPS
        horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1, keepdims=True) + EPS
        heading = np.arctan2(velocity[:, 1:2], velocity[:, 0:1] + EPS)
        flight_path_angle = np.arctan2(velocity[:, 2:3], horizontal_speed)

        feature = np.concatenate(
            [
                deltas,
                acceleration,
                speed,
                horizontal_speed,
                np.sin(heading),
                np.cos(heading),
                np.sin(flight_path_angle),
                np.cos(flight_path_angle),
            ],
            axis=-1,
        ).astype(np.float32)

        if feature.shape[1] != FEATURE_DIM:
            return None
        feature = np.clip(feature, -1e5, 1e5)
        feature[~np.isfinite(feature)] = 0.0
        return str(flight_id), feature
    except Exception:
        return None


def compute_future_maneuver_intensity(future_deltas: np.ndarray) -> float:
    """与训练、验证、测试完全一致的未来机动强度定义。"""
    if future_deltas is None or len(future_deltas) == 0:
        return 0.0
    move_norm = np.linalg.norm(future_deltas, axis=-1)
    if float(np.mean(move_norm)) <= 1e-8:
        return 0.0

    if len(future_deltas) >= 2:
        linear_change = float(np.mean(np.linalg.norm(np.diff(future_deltas, axis=0), axis=-1)))
    else:
        linear_change = 0.0
    speed_std = float(np.std(move_norm))
    return float(0.7 * linear_change + 0.3 * speed_std)


def build_window_records(seq_list: List[np.ndarray], history_len: int, horizon: int, stride: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    min_len = history_len + horizon
    for seq_idx, seq in enumerate(seq_list):
        if len(seq) < min_len:
            continue
        for start in range(0, len(seq) - min_len + 1, stride):
            future_deltas = seq[start + history_len : start + history_len + horizon, :3]
            intensity = compute_future_maneuver_intensity(future_deltas)
            if intensity <= 0.0:
                continue
            records.append({
                "seq_idx": int(seq_idx),
                "start": int(start),
                "intensity": float(intensity),
                "label": -1,
            })
    if len(records) == 0:
        raise RuntimeError("未构建到有效样本窗口，请检查 history_len、horizon、stride 与原始数据。")
    return records


def assign_intent_labels(records: List[Dict[str, Any]], q25: float, q75: float) -> List[Dict[str, Any]]:
    for item in records:
        value = float(item["intensity"])
        item["label"] = 0 if value < q25 else (1 if value < q75 else 2)
    return records


def build_fixed_groups_from_labels(records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    labels = np.asarray([int(r["label"]) for r in records], dtype=np.int64)
    return {
        "low": np.where(labels == 0)[0],
        "mid": np.where(labels == 1)[0],
        "high": np.where(labels == 2)[0],
    }


def _load_or_build_flight_cache(config: Dict[str, Any], min_length: int) -> Dict[str, np.ndarray]:
    cache_dir = ensure_dir(config["cache_dir"])
    cache_path = os.path.join(cache_dir, "processed_flights.joblib")
    meta_path = os.path.join(cache_dir, "processed_flights_meta.json")

    expected_meta = {
        "feature_dim": FEATURE_DIM,
        "feature_layout": FEATURE_LAYOUT_VERSION,
        "position_filter": str(config.get("position_filter", "none")),
        "filter_window": int(config.get("causal_savgol_window", 5)),
        "filter_polyorder": int(config.get("causal_savgol_polyorder", 2)),
        "sample_interval_s": float(config["sample_interval_s"]),
        "min_length": int(min_length),
        "data_dir": str(config["data_dir"]),
    }

    if os.path.exists(cache_path) and os.path.exists(meta_path) and not config.get("force_rebuild_cache", False):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
            if all(saved_meta.get(k) == v for k, v in expected_meta.items()):
                flights = joblib.load(cache_path)
                sample = next(iter(flights.values()))
                if sample.shape[1] == FEATURE_DIM:
                    print(
                        f"✅ 载入特征缓存：{len(flights)} 条轨迹，"
                        f"{FEATURE_DIM}维 {FEATURE_LAYOUT_VERSION} 特征。"
                    )
                    return flights
        except Exception as exc:
            print(f"⚠️ 特征缓存不可用，将重新构建：{exc}")

    parquet_files = sorted(glob.glob(os.path.join(config["data_dir"], "*.parquet")))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"未在 {config['data_dir']} 找到 parquet 文件。")

    tasks: List[Tuple[str, Any, int, str, int, int, float]] = []
    seen_ids = set()
    for file_path in parquet_files:
        print(f"读取：{os.path.basename(file_path)}")
        frame = pd.read_parquet(file_path)
        log_col = "logs" if "logs" in frame.columns else "log"
        if "id" not in frame.columns or log_col not in frame.columns:
            raise KeyError(f"{file_path} 不包含 id 与 {log_col} 列。")
        unique_frame = frame.drop_duplicates("id")[["id", log_col]]
        for fid, log_content in zip(unique_frame["id"].values, unique_frame[log_col].values):
            sid = str(fid)
            if sid not in seen_ids:
                seen_ids.add(sid)
                tasks.append(
                    (
                        sid,
                        log_content,
                        min_length,
                        str(config.get("position_filter", "none")),
                        int(config.get("causal_savgol_window", 5)),
                        int(config.get("causal_savgol_polyorder", 2)),
                        float(config["sample_interval_s"]),
                    )
                )
        del frame, unique_frame
        gc.collect()

    print(f"⏳ 开始 CPU 特征提取：{len(tasks)} 条轨迹。")
    cpu_count = os.cpu_count() or 4
    workers = int(config.get("feature_workers", min(max(cpu_count - 2, 2), 24)))
    chunk_size = max(1, len(tasks) // max(workers * 8, 1))

    flights: Dict[str, np.ndarray] = {}
    failed_tracks = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=100) as executor:
        iterator = executor.map(process_single_flight, tasks, chunksize=chunk_size)
        for output in tqdm(iterator, total=len(tasks), desc="Feature extraction", ncols=80):
            if output is None:
                failed_tracks += 1
            else:
                flights[output[0]] = output[1]

    if len(flights) == 0:
        raise RuntimeError("没有成功提取任何轨迹特征。")

    print(f"✅ 特征提取完成：成功 {len(flights)}，跳过/失败 {failed_tracks}。")
    joblib.dump(flights, cache_path, compress=3)
    save_json(meta_path, expected_meta)
    return flights


def _fit_scalers(train_raw: List[np.ndarray], feature_dim: int, cache_dir: str, config: Dict[str, Any]):
    scaler_path = os.path.join(cache_dir, "train_only_scalers.joblib")
    if os.path.exists(scaler_path) and not config.get("force_rebuild_cache", False):
        try:
            return joblib.load(scaler_path)
        except Exception:
            pass

    rng = np.random.default_rng(int(config["split_seed"]))
    target_samples = int(config.get("scaler_target_rows", 50000))
    rows = []
    collected = 0
    for seq in train_raw:
        if collected >= target_samples:
            break
        n_take = min(50, len(seq), target_samples - collected)
        if n_take <= 0:
            continue
        indices = rng.choice(len(seq), size=n_take, replace=False)
        rows.append(seq[indices])
        collected += n_take
    if len(rows) == 0:
        raise RuntimeError("训练集为空，无法拟合归一化器。")

    sample = np.vstack(rows).astype(np.float32)
    if feature_dim != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} features for {FEATURE_LAYOUT_VERSION}, "
            f"received {feature_dim}."
        )
    scale_mask = np.ones(feature_dim, dtype=bool)
    scale_mask[:3] = False
    scale_mask[8:12] = False

    delta_sample = sample[:, :3]
    valid_delta = np.linalg.norm(delta_sample, axis=1) > 1e-3
    valid_delta_sample = delta_sample[valid_delta]
    if len(valid_delta_sample) == 0:
        valid_delta_sample = delta_sample

    xy_std = max(float(np.std(valid_delta_sample[:, :2])), 0.05)
    z_std = max(float(np.std(valid_delta_sample[:, 2])), 0.01)

    delta_scaler = StandardScaler(with_mean=False)
    delta_scaler.fit(valid_delta_sample)
    delta_scaler.scale_ = np.asarray([xy_std, xy_std, z_std], dtype=np.float64)

    sample_clip = sample.copy()
    feature_scaler = StandardScaler().fit(
        np.clip(sample_clip[:, scale_mask], -500.0, 500.0)
    )
    feature_scaler.scale_ = np.maximum(feature_scaler.scale_, 1e-3)

    joblib.dump((delta_scaler, feature_scaler, scale_mask), scaler_path)
    return delta_scaler, feature_scaler, scale_mask


def _normalize_sequence(seq: np.ndarray, delta_scaler: StandardScaler, feature_scaler: StandardScaler, scale_mask: np.ndarray) -> np.ndarray:
    result = seq.astype(np.float32).copy()
    result[:, :3] = delta_scaler.transform(result[:, :3])
    result[:, scale_mask] = np.clip(
        feature_scaler.transform(np.clip(result[:, scale_mask], -500.0, 500.0)),
        -10.0,
        10.0,
    )
    return result


def _estimate_kinematic_reference_thresholds(
    train_raw: List[np.ndarray],
    dt: float,
    percentile: float = 10.0,
    max_points: int = 200000,
) -> Dict[str, float]:
    """
    Estimate robust physical reference floors from training trajectories only.

    These values are used only for metric normalization/thresholding. They are
    not exposed to the model and therefore do not leak validation/test data.
    """
    rng = np.random.default_rng(20260709)
    accel_values, jerk_values = [], []
    collected = 0
    for seq in train_raw:
        if collected >= max_points:
            break
        deltas = np.asarray(seq[:, :3], dtype=np.float64)
        if len(deltas) < 4:
            continue
        traj = np.cumsum(deltas, axis=0)[None, ...]
        origin = np.zeros((1, 1, 3), dtype=np.float64)
        v = np.diff(np.concatenate([origin, traj], axis=1), axis=1) / max(float(dt), 1e-6)
        a = np.diff(v, axis=1) / max(float(dt), 1e-6)
        j = np.diff(a, axis=1) / max(float(dt), 1e-6)
        a_norm = np.linalg.norm(a.reshape(-1, 3), axis=-1)
        j_norm = np.linalg.norm(j.reshape(-1, 3), axis=-1)
        values = np.concatenate([a_norm, j_norm], axis=0)
        if values.size == 0:
            continue
        # sample indices jointly for bounded memory, then split back by masks
        take = min(values.size, max_points - collected)
        if take <= 0:
            break
        if take < values.size:
            idx = rng.choice(values.size, size=take, replace=False)
            sampled = values[idx]
            # This joint sampling is only used to bound collection. Separate
            # robust floors are still estimated from positive portions below.
            a_sample = sampled[: max(1, min(len(sampled), len(sampled) // 2))]
            j_sample = sampled[max(1, min(len(sampled), len(sampled) // 2)) :]
        else:
            a_sample = a_norm
            j_sample = j_norm
        accel_values.append(a_sample[np.isfinite(a_sample)])
        jerk_values.append(j_sample[np.isfinite(j_sample)])
        collected += take

    def _robust_floor(chunks: List[np.ndarray], default: float) -> float:
        if not chunks:
            return float(default)
        values = np.concatenate(chunks)
        values = values[np.isfinite(values) & (values > 1e-8)]
        if values.size == 0:
            return float(default)
        return float(max(np.percentile(values, float(percentile)), default))

    return {
        "a_ref_floor_mps2": _robust_floor(accel_values, 1e-3),
        "j_ref_floor_mps3": _robust_floor(jerk_values, 1e-3),
        "reference_percentile": float(percentile),
        "source": "train trajectories only",
    }


def prepare_shared_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """建立固定 Train/Val/Test 数据包；所有模型、所有 seed 完全共享。

    重要修正：这里返回原始物理特征序列，而不是预先归一化后的序列。
    Dataset 在 __getitem__ 中先完成航向对齐/旋转增强，再使用训练集拟合的
    scaler 进行归一化，避免“归一化后再旋转”破坏三维向量的物理意义。
    """
    cache_dir = ensure_dir(config["cache_dir"])
    history_len = int(config["history_len"])
    horizon = int(config["horizon"])
    min_length = history_len + horizon

    flights = _load_or_build_flight_cache(config, min_length)
    ids = sorted(flights.keys())
    if len(ids) < 10:
        raise RuntimeError("有效轨迹数量过少，无法进行稳定的 Train/Val/Test 划分。")

    split_path = os.path.join(cache_dir, "trajectory_id_split_70_10_20.json")
    reference_split_path = config.get("reference_split_path")

    # Prefer the manifest used by the original formal comparison. This makes
    # the F12/F31 feature comparison trajectory-for-trajectory identical.
    if reference_split_path and os.path.exists(reference_split_path):
        with open(reference_split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_ids = list(map(str, split["train_ids"]))
        val_ids = list(map(str, split["val_ids"]))
        test_ids = list(map(str, split["test_ids"]))
        split = {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "source_manifest": str(reference_split_path),
        }
        save_json(split_path, split)
        print(f"✅ 复用正式对比实验轨迹划分：{reference_split_path}")
    elif os.path.exists(split_path) and not config.get("force_rebuild_split", False):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_ids = list(map(str, split["train_ids"]))
        val_ids = list(map(str, split["val_ids"]))
        test_ids = list(map(str, split["test_ids"]))
    else:
        train_ids, remain_ids = train_test_split(ids, test_size=0.30, random_state=int(config["split_seed"]))
        val_ids, test_ids = train_test_split(remain_ids, test_size=2.0 / 3.0, random_state=int(config["split_seed"]))
        split = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
        save_json(split_path, split)

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Train/Val/Test轨迹ID存在交集。")
    missing_ids = (train_set | val_set | test_set) - set(ids)
    if missing_ids:
        raise RuntimeError(
            f"固定划分中的{len(missing_ids)}条轨迹未能生成F12特征。"
        )

    train_raw = [flights[x] for x in train_ids]
    val_raw = [flights[x] for x in val_ids]
    test_raw = [flights[x] for x in test_ids]
    feature_dim = int(train_raw[0].shape[-1])

    # 仅训练集确定意图阈值。标签可以由未来窗口构造，但只作为训练监督，推理不输入真实标签。
    train_records = build_window_records(train_raw, history_len, horizon, int(config["train_stride"]))
    val_records = build_window_records(val_raw, history_len, horizon, int(config["eval_stride"]))
    test_records = build_window_records(test_raw, history_len, horizon, int(config["eval_stride"]))

    train_intensity = np.asarray([r["intensity"] for r in train_records], dtype=np.float64)
    q25, q75 = np.percentile(train_intensity, [25.0, 75.0])
    train_records = assign_intent_labels(train_records, float(q25), float(q75))
    val_records = assign_intent_labels(val_records, float(q25), float(q75))
    test_records = assign_intent_labels(test_records, float(q25), float(q75))

    save_json(os.path.join(cache_dir, "intent_thresholds_train_only.json"), {
        "q25": float(q25),
        "q75": float(q75),
        "definition": "0=steady, 1=ordinary maneuver, 2=sharp maneuver",
        "formula": "0.7*mean_norm(diff(future_delta))+0.3*std(norm(future_delta))",
    })

    delta_scaler, feature_scaler, scale_mask = _fit_scalers(train_raw, feature_dim, cache_dir, config)

    kin_ref = _estimate_kinematic_reference_thresholds(
        train_raw=train_raw,
        dt=float(config["sample_interval_s"]),
        percentile=float(config.get("a_keer_ref_floor_percentile", 10.0)),
    )
    config["a_keer_ref_floor_mps2"] = float(kin_ref["a_ref_floor_mps2"])
    config["j_ref_floor_mps3"] = float(kin_ref["j_ref_floor_mps3"])
    save_json(os.path.join(cache_dir, "train_only_kinematic_reference.json"), kin_ref)

    def _count(records):
        labels = np.asarray([r["label"] for r in records])
        return {"steady": int((labels == 0).sum()), "ordinary": int((labels == 1).sum()), "sharp": int((labels == 2).sum())}

    print("\n固定数据划分：")
    print(f"  Trajectories: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"  Windows:      train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    print(f"  Intent threshold from train only: q25={q25:.6f}, q75={q75:.6f}")
    print(f"  Train labels: {_count(train_records)}")
    print(f"  Val labels:   {_count(val_records)}")
    print(f"  Test labels:  {_count(test_records)}")
    print(f"  A-KEER train-only accel floor: {kin_ref['a_ref_floor_mps2']:.6f} m/s^2")

    return {
        # 保留键名 train_seqs/val_seqs/test_seqs，减少下游代码改动；其内容现在是原始物理特征。
        "train_seqs": train_raw,
        "val_seqs": val_raw,
        "test_seqs": test_raw,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "val_groups": build_fixed_groups_from_labels(val_records),
        "test_groups": build_fixed_groups_from_labels(test_records),
        "delta_scaler": delta_scaler,
        "feature_scaler": feature_scaler,
        "scale_mask": scale_mask,
        "feature_dim": feature_dim,
        "history_len": history_len,
        "horizon": horizon,
        "intent_thresholds": {"q25": float(q25), "q75": float(q75)},
        "kinematic_reference": kin_ref,
        "split": split,
    }

# ============================================================================
# Dataset 与 DataLoader
# ============================================================================
class OnTheFlyUAVDataset(Dataset):
    """
    Dataset with physically consistent canonical-frame augmentation.

    修正点：旋转/航向对齐在原始物理特征空间完成，然后再归一化。这样
    位移、速度、加速度、jerk 等三维向量不会因为通道独立标准化而失去
    旋转等价性。
    """

    def __init__(
        self,
        seq_list: List[np.ndarray],
        records: List[Dict[str, Any]],
        history_len: int,
        horizon: int,
        is_training: bool = False,
        augmentation_seed: int = 0,
        delta_scaler: Optional[StandardScaler] = None,
        feature_scaler: Optional[StandardScaler] = None,
        scale_mask: Optional[np.ndarray] = None,
        rotate_before_normalize: bool = True,
    ):
        self.seq_list = seq_list
        self.records = records
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.is_training = bool(is_training)
        self.augmentation_seed = int(augmentation_seed)
        self.delta_scaler = delta_scaler
        self.feature_scaler = feature_scaler
        self.scale_mask = scale_mask
        self.rotate_before_normalize = bool(rotate_before_normalize)
        if self.delta_scaler is None or self.feature_scaler is None or self.scale_mask is None:
            raise ValueError("OnTheFlyUAVDataset requires train-only scalers for post-rotation normalization.")
        # shared Value可被persistent worker读取；主进程每epoch更新一次。
        self._epoch_shared = mp.Value("i", 0, lock=False)

    def set_epoch(self, epoch: int) -> None:
        self._epoch_shared.value = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _normalize_feature_window(self, value: np.ndarray) -> np.ndarray:
        return _normalize_sequence(value, self.delta_scaler, self.feature_scaler, self.scale_mask)

    def _normalize_delta_array(self, value: np.ndarray) -> np.ndarray:
        shape = value.shape
        return self.delta_scaler.transform(value.reshape(-1, 3)).reshape(shape).astype(np.float32)

    def _rotate_feature(
        self,
        value: np.ndarray,
        rotation_matrix: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        out = value.copy()
        # 12-D layout: delta[0:3], acceleration[3:6], scalar speeds[6:8],
        # heading sin/cos[8:10], flight-path-angle sin/cos[10:12].
        if value.shape[-1] >= 3:
            out[..., :3] = np.matmul(value[..., :3], rotation_matrix.T)
        if value.shape[-1] >= 6:
            out[..., 3:6] = np.matmul(value[..., 3:6], rotation_matrix.T)

        # Heading rotates with the local xy frame. Scalar speeds and the
        # flight-path angle are invariant to this yaw rotation.
        if value.shape[-1] >= FEATURE_DIM:
            sin_yaw = value[..., 8:9]
            cos_yaw = value[..., 9:10]
            yaw_vec = np.concatenate([cos_yaw, sin_yaw], axis=-1)
            yaw_rot = np.matmul(yaw_vec, rotation_matrix[:2, :2].T)
            out[..., 9:10] = yaw_rot[..., 0:1]
            out[..., 8:9] = yaw_rot[..., 1:2]
        return out

    def _augmentation_angle(self, index: int) -> float:
        if not self.is_training:
            return 0.0
        epoch = int(self._epoch_shared.value)
        # 64位混合种子；避免依赖worker ID与预取顺序。
        local_seed = (
            self.augmentation_seed
            + 1_000_003 * epoch
            + 9_176 * int(index)
        ) % (2 ** 63 - 1)
        return float(np.random.default_rng(local_seed).uniform(0.0, 2.0 * np.pi))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        seq_idx = int(record["seq_idx"])
        start = int(record["start"])
        label = int(record["label"])

        raw_window = self.seq_list[seq_idx][
            start : start + self.history_len + self.horizon
        ].astype(np.float32, copy=False)
        x_seq_raw = raw_window[: self.history_len]
        last_delta_raw = raw_window[self.history_len - 1, :3]
        target_raw = raw_window[self.history_len :, :3]

        # Canonical heading alignment is computed in physical coordinates.
        heading = -np.arctan2(last_delta_raw[1], last_delta_raw[0] + EPS)
        angle = float(heading + self._augmentation_angle(index))
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrix = np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        x_seq_rot = self._rotate_feature(x_seq_raw, rotation_matrix, angle)
        last_delta_rot = self._rotate_feature(
            last_delta_raw.reshape(1, 3),
            rotation_matrix,
            angle,
        ).reshape(3)
        target_rot = self._rotate_feature(target_raw, rotation_matrix, angle)

        # Normalize only after the physical rotation has been applied.
        x_seq = self._normalize_feature_window(x_seq_rot)
        last_delta = self._normalize_delta_array(last_delta_rot.reshape(1, 3)).reshape(3)
        target = self._normalize_delta_array(target_rot)

        return {
            "seq": torch.from_numpy(x_seq).float(),
            "last_delta": torch.from_numpy(last_delta).float(),
            "target": torch.from_numpy(target).float(),
            "intent": torch.tensor(label, dtype=torch.long),
        }


def _count_open_file_descriptors() -> int:
    """Return the current process FD count on Linux; -1 when unavailable."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return -1


def _configure_file_descriptor_limit(target: int = 8192) -> None:
    """Best-effort soft-limit raise; never rely on this as the only protection."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = min(int(hard), max(int(soft), int(target)))
        if desired > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except Exception:
        pass


class EpochAwareWeightedSampler(Sampler[int]):
    """
    A reusable, epoch-aware sampler.

    The DataLoader is created only once for a full model×seed run.  At each
    epoch we only update this sampler's epoch state; worker processes, pipes,
    shared-memory handles and queues are not rebuilt.
    """

    def __init__(
        self,
        labels: List[int],
        seed: int,
        num_samples: Optional[int] = None,
        high_maneuver_curriculum: bool = False,
    ) -> None:
        labels_np = np.asarray(labels, dtype=np.int64)
        if labels_np.ndim != 1 or labels_np.size == 0:
            raise ValueError("Sampler labels must be a non-empty 1D array.")
        if np.any((labels_np < 0) | (labels_np > 2)):
            raise ValueError("Intent labels must be in {0, 1, 2}.")

        self.labels = torch.as_tensor(labels_np, dtype=torch.long)
        counts = np.bincount(labels_np, minlength=3).astype(np.float64)
        self.base_class_weight = torch.as_tensor(
            1.0 / np.maximum(counts, 1.0), dtype=torch.double
        )
        self.seed = int(seed)
        self.num_samples = int(num_samples or labels_np.size)
        self.high_maneuver_curriculum = bool(high_maneuver_curriculum)
        self.epoch = 0
        self.total_epochs = 1
        self._weights = torch.empty(labels_np.size, dtype=torch.double)
        self._refresh_weights()

    def set_epoch(self, epoch: int, total_epochs: int = 1) -> None:
        self.epoch = int(epoch)
        self.total_epochs = max(int(total_epochs), 1)
        self._refresh_weights()

    def _refresh_weights(self) -> None:
        class_weight = self.base_class_weight.clone()
        if self.high_maneuver_curriculum:
            progress = self.epoch / max(self.total_epochs - 1, 1)
            high_multiplier = 0.05 + 4.95 / (1.0 + math.exp(-15.0 * (progress - 0.15)))
            class_weight[2] *= high_multiplier
        self._weights = class_weight[self.labels].double().contiguous()

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1009)
        selected = torch.multinomial(
            self._weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        return iter(selected.tolist())

    def __len__(self) -> int:
        return self.num_samples


def _loader_kwargs(
    num_workers: int,
    generator: torch.Generator,
    persistent_workers: bool = True,
    prefetch_factor: int = 1,
) -> Dict[str, Any]:
    """Build DataLoader kwargs safely for both multiprocess and single-process modes."""
    workers = max(0, int(num_workers))
    kwargs: Dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "worker_init_fn": seed_worker if workers > 0 else None,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return kwargs


@dataclass
class LoaderBundle:
    """
    Owns all loaders for one model×seed run.

    Train/Val/Test DataLoader are built exactly once. `set_epoch()` changes
    only the sampler order/weights and the training dataset's shared epoch
    value; no worker, pipe or shared-memory queue is recreated.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_sampler: EpochAwareWeightedSampler

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        self.train_sampler.set_epoch(epoch=epoch, total_epochs=total_epochs)
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def close(self) -> None:
        shutdown_dataloader(self.train_loader)
        shutdown_dataloader(self.val_loader)
        shutdown_dataloader(self.test_loader)


def make_loader_bundle(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    train_records: List[Dict[str, Any]],
    train_batch_size: int,
    eval_batch_size: int,
    seed: int,
    num_workers: int,
    persistent_workers: bool = True,
    prefetch_factor: int = 1,
    high_maneuver_curriculum: bool = False,
) -> LoaderBundle:
    """
    Create exactly three DataLoaders for one model×seed run.

    Do not call this function inside an epoch loop.  `set_epoch()` changes the
    sample order without recreating workers or descriptors.
    """
    _configure_file_descriptor_limit()
    workers = max(0, int(num_workers))
    labels = [int(row["label"]) for row in train_records]
    sampler = EpochAwareWeightedSampler(
        labels=labels,
        seed=int(seed),
        high_maneuver_curriculum=high_maneuver_curriculum,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed) + 101)
    val_generator = torch.Generator()
    val_generator.manual_seed(int(seed) + 202)
    test_generator = torch.Generator()
    test_generator.manual_seed(int(seed) + 303)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_batch_size),
        sampler=sampler,
        **_loader_kwargs(
            workers,
            train_generator,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(eval_batch_size),
        shuffle=False,
        **_loader_kwargs(
            workers,
            val_generator,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(eval_batch_size),
        shuffle=False,
        **_loader_kwargs(
            workers,
            test_generator,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        ),
    )
    return LoaderBundle(train_loader, val_loader, test_loader, sampler)


def shutdown_dataloader(loader: Optional[DataLoader]) -> None:
    """Best-effort teardown for persistent worker processes at run completion."""
    if loader is None:
        return
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            loader._iterator = None
    except Exception:
        # Do not hide the primary training error because cleanup is best effort.
        pass

# ============================================================================
# 模型：Ours（不使用未来真实意图标签）
# ============================================================================
class ScaleAttentionGate(nn.Module):
    def __init__(self, d_model: int, n_scales: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model),
            nn.Tanh(),
            nn.Linear(d_model, n_scales),
        )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat(features, dim=-1)
        weights = torch.softmax(self.gate(concat).float(), dim=-1).to(concat.dtype)
        stacked = torch.stack(features, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class KinematicProjectionLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = DROPOUT_BASE):
        super().__init__()
        self.global_proj = nn.Sequential(nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.xy_proj = nn.Sequential(nn.Linear(2, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, d_model // 2))
        self.z_proj = nn.Sequential(nn.Linear(1, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, d_model // 2))
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))
        nn.init.normal_(self.fuse[-1].weight, std=0.02)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        global_feature = self.global_proj(delta)
        decoupled = torch.cat([self.xy_proj(delta[:, :2]), self.z_proj(delta[:, 2:3])], dim=-1)
        return global_feature + self.fuse(torch.cat([global_feature, decoupled], dim=-1))


class EnhancedMambaBlock(nn.Module):
    def __init__(self, d_model: int, mamba_main_dir: Optional[str] = None):
        super().__init__()
        mamba_cls = get_mamba2(mamba_main_dir)
        self.use_mamba = mamba_cls is not None
        self.norm = nn.LayerNorm(d_model)
        if self.use_mamba:
            self.backbone = mamba_cls(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.backbone = nn.GRU(d_model, d_model, batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model),
            nn.Conv1d(d_model, d_model, 7, padding=3, groups=d_model),
        ])
        self.channel_attention = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, d_model), nn.Sigmoid())
        self.dropout = nn.Dropout(DROPOUT_BASE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mamba:
            encoded = self.backbone(self.norm(x))
        else:
            encoded, _ = self.backbone(self.norm(x))
        residual = encoded + x
        conv_input = self.norm(residual).transpose(1, 2)
        conv_output = sum(conv(conv_input) for conv in self.convs) / len(self.convs)
        conv_output = conv_output.transpose(1, 2)
        return self.dropout(F.silu(conv_output * self.channel_attention(conv_output) + residual)) * 0.8 + x * 0.2


class ParallelTemporalDecoder(nn.Module):
    def __init__(self, d_model: int, horizon: int, n_heads: int = 4, dropout: float = DROPOUT_BASE):
        super().__init__()
        self.horizon = horizon
        self.time_queries = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.time_embedding = nn.Embedding(horizon, d_model)
        nn.init.normal_(self.time_embedding.weight, std=0.02)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.delta_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 3))
        nn.init.normal_(self.delta_head[-1].weight, std=0.02)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, context_sequence: torch.Tensor, intent_bias: torch.Tensor) -> torch.Tensor:
        batch = context_sequence.shape[0]
        time_ids = torch.arange(self.horizon, device=context_sequence.device)
        query = (self.time_queries + self.time_embedding(time_ids)).unsqueeze(0).expand(batch, -1, -1)
        query = query + intent_bias.unsqueeze(1)
        z, _ = self.self_attn(query, query, query)
        query = self.norm0(query + z)
        z, _ = self.cross_attn(query, context_sequence, context_sequence)
        query = self.norm1(query + z)
        query = self.norm2(query + self.ffn(query))
        return self.delta_head(query)


class PhysicsGatedPredictor(nn.Module):
    """UA-PGD 局部自回归预测器；输入维度已修正为 d_model + 32。"""

    def __init__(self, d_model: int, horizon: int, hidden_dim: int = 128):
        super().__init__()
        self.horizon = horizon
        self.time_embedding = nn.Embedding(horizon, 16)
        nn.init.normal_(self.time_embedding.weight, std=0.02)

        motion_context_dim = 3 + 10 + 3 + 16  # npd + mac + intent probability + time embedding
        self.residual_predictor = nn.Sequential(
            nn.Linear(d_model + motion_context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT_BASE),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )
        self.context_update = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))
        self.gru = nn.GRU(input_size=9, hidden_size=hidden_dim, batch_first=True)
        self.gru_init = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.Tanh())
        self.direction_residual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(DROPOUT_BASE), nn.Linear(hidden_dim // 2, 3))
        nn.init.zeros_(self.direction_residual[-1].weight)
        nn.init.zeros_(self.direction_residual[-1].bias)

        self.magnitude_head = nn.Sequential(nn.Linear(hidden_dim + d_model, hidden_dim // 2), nn.GELU(), nn.Dropout(DROPOUT_BASE), nn.Linear(hidden_dim // 2, 1))
        nn.init.zeros_(self.magnitude_head[-1].weight)
        nn.init.constant_(self.magnitude_head[-1].bias, 0.2)
        self.high_magnitude_boost = nn.Parameter(torch.tensor(0.3))
        self.alpha_per_intent = nn.Parameter(torch.tensor([-1.5, -0.4, 0.62]))
        self.kpl = KinematicProjectionLayer(d_model)
        self.kpl_mix = nn.Parameter(torch.tensor(-2.0))
        time_bias = torch.log(torch.arange(1, horizon + 1, dtype=torch.float32)) - 1.0
        self.time_log_var_bias = nn.Parameter(time_bias.unsqueeze(1).repeat(1, 3).contiguous())

    def init_state(self, context: torch.Tensor) -> torch.Tensor:
        return self.gru_init(context).unsqueeze(0)

    def forward(
        self,
        context: torch.Tensor,
        current_delta: torch.Tensor,
        previous_delta: torch.Tensor,
        recurrent_state: torch.Tensor,
        intent_prob: torch.Tensor,
        motion_context: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = current_delta.size(0)
        enriched = context + torch.sigmoid(self.kpl_mix) * self.kpl(current_delta)
        time_feature = self.time_embedding(torch.tensor(step, device=current_delta.device)).unsqueeze(0).expand(batch, -1)
        condition = torch.cat([current_delta, motion_context, intent_prob, time_feature], dim=-1)
        output = self.residual_predictor(torch.cat([enriched, condition], dim=-1))
        action_base, logvar_offset = output[:, :3], output[:, 3:]

        recurrent_input = torch.cat([current_delta, current_delta - previous_delta, intent_prob], dim=-1).unsqueeze(1)
        recurrent_output, next_state = self.gru(recurrent_input, recurrent_state)
        recurrent_feature = recurrent_output.squeeze(1)
        direction_delta = self.direction_residual(recurrent_feature)

        high_prob = intent_prob[:, 2:3]
        magnitude_logit = self.magnitude_head(torch.cat([recurrent_feature, enriched], dim=-1)) + high_prob * self.high_magnitude_boost
        magnitude_lower = -1.5 + 0.5 * (1.0 - high_prob)
        magnitude_upper = 1.5 + 1.5 * high_prob
        magnitude = torch.exp(torch.clamp(magnitude_logit, min=magnitude_lower, max=magnitude_upper))

        alpha = (intent_prob * (torch.sigmoid(self.alpha_per_intent) * 0.5)).sum(dim=-1, keepdim=True)
        alpha = alpha * (1.0 + 1.5 * high_prob)
        action = action_base + alpha * direction_delta
        action = torch.cat([action[:, :2] * magnitude, action[:, 2:3]], dim=-1)

        intent_logvar_bias = (intent_prob * torch.tensor([0.0, 0.3, 1.5], device=current_delta.device, dtype=current_delta.dtype)).sum(dim=-1, keepdim=True)
        logvar_floor = 2.0 * torch.log(torch.linspace(0.015, 0.12, self.horizon, device=current_delta.device)[step])
        # Keep uncertainty target-specific. Using high_prob.mean() would make
        # one target's predictive variance depend on other targets in the batch.
        logvar_ceiling = 3.0 + 0.18 * step + 1.5 * high_prob
        logvar = torch.clamp(logvar_offset + self.time_log_var_bias[step] + intent_logvar_bias, min=logvar_floor, max=logvar_ceiling)
        return action, self.context_update(enriched), logvar, next_state


class IntentAwarePredictor(nn.Module):
    """
    完整方法。关键约束：forward() 不接收真实 intent_targets，
    解码器条件始终来自模型自身预测的 intent probability。
    """

    def __init__(self, feature_dim: int, d_model: int, horizon: int, scales: List[int], mamba_main_dir: Optional[str] = None):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.scales = sorted(scales)
        self.mc_sampling_active = False
        self.goal_correction_strength = 0.0
        self.mc_latent_noise_scale = 0.30
        self.mc_step_noise_base = 0.04
        self.mc_step_noise_growth = 4.5

        self.input_projection = nn.Linear(feature_dim, d_model)
        self.encoder = EnhancedMambaBlock(d_model, mamba_main_dir=mamba_main_dir)
        self.scale_gate = ScaleAttentionGate(d_model, len(self.scales))
        self.global_dropout = nn.Dropout(DROPOUT_BASE * 1.5)

        self.intent_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_HEAVY),
            nn.Linear(d_model // 2, 3),
        )
        self.intent_embedding = nn.Embedding(3, d_model)
        self.context_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.parallel_decoder = ParallelTemporalDecoder(d_model, horizon)
        self.global_mix_logit = nn.Parameter(torch.tensor(-0.85))
        self.physics_decoder = PhysicsGatedPredictor(d_model, horizon)

        self.smoother = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, 3, kernel_size=3, padding=1),
        )
        self.output_scale = nn.Parameter(torch.ones(3) * 1.5)
        self.turn_rate_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(d_model // 2, horizon))

        self.num_goals = 3
        self.goal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(DROPOUT_BASE),
            nn.Linear(d_model // 2, self.num_goals * 3 + self.num_goals),
        )
        nn.init.normal_(self.goal_head[-1].weight, std=0.01)
        nn.init.zeros_(self.goal_head[-1].bias)

    def _pyramid_features(self, encoded: torch.Tensor) -> List[torch.Tensor]:
        time_len = encoded.shape[1]
        return [encoded[:, -1, :] - encoded[:, max(0, time_len - scale), :] for scale in self.scales]

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        tau: float = 1.0,
        target_deltas: Optional[torch.Tensor] = None,
        teacher_forcing_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        sequence = x_dict["seq"]
        current_delta = x_dict["last_delta"]
        batch = sequence.size(0)
        device = sequence.device

        encoded = self.encoder(self.input_projection(sequence))
        context_raw, _ = self.scale_gate(self._pyramid_features(encoded))

        if self.training or self.mc_sampling_active:
            # 高机动概率只用于调节噪声大小；不是未来真实标签。
            preliminary_logits = self.intent_head(context_raw)
            high_probability = torch.softmax(preliminary_logits, dim=-1)[:, 2:3]
            noise_scale = self.mc_latent_noise_scale * (1.0 + 1.5 * high_probability) if self.mc_sampling_active else self.mc_latent_noise_scale
            context = self.global_dropout(context_raw + torch.randn_like(context_raw) * noise_scale)
        else:
            context = self.global_dropout(context_raw)

        intent_logits = self.intent_head(context)
        intent_probability = torch.softmax(intent_logits / max(float(tau), 1e-4), dim=-1)
        if self.mc_sampling_active:
            sampled_intent = torch.distributions.Categorical(probs=intent_probability.float()).sample()
            intent_condition = F.one_hot(sampled_intent, num_classes=3).float()
        else:
            intent_condition = intent_probability

        intent_bias = intent_condition @ self.intent_embedding.weight
        global_context = self.context_projection(context + intent_bias)
        global_deltas = self.parallel_decoder(encoded, intent_bias)
        turn_rate = self.turn_rate_head(context)

        recurrent_state = self.physics_decoder.init_state(context)
        previous_delta = sequence[:, -2, :3].detach() if sequence.shape[1] >= 2 else torch.zeros_like(current_delta)
        rolling_buffer = sequence[:, -min(15, sequence.shape[1]) :, :3].detach()
        if rolling_buffer.shape[1] < 15:
            pad = torch.zeros(batch, 15 - rolling_buffer.shape[1], 3, device=device, dtype=rolling_buffer.dtype)
            rolling_buffer = torch.cat([pad, rolling_buffer], dim=1)

        predicted_deltas, predicted_logvars = [], []
        context_state = context
        high_prob = intent_condition[:, 2:3]
        mix = torch.sigmoid(self.global_mix_logit) * (1.0 - 0.5 * high_prob)

        for step in range(self.horizon):
            if step > 0 and step % 5 == 0:
                refresh = min(0.3, 0.1 + 0.05 * (step / 5.0))
                context_state = (1.0 - refresh) * context_state + refresh * global_context

            sum_5 = rolling_buffer[:, -5:].sum(dim=1)
            sum_15 = rolling_buffer[:, -15:].sum(dim=1)
            dir_5 = F.normalize(sum_5[:, :2], dim=-1, eps=EPS)
            dir_15 = F.normalize(sum_15[:, :2], dim=-1, eps=EPS)
            motion_context = torch.cat([sum_5, sum_15, dir_5, dir_15], dim=-1)

            ar_action, context_state, logvar, recurrent_state = self.physics_decoder(
                context_state,
                current_delta,
                previous_delta,
                recurrent_state,
                intent_condition,
                motion_context,
                step,
            )
            action = ((1.0 - mix) * ar_action + mix * global_deltas[:, step, :]) * self.output_scale.clamp(0.1, 2.0)

            action_xy = action[:, :2]
            action_z = action[:, 2:3]
            current_dir = F.normalize(current_delta[:, :2], dim=-1, eps=EPS)
            action_dir = F.normalize(action_xy, dim=-1, eps=EPS)
            base_blend = max(0.0, 0.4 - step * 0.025)
            blend = (base_blend * (1.0 - torch.sqrt(high_prob + EPS))).clamp(0.0, 1.0)
            blended_dir = F.normalize(blend * current_dir + (1.0 - blend) * action_dir, dim=-1, eps=EPS)
            action_xy = blended_dir * torch.norm(action_xy, dim=-1, keepdim=True).clamp(min=EPS)
            action = torch.cat([torch.clamp(action_xy, -6.0, 6.0), torch.clamp(action_z, -3.0, 3.0)], dim=-1)

            if self.mc_sampling_active:
                noise_std = self.mc_step_noise_base * (1.0 + step / max(self.horizon - 1, 1) * self.mc_step_noise_growth)
                action = action + torch.randn_like(action) * noise_std * (1.0 + 1.2 * high_prob)

            predicted_deltas.append(action)
            predicted_logvars.append(logvar)

            if self.training and target_deltas is not None and teacher_forcing_mask is not None:
                next_delta = torch.where(teacher_forcing_mask[:, step : step + 1], target_deltas[:, step], action.detach())
            else:
                next_delta = action.detach()
            previous_delta = current_delta.detach()
            current_delta = next_delta
            rolling_buffer = torch.cat([rolling_buffer[:, 1:], action.unsqueeze(1).detach()], dim=1)

        raw_prediction = torch.stack(predicted_deltas, dim=1)
        logvar_prediction = torch.stack(predicted_logvars, dim=1)
        smoother_residual = self.smoother(raw_prediction.detach().transpose(1, 2)).transpose(1, 2)
        smooth_weight = 1.0 * intent_condition[:, 0:1, None] + 0.6 * intent_condition[:, 1:2, None] + 0.25 * intent_condition[:, 2:3, None]
        final_prediction = raw_prediction + smooth_weight * smoother_residual

        goal_raw = self.goal_head(context)
        goals = goal_raw[:, : self.num_goals * 3].view(batch, self.num_goals, 3)
        goal_logits = goal_raw[:, self.num_goals * 3 :]
        goal_probability = torch.softmax(goal_logits, dim=-1)

        # Goal anchor 仅用于 MC 样本生成；不读取真实未来终点。
        if self.mc_sampling_active and self.goal_correction_strength > 0.0:
            selected = torch.distributions.Categorical(probs=goal_probability.float()).sample()
            selected_goal = torch.gather(goals, 1, selected.view(batch, 1, 1).expand(-1, 1, 3)).squeeze(1)
            endpoint = torch.cumsum(final_prediction, dim=1)[:, -1]
            correction = (selected_goal - endpoint) * (0.15 * self.goal_correction_strength) * (1.0 + high_prob)
            per_step = correction.unsqueeze(1) / max(self.horizon // 2, 1)
            mask = torch.zeros(1, self.horizon, 1, device=device)
            mask[:, self.horizon // 2 :, :] = 1.0
            final_prediction = final_prediction + per_step * mask

        if return_aux:
            return final_prediction, logvar_prediction, intent_logits, turn_rate, {
                "raw_prediction": raw_prediction,
                "global_deltas": global_deltas,
                "intent_probability": intent_probability,
                "goals": goals,
                "goal_logits": goal_logits,
            }
        return final_prediction, logvar_prediction, intent_logits, turn_rate

    @torch.no_grad()
    def sample_trajectories(self, x_dict: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        """生成 K 条真实 MC 轨迹，不进行伪样本复制。"""
        previous_training = self.training
        previous_mc = self.mc_sampling_active

        def enable_dropout(module: nn.Module):
            if isinstance(module, nn.Dropout):
                module.train()

        self.eval()
        self.apply(enable_dropout)
        self.mc_sampling_active = True
        samples = []
        try:
            for _ in range(int(num_samples)):
                prediction, logvar, _, _ = self.forward(x_dict)
                # Sample the same heteroscedastic displacement distribution
                # optimized by the Gaussian NLL. Dropout and decoder noise
                # account for epistemic variation; this term adds the predicted
                # target- and horizon-dependent aleatoric variation.
                std = torch.exp(
                    0.5 * torch.clamp(logvar, min=-6.0, max=4.0)
                )
                samples.append(
                    prediction + torch.randn_like(prediction) * std
                )
        finally:
            self.mc_sampling_active = previous_mc
            self.train(previous_training)
        return torch.stack(samples, dim=0)

# ============================================================================
# Baseline 模型：全部使用完整 history_len 输入，禁止内部截断为15步。
# ============================================================================
class VanillaMambaBlock(nn.Module):
    def __init__(self, d_model: int, mamba_main_dir: Optional[str] = None):
        super().__init__()
        mamba_cls = get_mamba2(mamba_main_dir)
        self.use_mamba = mamba_cls is not None
        self.norm = nn.LayerNorm(d_model)
        self.net = mamba_cls(d_model=d_model, d_state=16, d_conv=4, expand=2) if self.use_mamba else nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mamba:
            return self.net(self.norm(x)) + x
        out, _ = self.net(self.norm(x))
        return out + x


class iTransformerAdapter(nn.Module):
    def __init__(self, seq_len: int, horizon: int, feature_dim: int, d_model: int, n_heads: int = 8, layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model, batch_first=True, dropout=DROPOUT_BASE)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.projector = nn.Linear(d_model, horizon)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x.transpose(1, 2))
        z = self.encoder(z)
        return self.projector(z).transpose(1, 2)[:, :, :3]


class PatchTSTAdapter(nn.Module):
    def __init__(self, seq_len: int, horizon: int, feature_dim: int, d_model: int, patch_len: int = 5, stride: int = 5):
        super().__init__()
        if seq_len < patch_len:
            raise ValueError("patch_len cannot exceed seq_len")
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 1 + (seq_len - patch_len) // stride
        self.feature_dim = feature_dim
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=2 * d_model, batch_first=True, dropout=DROPOUT_BASE)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(self.num_patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_len, channels = x.shape
        if channels != self.feature_dim:
            raise RuntimeError("PatchTST feature dimension mismatch")
        patches = x.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch * channels, self.num_patches, self.patch_len)
        z = self.encoder(self.value_embedding(patches) + self.position_embedding)
        output = self.head(z.reshape(batch * channels, -1))
        return output.view(batch, channels, -1).transpose(1, 2)[:, :, :3]


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1].repeat(1, pad, 1)
        tail = x[:, -1:].repeat(1, pad, 1)
        z = torch.cat([front, x, tail], dim=1)
        return self.pool(z.permute(0, 2, 1)).permute(0, 2, 1)


class DLinearAdapter(nn.Module):
    def __init__(self, seq_len: int, horizon: int, feature_dim: int):
        super().__init__()
        kernel = min(25, seq_len if seq_len % 2 == 1 else seq_len - 1)
        kernel = max(kernel, 3)
        self.average = MovingAverage(kernel)
        self.seasonal = nn.Linear(seq_len, horizon)
        self.trend = nn.Linear(seq_len, horizon)
        self.project = nn.Linear(feature_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.average(x)
        seasonal = x - trend
        seasonal_out = self.seasonal(seasonal.permute(0, 2, 1))
        trend_out = self.trend(trend.permute(0, 2, 1))
        return self.project((seasonal_out + trend_out).permute(0, 2, 1))


class CVAEAdapter(nn.Module):
    """Conditional CVAE baseline with learned p(z|history).

    The original lightweight CVAE sampled from N(0, I) at inference. This
    version learns a conditional prior from the history context, which is a
    fairer probabilistic baseline for trajectory forecasting.
    """

    def __init__(self, feature_dim: int, d_model: int, horizon: int, latent_dim: int = 32):
        super().__init__()
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.history_encoder = nn.LSTM(feature_dim, d_model, batch_first=True)
        self.future_encoder = nn.LSTM(3, d_model, batch_first=True)
        self.posterior_mu_head = nn.Linear(2 * d_model, latent_dim)
        self.posterior_logvar_head = nn.Linear(2 * d_model, latent_dim)
        self.prior_mu_head = nn.Linear(d_model, latent_dim)
        self.prior_logvar_head = nn.Linear(d_model, latent_dim)
        self.decoder_input = nn.Linear(d_model + latent_dim, d_model)
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.output_head = nn.Linear(d_model, 3)

    def _decode(self, history_context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        decode = self.decoder_input(torch.cat([history_context, z], dim=-1)).unsqueeze(1).expand(-1, self.horizon, -1)
        output, _ = self.decoder(decode)
        return self.output_head(output)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None, deterministic: bool = False):
        _, (history_h, _) = self.history_encoder(x)
        history_context = history_h[-1]
        prior_mu = self.prior_mu_head(history_context)
        prior_logvar = torch.clamp(self.prior_logvar_head(history_context), -5.0, 5.0)

        if self.training and target is not None:
            _, (future_h, _) = self.future_encoder(target)
            latent_input = torch.cat([history_context, future_h[-1]], dim=-1)
            posterior_mu = self.posterior_mu_head(latent_input)
            posterior_logvar = torch.clamp(self.posterior_logvar_head(latent_input), -5.0, 5.0)
            z = posterior_mu if deterministic else posterior_mu + torch.randn_like(posterior_mu) * torch.exp(0.5 * posterior_logvar)
            return self._decode(history_context, z), posterior_mu, posterior_logvar, prior_mu, prior_logvar

        if deterministic:
            z = prior_mu
        else:
            z = prior_mu + torch.randn_like(prior_mu) * torch.exp(0.5 * prior_logvar)
        return self._decode(history_context, z)


class DiffusionAdapter(nn.Module):
    """简化 DDPM。最终 K 个样本均由真实反向采样生成，不复制补齐。"""

    def __init__(self, feature_dim: int, d_model: int, horizon: int, diffusion_steps: int = 20):
        super().__init__()
        self.horizon = horizon
        self.diffusion_steps = diffusion_steps
        self.history_encoder = nn.LSTM(feature_dim, d_model, batch_first=True)
        self.time_embedding = nn.Embedding(diffusion_steps, d_model)
        self.denoiser = nn.Sequential(
            nn.Linear(horizon * 3 + 2 * d_model, 2 * d_model), nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model), nn.SiLU(),
            nn.Linear(2 * d_model, horizon * 3),
        )
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def training_forward(self, x: torch.Tensor, target: torch.Tensor):
        _, (history_h, _) = self.history_encoder(x)
        condition = history_h[-1]
        batch = target.size(0)
        t = torch.randint(0, self.diffusion_steps, (batch,), device=x.device)
        noise = torch.randn_like(target)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1.0 - alpha_bar) * noise
        time_feature = self.time_embedding(t)
        network_input = torch.cat([noisy.reshape(batch, -1), condition, time_feature], dim=-1)
        predicted_noise = self.denoiser(network_input).view(batch, self.horizon, 3)
        x0 = (noisy - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar + EPS)
        return x0, predicted_noise, noise

    @torch.no_grad()
    def sample(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        _, (history_h, _) = self.history_encoder(x)
        condition = history_h[-1]
        batch = x.size(0)
        current = torch.zeros(batch, self.horizon, 3, device=x.device) if deterministic else torch.randn(batch, self.horizon, 3, device=x.device)
        for t_idx in reversed(range(self.diffusion_steps)):
            t = torch.full((batch,), t_idx, device=x.device, dtype=torch.long)
            time_feature = self.time_embedding(t)
            network_input = torch.cat([current.reshape(batch, -1), condition, time_feature], dim=-1)
            predicted_noise = self.denoiser(network_input).view(batch, self.horizon, 3)
            alpha = self.alphas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            if deterministic or t_idx == 0:
                noise = torch.zeros_like(current)
            else:
                noise = torch.randn_like(current)
            current = (current - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar + EPS)) * predicted_noise) / torch.sqrt(alpha + EPS) + torch.sqrt(self.betas[t_idx]) * noise
        return current



class GaussianLSTMAdapter(nn.Module):
    """Gaussian-LSTM baseline: LSTM history encoder + Gaussian horizon decoder."""

    def __init__(self, feature_dim: int, d_model: int, horizon: int):
        super().__init__()
        self.horizon = int(horizon)
        self.encoder = nn.LSTM(feature_dim, d_model, num_layers=2, batch_first=True, dropout=DROPOUT_BASE)
        self.time_embedding = nn.Embedding(horizon, d_model)
        self.decoder = nn.GRU(2 * d_model, d_model, batch_first=True)
        self.init_hidden = nn.Linear(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(DROPOUT_BASE), nn.Linear(d_model, 6)
        )
        nn.init.normal_(self.time_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        _, (hidden, _) = self.encoder(x)
        context = hidden[-1]
        time_ids = torch.arange(self.horizon, device=x.device)
        time_feature = self.time_embedding(time_ids).unsqueeze(0).expand(batch, -1, -1)
        decode_input = torch.cat([context.unsqueeze(1).expand(-1, self.horizon, -1), time_feature], dim=-1)
        init = torch.tanh(self.init_hidden(context)).unsqueeze(0)
        out, _ = self.decoder(decode_input, init)
        stats = self.head(out)
        mean = stats[..., :3]
        logvar = torch.clamp(stats[..., 3:], -6.0, 4.0)
        return mean, logvar


class GaussianTransformerAdapter(nn.Module):
    """Gaussian-Transformer baseline with cross-attentive horizon queries."""

    def __init__(self, seq_len: int, feature_dim: int, d_model: int, horizon: int, n_heads: int = 4, layers: int = 2):
        super().__init__()
        self.horizon = int(horizon)
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            batch_first=True, dropout=DROPOUT_BASE
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.time_embedding = nn.Embedding(horizon, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=DROPOUT_BASE)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(DROPOUT_BASE), nn.Linear(2 * d_model, d_model))
        self.head = nn.Linear(d_model, 6)
        nn.init.normal_(self.time_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        encoded = self.encoder(self.input_projection(x) + self.position)
        time_ids = torch.arange(self.horizon, device=x.device)
        q = self.query.expand(batch, -1, -1) + self.time_embedding(time_ids).unsqueeze(0)
        z, _ = self.cross_attn(q, encoded, encoded)
        z = self.norm(q + z)
        z = self.norm(z + self.ffn(z))
        stats = self.head(z)
        return stats[..., :3], torch.clamp(stats[..., 3:], -6.0, 4.0)


class DeepARAdapter(nn.Module):
    """DeepAR-style autoregressive Gaussian decoder for continuous 3-D deltas."""

    def __init__(self, feature_dim: int, d_model: int, horizon: int):
        super().__init__()
        self.horizon = int(horizon)
        self.encoder = nn.LSTM(feature_dim, d_model, num_layers=2, batch_first=True, dropout=DROPOUT_BASE)
        self.time_embedding = nn.Embedding(horizon, d_model)
        self.decoder_cell = nn.GRUCell(d_model + 3, d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(DROPOUT_BASE), nn.Linear(d_model, 6))
        nn.init.normal_(self.time_embedding.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        last_delta: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tf_prob: float = 0.0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        _, (hidden, _) = self.encoder(x)
        state = hidden[-1]
        prev_delta = last_delta
        means, logvars = [], []
        for step in range(self.horizon):
            time_feature = self.time_embedding(torch.tensor(step, device=x.device)).unsqueeze(0).expand(batch, -1)
            state = self.decoder_cell(torch.cat([prev_delta, time_feature], dim=-1), state)
            stats = self.head(state)
            mean = stats[:, :3]
            logvar = torch.clamp(stats[:, 3:], -6.0, 4.0)
            means.append(mean)
            logvars.append(logvar)
            if self.training and target is not None:
                teacher_mask = (torch.rand(batch, 1, device=x.device) < float(tf_prob)).to(mean.dtype)
                prev_delta = teacher_mask * target[:, step] + (1.0 - teacher_mask) * mean.detach()
            elif deterministic:
                prev_delta = mean.detach()
            else:
                prev_delta = (mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)).detach()
        return torch.stack(means, dim=1), torch.stack(logvars, dim=1)


class TFTQuantileAdapter(nn.Module):
    """TFT-Quantile inspired baseline with variable gating, recurrent encoding and quantile outputs."""

    def __init__(self, feature_dim: int, d_model: int, horizon: int, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.horizon = int(horizon)
        self.num_quantiles = len(quantiles)
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.variable_gate = nn.Sequential(
            nn.Linear(feature_dim, d_model), nn.GELU(), nn.Linear(d_model, feature_dim), nn.Sigmoid()
        )
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.encoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.time_embedding = nn.Embedding(horizon, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=DROPOUT_BASE)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(DROPOUT_BASE), nn.Linear(d_model, 3 * self.num_quantiles)
        )
        nn.init.normal_(self.time_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        weights = self.variable_gate(x)
        z = self.input_projection(x * weights)
        encoded, _ = self.encoder(z)
        time_ids = torch.arange(self.horizon, device=x.device)
        q = self.query.expand(batch, -1, -1) + self.time_embedding(time_ids).unsqueeze(0)
        attn, _ = self.cross_attn(q, encoded, encoded)
        z = self.norm(q + self.gate(attn) * attn)
        raw = self.head(z).view(batch, self.horizon, 3, self.num_quantiles)
        # Enforce non-crossing quantiles by sorting. The median remains the central quantile.
        return torch.sort(raw, dim=-1).values

    def median(self, quantile_prediction: torch.Tensor) -> torch.Tensor:
        median_idx = int(torch.argmin(torch.abs(self.quantiles - 0.5)).item())
        return quantile_prediction[..., median_idx]

    def sample_from_quantiles(self, quantile_prediction: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Approximate inverse-CDF sampling from the predicted quantile grid."""
        q_levels = self.quantiles.to(quantile_prediction.device)
        q_pred = quantile_prediction
        samples = []
        for _ in range(int(num_samples)):
            u = torch.rand(*q_pred.shape[:-1], 1, device=q_pred.device)
            # Three-quantile default: below q10, q10-q50, q50-q90, above q90.
            lower = q_pred[..., 0:1]
            median = q_pred[..., min(1, self.num_quantiles - 1):min(2, self.num_quantiles)]
            upper = q_pred[..., -1:]
            if self.num_quantiles == 1:
                sample = lower.squeeze(-1)
            else:
                q_low = q_levels[0]
                q_mid = q_levels[min(1, self.num_quantiles - 1)]
                q_high = q_levels[-1]
                left_slope = (median - lower) / torch.clamp(q_mid - q_low, min=1e-3)
                right_slope = (upper - median) / torch.clamp(q_high - q_mid, min=1e-3)
                sample_left = lower + (u - q_low) * left_slope
                sample_mid_left = lower + (u - q_low) / torch.clamp(q_mid - q_low, min=1e-3) * (median - lower)
                sample_mid_right = median + (u - q_mid) / torch.clamp(q_high - q_mid, min=1e-3) * (upper - median)
                sample_right = upper + (u - q_high) * right_slope
                sample = torch.where(u < q_low, sample_left, torch.where(u <= q_mid, sample_mid_left, torch.where(u <= q_high, sample_mid_right, sample_right)))
                sample = sample.squeeze(-1)
            samples.append(sample)
        return torch.stack(samples, dim=0)


class TrajectronPPAdapter(nn.Module):
    """Single-agent Trajectron++-style CVAE baseline.

    The original Trajectron++ uses dynamic interaction graphs. The present UAV
    dataset is single-target after ID grouping, so this adapter keeps the
    recurrent CVAE and multi-modal latent prior while omitting unavailable
    neighbor-edge encoders.
    """

    def __init__(self, feature_dim: int, d_model: int, horizon: int, latent_dim: int = 24, num_modes: int = 5):
        super().__init__()
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.num_modes = int(num_modes)
        self.history_encoder = nn.LSTM(feature_dim, d_model, num_layers=2, batch_first=True, dropout=DROPOUT_BASE)
        self.future_encoder = nn.LSTM(3, d_model, batch_first=True)
        self.prior_head = nn.Linear(d_model, num_modes * (2 * latent_dim) + num_modes)
        self.posterior_head = nn.Linear(2 * d_model, 2 * latent_dim)
        self.mode_embedding = nn.Embedding(num_modes, d_model)
        self.decoder_cell = nn.GRUCell(3 + d_model + latent_dim, d_model)
        self.init_state = nn.Linear(d_model + latent_dim + d_model, d_model)
        self.output_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(DROPOUT_BASE), nn.Linear(d_model, 3))

    def _prior(self, context: torch.Tensor):
        # The discrete prior is particularly sensitive to FP16 overflow.  Run
        # this head in FP32 even if a caller accidentally enables autocast.
        with torch.amp.autocast(device_type=context.device.type, enabled=False):
            raw = self.prior_head(context.float())
        logits = raw[:, :self.num_modes]
        params = raw[:, self.num_modes:].view(context.size(0), self.num_modes, 2 * self.latent_dim)
        mu = params[..., :self.latent_dim]
        logvar = torch.clamp(params[..., self.latent_dim:], -5.0, 5.0)
        return logits, mu, logvar

    def _decode(self, context: torch.Tensor, z: torch.Tensor, mode_ids: torch.Tensor, last_delta: torch.Tensor, target: Optional[torch.Tensor] = None, tf_prob: float = 0.0):
        batch = context.size(0)
        mode_emb = self.mode_embedding(mode_ids)
        state = torch.tanh(self.init_state(torch.cat([context, z, mode_emb], dim=-1)))
        prev_delta = last_delta
        outputs = []
        for step in range(self.horizon):
            state = self.decoder_cell(torch.cat([prev_delta, mode_emb, z], dim=-1), state)
            delta = self.output_head(state)
            outputs.append(delta)
            if self.training and target is not None:
                mask = (torch.rand(batch, 1, device=context.device) < float(tf_prob)).to(delta.dtype)
                prev_delta = mask * target[:, step] + (1.0 - mask) * delta.detach()
            else:
                prev_delta = delta.detach()
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, last_delta: torch.Tensor, target: Optional[torch.Tensor] = None, tf_prob: float = 0.0, deterministic: bool = False):
        _, (hidden, _) = self.history_encoder(x)
        context = hidden[-1]
        prior_logits, prior_mu_all, prior_logvar_all = self._prior(context)
        if self.training and target is not None:
            _, (future_h, _) = self.future_encoder(target)
            posterior_raw = self.posterior_head(torch.cat([context, future_h[-1]], dim=-1))
            posterior_mu, posterior_logvar = posterior_raw[:, :self.latent_dim], torch.clamp(posterior_raw[:, self.latent_dim:], -5.0, 5.0)
            mode_ids = torch.argmax(prior_logits, dim=-1)
            prior_mu = torch.gather(prior_mu_all, 1, mode_ids.view(-1, 1, 1).expand(-1, 1, self.latent_dim)).squeeze(1)
            prior_logvar = torch.gather(prior_logvar_all, 1, mode_ids.view(-1, 1, 1).expand(-1, 1, self.latent_dim)).squeeze(1)
            z = posterior_mu if deterministic else posterior_mu + torch.randn_like(posterior_mu) * torch.exp(0.5 * posterior_logvar)
            prediction = self._decode(context, z, mode_ids, last_delta, target=target, tf_prob=tf_prob)
            return prediction, posterior_mu, posterior_logvar, prior_mu, prior_logvar, prior_logits
        if deterministic:
            mode_ids = torch.argmax(prior_logits, dim=-1)
        else:
            sampling_logits = prior_logits.float()
            if not torch.isfinite(sampling_logits).all():
                finite = sampling_logits[torch.isfinite(sampling_logits)]
                finite_range = (
                    f"[{finite.min().item():.6g}, {finite.max().item():.6g}]"
                    if finite.numel() > 0 else "empty"
                )
                raise FloatingPointError(
                    "TrajectronPP prior logits became non-finite before "
                    f"Categorical sampling: nan={torch.isnan(sampling_logits).sum().item()}, "
                    f"posinf={torch.isposinf(sampling_logits).sum().item()}, "
                    f"neginf={torch.isneginf(sampling_logits).sum().item()}, "
                    f"finite_range={finite_range}."
                )
            # Clipping only removes numerically irrelevant extreme log-odds;
            # it does not replace NaN/Inf values.
            sampling_logits = sampling_logits.clamp(-30.0, 30.0)
            mode_ids = torch.distributions.Categorical(logits=sampling_logits).sample()
        prior_mu = torch.gather(prior_mu_all, 1, mode_ids.view(-1, 1, 1).expand(-1, 1, self.latent_dim)).squeeze(1)
        prior_logvar = torch.gather(prior_logvar_all, 1, mode_ids.view(-1, 1, 1).expand(-1, 1, self.latent_dim)).squeeze(1)
        z = prior_mu if deterministic else prior_mu + torch.randn_like(prior_mu) * torch.exp(0.5 * prior_logvar)
        return self._decode(context, z, mode_ids, last_delta, target=None, tf_prob=0.0)


class CVMDAdapter(nn.Module):
    """cVMD-style deterministic baseline using fixed multi-scale mode decomposition.

    This avoids an external VMD dependency while preserving the comparison idea:
    decompose the observed sequence into low/mid/high-frequency motion modes,
    encode each mode separately, and conditionally fuse them for prediction.
    """

    def __init__(self, seq_len: int, feature_dim: int, d_model: int, horizon: int, kernels: Tuple[int, ...] = (3, 9, 21)):
        super().__init__()
        self.horizon = int(horizon)
        self.kernels = tuple(int(min(k, seq_len if seq_len % 2 == 1 else seq_len - 1)) for k in kernels)
        self.kernels = tuple(max(3, k if k % 2 == 1 else k - 1) for k in self.kernels)
        self.mode_count = len(self.kernels) + 1
        self.encoders = nn.ModuleList([nn.GRU(feature_dim, d_model, batch_first=True) for _ in range(self.mode_count)])
        self.fuse = nn.Sequential(
            nn.Linear(self.mode_count * d_model + 3, 2 * d_model), nn.GELU(), nn.Dropout(DROPOUT_BASE),
            nn.Linear(2 * d_model, horizon * 3)
        )

    def _moving_average(self, x: torch.Tensor, kernel: int) -> torch.Tensor:
        pad = (kernel - 1) // 2
        z = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        return F.avg_pool1d(z, kernel_size=kernel, stride=1).transpose(1, 2)

    def _decompose(self, x: torch.Tensor) -> List[torch.Tensor]:
        smooths = [self._moving_average(x, k) for k in self.kernels]
        modes = [x - smooths[0]]
        for i in range(len(smooths) - 1):
            modes.append(smooths[i] - smooths[i + 1])
        modes.append(smooths[-1])
        return modes

    def forward(self, x: torch.Tensor, last_delta: torch.Tensor) -> torch.Tensor:
        contexts = []
        for mode, encoder in zip(self._decompose(x), self.encoders):
            _, hidden = encoder(mode)
            contexts.append(hidden[-1])
        fused = torch.cat(contexts + [last_delta], dim=-1)
        return self.fuse(fused).view(x.size(0), self.horizon, 3)


class BaselineWrapper(nn.Module):
    def __init__(self, model_type: str, feature_dim: int, d_model: int, horizon: int, history_len: int, diffusion_steps: int = 20, mamba_main_dir: Optional[str] = None):
        super().__init__()
        self.model_type = model_type
        self.horizon = horizon
        self.is_probabilistic = model_type in {
            "mc_dropout", "cvae", "diffusion", "gaussian_lstm", "gaussian_transformer",
            "deepar", "tft_quantile", "trajectronpp"
        }

        if model_type == "lstm":
            self.net = nn.LSTM(feature_dim, d_model, num_layers=2, batch_first=True, dropout=DROPOUT_BASE)
            self.head = nn.Linear(d_model, horizon * 3)
        elif model_type == "gru":
            self.net = nn.GRU(feature_dim, d_model, num_layers=2, batch_first=True, dropout=DROPOUT_BASE)
            self.head = nn.Linear(d_model, horizon * 3)
        elif model_type == "transformer":
            self.input_projection = nn.Linear(feature_dim, d_model)
            self.position = nn.Parameter(torch.randn(1, history_len, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=2 * d_model, batch_first=True, dropout=DROPOUT_BASE)
            self.net = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(d_model, horizon * 3)
        elif model_type == "vanilla_mamba":
            self.input_projection = nn.Linear(feature_dim, d_model)
            self.net = VanillaMambaBlock(d_model, mamba_main_dir=mamba_main_dir)
            self.head = nn.Linear(d_model, horizon * 3)
        elif model_type == "itransformer":
            self.net = iTransformerAdapter(history_len, horizon, feature_dim, d_model)
        elif model_type == "patchtst":
            self.net = PatchTSTAdapter(history_len, horizon, feature_dim, d_model)
        elif model_type == "dlinear":
            self.net = DLinearAdapter(history_len, horizon, feature_dim)
        elif model_type == "mc_dropout":
            self.net = nn.LSTM(feature_dim, d_model, num_layers=2, batch_first=True, dropout=0.0)
            self.dropout = nn.Dropout(0.30)
            self.head = nn.Linear(d_model, horizon * 3)
        elif model_type == "cvae":
            self.net = CVAEAdapter(feature_dim, d_model, horizon)
        elif model_type == "diffusion":
            self.net = DiffusionAdapter(feature_dim, d_model, horizon, diffusion_steps=diffusion_steps)
        elif model_type == "gaussian_lstm":
            self.net = GaussianLSTMAdapter(feature_dim, d_model, horizon)
        elif model_type == "gaussian_transformer":
            self.net = GaussianTransformerAdapter(history_len, feature_dim, d_model, horizon)
        elif model_type == "deepar":
            self.net = DeepARAdapter(feature_dim, d_model, horizon)
        elif model_type == "tft_quantile":
            self.net = TFTQuantileAdapter(feature_dim, d_model, horizon)
        elif model_type == "trajectronpp":
            self.net = TrajectronPPAdapter(feature_dim, d_model, horizon)
        elif model_type == "cvmd":
            self.net = CVMDAdapter(history_len, feature_dim, d_model, horizon)
        else:
            raise ValueError(f"Unknown baseline model_type: {model_type}")

    def forward(self, x_dict: Dict[str, torch.Tensor], target: Optional[torch.Tensor] = None, deterministic: bool = False, tf_prob: float = 0.0):
        x = x_dict["seq"]
        batch = x.size(0)
        if self.model_type in {"lstm", "gru"}:
            out, _ = self.net(x)
            return self.head(out[:, -1]).view(batch, self.horizon, 3)
        if self.model_type == "transformer":
            out = self.net(self.input_projection(x) + self.position)
            return self.head(out[:, -1]).view(batch, self.horizon, 3)
        if self.model_type == "vanilla_mamba":
            out = self.net(self.input_projection(x))
            return self.head(out[:, -1]).view(batch, self.horizon, 3)
        if self.model_type in {"itransformer", "patchtst", "dlinear"}:
            return self.net(x)
        if self.model_type == "mc_dropout":
            out, _ = self.net(x)
            out = self.dropout(out[:, -1])
            return self.head(out).view(batch, self.horizon, 3)
        if self.model_type == "cvae":
            return self.net(x, target=target, deterministic=deterministic)
        if self.model_type == "diffusion":
            if self.training and target is not None:
                return self.net.training_forward(x, target)
            return self.net.sample(x, deterministic=deterministic)
        if self.model_type in {"gaussian_lstm", "gaussian_transformer"}:
            return self.net(x)
        if self.model_type == "deepar":
            return self.net(x, x_dict["last_delta"], target=target, tf_prob=tf_prob, deterministic=deterministic)
        if self.model_type == "tft_quantile":
            return self.net(x)
        if self.model_type == "trajectronpp":
            return self.net(x, x_dict["last_delta"], target=target, tf_prob=tf_prob, deterministic=deterministic)
        if self.model_type == "cvmd":
            return self.net(x, x_dict["last_delta"])
        raise RuntimeError("Unreachable model branch")

    @torch.no_grad()
    def point_prediction(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model_type == "cvae":
            return self.forward(x_dict, deterministic=True)
        if self.model_type == "diffusion":
            return self.forward(x_dict, deterministic=True)
        if self.model_type in {"gaussian_lstm", "gaussian_transformer", "deepar"}:
            mean, _ = self.forward(x_dict, deterministic=True)
            return mean
        if self.model_type == "tft_quantile":
            quantiles = self.forward(x_dict)
            return self.net.median(quantiles)
        if self.model_type == "trajectronpp":
            return self.forward(x_dict, deterministic=True)
        return self.forward(x_dict)

    @torch.no_grad()
    def sample_trajectories(self, x_dict: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        num_samples = int(num_samples)
        was_training = self.training
        samples = []
        try:
            if self.model_type == "mc_dropout":
                self.eval()
                def enable_dropout(module: nn.Module):
                    if isinstance(module, nn.Dropout):
                        module.train()
                self.apply(enable_dropout)
                for _ in range(num_samples):
                    samples.append(self.forward(x_dict))
            elif self.model_type == "cvae":
                self.eval()
                for _ in range(num_samples):
                    samples.append(self.forward(x_dict, deterministic=False))
            elif self.model_type == "diffusion":
                self.eval()
                for _ in range(num_samples):
                    samples.append(self.forward(x_dict, deterministic=False))
            elif self.model_type in {"gaussian_lstm", "gaussian_transformer", "deepar"}:
                self.eval()
                mean, logvar = self.forward(x_dict, deterministic=True)
                std = torch.exp(0.5 * torch.clamp(logvar, -6.0, 4.0))
                for _ in range(num_samples):
                    samples.append(mean + torch.randn_like(mean) * std)
            elif self.model_type == "tft_quantile":
                self.eval()
                quantiles = self.forward(x_dict)
                return self.net.sample_from_quantiles(quantiles, num_samples)
            elif self.model_type == "trajectronpp":
                self.eval()
                for _ in range(num_samples):
                    samples.append(self.forward(x_dict, deterministic=False))
            else:
                self.eval()
                point = self.point_prediction(x_dict)
                samples = [point for _ in range(num_samples)]
        finally:
            self.train(was_training)
        return torch.stack(samples, dim=0)


# ============================================================================
# 训练损失：所有模型共享的基础轨迹项；Ours仅附加其方法内在的辅助任务。
# ============================================================================
def trajectory_loss_base(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """不使用意图标签的公共基础轨迹损失。"""
    pred_traj = torch.cumsum(prediction.float(), dim=1)
    true_traj = torch.cumsum(target.float(), dim=1)
    loss_position = F.smooth_l1_loss(pred_traj, true_traj)
    loss_fde = F.smooth_l1_loss(pred_traj[:, -1], true_traj[:, -1])
    loss_step_mag = F.mse_loss(torch.norm(prediction.float(), dim=-1), torch.norm(target.float(), dim=-1))

    pred_yaw = torch.atan2(prediction[..., 1], prediction[..., 0] + EPS)
    true_yaw = torch.atan2(target[..., 1], target[..., 0] + EPS)
    valid = torch.norm(target[..., :2], dim=-1) > 0.05
    if valid.any():
        loss_heading = (1.0 - torch.cos(pred_yaw - true_yaw))[valid].mean()
    else:
        loss_heading = prediction.new_zeros(())

    if prediction.size(1) >= 3:
        jerk_pred = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
        jerk_true = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
        loss_jerk = F.smooth_l1_loss(jerk_pred, jerk_true)
    else:
        loss_jerk = prediction.new_zeros(())

    return loss_position + 0.5 * loss_fde + 0.5 * loss_step_mag + 0.25 * loss_heading + 0.05 * loss_jerk


def ours_loss(
    prediction: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    intent_logits: torch.Tensor,
    intent_target: torch.Tensor,
    aux_info: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Ours 的总训练损失；真实意图标签只用于分类监督，不进入解码条件。"""
    base = trajectory_loss_base(prediction, target)
    clipped_logvar = torch.clamp(logvar.float(), -5.0, 5.0)
    variance = torch.exp(clipped_logvar) + 1e-6
    hetero_nll = 0.5 * (clipped_logvar + (prediction.float() - target.float()) ** 2 / variance).mean()
    intent_ce = F.cross_entropy(intent_logits, intent_target, weight=torch.tensor([1.0, 1.5, 3.0], device=prediction.device))

    goals = aux_info["goals"]
    goal_logits = aux_info["goal_logits"]
    endpoint = torch.cumsum(target.float(), dim=1)[:, -1]
    distance = torch.norm(goals - endpoint.unsqueeze(1), dim=-1)
    closest = torch.argmin(distance, dim=-1)
    goal_regression = distance.gather(1, closest.unsqueeze(1)).mean()
    goal_classification = F.cross_entropy(goal_logits, closest)
    if goals.size(1) > 1:
        pairwise = torch.cdist(goals[:, :, :2], goals[:, :, :2])
        eye = torch.eye(goals.size(1), device=goals.device).unsqueeze(0)
        mean_pairwise = (pairwise * (1.0 - eye)).sum(dim=(1, 2)) / float(goals.size(1) * (goals.size(1) - 1))
        diversity = torch.relu(1.5 - mean_pairwise).mean()
    else:
        diversity = prediction.new_zeros(())
    goal_loss = goal_regression + 0.3 * goal_classification + 0.3 * diversity

    return base + 0.30 * hetero_nll + 0.50 * intent_ce + 0.30 * goal_loss


def cvae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    progress: float,
) -> torch.Tensor:
    reconstruction = trajectory_loss_base(prediction, target)
    posterior_logvar = torch.clamp(posterior_logvar, -5.0, 5.0)
    prior_logvar = torch.clamp(prior_logvar, -5.0, 5.0)
    # KL(q(z|x,y) || p(z|x)) for diagonal Gaussians.
    kl = 0.5 * (
        prior_logvar - posterior_logvar
        + (posterior_logvar.exp() + (posterior_mu - prior_mu).pow(2)) / (prior_logvar.exp() + 1e-8)
        - 1.0
    ).sum(dim=1).mean()
    beta = 0.001 * min(1.0, progress / 0.10)
    return reconstruction + beta * kl


def diffusion_loss(x0_prediction: torch.Tensor, predicted_noise: torch.Tensor, true_noise: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_noise, true_noise) + 0.10 * trajectory_loss_base(x0_prediction, target)



def gaussian_trajectory_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL plus the shared trajectory-shape loss on the predictive mean."""
    clipped = torch.clamp(logvar.float(), -6.0, 4.0)
    var = torch.exp(clipped) + 1e-6
    nll = 0.5 * (clipped + (target.float() - mean.float()).pow(2) / var).mean()
    return trajectory_loss_base(mean, target) + 0.30 * nll


def quantile_pinball_loss(quantile_prediction: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Pinball loss for TFT-Quantile; target [B,H,3], prediction [B,H,3,Q]."""
    q = quantiles.to(target.device).view(1, 1, 1, -1)
    error = target.unsqueeze(-1).float() - quantile_prediction.float()
    pinball = torch.maximum(q * error, (q - 1.0) * error).mean()
    median_idx = int(torch.argmin(torch.abs(quantiles.to(target.device) - 0.5)).item())
    median = quantile_prediction[..., median_idx]
    return pinball + 0.50 * trajectory_loss_base(median, target)


def trajectronpp_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    prior_logits: torch.Tensor,
    progress: float,
) -> torch.Tensor:
    # Evaluate the complete CVAE objective in FP32.  Casting only the final
    # logits is too late because FP16 overflow may already have occurred.
    prediction = prediction.float()
    target = target.float()
    posterior_mu = posterior_mu.float()
    posterior_logvar = torch.clamp(posterior_logvar.float(), -5.0, 5.0)
    prior_mu = prior_mu.float()
    prior_logvar = torch.clamp(prior_logvar.float(), -5.0, 5.0)
    prior_logits = prior_logits.float()

    named_tensors = {
        "prediction": prediction,
        "target": target,
        "posterior_mu": posterior_mu,
        "posterior_logvar": posterior_logvar,
        "prior_mu": prior_mu,
        "prior_logvar": prior_logvar,
        "prior_logits": prior_logits,
    }
    invalid = [name for name, value in named_tensors.items() if not torch.isfinite(value).all()]
    if invalid:
        raise FloatingPointError(
            "Non-finite TrajectronPP loss inputs: " + ", ".join(invalid)
        )

    reconstruction = trajectory_loss_base(prediction, target)
    kl = 0.5 * (
        prior_logvar - posterior_logvar
        + (posterior_logvar.exp() + (posterior_mu - prior_mu).pow(2)) / (prior_logvar.exp() + 1e-8)
        - 1.0
    ).sum(dim=1).mean()
    # log_softmax is stable for large finite logits and avoids softmax/log
    # cancellation in the discrete-prior entropy term.
    log_prior_prob = F.log_softmax(prior_logits, dim=-1)
    prior_prob = log_prior_prob.exp()
    entropy = -(prior_prob * log_prior_prob).sum(dim=-1).mean()
    beta = 0.002 * min(1.0, progress / 0.15)
    total = reconstruction + beta * kl - 0.001 * entropy
    if not torch.isfinite(total):
        raise FloatingPointError(
            "Non-finite TrajectronPP objective: "
            f"reconstruction={reconstruction.detach().item():.6g}, "
            f"kl={kl.detach().item():.6g}, entropy={entropy.detach().item():.6g}."
        )
    return total

# ============================================================================
# 物理坐标转换与评价指标
# ============================================================================
def inverse_deltas(x: np.ndarray | torch.Tensor, delta_scaler: StandardScaler) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        array = x.detach().float().cpu().numpy()
    else:
        array = np.asarray(x, dtype=np.float32)
    original_shape = array.shape
    return delta_scaler.inverse_transform(array.reshape(-1, 3)).reshape(original_shape)


def deltas_to_trajectories(deltas: np.ndarray) -> np.ndarray:
    return np.cumsum(deltas, axis=-2)


def energy_score_batch(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """多变量轨迹 Energy Score；返回每个样本的 ES。"""
    # samples [K,B,H,3], truth [B,H,3]
    k, batch = samples.shape[:2]
    scores = np.empty(batch, dtype=np.float64)
    for b in range(batch):
        sample_vec = samples[:, b].reshape(k, -1)
        truth_vec = truth[b].reshape(-1)
        first = np.linalg.norm(sample_vec - truth_vec[None, :], axis=1).mean()
        pairwise = np.linalg.norm(sample_vec[:, None, :] - sample_vec[None, :, :], axis=-1).mean()
        scores[b] = first - 0.5 * pairwise
    return scores


def kinematic_metrics_batch(
    pred_traj: np.ndarray,
    true_traj: np.ndarray,
    dt: float,
    gamma: float,
    max_accel: float,
    max_jerk: float,
    a_ref_floor: float = 1e-3,
) -> Dict[str, Any]:
    """速度、加速度、加加速度误差及物理可行性指标。

    A-KEER 修正：使用训练集估计的加速度参考下限，避免平稳飞行中
    max(a_true) 过小而导致轻微预测扰动被放大为异常能量越限。
    """
    if pred_traj.shape[1] < 4:
        return {
            "Ev_sum": 0.0, "Ev_count": 0,
            "Ea_sum": 0.0, "Ea_count": 0,
            "Ej_sum": 0.0, "Ej_count": 0,
            "AKEER_sum": 0.0, "AKEER_count": 0,
            "NJER_dev_sum": 0.0, "NJER_count": 0,
            "Violation_sum": 0.0, "Violation_count": 0,
        }

    origin = np.zeros((pred_traj.shape[0], 1, 3), dtype=pred_traj.dtype)
    v_pred = np.diff(np.concatenate([origin, pred_traj], axis=1), axis=1) / dt
    v_true = np.diff(np.concatenate([origin, true_traj], axis=1), axis=1) / dt
    a_pred = np.diff(v_pred, axis=1) / dt
    a_true = np.diff(v_true, axis=1) / dt
    j_pred = np.diff(a_pred, axis=1) / dt
    j_true = np.diff(a_true, axis=1) / dt

    a_pred_norm = np.linalg.norm(a_pred, axis=-1)
    a_true_norm = np.linalg.norm(a_true, axis=-1)
    j_pred_norm = np.linalg.norm(j_pred, axis=-1)
    j_true_norm = np.linalg.norm(j_true, axis=-1)

    floor_value = max(float(a_ref_floor), 1e-3)
    a_ref = np.maximum(np.max(a_true_norm, axis=1, keepdims=True), floor_value)
    a_keer = (a_pred_norm > gamma * a_ref).astype(np.float64)

    jerk_energy_pred = np.sum(j_pred_norm ** 2, axis=1)
    jerk_energy_true = np.sum(j_true_norm ** 2, axis=1)
    njer = jerk_energy_pred / (jerk_energy_true + EPS)

    acceleration_violation = a_pred_norm > max_accel
    jerk_violation = j_pred_norm > max_jerk
    violation_rate_sum = acceleration_violation.sum() + jerk_violation.sum()
    violation_rate_count = acceleration_violation.size + jerk_violation.size

    return {
        "Ev_sum": float(np.linalg.norm(v_pred - v_true, axis=-1).sum()), "Ev_count": int(v_pred.shape[0] * v_pred.shape[1]),
        "Ea_sum": float(np.linalg.norm(a_pred - a_true, axis=-1).sum()), "Ea_count": int(a_pred.shape[0] * a_pred.shape[1]),
        "Ej_sum": float(np.linalg.norm(j_pred - j_true, axis=-1).sum()), "Ej_count": int(j_pred.shape[0] * j_pred.shape[1]),
        "AKEER_sum": float(a_keer.sum()), "AKEER_count": int(a_keer.size),
        "NJER_dev_sum": float(np.abs(njer - 1.0).sum()), "NJER_count": int(njer.size),
        "Violation_sum": float(violation_rate_sum), "Violation_count": int(violation_rate_count),
    }


class MetricAccumulator:
    def __init__(self, horizon: int, config: Dict[str, Any], probability_enabled: bool):
        self.horizon = horizon
        self.probability_enabled = probability_enabled
        self.dt = float(config["sample_interval_s"])
        self.distance_ratio = float(config["distance_accuracy_ratio"])
        self.distance_floor = float(config["distance_accuracy_min_threshold_m"])
        self.direction_threshold = np.deg2rad(float(config["direction_accuracy_threshold_deg"]))
        self.direction_speed_threshold = float(config["direction_min_speed_m"])
        self.gamma = float(config["a_keer_gamma"])
        self.max_accel = float(config["physical_max_accel_mps2"])
        self.max_jerk = float(config["physical_max_jerk_mps3"])
        self.a_ref_floor = float(config.get("a_keer_ref_floor_mps2", 1e-3))

        self.n = 0
        self.err_sum = np.zeros(horizon, dtype=np.float64)
        self.err_sq_sum = np.zeros(horizon, dtype=np.float64)
        self.dist_correct = np.zeros(horizon, dtype=np.float64)
        self.dist_total = np.zeros(horizon, dtype=np.float64)
        self.dir_correct = np.zeros(horizon, dtype=np.float64)
        self.dir_total = np.zeros(horizon, dtype=np.float64)
        self.rpe_sum = {5: 0.0, 10: 0.0}
        self.rpe_count = {5: 0, 10: 0}
        self.prob_sum = {
            "minADE": 0.0,
            "minFDE": 0.0,
            "EnergyScore": 0.0,
            "NLL_sum": 0.0,
            "NLL_count": 0,
            "mPIC95_correct": 0.0,
            "mPIC95_total": 0,
            "MIW95_sum": 0.0,
            "MIW95_count": 0,
            "count": 0,
        }
        # Stepwise probability metrics are accumulated in position space from
        # exactly the same predictive samples used for interval estimation.
        self.step_nll_sum = np.zeros(horizon, dtype=np.float64)
        self.step_nll_count = np.zeros(horizon, dtype=np.int64)
        self.step_pic95_correct = np.zeros(horizon, dtype=np.float64)
        self.step_pic95_total = np.zeros(horizon, dtype=np.int64)
        self.step_miw95_sum = np.zeros(horizon, dtype=np.float64)
        self.step_miw95_count = np.zeros(horizon, dtype=np.int64)
        self.kin_sum = {key: 0.0 for key in ["Ev_sum", "Ea_sum", "Ej_sum", "AKEER_sum", "NJER_dev_sum", "Violation_sum"]}
        self.kin_count = {key: 0 for key in ["Ev_count", "Ea_count", "Ej_count", "AKEER_count", "NJER_count", "Violation_count"]}

    def update(self, pred_traj: np.ndarray, true_traj: np.ndarray, sample_trajs: Optional[np.ndarray] = None) -> None:
        if len(pred_traj) == 0:
            return
        batch = pred_traj.shape[0]
        self.n += batch
        error = np.linalg.norm(pred_traj - true_traj, axis=-1)
        self.err_sum += error.sum(axis=0)
        self.err_sq_sum += (error ** 2).sum(axis=0)

        true_deltas = np.diff(np.concatenate([np.zeros((batch, 1, 3), dtype=true_traj.dtype), true_traj], axis=1), axis=1)
        path_length = np.cumsum(np.linalg.norm(true_deltas, axis=-1), axis=1)
        threshold = np.maximum(self.distance_floor, self.distance_ratio * path_length)
        self.dist_correct += (error <= threshold).sum(axis=0)
        self.dist_total += batch

        pred_delta = np.diff(np.concatenate([np.zeros((batch, 1, 3), dtype=pred_traj.dtype), pred_traj], axis=1), axis=1)
        pred_xy = pred_delta[:, :, :2]
        true_xy = true_deltas[:, :, :2]
        pred_norm = np.linalg.norm(pred_xy, axis=-1)
        true_norm = np.linalg.norm(true_xy, axis=-1)
        valid = true_norm > self.direction_speed_threshold
        cos_value = np.sum(pred_xy * true_xy, axis=-1) / (pred_norm * true_norm + EPS)
        angle = np.arccos(np.clip(cos_value, -1.0, 1.0))
        self.dir_correct += ((angle <= self.direction_threshold) & valid).sum(axis=0)
        self.dir_total += valid.sum(axis=0)

        for delta in [5, 10]:
            if self.horizon > delta:
                rel_pred = pred_traj[:, delta:] - pred_traj[:, :-delta]
                rel_true = true_traj[:, delta:] - true_traj[:, :-delta]
                rel_error = np.linalg.norm(rel_pred - rel_true, axis=-1)
                self.rpe_sum[delta] += float(rel_error.sum())
                self.rpe_count[delta] += int(rel_error.size)

        dynamic = kinematic_metrics_batch(
            pred_traj,
            true_traj,
            self.dt,
            self.gamma,
            self.max_accel,
            self.max_jerk,
            self.a_ref_floor,
        )
        for key in self.kin_sum:
            self.kin_sum[key] += dynamic[key]
        for key in self.kin_count:
            self.kin_count[key] += dynamic[key]

        if sample_trajs is not None:
            if sample_trajs.ndim != 4 or sample_trajs.shape[1:] != true_traj.shape:
                raise ValueError(
                    "sample_trajs must have shape [K, B, H, 3] matching true_traj"
                )
            sample_error = np.linalg.norm(sample_trajs - true_traj[None, ...], axis=-1)
            min_ade = np.min(sample_error.mean(axis=2), axis=0)
            min_fde = np.min(sample_error[:, :, -1], axis=0)
            energy = energy_score_batch(sample_trajs, true_traj)
            lower = np.quantile(sample_trajs, 0.025, axis=0)
            upper = np.quantile(sample_trajs, 0.975, axis=0)
            inside = (true_traj >= lower) & (true_traj <= upper)
            width = upper - lower

            # Moment-matched diagonal Gaussian NLL in physical position space.
            # A scale-aware floor avoids singular likelihoods when a finite
            # sample set has nearly zero spread at an early horizon.
            sample_mean = np.mean(sample_trajs, axis=0)
            sample_var = np.var(sample_trajs, axis=0, ddof=0)
            scale_floor = max(
                float(np.median(np.linalg.norm(true_deltas, axis=-1))) * 1e-3,
                1e-4,
            )
            variance = np.maximum(sample_var, scale_floor ** 2)
            nll_element = 0.5 * (
                np.log(2.0 * np.pi * variance)
                + (true_traj - sample_mean) ** 2 / variance
            )
            nll_step_sum = nll_element.sum(axis=(0, 2))
            nll_step_count = np.full(
                self.horizon,
                batch * true_traj.shape[2],
                dtype=np.int64,
            )

            pic_step_correct = inside.sum(axis=(0, 2))
            pic_step_total = np.full(
                self.horizon,
                batch * true_traj.shape[2],
                dtype=np.int64,
            )
            miw_step_sum = width.sum(axis=(0, 2))
            miw_step_count = np.full(
                self.horizon,
                batch * width.shape[2],
                dtype=np.int64,
            )

            self.step_nll_sum += nll_step_sum
            self.step_nll_count += nll_step_count
            self.step_pic95_correct += pic_step_correct
            self.step_pic95_total += pic_step_total
            self.step_miw95_sum += miw_step_sum
            self.step_miw95_count += miw_step_count

            self.prob_sum["minADE"] += float(min_ade.sum())
            self.prob_sum["minFDE"] += float(min_fde.sum())
            self.prob_sum["EnergyScore"] += float(energy.sum())
            self.prob_sum["NLL_sum"] += float(nll_step_sum.sum())
            self.prob_sum["NLL_count"] += int(nll_step_count.sum())
            self.prob_sum["mPIC95_correct"] += float(inside.sum())
            self.prob_sum["mPIC95_total"] += int(inside.size)
            self.prob_sum["MIW95_sum"] += float(width.sum())
            self.prob_sum["MIW95_count"] += int(width.size)
            self.prob_sum["count"] += batch

    def summary(self) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        count = max(self.n, 1)
        step_error = self.err_sum / count
        step_rmse = np.sqrt(self.err_sq_sum / count)
        step_dist_acc = 100.0 * self.dist_correct / np.maximum(self.dist_total, 1.0)
        step_dir_acc = 100.0 * self.dir_correct / np.maximum(self.dir_total, 1.0)

        summary = {
            "n_samples": int(self.n),
            "ADE": float(step_error.mean()),
            "FDE": float(step_error[-1]),
            "RPE5": float(self.rpe_sum[5] / max(self.rpe_count[5], 1)),
            "RPE10": float(self.rpe_sum[10] / max(self.rpe_count[10], 1)),
            "DistAcc": float(step_dist_acc.mean()),
            "DirAcc": float(self.dir_correct.sum() / max(self.dir_total.sum(), 1.0) * 100.0),
            "Ev": float(self.kin_sum["Ev_sum"] / max(self.kin_count["Ev_count"], 1)),
            "Ea": float(self.kin_sum["Ea_sum"] / max(self.kin_count["Ea_count"], 1)),
            "Ej": float(self.kin_sum["Ej_sum"] / max(self.kin_count["Ej_count"], 1)),
            "A_KEER": float(self.kin_sum["AKEER_sum"] / max(self.kin_count["AKEER_count"], 1) * 100.0),
            "NJERDev": float(self.kin_sum["NJER_dev_sum"] / max(self.kin_count["NJER_count"], 1)),
            "ViolationRate": float(self.kin_sum["Violation_sum"] / max(self.kin_count["Violation_count"], 1) * 100.0),
        }

        if self.prob_sum["count"] > 0:
            prob_count = self.prob_sum["count"]
            pic95_value = float(
                self.prob_sum["mPIC95_correct"]
                / max(self.prob_sum["mPIC95_total"], 1)
                * 100.0
            )
            summary.update({
                "minADE": float(self.prob_sum["minADE"] / prob_count),
                "minFDE": float(self.prob_sum["minFDE"] / prob_count),
                "EnergyScore": float(self.prob_sum["EnergyScore"] / prob_count),
                "NLL": float(
                    self.prob_sum["NLL_sum"]
                    / max(self.prob_sum["NLL_count"], 1)
                ),
                "PIC95": pic95_value,
                "mPIC95": pic95_value,
                "MIW95": float(self.prob_sum["MIW95_sum"] / max(self.prob_sum["MIW95_count"], 1)),
            })
        else:
            summary.update({"minADE": float("nan"), "minFDE": float("nan"), "EnergyScore": float("nan"), "NLL": float("nan"), "PIC95": float("nan"), "mPIC95": float("nan"), "MIW95": float("nan")})

        step_rows = []
        for step in range(self.horizon):
            probability_available = self.step_nll_count[step] > 0
            step_rows.append({
                "step": step + 1,
                "time_s": float((step + 1) * self.dt),
                "DistErr": float(step_error[step]),
                "RMSE": float(step_rmse[step]),
                "DistAcc": float(step_dist_acc[step]),
                "DirAcc": float(step_dir_acc[step]) if self.dir_total[step] > 0 else float("nan"),
                "NLL": float(
                    self.step_nll_sum[step]
                    / max(self.step_nll_count[step], 1)
                ) if probability_available else float("nan"),
                "PIC95": float(
                    100.0 * self.step_pic95_correct[step]
                    / max(self.step_pic95_total[step], 1)
                ) if probability_available else float("nan"),
                "MIW95": float(
                    self.step_miw95_sum[step]
                    / max(self.step_miw95_count[step], 1)
                ) if probability_available else float("nan"),
            })
        return summary, step_rows

# ============================================================================
# 模型调用、验证与最终测试评估
# ============================================================================
def is_probabilistic_model(model_type: str) -> bool:
    return model_type in {
        "ours", "mc_dropout", "cvae", "diffusion",
        "gaussian_lstm", "gaussian_transformer", "deepar",
        "tft_quantile", "trajectronpp",
    }


def point_predict(model: nn.Module, model_type: str, x_dict: Dict[str, torch.Tensor]):
    if model_type == "ours":
        prediction, _, intent_logits, _ = model(x_dict)
        return prediction, intent_logits
    prediction = model.point_prediction(x_dict)
    return prediction, None


def sample_predict(model: nn.Module, model_type: str, x_dict: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
    if model_type == "ours":
        return model.sample_trajectories(x_dict, num_samples)
    return model.sample_trajectories(x_dict, num_samples)


def apply_horizon_interval_calibration(
    sample_trajs: np.ndarray,
    calibration_scale: Optional[np.ndarray],
) -> np.ndarray:
    """Rescale predictive deviations without changing the predictive mean.

    ``calibration_scale`` has shape [H, 3] and is estimated on validation data
    only. A separate factor is used at every horizon and coordinate, preserving
    target-specific H-PGD uncertainty while correcting systematic horizon-wise
    under- or over-dispersion.
    """
    if calibration_scale is None:
        return sample_trajs
    scale = np.asarray(calibration_scale, dtype=sample_trajs.dtype)
    if scale.shape != sample_trajs.shape[2:]:
        raise ValueError(
            f"calibration_scale must have shape {sample_trajs.shape[2:]}, got {scale.shape}"
        )
    center = np.mean(sample_trajs, axis=0, keepdims=True)
    return center + (sample_trajs - center) * scale[None, None, :, :]


@torch.no_grad()
def fit_horizon_interval_calibration(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    delta_scaler: StandardScaler,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fit the narrowest validation-only scale attaining nominal marginal PIC.

    The procedure uses no test labels. For each horizon and coordinate, it
    computes the empirical quantile of the ratio between the absolute residual
    and the corresponding raw 95% interval radius. Multiplying deviations from
    the predictive sample mean by this ratio corrects systematic dispersion.
    The same protocol is applied to every probabilistic comparator.
    """
    horizon = int(config["horizon"])
    if not is_probabilistic_model(model_type):
        return np.ones((horizon, 3), dtype=np.float64), pd.DataFrame()

    k_samples = int(config.get("calibration_prob_samples", config["final_prob_samples"]))
    nominal = float(config.get("interval_nominal_coverage", 0.95))
    min_scale = float(config.get("interval_calibration_min_scale", 0.10))
    max_scale = float(config.get("interval_calibration_max_scale", 10.0))
    ratio_blocks: List[np.ndarray] = []

    model.eval()
    for batch in tqdm(loader, desc=f"Calibrate {model_type}", leave=False):
        x_dict = {
            "seq": batch["seq"].to(device, non_blocking=True),
            "last_delta": batch["last_delta"].to(device, non_blocking=True),
        }
        target = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=AMP_DTYPE,
            enabled=amp_enabled_for_model(model_type),
        ):
            sample_delta = sample_predict(model, model_type, x_dict, k_samples)
        sample_traj = deltas_to_trajectories(inverse_deltas(sample_delta, delta_scaler))
        true_traj = deltas_to_trajectories(inverse_deltas(target, delta_scaler))

        center = np.mean(sample_traj, axis=0)
        lower = np.quantile(sample_traj, 0.025, axis=0)
        upper = np.quantile(sample_traj, 0.975, axis=0)
        side_radius = np.where(true_traj >= center, upper - center, center - lower)
        radius_floor = np.maximum(
            np.median(np.abs(sample_traj - center[None, ...]), axis=(0, 1)) * 1e-3,
            1e-6,
        )
        ratio_blocks.append(
            np.abs(true_traj - center)
            / np.maximum(side_radius, radius_floor[None, :, :])
        )

    if not ratio_blocks:
        raise RuntimeError("Validation loader produced no samples for interval calibration.")

    ratios = np.concatenate(ratio_blocks, axis=0)
    n_val = int(ratios.shape[0])
    # Finite-sample upper empirical quantile. Overlapping windows mean this is
    # an empirical validation calibration, not an exchangeability guarantee.
    quantile_level = min(1.0, math.ceil((n_val + 1) * nominal) / max(n_val, 1))
    try:
        raw_scale = np.quantile(ratios, quantile_level, axis=0, method="higher")
    except TypeError:  # NumPy < 1.22
        raw_scale = np.quantile(ratios, quantile_level, axis=0, interpolation="higher")
    scale = np.clip(raw_scale, min_scale, max_scale)

    rows = []
    axis_names = ["x", "y", "z"]
    for step in range(horizon):
        for axis, axis_name in enumerate(axis_names):
            values = ratios[:, step, axis]
            rows.append({
                "step": step + 1,
                "time_s": (step + 1) * float(config["sample_interval_s"]),
                "axis": axis_name,
                "scale": float(scale[step, axis]),
                "raw_scale": float(raw_scale[step, axis]),
                "clipped": bool(not np.isclose(scale[step, axis], raw_scale[step, axis])),
                "val_PIC95_before": float(100.0 * np.mean(values <= 1.0)),
                "val_PIC95_after": float(100.0 * np.mean(values <= scale[step, axis])),
                "n_validation_windows": n_val,
                "nominal_coverage": 100.0 * nominal,
            })
    return scale.astype(np.float64), pd.DataFrame(rows)


@torch.no_grad()
def validation_selection_score(model: nn.Module, model_type: str, loader: DataLoader, delta_scaler: StandardScaler, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_error = 0.0
    total_fde = 0.0
    total_count = 0
    for batch in loader:
        x_dict = {"seq": batch["seq"].to(device, non_blocking=True), "last_delta": batch["last_delta"].to(device, non_blocking=True)}
        target = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=AMP_DTYPE,
            enabled=amp_enabled_for_model(model_type),
        ):
            prediction, _ = point_predict(model, model_type, x_dict)
        pred_traj = deltas_to_trajectories(inverse_deltas(prediction, delta_scaler))
        true_traj = deltas_to_trajectories(inverse_deltas(target, delta_scaler))
        error = np.linalg.norm(pred_traj - true_traj, axis=-1)
        total_error += float(error.sum())
        total_fde += float(error[:, -1].sum())
        total_count += int(error.shape[0])
    ade = total_error / max(total_count * loader.dataset.horizon, 1)
    fde = total_fde / max(total_count, 1)
    return float(ade + 0.5 * fde), float(ade), float(fde)




@torch.no_grad()
def validation_probability_selection_score(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    delta_scaler: StandardScaler,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validation score for probabilistic models, aligned with final sampled evaluation.

    The score is intentionally based on sampled trajectories rather than only
    deterministic z=0/noiseless output, preventing probability models from being
    selected by a criterion different from their reported probabilistic metrics.
    """
    model.eval()
    horizon = int(config["horizon"])
    k_samples = int(config.get("val_prob_samples", 20))
    accumulator = MetricAccumulator(horizon, config, probability_enabled=True)
    for batch in loader:
        x_dict = {"seq": batch["seq"].to(device, non_blocking=True), "last_delta": batch["last_delta"].to(device, non_blocking=True)}
        target = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=AMP_DTYPE,
            enabled=amp_enabled_for_model(model_type),
        ):
            sample_delta = sample_predict(model, model_type, x_dict, k_samples)
        sample_traj = deltas_to_trajectories(inverse_deltas(sample_delta, delta_scaler))
        pred_traj = sample_traj.mean(axis=0)
        true_traj = deltas_to_trajectories(inverse_deltas(target, delta_scaler))
        accumulator.update(pred_traj, true_traj, sample_traj)

    summary, _ = accumulator.summary()
    target_pic = float(config.get("prob_selection_target_pic", 95.0))
    score = (
        float(summary["minADE"])
        + 0.50 * float(summary["minFDE"])
        + float(config.get("prob_selection_energy_weight", 0.05)) * float(summary["EnergyScore"])
        + float(config.get("prob_selection_pic_weight", 0.01)) * abs(float(summary["mPIC95"]) - target_pic)
        + float(config.get("prob_selection_miw_weight", 0.05)) * float(summary["MIW95"])
    )
    summary["ProbSelectionScore"] = float(score)
    return float(score), summary


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    x_dict = {"seq": batch["seq"].to(device, non_blocking=True), "last_delta": batch["last_delta"].to(device, non_blocking=True)}
    target = batch["target"].to(device, non_blocking=True)
    labels = batch["intent"].cpu().numpy()
    return x_dict, target, labels


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    delta_scaler: StandardScaler,
    config: Dict[str, Any],
    device: torch.device,
    calibration_scale: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """最终测试评估。概率模型以 K 条真实样本的均值作为点预测。"""
    model.eval()
    horizon = int(config["horizon"])
    probabilistic = is_probabilistic_model(model_type)
    k_samples = int(config["final_prob_samples"]) if probabilistic else 0

    accumulators = {
        "global": MetricAccumulator(horizon, config, probability_enabled=probabilistic),
        "low": MetricAccumulator(horizon, config, probability_enabled=probabilistic),
        "mid": MetricAccumulator(horizon, config, probability_enabled=probabilistic),
        "high": MetricAccumulator(horizon, config, probability_enabled=probabilistic),
    }
    group_map = {0: "low", 1: "mid", 2: "high"}
    intent_true, intent_pred = [], []

    for batch in tqdm(loader, desc=f"Test {model_type}", leave=False):
        x_dict, target, labels = _batch_to_device(batch, device)
        with torch.amp.autocast(
            device_type="cuda", dtype=AMP_DTYPE,
            enabled=amp_enabled_for_model(model_type),
        ):
            if probabilistic:
                sample_delta = sample_predict(model, model_type, x_dict, k_samples)
                sample_traj = deltas_to_trajectories(inverse_deltas(sample_delta, delta_scaler))
                sample_traj = apply_horizon_interval_calibration(
                    sample_traj,
                    calibration_scale,
                )
                pred_traj = sample_traj.mean(axis=0)
                if model_type == "ours":
                    # 额外计算意图识别结果，但不参与对比模型排名。
                    _, _, logits, _ = model(x_dict)
                    intent_pred.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                    intent_true.extend(labels.tolist())
            else:
                point_delta, _ = point_predict(model, model_type, x_dict)
                pred_traj = deltas_to_trajectories(inverse_deltas(point_delta, delta_scaler))
                sample_traj = None

        true_traj = deltas_to_trajectories(inverse_deltas(target, delta_scaler))
        accumulators["global"].update(pred_traj, true_traj, sample_traj)
        for cls, group_name in group_map.items():
            mask = labels == cls
            if np.any(mask):
                subset_samples = sample_traj[:, mask] if sample_traj is not None else None
                accumulators[group_name].update(pred_traj[mask], true_traj[mask], subset_samples)

    global_summary, global_steps = accumulators["global"].summary()
    group_rows = []
    step_rows = []
    group_names = {"low": "Steady flight", "mid": "Ordinary maneuver", "high": "Sharp maneuver"}

    # Retain the global horizon curve in the same schema as maneuver-specific
    # curves so the paper can report T+1 through T+20 for all test trajectories.
    for row in global_steps:
        row.update({
            "group_key": "global",
            "group_name": "All test trajectories",
        })
        step_rows.append(row)

    for key in ["low", "mid", "high"]:
        summary, steps = accumulators[key].summary()
        summary["group_key"] = key
        summary["group_name"] = group_names[key]
        group_rows.append(summary)
        for row in steps:
            row.update({"group_key": key, "group_name": group_names[key]})
            step_rows.append(row)

    intent_metrics = {"IntentAcc": float("nan"), "IntentMacroF1": float("nan")}
    if len(intent_true) > 0:
        intent_metrics["IntentAcc"] = float(np.mean(np.asarray(intent_true) == np.asarray(intent_pred)) * 100.0)
        intent_metrics["IntentMacroF1"] = float(f1_score(intent_true, intent_pred, average="macro") * 100.0)

    return global_summary, pd.DataFrame(group_rows), pd.DataFrame(step_rows), intent_metrics

# ============================================================================
# 硬件效率
# ============================================================================
class PointPredictionWrapper(nn.Module):
    def __init__(self, model: nn.Module, model_type: str):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, sequence: torch.Tensor, last_delta: torch.Tensor) -> torch.Tensor:
        x_dict = {"seq": sequence, "last_delta": last_delta}
        if self.model_type == "ours":
            return self.model(x_dict)[0]
        return self.model.point_prediction(x_dict)


def _time_callable(fn, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        values = []
        for _ in range(repeats):
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize(device)
            values.append(start.elapsed_time(end))
        return float(np.mean(values))
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = fn()
        values.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(values))


@torch.no_grad()
def measure_hardware(model: nn.Module, model_type: str, feature_dim: int, history_len: int, device: torch.device, config: Dict[str, Any]) -> Dict[str, float]:
    model.eval()
    dummy = {
        "seq": torch.randn(1, history_len, feature_dim, device=device),
        "last_delta": torch.randn(1, 3, device=device),
    }
    point_latency = _time_callable(lambda: point_predict(model, model_type, dummy)[0], device, warmup=5, repeats=int(config["hardware_point_repeats"]))
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    probability_latency = float("nan")
    if is_probabilistic_model(model_type):
        k_hw = int(config["hardware_prob_samples"])
        probability_latency = _time_callable(lambda: sample_predict(model, model_type, dummy, k_hw), device, warmup=1, repeats=int(config["hardware_prob_repeats"]))

    flops_g = float("nan")
    if thop_profile is not None:
        try:
            wrapper = PointPredictionWrapper(model, model_type).to(device)
            macs, _ = thop_profile(wrapper, inputs=(dummy["seq"], dummy["last_delta"]), verbose=False)
            flops_g = float(2.0 * macs / 1e9)
        except Exception:
            pass

    return {
        "Params_M": float(params_m),
        "FLOPs_G": flops_g,
        "Latency_K1_ms": float(point_latency),
        f"Latency_K{int(config['hardware_prob_samples'])}_ms": probability_latency,
    }

# ============================================================================
# 训练与单次运行
# ============================================================================
def build_model(model_type: str, data: Dict[str, Any], config: Dict[str, Any], device: torch.device) -> nn.Module:
    d_model = int(config["d_model"])
    if model_type == "ours":
        model = IntentAwarePredictor(
            feature_dim=data["feature_dim"],
            d_model=d_model,
            horizon=data["horizon"],
            scales=config["scales"],
            mamba_main_dir=config.get("mamba_main_dir"),
        )
    else:
        model = BaselineWrapper(
            model_type=model_type,
            feature_dim=data["feature_dim"],
            d_model=d_model,
            horizon=data["horizon"],
            history_len=data["history_len"],
            diffusion_steps=int(config["diffusion_steps"]),
            mamba_main_dir=config.get("mamba_main_dir"),
        )
    return model.to(device)


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with a validation-fixed stable LR for the recurrent CVAE."""
    model_type = str(config.get("model_type", "")).lower()
    if model_type == "trajectronpp":
        learning_rate = float(
            config.get("trajectronpp_learning_rate", config["learning_rate"])
        )
    else:
        learning_rate = float(config["learning_rate"])
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )


def compute_training_loss(model: nn.Module, model_type: str, x_dict: Dict[str, torch.Tensor], target: torch.Tensor, labels: torch.Tensor, progress: float, tf_prob: float):
    if model_type == "ours":
        tf_mask = torch.rand(target.shape[0], target.shape[1], device=target.device) < tf_prob
        prediction, logvar, intent_logits, _, aux = model(
            x_dict,
            tau=max(1.0, 3.0 - 2.0 * progress),
            target_deltas=target,
            teacher_forcing_mask=tf_mask,
            return_aux=True,
        )
        return ours_loss(prediction, logvar, target, intent_logits, labels, aux)

    if model_type == "cvae":
        prediction, posterior_mu, posterior_logvar, prior_mu, prior_logvar = model(x_dict, target=target)
        return cvae_loss(prediction, target, posterior_mu, posterior_logvar, prior_mu, prior_logvar, progress)

    if model_type == "diffusion":
        x0, predicted_noise, true_noise = model(x_dict, target=target)
        return diffusion_loss(x0, predicted_noise, true_noise, target)

    if model_type in {"gaussian_lstm", "gaussian_transformer"}:
        mean, logvar = model(x_dict)
        return gaussian_trajectory_loss(mean, logvar, target)

    if model_type == "deepar":
        mean, logvar = model(x_dict, target=target, tf_prob=tf_prob)
        return gaussian_trajectory_loss(mean, logvar, target)

    if model_type == "tft_quantile":
        quantile_prediction = model(x_dict)
        return quantile_pinball_loss(quantile_prediction, target, model.net.quantiles)

    if model_type == "trajectronpp":
        prediction, posterior_mu, posterior_logvar, prior_mu, prior_logvar, prior_logits = model(x_dict, target=target, tf_prob=tf_prob)
        return trajectronpp_loss(prediction, target, posterior_mu, posterior_logvar, prior_mu, prior_logvar, prior_logits, progress)

    prediction = model(x_dict)
    return trajectory_loss_base(prediction, target)


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    """Avoid half-written checkpoints when a process is interrupted."""
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore process RNGs. Dataset augmentation remains epoch-index deterministic."""
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        print(f"⚠️ RNG状态恢复失败，将继续训练：{exc}")


def _resume_signature(run_config: Dict[str, Any], data: Dict[str, Any]) -> str:
    """
    Signature excludes epochs so that a 10-epoch debug run can be extended
    to 300 epochs, but rejects incompatible model/data/task checkpoints.
    """
    keys = [
        "model_name", "model_type", "seed", "split_seed", "sample_interval_s",
        "history_len", "horizon", "d_model", "diffusion_steps",
        "learning_rate", "weight_decay", "train_batch_size",
        "high_maneuver_curriculum", "probabilistic_selection_on_val",
        "val_prob_samples", "diffusion_steps", "output_dir",
        "feature_layout", "position_filter", "causal_savgol_window",
        "causal_savgol_polyorder",
    ]
    payload = {key: run_config.get(key) for key in keys}
    payload["data_feature_dim"] = int(data["feature_dim"])
    payload["data_history_len"] = int(data["history_len"])
    payload["data_horizon"] = int(data["horizon"])
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    selection_score: float,
    config: Dict[str, Any],
) -> None:
    _atomic_torch_save(
        {
            "epoch": int(epoch),
            "selection_score": float(selection_score),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def save_training_state(
    path: str,
    *,
    completed_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    best_score: float,
    best_epoch: int,
    history_rows: List[Dict[str, Any]],
    resume_signature: str,
) -> None:
    """Persist enough state to continue after a crash without restarting Ours."""
    _atomic_torch_save(
        {
            "state_version": 1,
            "completed_epoch": int(completed_epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "amp_scaler_state_dict": amp_scaler.state_dict(),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "history_rows": history_rows,
            "rng_state": _capture_rng_state(),
            "resume_signature": resume_signature,
        },
        path,
    )


def _load_trusted_checkpoint(path: str, map_location: Any) -> Dict[str, Any]:
    """Load a checkpoint generated by this experiment script.

    PyTorch 2.6 defaults torch.load to weights_only=True. Full training-state
    checkpoints also contain optimizer, scheduler, history, and NumPy RNG
    objects, so trusted local checkpoints require weights_only=False. The
    fallback keeps compatibility with older PyTorch releases that do not expose
    the argument.
    """
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_training_state(
    path: str,
    *,
    expected_signature: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    strict_signature: bool = True,
) -> Tuple[int, float, int, List[Dict[str, Any]]]:
    """
    Return (start_epoch, best_score, best_epoch, history_rows).
    Start from completed_epoch + 1.
    """
    state = _load_trusted_checkpoint(path, map_location="cpu")
    stored_signature = state.get("resume_signature")
    if strict_signature and stored_signature != expected_signature:
        raise RuntimeError(
            "训练状态与当前模型/数据协议不匹配。"
            "请设置 resume=False 或更换输出目录。"
        )

    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if "amp_scaler_state_dict" in state:
        amp_scaler.load_state_dict(state["amp_scaler_state_dict"])
    _restore_rng_state(state.get("rng_state"))

    completed_epoch = int(state.get("completed_epoch", -1))
    history_rows = list(state.get("history_rows", []))
    return (
        completed_epoch + 1,
        float(state.get("best_score", float("inf"))),
        int(state.get("best_epoch", -1)),
        history_rows,
    )



def train_single_run(run_config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    单个模型×随机种子运行。

    具备两级防重复机制：
    1. run_repeated_comparison()会跳过已有完成标记的任务；
    2. 本函数会从last_training_state.pt恢复中断训练，而不是重新从epoch=1开始。
    """
    model_name = run_config["model_name"]
    model_type = run_config["model_type"]
    seed = int(run_config["seed"])
    run_dir = ensure_dir(run_config["run_dir"])
    seed_everything(seed, deterministic=bool(run_config["deterministic"]))
    save_json(os.path.join(run_dir, "config.json"), run_config)

    completion_path = os.path.join(run_dir, "completed.json")
    metric_path = os.path.join(run_dir, "test_global_metrics.csv")
    if bool(run_config.get("resume", True)) and os.path.exists(completion_path) and os.path.exists(metric_path):
        completed = pd.read_csv(metric_path)
        if len(completed) > 0:
            print(f"⏭️ 已完成，直接读取结果：{model_name} | seed={seed}")
            return completed.iloc[-1].to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = max(0, int(run_config["num_workers"]))

    dataset_scaler_kwargs = {
        "delta_scaler": data["delta_scaler"],
        "feature_scaler": data["feature_scaler"],
        "scale_mask": data["scale_mask"],
        "rotate_before_normalize": True,
    }
    train_dataset = OnTheFlyUAVDataset(
        data["train_seqs"], data["train_records"],
        data["history_len"], data["horizon"],
        is_training=True, augmentation_seed=seed,
        **dataset_scaler_kwargs,
    )
    val_dataset = OnTheFlyUAVDataset(
        data["val_seqs"], data["val_records"],
        data["history_len"], data["horizon"],
        is_training=False, augmentation_seed=seed + 1,
        **dataset_scaler_kwargs,
    )
    test_dataset = OnTheFlyUAVDataset(
        data["test_seqs"], data["test_records"],
        data["history_len"], data["horizon"],
        is_training=False, augmentation_seed=seed + 2,
        **dataset_scaler_kwargs,
    )

    loader_bundle = make_loader_bundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_records=data["train_records"],
        train_batch_size=int(run_config["train_batch_size"]),
        eval_batch_size=int(run_config["eval_batch_size"]),
        seed=seed,
        num_workers=workers,
        persistent_workers=bool(run_config.get("dataloader_persistent_workers", True)),
        prefetch_factor=int(run_config.get("dataloader_prefetch_factor", 1)),
        high_maneuver_curriculum=bool(run_config.get("high_maneuver_curriculum", False)),
    )
    train_loader = loader_bundle.train_loader
    val_loader = loader_bundle.val_loader
    test_loader = loader_bundle.test_loader

    model = None
    try:
        model = build_model(model_type, data, run_config, device)
        optimizer = build_optimizer(model, run_config)
        warmup_epochs = min(
            int(run_config["warmup_epochs"]),
            max(int(run_config["epochs"]) - 1, 1),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1)),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max(int(run_config["epochs"]) - warmup_epochs, 1),
                    eta_min=float(run_config["eta_min"]),
                ),
            ],
            milestones=[max(warmup_epochs, 1)],
        )
        run_amp_enabled = amp_enabled_for_model(model_type)
        amp_scaler = torch.amp.GradScaler("cuda", enabled=run_amp_enabled)

        best_score = float("inf")
        best_epoch = -1
        start_epoch = 0
        history_rows: List[Dict[str, Any]] = []
        checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")
        state_path = os.path.join(run_dir, "last_training_state.pt")
        signature = _resume_signature(run_config, data)

        if bool(run_config.get("resume", True)) and os.path.exists(state_path):
            try:
                start_epoch, best_score, best_epoch, history_rows = load_training_state(
                    state_path,
                    expected_signature=signature,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    amp_scaler=amp_scaler,
                    strict_signature=bool(run_config.get("resume_strict_signature", True)),
                )
                print(
                    f"♻️ 继续训练：{model_name} | seed={seed} | "
                    f"从 epoch={start_epoch + 1} 开始；"
                    f"已保存最佳 epoch={best_epoch}"
                )
            except Exception as exc:
                print(f"⚠️ 无法恢复已有状态，将从头训练：{exc}")
                if bool(run_config.get("discard_incompatible_resume", False)):
                    try:
                        os.remove(state_path)
                    except OSError:
                        pass
                start_epoch = 0
                best_score = float("inf")
                best_epoch = -1
                history_rows = []

        if start_epoch >= int(run_config["epochs"]):
            print(
                f"ℹ️ 训练已完成到目标epochs={run_config['epochs']}，"
                "直接执行最终测试评估。"
            )

        save_interval = max(1, int(run_config.get("resume_save_interval", 5)))

        for epoch in range(start_epoch, int(run_config["epochs"])):
            loader_bundle.set_epoch(epoch, int(run_config["epochs"]))
            model.train()
            progress = epoch / max(int(run_config["epochs"]) - 1, 1)
            tf_prob = float(run_config["tf_start"]) + (
                float(run_config["tf_end"]) - float(run_config["tf_start"])
            ) * progress

            if model_type == "ours":
                model.goal_correction_strength = max(
                    0.0, min(1.0, (progress - 0.30) / 0.30)
                )

            train_loss_sum = 0.0
            train_batches = 0
            last_batch_refs = None

            for batch_index, batch in enumerate(train_loader):
                x_dict, target, _ = _batch_to_device(batch, device)
                labels = batch["intent"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type="cuda", dtype=AMP_DTYPE,
                    enabled=run_amp_enabled,
                ):
                    loss = compute_training_loss(
                        model, model_type, x_dict, target, labels, progress, tf_prob
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss before backward: model={model_type}, "
                        f"epoch={epoch + 1}, batch={batch_index}, loss={loss.detach().item()}."
                    )
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                clip_norm = (
                    float(run_config.get("trajectronpp_grad_clip_norm", 0.5))
                    if model_type == "trajectronpp"
                    else float(run_config["grad_clip_norm"])
                )
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_norm,
                )
                if not torch.isfinite(total_grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite gradient norm: model={model_type}, "
                        f"epoch={epoch + 1}, batch={batch_index}."
                    )
                amp_scaler.step(optimizer)
                amp_scaler.update()

                if model_type == "trajectronpp" and batch_index % 50 == 0:
                    invalid_parameters = [
                        name for name, parameter in model.named_parameters()
                        if not torch.isfinite(parameter).all()
                    ]
                    if invalid_parameters:
                        raise FloatingPointError(
                            "Non-finite TrajectronPP parameters after optimizer step: "
                            + ", ".join(invalid_parameters[:8])
                        )

                if model_type == "ours":
                    with torch.no_grad():
                        model.output_scale.data.clamp_(0.5, 1.5)

                train_loss_sum += float(loss.item())
                train_batches += 1
                last_batch_refs = (batch, x_dict, target, labels, loss)

            if last_batch_refs is not None:
                del last_batch_refs
            gc.collect()

            model.eval()
            point_selection_score, val_ade, val_fde = validation_selection_score(
                model, model_type, val_loader, data["delta_scaler"], device
            )
            prob_selection_enabled = (
                is_probabilistic_model(model_type)
                and bool(run_config.get("probabilistic_selection_on_val", True))
            )
            selection_mode = "point_ADE_0p5FDE"
            selection_score = point_selection_score
            checkpoint_eligible = True
            prob_val = {"minADE": np.nan, "minFDE": np.nan, "EnergyScore": np.nan, "mPIC95": np.nan, "MIW95": np.nan, "ProbSelectionScore": np.nan}
            do_prob_val = (
                prob_selection_enabled
                and (
                    epoch == start_epoch
                    or (epoch + 1) % int(run_config.get("val_prob_eval_interval", 5)) == 0
                    or epoch + 1 == int(run_config["epochs"])
                )
            )
            if do_prob_val:
                selection_score, prob_val = validation_probability_selection_score(
                    model, model_type, val_loader, data["delta_scaler"], run_config, device
                )
                selection_mode = "probabilistic_sampled_val"
            elif prob_selection_enabled:
                # Do not compare point-score epochs against probabilistic-score epochs.
                selection_score = float("nan")
                selection_mode = "probabilistic_sampled_val_skipped"
                checkpoint_eligible = False
            scheduler.step()

            history_rows.append({
                "epoch": epoch + 1,
                "train_loss": train_loss_sum / max(train_batches, 1),
                "val_ADE": val_ade,
                "val_FDE": val_fde,
                "val_point_selection_score": point_selection_score,
                "val_minADE": prob_val.get("minADE", np.nan),
                "val_minFDE": prob_val.get("minFDE", np.nan),
                "val_EnergyScore": prob_val.get("EnergyScore", np.nan),
                "val_mPIC95": prob_val.get("mPIC95", np.nan),
                "val_MIW95": prob_val.get("MIW95", np.nan),
                "selection_score": selection_score,
                "selection_mode": selection_mode,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "teacher_forcing_probability": tf_prob,
            })

            if checkpoint_eligible and np.isfinite(selection_score) and selection_score < best_score:
                best_score = selection_score
                best_epoch = epoch + 1
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    best_epoch,
                    best_score,
                    run_config,
                )

            pd.DataFrame(history_rows).to_csv(
                os.path.join(run_dir, "training_history.csv"),
                index=False,
            )

            if (
                (epoch + 1) % save_interval == 0
                or epoch + 1 == int(run_config["epochs"])
            ):
                save_training_state(
                    state_path,
                    completed_epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    amp_scaler=amp_scaler,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    history_rows=history_rows,
                    resume_signature=signature,
                )

            if (
                (epoch + 1) % int(run_config["log_interval"]) == 0
                or epoch == start_epoch
                or epoch + 1 == int(run_config["epochs"])
            ):
                fd_count = _count_open_file_descriptors()
                fd_suffix = f" fd={fd_count}" if fd_count >= 0 else ""
                print(
                    f"[{model_name} | seed={seed}] epoch={epoch + 1:03d} "
                    f"train={history_rows[-1]['train_loss']:.4f} "
                    f"valADE={val_ade:.4f} valFDE={val_fde:.4f} "
                    f"select={selection_score:.4f}{fd_suffix}"
                )

            gc.collect()

        if not os.path.exists(checkpoint_path):
            raise RuntimeError("未产生best_checkpoint.pt，无法进行最终测试。")

        checkpoint = _load_trusted_checkpoint(
            checkpoint_path,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        calibration_scale = None
        calibration_df = pd.DataFrame()
        if (
            is_probabilistic_model(model_type)
            and bool(run_config.get("calibrate_intervals_on_val", True))
        ):
            # Keep calibration and final Monte Carlo streams deterministic and
            # independent of how many training epochs were executed/resumed.
            seed_everything(seed + 40000, deterministic=bool(run_config["deterministic"]))
            calibration_scale, calibration_df = fit_horizon_interval_calibration(
                model,
                model_type,
                val_loader,
                data["delta_scaler"],
                run_config,
                device,
            )
            calibration_df.insert(0, "Model", model_name)
            calibration_df.insert(1, "Seed", seed)
            calibration_df.to_csv(
                os.path.join(run_dir, "validation_interval_calibration.csv"),
                index=False,
            )
            np.save(
                os.path.join(run_dir, "validation_interval_scale.npy"),
                calibration_scale,
            )

        seed_everything(seed + 50000, deterministic=bool(run_config["deterministic"]))

        global_metrics, group_df, step_df, intent_metrics = evaluate_test_set(
            model,
            model_type,
            test_loader,
            data["delta_scaler"],
            run_config,
            device,
            calibration_scale=calibration_scale,
        )
        hardware = measure_hardware(
            model,
            model_type,
            data["feature_dim"],
            data["history_len"],
            device,
            run_config,
        )

        group_df.insert(0, "Model", model_name)
        group_df.insert(1, "Seed", seed)
        step_df.insert(0, "Model", model_name)
        step_df.insert(1, "Seed", seed)
        group_df.to_csv(os.path.join(run_dir, "test_group_summary.csv"), index=False)
        step_df.to_csv(os.path.join(run_dir, "test_step_metrics.csv"), index=False)

        result: Dict[str, Any] = {
            "Model": model_name,
            "ModelType": model_type,
            "Seed": seed,
            "RunStatus": "completed",
            "BestEpoch": int(best_epoch),
            "SelectionScore": float(best_score),
            "SelectionMode": "probabilistic_sampled_val" if (is_probabilistic_model(model_type) and bool(run_config.get("probabilistic_selection_on_val", True))) else "point_ADE_0p5FDE",
            **global_metrics,
            **intent_metrics,
            **hardware,
        }
        for _, row in group_df.iterrows():
            prefix = str(row["group_key"]).upper()
            for metric in [
                "ADE", "FDE", "RPE5", "RPE10", "DistAcc", "DirAcc",
                "Ev", "Ea", "Ej", "A_KEER", "NJERDev",
                "ViolationRate", "minADE", "minFDE", "EnergyScore",
                "NLL", "PIC95", "mPIC95", "MIW95",
            ]:
                result[f"{prefix}_{metric}"] = finite_or_nan(row.get(metric, np.nan))

        pd.DataFrame([result]).to_csv(metric_path, index=False)
        save_json(
            completion_path,
            {
                "status": "completed",
                "model_name": model_name,
                "model_type": model_type,
                "seed": seed,
                "best_epoch": int(best_epoch),
                "result_file": "test_global_metrics.csv",
            },
        )
        print(
            f"✅ [{model_name} | seed={seed}] test: "
            f"ADE={result['ADE']:.4f}, FDE={result['FDE']:.4f}, "
            f"NLL={result['NLL']:.4f}, PIC95={result['mPIC95']:.2f}%, "
            f"MIW95={result['MIW95']:.4f}, NJERDev={result['NJERDev']:.4f}"
        )
        return result

    finally:
        # 无论成功、崩溃或Ctrl+C，都终止persistent worker，防止跨run资源残留。
        try:
            loader_bundle.close()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# 多种子实验与汇总
# ============================================================================
def _format_mean_std(mean: float, std: float, precision: int = 4) -> str:
    if not np.isfinite(mean):
        return "—"
    if not np.isfinite(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"



def summarize_comparison(raw_df: pd.DataFrame, root_dir: str, final_prob_samples: int) -> pd.DataFrame:
    """汇总成功run；即使全部失败也只输出提示文件而不再次崩溃。"""
    valid = raw_df.copy()
    if "Error" in valid.columns:
        valid = valid[valid["Error"].isna()].copy()

    if valid.empty:
        failure_path = os.path.join(root_dir, "comparison_all_runs_failed.csv")
        raw_df.to_csv(failure_path, index=False, encoding="utf-8-sig")
        print(
            "⚠️ 所有运行均失败，未生成均值±标准差表。"
            f"失败记录已保存：{failure_path}"
        )
        return pd.DataFrame()

    numeric_cols = [
        col for col in valid.select_dtypes(include=[np.number]).columns
        if col not in {"Seed", "BestEpoch"}
    ]
    if not numeric_cols:
        failure_path = os.path.join(root_dir, "comparison_no_numeric_metrics.csv")
        valid.to_csv(failure_path, index=False, encoding="utf-8-sig")
        print(f"⚠️ 没有可汇总的数值指标：{failure_path}")
        return pd.DataFrame()

    summary = valid.groupby(["Model", "ModelType"], dropna=False)[numeric_cols].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(root_dir, "comparison_mean_std.csv"), index=False)

    def make_paper_table(
        metrics: List[str],
        output_name: str,
        include_types: Optional[List[str]] = None,
    ) -> None:
        frame = summary.copy()
        if include_types is not None:
            frame = frame[frame["ModelType"].isin(include_types)].copy()
        table = pd.DataFrame({"Model": frame["Model"]})
        for metric in metrics:
            mean_col, std_col = f"{metric}_mean", f"{metric}_std"
            if mean_col in frame.columns:
                std_values = frame[std_col] if std_col in frame.columns else pd.Series(np.nan, index=frame.index)
                table[metric] = [
                    _format_mean_std(mean, std)
                    for mean, std in zip(frame[mean_col], std_values)
                ]
        table.to_csv(
            os.path.join(root_dir, output_name), index=False, encoding="utf-8-sig"
        )

    make_paper_table(
        ["ADE", "FDE", "RPE5", "RPE10", "DistAcc", "DirAcc", "Params_M", "Latency_K1_ms"],
        "paper_table_deterministic.csv",
    )
    make_paper_table(
        ["NLL", "PIC95", "MIW95", "minADE", "minFDE", "EnergyScore", f"Latency_K{final_prob_samples}_ms"],
        "paper_table_probabilistic.csv",
        [
            "ours", "mc_dropout", "cvae", "diffusion",
            "gaussian_lstm", "gaussian_transformer", "deepar",
            "tft_quantile", "trajectronpp",
        ],
    )
    make_paper_table(
        ["Ev", "Ea", "Ej", "A_KEER", "NJERDev", "ViolationRate"],
        "paper_table_kinematic.csv",
    )
    # One consolidated global table matching the manuscript's primary results:
    # point accuracy, calibrated uncertainty, and physical consistency.
    make_paper_table(
        [
            "ADE", "FDE", "NLL", "PIC95", "MIW95",
            "Ev", "Ea", "Ej", "A_KEER", "NJERDev", "ViolationRate",
        ],
        "paper_table_global_main.csv",
    )
    return summary


def aggregate_maneuver_tables(root_dir: str, raw_df: pd.DataFrame) -> None:
    """Aggregate steady/ordinary/sharp deterministic and probability metrics."""
    frames = []
    valid = raw_df.copy()
    if "Error" in valid.columns:
        valid = valid[valid["Error"].isna()].copy()
    for _, row in valid.iterrows():
        path = os.path.join(
            root_dir,
            "runs",
            safe_name(row["Model"]),
            f"seed_{int(row['Seed'])}",
            "test_group_summary.csv",
        )
        if os.path.exists(path):
            frame = pd.read_csv(path)
            frame["ModelType"] = row["ModelType"]
            frames.append(frame)
    if not frames:
        return

    groups = pd.concat(frames, ignore_index=True)
    groups.to_csv(
        os.path.join(root_dir, "maneuver_metrics_all_seeds.csv"),
        index=False,
    )
    metrics = ["n_samples", "ADE", "FDE", "NLL", "PIC95", "MIW95"]
    metrics = [metric for metric in metrics if metric in groups.columns]
    summary = groups.groupby(
        ["Model", "ModelType", "group_key", "group_name"],
        dropna=False,
    )[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join([part for part in column if part])
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(
        os.path.join(root_dir, "maneuver_metrics_mean_std.csv"),
        index=False,
    )

    table = summary[["Model", "ModelType", "group_key", "group_name"]].copy()
    for metric in ["ADE", "FDE", "NLL", "PIC95", "MIW95"]:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col in summary.columns:
            table[metric] = [
                _format_mean_std(mean, std)
                for mean, std in zip(summary[mean_col], summary[std_col])
            ]
    if "n_samples_mean" in summary.columns:
        table["n_samples"] = summary["n_samples_mean"].round().astype("Int64")
    table.to_csv(
        os.path.join(root_dir, "paper_table_maneuver_state.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def plot_stepwise_probability_curves(
    step_summary: pd.DataFrame,
    root_dir: str,
) -> None:
    """Export a publication-ready global T+1...T+20 three-panel figure."""
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: stepwise CSV files were saved, but plotting was skipped: {exc}")
        return

    global_rows = step_summary[step_summary["group_key"] == "global"].copy()
    probabilistic_types = {
        "ours", "gaussian_lstm", "gaussian_transformer", "deepar",
        "tft_quantile", "trajectronpp",
    }
    if "ModelType" in global_rows.columns:
        global_rows = global_rows[
            global_rows["ModelType"].isin(probabilistic_types)
        ].copy()
    if global_rows.empty:
        return

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    colors = ["#3B6FB6", "#D07C32", "#3A8D74", "#A75D8B", "#7A7A7A", "#B6A23B"]
    markers = ["o", "s", "^", "D", "v", "P"]
    models = list(dict.fromkeys(global_rows["Model"].tolist()))
    if "Ours_Full" in models:
        models = ["Ours_Full"] + [name for name in models if name != "Ours_Full"]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)
    panels = [
        ("DistErr", "Distance error (m)"),
        ("PIC95", r"PIC$_{95}$ (%)"),
        ("MIW95", r"MIW$_{95}$ (m)"),
    ]
    for panel_index, (metric, ylabel) in enumerate(panels):
        axis = axes[panel_index]
        for model_index, model_name in enumerate(models):
            subset = global_rows[global_rows["Model"] == model_name].sort_values("step")
            mean_col, std_col = f"{metric}_mean", f"{metric}_std"
            if subset.empty or mean_col not in subset.columns:
                continue
            y = subset[mean_col].to_numpy(dtype=float)
            if not np.any(np.isfinite(y)):
                continue
            x = subset["step"].to_numpy(dtype=float)
            color = colors[model_index % len(colors)]
            linewidth = 1.8 if model_name == "Ours_Full" else 1.1
            zorder = 5 if model_name == "Ours_Full" else 2
            axis.plot(
                x, y,
                color=color,
                marker=markers[model_index % len(markers)],
                markersize=2.4,
                markevery=2,
                linewidth=linewidth,
                label=model_name.replace("_", " "),
                zorder=zorder,
            )
            if std_col in subset.columns:
                std = subset[std_col].to_numpy(dtype=float)
                if np.any(np.isfinite(std)):
                    axis.fill_between(x, y - std, y + std, color=color, alpha=0.12, linewidth=0)
        if metric == "PIC95":
            axis.axhline(95.0, color="#555555", linestyle="--", linewidth=0.9, label="Nominal 95%")
            axis.set_ylim(0.0, 100.5)
        axis.set_xlabel("Prediction step")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, int(global_rows["step"].max()))
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
        axis.text(-0.18, 1.03, chr(ord("a") + panel_index), transform=axis.transAxes, fontweight="bold", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    handles_pic, labels_pic = axes[1].get_legend_handles_labels()
    if "Nominal 95%" in labels_pic:
        index = labels_pic.index("Nominal 95%")
        handles.append(handles_pic[index])
        labels.append(labels_pic[index])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=4, columnspacing=1.2, handlelength=2.2)
    base = os.path.join(root_dir, "fig_stepwise_global_probability")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def aggregate_step_curves(root_dir: str, raw_df: pd.DataFrame) -> None:
    all_steps = []
    valid = raw_df.copy()
    if "Error" in valid.columns:
        valid = valid[valid["Error"].isna()].copy()
    for _, row in valid.iterrows():
        path = os.path.join(root_dir, "runs", safe_name(row["Model"]), f"seed_{int(row['Seed'])}", "test_step_metrics.csv")
        if os.path.exists(path):
            frame = pd.read_csv(path)
            frame["ModelType"] = row["ModelType"]
            all_steps.append(frame)
    if len(all_steps) == 0:
        return
    steps = pd.concat(all_steps, ignore_index=True)
    steps.to_csv(os.path.join(root_dir, "step_curves_all_seeds.csv"), index=False)
    metrics = ["DistErr", "RMSE", "DistAcc", "DirAcc", "NLL", "PIC95", "MIW95"]
    metrics = [metric for metric in metrics if metric in steps.columns]
    grouped = steps.groupby(
        ["Model", "ModelType", "group_key", "group_name", "step", "time_s"],
        dropna=False,
    )[metrics].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in grouped.columns]
    grouped.to_csv(os.path.join(root_dir, "step_curves_mean_std.csv"), index=False)
    paper_columns = [
        "Model", "ModelType", "step", "time_s",
        "DistErr_mean", "DistErr_std", "NLL_mean", "NLL_std",
        "PIC95_mean", "PIC95_std", "MIW95_mean", "MIW95_std",
    ]
    paper_global = grouped[grouped["group_key"] == "global"][
        [column for column in paper_columns if column in grouped.columns]
    ].copy()
    paper_global.to_csv(
        os.path.join(root_dir, "paper_stepwise_global_probability.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    plot_stepwise_probability_curves(grouped, root_dir)


def _run_key(model_name: str, model_type: str, seed: int) -> Tuple[str, str, int]:
    return str(model_name), str(model_type), int(seed)


def _result_frame_from_records(records: Dict[Tuple[str, str, int], Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(list(records.values()))
    sort_cols = [col for col in ["Model", "ModelType", "Seed"] if col in frame.columns]
    return frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame


def _load_existing_run_records(raw_path: str) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """Load prior seed results so restarting the script does not restart Ours_Full."""
    records: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    if not os.path.exists(raw_path):
        return records
    try:
        old = pd.read_csv(raw_path)
    except Exception as exc:
        print(f"⚠️ 无法读取已有结果表，将从空表开始：{exc}")
        return records

    required = {"Model", "ModelType", "Seed"}
    if not required.issubset(old.columns):
        return records

    for _, row in old.iterrows():
        record = row.to_dict()
        key = _run_key(record["Model"], record["ModelType"], int(record["Seed"]))
        records[key] = record
    return records


def _load_completed_result_from_disk(
    root_dir: str,
    model_name: str,
    model_type: str,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Recover completed metrics even when comparison_seed_results.csv was lost."""
    run_dir = os.path.join(root_dir, "runs", safe_name(model_name), f"seed_{int(seed)}")
    completion_path = os.path.join(run_dir, "completed.json")
    metric_path = os.path.join(run_dir, "test_global_metrics.csv")
    if not (os.path.exists(completion_path) and os.path.exists(metric_path)):
        return None
    try:
        marker = json.load(open(completion_path, "r", encoding="utf-8"))
        if marker.get("status") != "completed":
            return None
        frame = pd.read_csv(metric_path)
        if frame.empty:
            return None
        record = frame.iloc[-1].to_dict()
        record.update(
            {
                "Model": model_name,
                "ModelType": model_type,
                "Seed": int(seed),
                "RunStatus": "completed",
            }
        )
        return record
    except Exception:
        return None


def _iter_tasks(
    experiments: List[Dict[str, Any]],
    seeds: List[int],
    order: str,
) -> Iterator[Tuple[Dict[str, Any], int]]:
    """
    seed_major: Ours(seed2026) then all baselines(seed2026), then next seed.
    model_major: all Ours seeds before LSTM; this is the original behavior.
    """
    if order == "seed_major":
        for seed in seeds:
            for experiment in experiments:
                yield experiment, int(seed)
    elif order == "model_major":
        for experiment in experiments:
            for seed in seeds:
                yield experiment, int(seed)
    else:
        raise ValueError("run_order must be 'seed_major' or 'model_major'.")


def run_repeated_comparison(
    base_config: Dict[str, Any],
    data: Dict[str, Any],
    experiments: List[Dict[str, Any]],
    seeds: List[int],
) -> pd.DataFrame:
    """
    Multi-seed scheduler with:
    - configurable task order;
    - completed-run skipping;
    - failed-run retry control;
    - interruption-safe immediate result persistence.

    It prevents a fresh interpreter launch from retraining Ours_Full seed=2026
    when that task already has a completed marker/result file.
    """
    root_dir = ensure_dir(base_config["output_dir"])
    raw_path = os.path.join(root_dir, "comparison_seed_results.csv")
    records = _load_existing_run_records(raw_path)

    selected_models = base_config.get("models_to_run")
    selected_models = set(selected_models) if selected_models else None
    retry_failed = bool(base_config.get("retry_failed_runs", True))
    resume_enabled = bool(base_config.get("resume", True))
    order = str(base_config.get("run_order", "seed_major"))
    max_runs = base_config.get("max_runs_this_session")
    max_runs = None if max_runs in (None, 0, "") else int(max_runs)
    executed = 0

    for experiment, seed in _iter_tasks(experiments, seeds, order):
        model_name = experiment["model_name"]
        model_type = experiment["model_type"]

        if selected_models is not None and model_name not in selected_models:
            continue

        key = _run_key(model_name, model_type, seed)
        existing = records.get(key)
        disk_completed = _load_completed_result_from_disk(
            root_dir,
            model_name,
            model_type,
            seed,
        )

        if resume_enabled and disk_completed is not None:
            records[key] = disk_completed
            print(f"⏭️ 跳过已完成任务：{model_name} | seed={seed}")
            _result_frame_from_records(records).to_csv(raw_path, index=False)
            continue

        if existing is not None:
            status = str(existing.get("RunStatus", "")).lower()
            has_error = pd.notna(existing.get("Error", np.nan))
            is_completed = (status == "completed") or (
                not has_error and pd.notna(existing.get("ADE", np.nan))
            )
            if resume_enabled and is_completed:
                print(f"⏭️ 跳过结果表中已完成任务：{model_name} | seed={seed}")
                continue
            if has_error and not retry_failed:
                print(f"⏭️ 跳过历史失败任务：{model_name} | seed={seed}")
                continue

        if max_runs is not None and executed >= max_runs:
            print(f"⏸️ 已达到 max_runs_this_session={max_runs}，安全停止。")
            break

        run_dir = os.path.join(
            root_dir,
            "runs",
            safe_name(model_name),
            f"seed_{seed}",
        )
        run_config = dict(base_config)
        run_config.update(experiment)
        run_config.update(
            {
                "seed": int(seed),
                "run_dir": run_dir,
            }
        )

        print("\n" + "=" * 100)
        print(
            f"Model={model_name} | Type={model_type} | Seed={seed} "
            f"| order={order} | fd={_count_open_file_descriptors()}"
        )
        print("=" * 100)

        try:
            metrics = train_single_run(run_config, data)
            metrics["Model"] = model_name
            metrics["ModelType"] = model_type
            metrics["Seed"] = int(seed)
            metrics["RunStatus"] = "completed"
            metrics.pop("Error", None)
            records[key] = metrics
        except KeyboardInterrupt:
            _result_frame_from_records(records).to_csv(raw_path, index=False)
            print("\n⏸️ 收到中断信号。训练状态已在各run目录保存；下次执行将自动续训。")
            raise
        except Exception as exc:
            print(f"❌ 运行失败：{model_name} | seed={seed}\n{exc}")
            records[key] = {
                "Model": model_name,
                "ModelType": model_type,
                "Seed": int(seed),
                "RunStatus": "failed",
                "Error": repr(exc),
            }
        finally:
            _result_frame_from_records(records).to_csv(raw_path, index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(float(base_config.get("inter_run_sleep_s", 2.0)))

        executed += 1

    raw_df = _result_frame_from_records(records)
    raw_df.to_csv(raw_path, index=False)
    summarize_comparison(raw_df, root_dir, int(base_config["final_prob_samples"]))
    aggregate_maneuver_tables(root_dir, raw_df)
    aggregate_step_curves(root_dir, raw_df)
    return raw_df


# ============================================================================
# 主程序
# ============================================================================
def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # -------------------------- 统一实验配置 --------------------------
    CONFIG: Dict[str, Any] = {
        "data_dir": "uavdataset/",
        # Reuse the unchanged causal 12-D feature cache, but isolate all new
        # checkpoints and result files from the previous metric schema.
        "cache_dir": "Compare_5Hz_125_Causal12D_SG7P2_Seed2026_Corrected/cache",
        "output_dir": "Compare_5Hz_125_Causal12D_SG7P2_Seed2026_StepwiseProb",
        "reference_split_path": "Compare_Extended_GenerativeBaselines_5Hz_125/cache/trajectory_id_split_70_10_20.json",
        "mamba_main_dir": "./mamba-main",

        # 固定数据划分与任务设置
        "split_seed": 42,
        "history_len": 45,
        "horizon": 20,
        "scales": [5, 10, 20, 30, 45],
        "train_stride": 5,
        "eval_stride": 2,
        "force_rebuild_cache": False,
        "force_rebuild_split": False,
        "feature_workers": 4,
        "feature_layout": FEATURE_LAYOUT_VERSION,
        # Right-edge SG(7,2) uses only the current and six past observations.
        "position_filter": "causal_savgol",
        "causal_savgol_window": 7,
        "causal_savgol_polyorder": 2,

        # 公平训练设置：所有模型统一
        "epochs": 200,
        "train_batch_size": 1024,
        "eval_batch_size": 128,
        "num_workers": 2,  # 0最稳；2为大多数Linux单GPU环境的推荐起点。
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 1,
        "high_maneuver_curriculum": False,  # 对比实验默认固定类别平衡；不重建Loader。

        # 任务调度与中断恢复
        "run_order": "seed_major",      # seed_major：同一seed先完成所有模型；不再连续运行5次Ours_Full。
        "resume": False,                # 稳定化后五个种子均从第1轮重新训练。
        "retry_failed_runs": True,      # 失败run下次自动重试；False则保留失败记录并跳过。
        "resume_save_interval": 5,      # 每5个epoch保存last_training_state.pt；中断最多回退4个epoch。
        "resume_strict_signature": True,
        "discard_incompatible_resume": False,
        "models_to_run": ["TrajectronPP"],  # 本次只补跑TrajectronPP。
        "max_runs_this_session": 5,          # 稳定化后重新运行TrajectronPP的全部五个种子。
        "inter_run_sleep_s": 2.0,
        "learning_rate": 5e-4,
        "trajectronpp_learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "warmup_epochs": 5,
        "eta_min": 1e-6,
        "grad_clip_norm": 1.0,
        "trajectronpp_grad_clip_norm": 0.5,
        "tf_start": 0.55,
        "tf_end": 0.05,
        "deterministic": True,
        "log_interval": 5,
        "d_model": 192,
        "diffusion_steps": 50,
        "probabilistic_selection_on_val": True,
        "val_prob_samples": 20,
        "val_prob_eval_interval": 5,
        "prob_selection_target_pic": 95.0,
        "prob_selection_energy_weight": 0.05,
        "prob_selection_pic_weight": 0.01,
        "prob_selection_miw_weight": 0.05,
        "a_keer_ref_floor_percentile": 10.0,

        # 最终概率评估：调试阶段可先改为20，正式实验建议100。
        "final_prob_samples": 100,
        "calibration_prob_samples": 100,
        "calibrate_intervals_on_val": True,
        "interval_nominal_coverage": 0.95,
        "interval_calibration_min_scale": 0.10,
        "interval_calibration_max_scale": 10.0,
        "hardware_prob_samples": 100,
        "hardware_point_repeats": 30,
        "hardware_prob_repeats": 2,

        # UAV-Flow trajectory states are aligned at 5 Hz; no 5-to-1
        # downsampling is performed in this pipeline.
        "sample_interval_s": 0.2,
        "distance_accuracy_ratio": 0.15,
        "distance_accuracy_min_threshold_m": 0.25,
        "direction_accuracy_threshold_deg": 25.0,
        "direction_min_speed_m": 0.05,
        "a_keer_gamma": 1.10,
        "physical_max_accel_mps2": 30.0,
        "physical_max_jerk_mps3": 80.0,
    }

    # 不同结构可进行预注册式、有限范围的超参数设定；输入长度、输出长度、数据与选择规则相同。
    EXPERIMENTS = [
        {"model_name": "Ours_Full", "model_type": "ours"},
        {"model_name": "Base_LSTM", "model_type": "lstm"},
        {"model_name": "Base_GRU", "model_type": "gru"},
        {"model_name": "Base_Transformer", "model_type": "transformer"},
        {"model_name": "Base_VanillaMamba", "model_type": "vanilla_mamba"},
        {"model_name": "Base_iTransformer", "model_type": "itransformer"},
        {"model_name": "Base_PatchTST", "model_type": "patchtst"},
        {"model_name": "Base_DLinear", "model_type": "dlinear"},
        {"model_name": "Gaussian_LSTM", "model_type": "gaussian_lstm"},
        {"model_name": "Gaussian_Transformer", "model_type": "gaussian_transformer"},
        {"model_name": "DeepAR", "model_type": "deepar"},
        {"model_name": "TFT_Quantile", "model_type": "tft_quantile"},
        {"model_name": "TrajectronPP", "model_type": "trajectronpp"},
        {"model_name": "cVMD", "model_type": "cvmd"},
    ]

    # 数值稳定化改变了TrajectronPP训练设置，因此五个种子全部重新运行。
    PROTOCOL_SEEDS = [2026, 2036, 2046, 2056, 2066]
    RUN_SEEDS = PROTOCOL_SEEDS.copy()

    ensure_dir(CONFIG["output_dir"])
    save_json(os.path.join(CONFIG["output_dir"], "experiment_protocol.json"), {
        "task": "UAV trajectory prediction comparison",
        "data_split": "trajectory-ID train/validation/test = 70/10/20",
        "selection_rule": "validation ADE + 0.5*FDE",
        "probability_models": ["Ours_Full", "Gaussian_LSTM", "Gaussian_Transformer", "DeepAR", "TFT_Quantile", "TrajectronPP"],
        "final_prob_samples": CONFIG["final_prob_samples"],
        "validation_selection": "Deterministic models use ADE+0.5*FDE; probabilistic models use sampled validation minADE/minFDE/Energy/mPIC/MIW composite.",
        "uncertainty_calibration": "For each probabilistic model, validation labels only are used to fit a separate multiplicative interval scale for every horizon and coordinate. The scale rescales sample deviations around the predictive mean and is frozen before test evaluation.",
        "probability_metric_space": "NLL is the per-coordinate moment-matched Gaussian NLL in physical position space. PIC95 is marginal coordinate-wise empirical coverage of the central 95% sample interval. MIW95 is its mean coordinate-wise width. All three are reported globally, by maneuver state, and for T+1 through T+20.",
        "nominal_interval_coverage_percent": 95.0,
        "comparison_scope": "Ours_Full is compared as a complete framework with auxiliary intent/probability/goal losses; Gaussian-LSTM/Gaussian-Transformer/DeepAR/TFT-Quantile/TrajectronPP are probabilistic baselines; cVMD is a deterministic decomposition baseline; module-level claims require ablation.",
        "feature_protocol": "12-D causal motion features derived only from observed 3-D positions; no attitude or telemetry-only inputs.",
        "smoothing_protocol": "Right-edge causal Savitzky-Golay SG(7,2) uses only the current and six previous position observations.",
        "augmentation_protocol": "Rotation/canonical heading alignment is applied in physical feature space before train-only normalization.",
        "kinematic_protocol": "A-KEER uses a train-only acceleration reference floor; NJERDev measures absolute deviation from the ideal jerk-energy ratio.",
        "trajectronpp_numerical_protocol": "TrajectronPP is trained and evaluated in FP32 with AdamW lr=1e-4 and gradient clipping at 0.5; its KL and discrete-prior entropy are evaluated in FP32. The setting is fixed for all five seeds.",
        "seeds": PROTOCOL_SEEDS,
        "seeds_executed_this_session": RUN_SEEDS,
        "config": CONFIG,
        "experiments": EXPERIMENTS,
    })

    get_mamba2(CONFIG.get("mamba_main_dir"))
    check_mamba_cuda()

    # 注意：只按 split_seed 预处理一次，run seed 不改变数据划分。
    seed_everything(CONFIG["split_seed"], deterministic=True)
    data = prepare_shared_dataset(CONFIG)
    run_repeated_comparison(CONFIG, data, EXPERIMENTS, RUN_SEEDS)


if __name__ == "__main__":
    main()
