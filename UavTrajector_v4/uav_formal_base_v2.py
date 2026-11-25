# ================================================================
# ⚠️ 必须在所有 import 之前执行，防止子进程继承 CUDA context 导致死锁
# ================================================================
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import sys
import time
import glob
import importlib
import concurrent.futures
import pandas as pd
import joblib
import orjson
import math
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from thop import profile
except ImportError:
    profile = None
    print("⚠️ 未找到 thop 库")

# ===================== 全局统一超参 =====================
# 🔥 统一所有 Dropout 率
DROPOUT_BASE = 0.15          # 主 dropout
DROPOUT_LIGHT = 0.075       # 轻量
DROPOUT_HEAVY = 0.3         # 意图头专用

# 🔥 混合精度固定配置
AMP_DTYPE = torch.float16    # 主流显卡最稳
USE_AMP = torch.cuda.is_available()

# 随机种子由每个 variant × seed 运行开始时的 seed_everything() 统一设置。

def _init_mamba():
    """只在主进程调用一次"""
    mamba_main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba-main")
    sys.path.append(mamba_main_dir)
    module = importlib.import_module("mamba_ssm.modules.mamba2")
    Mamba2_cls = getattr(module, "Mamba2")
    print("✅ 动态导入成功")
    return Mamba2_cls

def _check_mamba_cuda():
    print("\n" + "="*50)
    print(" 🚀 正在自检 Mamba2 硬件加速引擎状态...")
    print("="*50)
    try:
        import causal_conv1d_cuda
        print("  [✅] causal_conv1d_cuda: 已连接")
    except ImportError:
        print("  [❌] causal_conv1d_cuda: 未连接")
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        print("  [✅] mamba2_triton_ops: 已连接")
    except ImportError as e:
        print(f"  [❌] mamba2_triton_ops: 未连接 ({e})")
    print("="*50 + "\n")


# ================================================================
# 概率与不确定性生成式指标计算
# ================================================================
def compute_probabilistic_metrics(y_true, y_samples, y_pred_mean, std_real):
    if np.max(std_real) < 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    y_true_exp = np.expand_dims(y_true, axis=0)
    err_samples = np.linalg.norm(y_samples - y_true_exp, axis=-1)
    min_ade = np.mean(np.min(np.mean(err_samples, axis=2), axis=0))
    min_fde = np.mean(np.min(err_samples[:, :, -1], axis=0))

    var_real = (std_real + 1e-8) ** 2
    nll = np.mean(0.5 * np.log(2 * np.pi * var_real) + ((y_true - y_pred_mean) ** 2) / (2 * var_real))

    in_bound = (y_true >= y_pred_mean - 1.96 * std_real) & (y_true <= y_pred_mean + 1.96 * std_real)
    pic_95 = np.mean(in_bound) * 100.0
    miw_95 = np.mean(2 * 1.96 * std_real)
    return min_ade, min_fde, nll, pic_95, miw_95

def calculate_global_accuracies(y_p_traj, y_t_traj, horizon):
    dist_acc_list, dir_acc_list = [], []
    err_matrix = np.linalg.norm(y_p_traj - y_t_traj, axis=2)
    for h in range(horizon):
        dist_acc_list.append(np.mean(err_matrix[:, h] < min(0.25 + h * 0.15, 1.5)) * 100)
        lookback = 2 if h >= 2 else h
        p_v_local = y_p_traj[:, h, :2] if h == 0 else y_p_traj[:, h, :2] - y_p_traj[:, h - lookback, :2]
        t_v_local = y_t_traj[:, h, :2] if h == 0 else y_t_traj[:, h, :2] - y_t_traj[:, h - lookback, :2]
        p_v_macro = y_p_traj[:, h, :2] - y_p_traj[:, 0, :2]
        t_v_macro = y_t_traj[:, h, :2] - y_t_traj[:, 0, :2]
        alpha_macro = min(1.0, h / 15.0)
        p_v = (1.0 - alpha_macro) * p_v_local + alpha_macro * p_v_macro
        t_v = (1.0 - alpha_macro) * t_v_local + alpha_macro * t_v_macro
        norm_p, norm_t = np.linalg.norm(p_v, axis=1), np.linalg.norm(t_v, axis=1)
        valid = norm_t > 0.05
        if valid.sum() > 0:
            cos_t = np.clip(np.sum(p_v[valid] * t_v[valid], axis=1) / (norm_p[valid] * norm_t[valid] + 1e-8), -1.0, 1.0)
            dir_acc_list.append(np.mean(np.arccos(cos_t) < np.deg2rad(25.0)) * 100)
    return float(np.mean(dist_acc_list)), float(np.mean(dir_acc_list))

def compute_advanced_metrics(y_samples, y_true, y_mean, dt=0.1, accel_limit=30.0):
    K, N, H, D = y_samples.shape
    if H < 3: return 0.0, 0.0, 0.0
    v = np.diff(y_samples, axis=2) / dt
    accel = np.diff(v, axis=2) / dt
    accel_norm = np.linalg.norm(accel, axis=-1)
    violations = np.sum(accel_norm > accel_limit) / accel_norm.size
    jerk = np.diff(accel, axis=2) / dt
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    jerk_mask = np.mean(jerk_norm, axis=2) < 5.0
    fde_samples = np.linalg.norm(y_samples[:, :, -1, :] - y_true[:, -1, :][np.newaxis], axis=-1)
    fde_samples[~jerk_mask] += 1e6
    jc_min_fde = np.mean(np.min(fde_samples, axis=0))
    brier_score = np.mean((y_mean - y_true) ** 2)
    return violations, jc_min_fde, brier_score

class ModelEMA:
    def __init__(self, model, decay=0.999, warmup_steps=2000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        d = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        d = min(d, self.decay)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

def uav_loss_v20(pred, log_var, by, last_pose,
                 device=None, intent_logits=None, intent_targets=None,
                 epoch=0, total_epochs=500, turn_rates_pred=None,
                 is_training=True, current_horizon=None, aux_info=None):
    device = pred.device
    H_actual = min(pred.shape[1], by.shape[1])
    H = min(current_horizon, H_actual) if current_horizon is not None else H_actual
    if H < H_actual:
        pred = pred[:, :H]; by = by[:, :H]
        if log_var is not None: log_var = log_var[:, :H]
        if turn_rates_pred is not None and turn_rates_pred.shape[1] >= H:
            turn_rates_pred = turn_rates_pred[:, :H]

    B = pred.shape[0]
    progress = epoch / max(total_epochs, 1)

    pred_fp32 = pred.float()
    by_fp32 = by.float()
    pred_traj = torch.cumsum(pred_fp32, dim=1)
    true_traj = torch.cumsum(by_fp32, dim=1)
    loss_fde = F.smooth_l1_loss(pred_traj[:, -1], true_traj[:, -1])

    SHORT, MID = min(5, H), min(15, H)
    loss_traj_short = F.smooth_l1_loss(pred_traj[:, :SHORT], true_traj[:, :SHORT])
    loss_traj_mid = F.smooth_l1_loss(pred_traj[:, SHORT:MID], true_traj[:, SHORT:MID]) if H > SHORT else torch.tensor(0.0, device=device)
    loss_traj_long = F.smooth_l1_loss(pred_traj[:, MID:], true_traj[:, MID:]) if H > MID else torch.tensor(0.0, device=device)

    loss_stable_normal_pos_mse = torch.tensor(0.0, device=device)
    if intent_targets is not None and H > 5:
        sn_mask = (intent_targets != 2)
        if sn_mask.sum() > 0:
            t_w = (1.0 + 2.0 * (torch.arange(H, device=device).float() / max(H - 1, 1)) ** 1.5).view(1, H, 1)
            stable_w = (intent_targets == 0).float().view(B, 1, 1) * 1.5
            normal_w = (intent_targets == 1).float().view(B, 1, 1) * 1.0
            pos_err_sq = (pred_traj - true_traj) ** 2
            loss_stable_normal_pos_mse = ((stable_w + normal_w) * t_w * pos_err_sq)[sn_mask].mean()

    pred_yaw = torch.atan2(pred[:, :, 1], pred[:, :, 0] + 1e-8)
    true_yaw = torch.atan2(by[:, :, 1], by[:, :, 0] + 1e-8)
    valid_mask = torch.norm(by[:, :, :2], dim=-1) > 0.05
    if valid_mask.sum() > 0:
        angle_diff = pred_yaw - true_yaw
        t_idx = torch.arange(H, device=device, dtype=torch.float32)
        time_w_dir = (1.0 + 2.5 * (t_idx / max(H - 1, 1)) ** 1.8).view(1, H)
        if intent_targets is not None:
            stable_mask = (intent_targets == 0).float().unsqueeze(1)
            high_mask = (intent_targets == 2).float().unsqueeze(1)
            dir_weight = 3.0 * stable_mask + 1.5 * (1.0 - stable_mask - high_mask) + 0.5 * high_mask
            loss_heading = ((1.0 - torch.cos(angle_diff)) * time_w_dir * dir_weight)[valid_mask].mean()
        else:
            loss_heading = ((1.0 - torch.cos(angle_diff)) * time_w_dir)[valid_mask].mean()
    else:
        loss_heading = torch.tensor(0.0, device=device)

    loss_dir_consistency = torch.tensor(0.0, device=device)
    if H > 1 and intent_targets is not None:
        pred_yaw_diff = (pred_yaw[:, 1:] - pred_yaw[:, :-1] + np.pi) % (2 * np.pi) - np.pi
        true_yaw_diff = (true_yaw[:, 1:] - true_yaw[:, :-1] + np.pi) % (2 * np.pi) - np.pi
        valid_consist = valid_mask[:, 1:] & valid_mask[:, :-1]
        if valid_consist.sum() > 0:
            consist_weight = (intent_targets == 0).float().unsqueeze(1) * 3.0 + (intent_targets == 1).float().unsqueeze(1) * 1.5
            loss_dir_consistency = ((1.0 - torch.cos(pred_yaw_diff - true_yaw_diff)) * consist_weight)[valid_consist].mean()

    loss_turn_rate = torch.tensor(0.0, device=device)
    if turn_rates_pred is not None and H > 1:
        true_vel_xy = by[..., :2]
        true_angles = torch.atan2(true_vel_xy[..., 1], true_vel_xy[..., 0] + 1e-8)
        true_turn = (true_angles[:, 1:] - true_angles[:, :-1] + np.pi) % (2 * np.pi) - np.pi
        true_speed = torch.norm(true_vel_xy, dim=-1)
        valid_turn_mask = (true_speed[:, :-1] > 0.15) & (true_speed[:, 1:] > 0.15)
        if valid_turn_mask.sum() > 0:
            tw = torch.where(intent_targets == 2, torch.tensor(2.5, device=device), torch.tensor(1.0, device=device)).unsqueeze(1) if intent_targets is not None else 1.0
            loss_turn_rate = F.smooth_l1_loss(turn_rates_pred[:, :H - 1] * tw, true_turn * tw, reduction='none')[valid_turn_mask].mean()

    loss_intent = F.cross_entropy(intent_logits, intent_targets, weight=torch.tensor([1.0, 1.5, 3.0], device=device)) if (intent_logits is not None and intent_targets is not None) else torch.tensor(0.0, device=device)

    pred_radial = torch.norm(pred_traj, dim=-1)
    true_radial = torch.norm(true_traj, dim=-1)
    loss_magnitude_global = (torch.linspace(1.0, 3.0, H, device=device).unsqueeze(0) * F.smooth_l1_loss(pred_radial, true_radial, reduction='none')).mean()

    pred_step_norm, true_step_norm = torch.norm(pred_fp32, dim=-1), torch.norm(by_fp32, dim=-1)
    if intent_targets is not None:
        step_mag_w = torch.where(intent_targets == 2, torch.tensor(2.5, device=device), torch.where(intent_targets == 0, torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))).unsqueeze(1)
        loss_step_magnitude = (step_mag_w * (pred_step_norm - true_step_norm) ** 2).mean()
    else:
        loss_step_magnitude = F.mse_loss(pred_step_norm, true_step_norm)

    loss_high_magnitude = loss_high_long = loss_high_pos_mse = loss_high_endpoint = torch.tensor(0.0, device=device)
    if intent_targets is not None:
        high_mask_radial = (intent_targets == 2)
        if high_mask_radial.sum().item() > 0:
            loss_high_magnitude = (torch.linspace(1.0, 3.0, H, device=device).unsqueeze(0) * F.smooth_l1_loss(pred_radial[high_mask_radial], true_radial[high_mask_radial], reduction='none')).mean()
            TAIL = min(8, H - 1)
            if H > TAIL:
                loss_high_pos_tail = F.smooth_l1_loss(pred_traj[high_mask_radial][:, TAIL:], true_traj[high_mask_radial][:, TAIL:])
                pred_macro_dir = F.normalize(pred_traj[high_mask_radial][:, -1, :2] - pred_traj[high_mask_radial][:, TAIL, :2], p=2, dim=1, eps=1e-8)
                true_macro_dir = F.normalize(true_traj[high_mask_radial][:, -1, :2] - true_traj[high_mask_radial][:, TAIL, :2], p=2, dim=1, eps=1e-8)
                loss_high_macro_dir = (1.0 - (pred_macro_dir * true_macro_dir).sum(-1)).mean()
                loss_high_vel = F.smooth_l1_loss(pred[high_mask_radial], by[high_mask_radial])
                loss_high_long = min(1.0, max(0.0, (progress - 0.05) / 0.15)) * (2.5 * loss_high_pos_tail + 1.0 * loss_high_vel + 1.5 * loss_high_macro_dir)
            pos_tail_start = max(H - 10, H // 2)
            if H > pos_tail_start:
                pos_t_weight = torch.linspace(1.5, 4.5, H - pos_tail_start, device=device).view(1, -1, 1) ** 1.2
                loss_high_pos_mse = ((pred_traj[high_mask_radial, pos_tail_start:] - true_traj[high_mask_radial, pos_tail_start:]) ** 2 * pos_t_weight).mean()
            loss_high_endpoint = F.smooth_l1_loss(pred_traj[high_mask_radial, -1, :], true_traj[high_mask_radial, -1, :])

    loss_macro_dir = torch.tensor(0.0, device=device)
    if intent_targets is not None and H > 1:
        macro_losses = []
        for intent_id, step_list in {0: [5, 10, 15, 20, H - 1], 1: [5, 10, 15, 20, H - 1], 2: [5, 10, 15]}.items():
            mask = (intent_targets == intent_id)
            if mask.sum() == 0: continue
            for t in step_list:
                if t >= H or t <= 0: continue
                valid = torch.norm(true_traj[mask, t, :2], dim=-1) > 0.1
                if valid.sum() > 0:
                    pred_dir_macro = F.normalize(pred_traj[mask, t, :2][valid], p=2, dim=-1, eps=1e-6)
                    true_dir_macro = F.normalize(true_traj[mask, t, :2][valid], p=2, dim=-1, eps=1e-6)
                    t_weight = 1.0 + (3.5 if intent_id == 0 else (2.0 if intent_id == 1 else 0.5)) * (t / H)
                    macro_losses.append((1.0 - (pred_dir_macro * true_dir_macro).sum(-1)).mean() * t_weight)
        if macro_losses: loss_macro_dir = torch.stack(macro_losses).mean()

    loss_jerk_main = loss_jerk_smoother = torch.tensor(0.0, device=device)
    if H >= 3:
        acc_true = by[:, 1:] - by[:, :-1]; jerk_true = acc_true[:, 1:] - acc_true[:, :-1]
        acc_raw = (aux_info['ps_raw'] if aux_info is not None and 'ps_raw' in aux_info else pred)[:, 1:] - (aux_info['ps_raw'] if aux_info is not None and 'ps_raw' in aux_info else pred)[:, :-1]
        loss_jerk_main = (torch.relu((acc_raw[:, 1:] - acc_raw[:, :-1]).abs() - jerk_true.abs() * 2.0) ** 2).mean()
        acc_pred = pred[:, 1:] - pred[:, :-1]
        excess_jerk_smooth = torch.relu((acc_pred[:, 1:] - acc_pred[:, :-1]).abs() - jerk_true.abs() * 1.2)
        jerk_weight = torch.where(intent_targets == 0, 5.0, torch.where(intent_targets == 1, 1.0, 0.05)).float().unsqueeze(1).unsqueeze(2) if intent_targets is not None else 1.0
        loss_jerk_smoother = (jerk_weight * (excess_jerk_smooth ** 2)).mean()

    loss_deterministic = (1.0 * loss_fde + 1.0 * loss_traj_short + 1.5 * loss_traj_mid + 2.0 * loss_traj_long
        + 2.0 * loss_stable_normal_pos_mse + 1.8 * loss_heading + 1.0 * loss_dir_consistency + 3.0 * loss_macro_dir
        + 6.0 * loss_magnitude_global + 3.5 * loss_step_magnitude + 0.5 * loss_turn_rate + 0.5 * loss_intent
        + 1.2 * loss_high_long + 2.5 * loss_high_magnitude + 4.0 * loss_high_pos_mse + 2.5 * loss_high_endpoint
        + 0.1 * loss_jerk_main + 2.0 * loss_jerk_smoother)

    loss_probabilistic = var_gap_loss = loss_kl = torch.tensor(0.0, device=device)
    if log_var is not None:
        log_var_clamped = torch.clamp(log_var.float(), min=-5.0, max=5.0)
        var = torch.exp(log_var_clamped) + 1e-6
        time_w_nll = torch.linspace(0.8, 1.2, H, device=device).view(1, H, 1)
        loss_nll = (time_w_nll * 0.5 * (log_var_clamped + (pred.float() - by.float()) ** 2 / var)).mean()
        if H > 1 and H // 2 > 0: var_gap_loss = torch.relu(1.0 - (log_var_clamped[:, H // 2:, :].mean(dim=1) - log_var_clamped[:, :H // 2, :].mean(dim=1))).mean() * 0.2
        if is_training and intent_targets is not None:
            prior_log_var = torch.where(intent_targets == 0, -1.5, torch.where(intent_targets == 1, -0.5, 1.2)).float().view(B, 1, 1).expand_as(log_var_clamped)
            loss_kl = (time_w_nll * 0.5 * ((log_var_clamped - prior_log_var).exp() + (log_var_clamped - prior_log_var) - 1.0)).mean() * 0.05
        loss_probabilistic = loss_nll + loss_kl + var_gap_loss

    loss_goal = torch.tensor(0.0, device=device)
    if aux_info is not None and 'goals_xyz' in aux_info and is_training:
        goals, goal_logits, true_endpoint = aux_info['goals_xyz'], aux_info['goal_logits'], true_traj[:, -1, :]
        closest = torch.argmin(torch.norm(goals - true_endpoint.unsqueeze(1), dim=-1), dim=-1)
        goal_weight = 1.0 + 1.5 * (intent_targets == 2).float() if intent_targets is not None else torch.ones(B, device=device)
        loss_goal_reg = (goal_weight * F.smooth_l1_loss(torch.gather(goals, 1, closest.view(B, 1, 1).expand(-1, 1, 3)).squeeze(1), true_endpoint, reduction='none').mean(-1)).mean()
        loss_goal_cls = (goal_weight * F.cross_entropy(goal_logits, closest, reduction='none')).mean()
        mask_off_diag = (1 - torch.eye(goals.shape[1], device=device)).unsqueeze(0)
        diversity_reg = torch.relu(1.5 - (torch.cdist(goals[:, :, :2], goals[:, :, :2]) * mask_off_diag).sum(dim=(1, 2)) / max(goals.shape[1] * (goals.shape[1] - 1), 1)).mean()
        loss_goal = loss_goal_reg + 0.3 * loss_goal_cls + 0.5 * diversity_reg
        uav_loss_v20._last_goal_loss = float(loss_goal.detach().item())

    return loss_deterministic + 0.3 * loss_probabilistic + 2.0 * loss_goal

def prepare_data_from_cache(config):
    """
    特征提取（多进程 CPU）+ 归一化 + 建索引
    内存优化：流式采样拟合 scaler，边归一化边释放原始数据
    """
    cache_dir  = os.path.join(config.get("maneuver_save_dir", "maneuver_results"), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file  = os.path.join(cache_dir, "processed_flights_cache.npz")
    scaler_file = os.path.join(cache_dir, "scalers.pkl")

    max_s      = max(config["scales"])
    horizon    = config["horizon"]
    min_length = max_s + horizon

    # ── 1. 加载或提取特征 ─────────────────────────────────────────
    all_flights = None
    if os.path.exists(cache_file) and not config.get("force_rebuild_cache", False):
        try:
            data = np.load(cache_file, allow_pickle=True)
            all_flights = {k: data[k] for k in data.files}
            sample_feat = next(iter(all_flights.values()))
            if sample_feat.shape[-1] != 31:
                print(f"⚠️ 缓存维度={sample_feat.shape[-1]} != 31，重建缓存...")
                all_flights = None
                if os.path.exists(scaler_file): os.remove(scaler_file)
            else:
                print(f"✅ 加载缓存：{len(all_flights)} 条轨迹，维度={sample_feat.shape[-1]}")
        except Exception as e:
            print(f"⚠️ 缓存损坏({e})，重新提取...")
            all_flights = None

    if all_flights is None:
        print("⏳ 启动多进程特征提取（31维，纯 CPU）...")

        # ✅ 每个 parquet 只读一次
        tasks = []
        seen_ids = set()
        for f in sorted(glob.glob(os.path.join(config.get("data_dir", "uavdataset/"), "*.parquet"))):
            print(f"  ├─ 读取: {os.path.basename(f)}")
            try:
                df = pd.read_parquet(f)
                col = 'logs' if 'logs' in df.columns else 'log'
                df_u = df.drop_duplicates('id')[['id', col]]
                for fid, lstr in zip(df_u['id'].values, df_u[col].values):
                    sid = str(fid)
                    if sid not in seen_ids:
                        tasks.append((sid, lstr, min_length))
                        seen_ids.add(sid)
                del df, df_u
                gc.collect()
            except Exception as e:
                print(f"  ⚠️ 读取失败: {e}")

        print(f"  ├─ 总任务: {len(tasks)} 条轨迹")

        # ✅ 根据服务器 CPU 数自动设置 worker 数
        cpu_total  = os.cpu_count() or 4
        num_workers = min(max(cpu_total - 2, 2), 32)
        print(f"  ├─ CPU 核心数: {cpu_total}，使用 {num_workers} workers")

        all_flights = {}
        failed = 0
        chunk = max(1, len(tasks) // (num_workers * 8))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            max_tasks_per_child=100   # ✅ 防子进程内存泄漏
        ) as executor:
            for r in tqdm(
                executor.map(process_single_flight, tasks, chunksize=chunk),
                total=len(tasks), desc="CPU特征提取", ncols=80
            ):
                if r is not None and r[0] is not None:
                    all_flights[r[0]] = r[1]
                else:
                    failed += 1

        print(f"  ├─ 成功: {len(all_flights)} | 失败/跳过: {failed}")
        if len(all_flights) == 0:
            raise ValueError("没有提取到任何有效轨迹！检查数据路径。")

        print(f"  ├─ 保存缓存...")
        np.savez_compressed(cache_file, **all_flights)
        print(f"  ✅ 缓存已保存 → {cache_file}")

    # ── 2. 划分训练/验证集 ──────────────────────────────────────────
    feat_dim = next(iter(all_flights.values())).shape[-1]
    uids = list(all_flights.keys())
    tr_ids, va_ids = train_test_split(uids, test_size=0.2, random_state=42)
    tr_s = [all_flights[i] for i in tr_ids]
    va_s = [all_flights[i] for i in va_ids]

    # ✅ all_flights 使命完成，立即释放
    del all_flights
    gc.collect()
    print(f"  ├─ all_flights 已释放，当前仅保留 tr_s/va_s")

    # ── 3. 先建索引（需要原始数据计算机动强度）──────────────────────
    scale_mask = np.ones(feat_dim, dtype=bool)
    scale_mask[0:3] = False  # deltas 由 d_scaler 处理
    scale_mask[12:18] = False  # sin/cos 已经在 [-1,1]
    # ✅ 新增:所有原始角度特征都不进 f_scaler
    scale_mask[21] = False  # ta (track angle, 需按 ang 旋转)
    scale_mask[22] = False  # fpa (xy 旋转不影响)
    scale_mask[25] = False  # slip_angle
    scale_mask[26] = False  # aoa

    def build_idx(orig_seqs, stride):
        idx, all_int = [], []
        for sid, orig in enumerate(orig_seqs):
            for st in range(0, len(orig) - min_length + 1, stride):
                seg = orig[st + max_s: st + min_length, :3]
                mean_move = float(np.mean(np.linalg.norm(seg, axis=-1)))
                if mean_move <= 0.01:
                    continue
                linear = float(np.mean(np.linalg.norm(
                    np.diff(seg, axis=0), axis=-1))) if len(seg) >= 2 else 0.0
                spd_std = float(np.std(np.linalg.norm(seg, axis=-1)))
                intensity = 0.7 * linear + 0.3 * spd_std
                idx.append({"seq_idx": int(sid), "start": int(st),
                            "intensity": float(intensity), "label": 0})
                all_int.append(intensity)
        if not all_int:
            raise ValueError("所有序列都被过滤掉！")
        p25, p75 = np.percentile(all_int, [25, 75])
        for item in idx:
            v = item["intensity"]
            item["label"] = int(0 if v < p25 else (1 if v < p75 else 2))
        return idx

    print(f"  ├─ 构建训练索引...")
    tr_idx = build_idx(tr_s, stride=5)
    print(f"  ├─ 构建验证索引...")
    va_idx = build_idx(va_s, stride=2)
    print(f"  ├─ train {len(tr_idx)} | val {len(va_idx)} 窗口")

    # ── 4. 拟合 scaler（流式采样，不 vstack 全量）────────────────────
    if os.path.exists(scaler_file) and not config.get("force_rebuild_cache", False):
        d_scaler, f_scaler = joblib.load(scaler_file)
        print(f"  ✅ 加载 scaler: d_scaler.scale_={d_scaler.scale_}")
    else:
        print(f"  ├─ 流式采样拟合 scaler（不 vstack，节省内存）...")
        rng = np.random.default_rng(42)
        sample_rows = []
        total_sampled = 0
        TARGET_SAMPLES = 50000

        for s in tr_s:
            if total_sampled >= TARGET_SAMPLES:
                break
            n = len(s)
            if n <= 0:
                continue
            n_take = min(50, n, TARGET_SAMPLES - total_sampled)
            chosen = rng.choice(n, size=n_take, replace=False)
            sample_rows.append(s[chosen])
            total_sampled += n_take

        atr_sample = np.vstack(sample_rows)
        del sample_rows
        print(f"  ├─ 采样 {total_sampled} 帧用于拟合 scaler")

        # d_scaler
        deltas       = atr_sample[:, :3]
        valid_d      = np.linalg.norm(deltas, axis=1) > 0.001
        deltas_valid = deltas[valid_d]
        xy_std = max(float(np.std(deltas_valid[:, :2])), 0.05)
        z_std  = max(float(np.std(deltas_valid[:, 2:3])), 0.01)
        d_scaler = StandardScaler(with_mean=False)
        d_scaler.fit(deltas_valid)
        d_scaler.scale_ = np.array([xy_std, xy_std, z_std])
        print(f"  ├─ d_scaler.scale_={d_scaler.scale_} (xy={xy_std:.4f}, z={z_std:.4f})")

        # f_scaler
        atr_clip = atr_sample.copy()
        atr_clip[:, 18:21] = np.clip(atr_sample[:, 18:21], -10, 10)
        atr_clip[:, 9:12]  = np.clip(atr_sample[:, 9:12],  -50, 50)
        f_scaler = StandardScaler().fit(np.clip(atr_clip[:, scale_mask], -500, 500))
        f_scaler.scale_ = np.maximum(f_scaler.scale_, 1e-3)

        del atr_sample, atr_clip, deltas, deltas_valid
        gc.collect()

        joblib.dump((d_scaler, f_scaler), scaler_file)
        print(f"  ✅ scaler 已保存 → {scaler_file}")

    # ── 5. 归一化（边处理边释放原始数据）────────────────────────────
    def _norm_one(s):
        sc = s.astype(np.float32)
        sc[:, :3]         = d_scaler.transform(sc[:, :3])
        sc[:, 18:21]      = np.clip(sc[:, 18:21], -10, 10)
        sc[:, 9:12]       = np.clip(sc[:, 9:12],  -50, 50)
        sc[:, scale_mask] = np.clip(
            f_scaler.transform(np.clip(sc[:, scale_mask], -500, 500)), -10, 10)
        return sc

    print("  ├─ 归一化训练序列（边处理边释放）...")
    tr_sc = []
    for i in tqdm(range(len(tr_s)), desc="norm train", ncols=60):
        tr_sc.append(_norm_one(tr_s[i]))
        tr_s[i] = None          # ✅ 立即释放原始引用
    del tr_s
    gc.collect()
    print("  ✅ 训练集原始数据已释放")

    print("  ├─ 归一化验证序列（边处理边释放）...")
    va_sc = []
    for i in tqdm(range(len(va_s)), desc="norm val", ncols=60):
        va_sc.append(_norm_one(va_s[i]))
        va_s[i] = None          # ✅ 立即释放原始引用
    del va_s
    gc.collect()
    print("  ✅ 验证集原始数据已释放")

    # ── 6. 验证归一化结果 ─────────────────────────────────────────
    s0 = tr_sc[0]
    print(f"  ├─ 归一化验证:")
    print(f"     [0:3]  位移增量: mean={np.abs(s0[:,0:3]).mean():.4f}, "
          f"max={np.abs(s0[:,0:3]).max():.4f}")
    print(f"     [3:6]  速度:     mean={np.abs(s0[:,3:6]).mean():.4f}")
    print(f"     [6:9]  加速度:   mean={np.abs(s0[:,6:9]).mean():.4f}")
    print(f"     [9:12] jerk:     mean={np.abs(s0[:,9:12]).mean():.4f}")
    print(f"     [18:21] pqr:     max={np.abs(s0[:,18:21]).max():.3f}")
    all_mean = np.mean([np.abs(s[:,0:3]).mean() for s in tr_sc[:50]])
    print(f"     前50条归一化位移均值: {all_mean:.4f} (正常范围 0.4-0.8)")
    print(f"  ├─ d_scaler.scale_ = {d_scaler.scale_}")

    return tr_sc, tr_idx, va_sc, va_idx, d_scaler, feat_dim, max_s, horizon

def process_single_flight(args):
    """
    多进程 worker：纯 CPU + numpy 运算，禁止 import torch/cuda
    """
    flight_id, log_str, min_length = args

    # worker 内部局部 import，避免主进程的模块污染子进程
    import numpy as _np
    import orjson as _orjson
    from scipy.signal import savgol_filter as _savgol

    def _wrap(a):
        return (a + _np.pi) % (2 * _np.pi) - _np.pi

    def _safe_div(num, den, eps=1e-8):
        mask = _np.abs(den) > eps
        out = _np.zeros_like(num)
        _np.divide(num, den, out=out, where=mask)
        return out

    try:
        log_data = _orjson.loads(log_str)
        raw_logs  = _np.array(log_data['raw_logs'],          dtype=_np.float64)
        prep_logs = _np.array(log_data['preprocessed_logs'], dtype=_np.float32)
    except Exception:
        return None

    if len(prep_logs) < min_length:
        return None

    dt = _np.clip(_np.diff(raw_logs[:, 6]), 0.01, 10.0)
    dt = _np.insert(dt, 0, dt[0]).reshape(-1, 1)

    pos  = prep_logs[:, :3]
    angs = prep_logs[:, 3:6]
    if _np.max(_np.abs(angs)) > 10.0:
        angs = _np.radians(angs)
    angs[:, 2] = _np.unwrap(angs[:, 2])

    wl    = min(7, len(pos) if len(pos) % 2 != 0 else len(pos) - 1)
    pos_s = _savgol(pos, wl, 3, axis=0) if len(pos) > 3 else pos

    deltas = _np.vstack([_np.zeros((1, 3)), _np.diff(pos_s, axis=0)])
    v    = deltas / dt
    acc  = _np.vstack([_np.zeros((1, 3)), _np.diff(v,   axis=0)]) / dt
    jerk = _np.vstack([_np.zeros((1, 3)), _np.diff(acc, axis=0)]) / dt

    d_angs = _np.vstack([_np.zeros((1, 3)), _np.diff(angs, axis=0)]) / dt
    d_angs = _wrap(d_angs)
    phi, theta = angs[:, 0:1], angs[:, 1:2]
    p = d_angs[:,0:1] - d_angs[:,2:3] * _np.sin(theta)
    q = d_angs[:,1:2] * _np.cos(phi) + d_angs[:,2:3] * _np.sin(phi) * _np.cos(theta)
    r = -d_angs[:,1:2] * _np.sin(phi) + d_angs[:,2:3] * _np.cos(phi) * _np.cos(theta)

    spd   = _np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    xy_sp = _np.linalg.norm(v[:, :2], axis=1, keepdims=True) + 1e-8
    ta    = _np.arctan2(v[:, 1:2], v[:, 0:1] + 1e-8)
    fpa   = _np.arctan2(v[:, 2:3], xy_sp)

    vu = v / spd
    an = _np.linalg.norm(acc - _np.sum(acc*vu, axis=1, keepdims=True)*vu,
                         axis=1, keepdims=True)
    vca  = _np.cross(v, acc)
    curv = _safe_div(_np.linalg.norm(vca, axis=1, keepdims=True), spd**3)

    slip_angle = _wrap(angs[:, 2:3] - ta)
    aoa        = _wrap(angs[:, 1:2] - fpa)

    Ek  = 0.5 * spd**2
    Ep  = 9.81 * (pos[:, 2:3] - pos[0, 2])
    Ps  = _np.concatenate([_np.zeros((1, 1)),
                            _np.diff(Ek + Ep, axis=0)]) / dt
    tor = _safe_div(
        _np.sum(vca * jerk, axis=1, keepdims=True),
        _np.linalg.norm(vca, axis=1, keepdims=True)**2)

    feat = _np.concatenate([
        deltas, v, acc, jerk,
        _np.sin(angs), _np.cos(angs),
        p, q, r,
        ta, fpa, curv, an,
        slip_angle, aoa,
        Ek, Ep, Ps, tor
    ], axis=-1).astype(_np.float32)

    feat = _np.clip(feat, -1e5, 1e5)
    feat[~_np.isfinite(feat)] = 0.0

    col_max = _np.abs(feat).max(axis=0)
    if (col_max > 1e4).sum() > feat.shape[1] // 3:
        return None

    return flight_id, feat

class FusionHealthMonitor:
    """
    融合权重 global_mix_alpha 健康监控 + 自动救援

    检测三类病态:
        1. 并行解码器死亡(α 收敛到 0)
        2. AR-GRU 死亡(α 收敛到 1)
        3. α 卡住不动(梯度过小)

    救援措施:
        1. 重置 α + 开启辅助损失 + 提高并行解码器 LR
        2. 重置 α + 开启 L2 正则
        3. 提高 α 的 LR
    """

    def __init__(self, model, optimizer,
                 # 检测阈值
                 dead_low=0.05,  # < 此值视为"死亡"(向 0)
                 dead_high=0.95,  # > 此值视为"死亡"(向 1)
                 stuck_range=0.02,  # 历史窗口内变化幅度 < 此值视为"卡住"
                 # 时间窗口
                 history_window=20,  # 卡住检测的窗口大小(epoch)
                 dead_consecutive=5,  # 死亡需要连续多少 epoch 触发
                 stuck_start_epoch=40,  # 卡住检测从第几个 epoch 开始
                 check_interval=1,  # 每隔几个 epoch 检测一次
                 # 救援参数
                 aux_parallel_weight=0.2,  # 辅助损失权重
                 mix_l2_weight=0.05,  # L2 正则权重
                 lr_boost_multiplier=5.0,  # α LR 提升倍数
                 reinit_value_p1=-0.40,  # 病态 1 救援后的 α 值(sig≈0.40)
                 reinit_value_p2=-1.40,  # 病态 2 救援后的 α 值(sig≈0.20)
                 # 预防性开关(也可在训练前直接开启)
                 aux_parallel_from_start=False,
                 mix_l2_from_start=False,
                 # 行为开关
                 verbose=True):
        self.model = model
        self.optimizer = optimizer

        self.dead_low = dead_low
        self.dead_high = dead_high
        self.stuck_range = stuck_range
        self.history_window = history_window
        self.dead_consecutive = dead_consecutive
        self.stuck_start_epoch = stuck_start_epoch
        self.check_interval = check_interval

        self.aux_parallel_weight = aux_parallel_weight
        self.mix_l2_weight = mix_l2_weight
        self.lr_boost_multiplier = lr_boost_multiplier
        self.reinit_value_p1 = reinit_value_p1
        self.reinit_value_p2 = reinit_value_p2

        self.verbose = verbose

        # 历史记录
        self.history = []

        # 救援开关
        self.aux_parallel_active = aux_parallel_from_start
        self.mix_l2_active = mix_l2_from_start
        self.lr_boosted = False

        # 防止重复触发
        self.fired = {'p1': False, 'p2': False, 'p3': False}

        if self.verbose:
            init_alpha = torch.sigmoid(self.model.global_mix_alpha).item()
            print(f"  🔍 FusionHealthMonitor 已启动")
            print(f"     初始 sigmoid(α) = {init_alpha:.3f}")
            print(f"     预防性辅助损失: {'开' if aux_parallel_from_start else '关'}")
            print(f"     预防性 L2 正则: {'开' if mix_l2_from_start else '关'}")

    # ====================================================================
    # 每个 epoch 末尾调用
    # ====================================================================
    def update(self, epoch):
        """检测并触发救援"""
        if (epoch + 1) % self.check_interval != 0:
            return

        alpha_sig = torch.sigmoid(self.model.global_mix_alpha).item()
        self.history.append(alpha_sig)

        # 不够数据,先返回
        if len(self.history) < self.dead_consecutive:
            return

        # =================== 病态 1:收敛到 0 ===================
        if not self.fired['p1']:
            recent = self.history[-self.dead_consecutive:]
            if all(v < self.dead_low for v in recent):
                self._rescue_pathology_1(epoch)
                self.fired['p1'] = True
                return

        # =================== 病态 2:收敛到 1 ===================
        if not self.fired['p2']:
            recent = self.history[-self.dead_consecutive:]
            if all(v > self.dead_high for v in recent):
                self._rescue_pathology_2(epoch)
                self.fired['p2'] = True
                return

        # =================== 病态 3:卡住不动 ===================
        if not self.fired['p3'] and epoch >= self.stuck_start_epoch:
            if len(self.history) >= self.history_window:
                window = self.history[-self.history_window:]
                if max(window) - min(window) < self.stuck_range:
                    # 还要排除"卡在合理位置"的情况(0.2-0.6 之间是健康的)
                    avg = sum(window) / len(window)
                    if avg < 0.15 or avg > 0.65:
                        self._rescue_pathology_3(epoch)
                        self.fired['p3'] = True

    # ====================================================================
    # 救援措施
    # ====================================================================
    def _rescue_pathology_1(self, epoch):
        """病态 1:并行解码器死亡 (α→0)"""
        if self.verbose:
            print("\n" + "!" * 60)
            print(f"  ⚠️ [Epoch {epoch + 1}] 检测到病态 1:并行解码器死亡")
            print(f"     最近 {self.dead_consecutive} epoch sigmoid(α) 均 < {self.dead_low}")
            print(f"     启动救援:重置 α + 开启辅助损失 + 提高并行解码器 LR")
            print("!" * 60)

        # ① 重置 α 到偏向并行的值
        with torch.no_grad():
            self.model.global_mix_alpha.data.fill_(self.reinit_value_p1)
        new_sig = torch.sigmoid(torch.tensor(self.reinit_value_p1)).item()

        # ② 开启辅助损失
        self.aux_parallel_active = True

        # ③ 提高并行解码器学习率
        for param_group in self.optimizer.param_groups:
            params_in_group = [p for p in param_group['params']]
            # 通过参数对象判断是哪个组(更鲁棒)
            for p in params_in_group:
                if any(p is mp for mp in self.model.parallel_decoder.parameters()):
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * 2.0
                    if self.verbose:
                        print(f"     ⚙️ parallel_decoder LR: {old_lr:.6f} → {param_group['lr']:.6f}")
                    break
            else:
                continue
            break

        if self.verbose:
            print(f"     🔄 global_mix_alpha → {self.reinit_value_p1:.2f} (sigmoid={new_sig:.3f})")
            print(f"     ✅ 辅助并行损失已开启 (权重={self.aux_parallel_weight})")

    def _rescue_pathology_2(self, epoch):
        """病态 2:AR-GRU 死亡 (α→1)"""
        if self.verbose:
            print("\n" + "!" * 60)
            print(f"  ⚠️ [Epoch {epoch + 1}] 检测到病态 2:AR-GRU 死亡")
            print(f"     最近 {self.dead_consecutive} epoch sigmoid(α) 均 > {self.dead_high}")
            print(f"     启动救援:重置 α + 开启 L2 正则")
            print("!" * 60)

        # ① 重置 α 到偏向 AR 的值
        with torch.no_grad():
            self.model.global_mix_alpha.data.fill_(self.reinit_value_p2)
        new_sig = torch.sigmoid(torch.tensor(self.reinit_value_p2)).item()

        # ② 开启 L2 正则(把 α 拉回中间)
        self.mix_l2_active = True

        if self.verbose:
            print(f"     🔄 global_mix_alpha → {self.reinit_value_p2:.2f} (sigmoid={new_sig:.3f})")
            print(f"     ✅ L2 正则已开启 (权重={self.mix_l2_weight})")

    def _rescue_pathology_3(self, epoch):
        """病态 3:α 卡住不动"""
        if self.verbose:
            print("\n" + "!" * 60)
            print(f"  ⚠️ [Epoch {epoch + 1}] 检测到病态 3:α 卡住不动")
            print(f"     最近 {self.history_window} epoch 变化幅度 < {self.stuck_range}")
            print(f"     启动救援:提高 α 的 LR")
            print("!" * 60)

        # 找到包含 global_mix_alpha 的参数组,提高 LR
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p is self.model.global_mix_alpha:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * self.lr_boost_multiplier
                    self.lr_boosted = True
                    if self.verbose:
                        print(f"     ⚙️ global_mix_alpha LR: {old_lr:.6f} → {param_group['lr']:.6f}")
                    return

        if self.verbose:
            print(f"     ❌ 未找到 global_mix_alpha 的参数组")

    # ====================================================================
    # 辅助损失计算(在主损失之上叠加)
    # ====================================================================
    def compute_aux_loss(self, by_truncated, aux_info):
        extra = torch.tensor(0.0, device=by_truncated.device)

        if self.aux_parallel_active and 'deltas_global' in aux_info:
            deltas = aux_info['deltas_global']
            # ✅ 显式对齐到最小维度,防止维度不一致引发崩溃
            H_min = min(deltas.shape[1], by_truncated.shape[1])
            pred_traj = torch.cumsum(deltas[:, :H_min].float(), dim=1)
            true_traj = torch.cumsum(by_truncated[:, :H_min].float(), dim=1)
            loss_aux = F.smooth_l1_loss(pred_traj, true_traj)
            extra = extra + self.aux_parallel_weight * loss_aux

        if self.mix_l2_active:
            extra = extra + self.mix_l2_weight * (self.model.global_mix_alpha) ** 2

        return extra

    # ====================================================================
    # 状态摘要(打印用)
    # ====================================================================
    def status_string(self):
        alpha_sig = torch.sigmoid(self.model.global_mix_alpha).item()
        flags = []
        if self.aux_parallel_active: flags.append('AUX+')
        if self.mix_l2_active: flags.append('L2+')
        if self.lr_boosted: flags.append('LR↑')
        flag_str = '|'.join(flags) if flags else 'OK'
        return f"α={alpha_sig:.3f} [{flag_str}]"

class OnTheFlyUAVDataset(Dataset):
    def __init__(self, seq_list, indices, max_s, horizon, is_training=False):
        self.seq_list = seq_list
        self.indices = indices
        self.max_s = max_s
        self.horizon = horizon
        self.is_training = is_training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        info = self.indices[idx]
        seq_idx = int(info["seq_idx"])
        start = int(info["start"])
        label = int(info["label"])

        w = self.seq_list[seq_idx][start: start + self.max_s + self.horizon]
        xs, ld, yt = w[:self.max_s], w[self.max_s - 1, :3], w[self.max_s:, :3]

        ang = -np.arctan2(ld[1], ld[0] + 1e-8) + (
            np.random.uniform(0, 2 * np.pi) if self.is_training else 0
        )
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)

        def rot(v):
            o = v.copy()
            if v.shape[-1] >= 3:  o[..., 0:3] = np.dot(v[..., 0:3], R.T)
            if v.shape[-1] >= 6:  o[..., 3:6] = np.dot(v[..., 3:6], R.T)
            if v.shape[-1] >= 9:  o[..., 6:9] = np.dot(v[..., 6:9], R.T)
            if v.shape[-1] >= 12: o[..., 9:12] = np.dot(v[..., 9:12], R.T)

            # 修正旋转后的 Yaw 姿态 (sin/cos_yaw)
            if v.shape[-1] >= 18:
                sin_yaw, cos_yaw = v[..., 14:15], v[..., 17:18]
                new_yaw_vec = np.dot(np.concatenate([cos_yaw, sin_yaw], axis=-1), R[:2, :2].T)
                o[..., 17:18] = new_yaw_vec[..., 0:1]
                o[..., 14:15] = new_yaw_vec[..., 1:2]

                # 🚀 修复点 4：同步旋转“全局航迹角 (ta)”，索引为 21
            # np.dot(..., R.T) 相当于坐标系旋转了 ang，所以全局角度特征也要增加 ang
            if v.shape[-1] >= 22:
                # 使用数学安全的方式将其包裹在 [-π, π] 内
                o[..., 21] = (o[..., 21] + ang + np.pi) % (2 * np.pi) - np.pi

            return o

        return {
            "seq": torch.from_numpy(rot(xs)).float(),
            "last_delta": torch.from_numpy(rot(ld.reshape(1, 3)).reshape(3)).float(),
            "target": torch.from_numpy(rot(yt)).float(),
            "intent": torch.tensor(label, dtype=torch.long),
        }

def measure_hardware_metrics(model, feature_dim, seq_len, device, exp_name, save_dir, num_iterations=50):
    print(f"\n⚙️ 正在压测 [{exp_name}] 的硬件效率指标...")
    model.eval()
    dummy_input = {"seq": torch.randn(1, seq_len, feature_dim).to(device), "last_delta": torch.randn(1, 3).to(device)}
    flops_g, params_m = 0.0, sum(p.numel() for p in model.parameters()) / 1e6
    try:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_g = (macs * 2) / 1e9
    except Exception as e:
        pass
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_iterations, 1))
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for _ in range(10): _ = model(dummy_input)
        for i in range(num_iterations):
            starter.record(); out = model(dummy_input); ender.record()
            torch.cuda.synchronize(); timings[i] = starter.elapsed_time(ender)
    latency_ms = float(np.mean(timings))
    print(f"  ├─ 参数量 (Params): {params_m:.3f} M | 计算量 (FLOPs):  {flops_g:.4f} G | 延迟(Lat): {latency_ms:.2f} ms")
    os.makedirs(save_dir, exist_ok=True)
    hw_csv = os.path.join(save_dir, f"{exp_name}_hardware_metrics.csv")
    pd.DataFrame([{"Model": exp_name, "Params(M)": params_m, "FLOPs(G)": flops_g, "Static_Latency(ms)": latency_ms}]).to_csv(hw_csv, mode='a', header=not os.path.exists(hw_csv), index=False)
    return params_m, flops_g, latency_ms

def _compute_intensity(y_real, y_traj):
    linear = np.mean(np.linalg.norm(np.diff(y_real, axis=1), axis=-1), axis=1) if y_real.shape[1] >= 2 else np.zeros(len(y_real))
    curv = np.zeros(len(y_real))
    if y_traj.shape[1] >= 3:
        v = np.diff(y_traj, axis=1); vu = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        curv = np.mean(1 - np.clip(np.sum(vu[:, :-1] * vu[:, 1:], axis=-1), -1, 1), axis=1)
    spd_std = np.std(np.linalg.norm(y_real, axis=-1), axis=1)
    norm01 = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    return 0.5 * norm01(linear) + 0.3 * norm01(curv) + 0.2 * norm01(spd_std)

def load_group_indices(exp_name, save_dir="maneuver_results_short_uncerity3.2"):
    npz_path = os.path.join(save_dir, exp_name, "group_indices.npz")
    if not os.path.exists(npz_path): return None, None, None, False
    data = np.load(npz_path, allow_pickle=False)
    return {'low': data['low'], 'mid': data['mid'], 'high': data['high']}, float(data['p25'][0]), float(data['p75'][0]), True

def to_real(x, scaler):
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().float().numpy().reshape(-1, 3)
    else:
        x_np = np.array(x, dtype=np.float32).reshape(-1, 3)
    return scaler.inverse_transform(x_np).reshape(x.shape)


def prebuild_maneuver_groups(val_loader, scaler, config, force_rebuild=False):
    exp_name, save_dir = config.get("exp_name", "exp"), config.get("maneuver_save_dir", "final_compare_result")
    if not force_rebuild:
        g_idx, p25, p75, succ = load_group_indices(exp_name, save_dir)
        if succ and sum(len(v) for v in g_idx.values()) == len(val_loader.dataset): return g_idx, p25, p75
    all_t_real = [to_real(b["target"].numpy(), scaler) for b in val_loader]
    y_t_real = np.concatenate(all_t_real, axis=0)
    y_t_traj = np.concatenate([np.cumsum(t, axis=1) for t in all_t_real], axis=0)
    intensity = _compute_intensity(y_t_real, y_t_traj)
    p25, p75 = float(np.percentile(intensity, 25)), float(np.percentile(intensity, 75))
    g_idx = {'low': np.where(intensity < p25)[0], 'mid': np.where((intensity >= p25) & (intensity < p75))[0], 'high': np.where(intensity >= p75)[0]}
    out_dir = os.path.join(save_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "group_indices.npz"), low=g_idx['low'], mid=g_idx['mid'], high=g_idx['high'], p25=np.array([p25]), p75=np.array([p75]))
    return g_idx, p25, p75

def calculate_kinematic_fidelity_adaptive(y_pred, y_true):
    """
    自适应动力学保真度评估 (A-KEER, NJER, KPC)
    y_pred, y_true: [N, H, 3] 反归一化后的累积物理坐标
    """
    H = y_pred.shape[1]
    if H < 3:
        return 0.0, 1.0, 0.0

    v_pred = np.diff(y_pred, axis=1)  # [N, H-1, 3]
    a_pred = np.diff(v_pred, axis=1)  # [N, H-2, 3]
    j_pred = np.diff(a_pred, axis=1)  # [N, H-3, 3]

    v_true = np.diff(y_true, axis=1)
    a_true = np.diff(v_true, axis=1)
    j_true = np.diff(a_true, axis=1)

    a_pred_scalar = np.linalg.norm(a_pred, axis=-1)  # [N, H-2]
    a_true_scalar = np.linalg.norm(a_true, axis=-1)  # [N, H-2]

    # 1. A-KEER (自适应包络越限率) - 允许 10% 裕度
    gamma = 1.1
    a_max_gt = np.max(a_true_scalar, axis=1, keepdims=True)
    a_max_gt_safe = np.maximum(a_max_gt, 1.0)
    violations = (a_pred_scalar > (gamma * a_max_gt_safe))
    a_keer_pct = float(np.mean(violations) * 100.0)

    # 2. NJER (归一化抖动能量比)
    j_energy_pred = np.sum(np.linalg.norm(j_pred, axis=-1) ** 2, axis=1)
    j_energy_true = np.sum(np.linalg.norm(j_true, axis=-1) ** 2, axis=1)
    njer_mean = float(np.mean(j_energy_pred / (j_energy_true + 1e-8)))

    # 3. KPC (加速度轮廓相关性)
    kpc_list = []
    for i in range(len(a_pred_scalar)):
        A_p, A_t = a_pred_scalar[i], a_true_scalar[i]
        if np.std(A_p) < 1e-5 or np.std(A_t) < 1e-5:
            kpc_list.append(0.0)
        else:
            kpc_list.append(np.corrcoef(A_p, A_t)[0, 1])
    kpc_mean = float(np.mean(kpc_list))

    return a_keer_pct, njer_mean, kpc_mean


def evaluate_by_fixed_groups(group_indices, err_matrix, y_p_traj, y_t_traj, y_samples, std_real, horizon, dir_thresh_deg=15.0):
    dir_thresh, group_names, group_results = np.deg2rad(dir_thresh_deg), {'low': '平稳', 'mid': '普通', 'high': '急剧'}, {}
    for key, indices in group_indices.items():
        if len(indices) == 0: continue
        g_err, g_p_traj, g_t_traj, g_std_real, g_samples = err_matrix[indices], y_p_traj[indices], y_t_traj[indices], std_real[indices], y_samples[:, indices]
        err_samples = np.linalg.norm(g_samples - g_t_traj[np.newaxis], axis=-1)
        mean_accel = max_accel = mean_jerk = 0.0
        if horizon >= 4:
            a_p = np.diff(np.diff(g_p_traj, axis=1), axis=1)
            mean_accel, max_accel, mean_jerk = float(np.mean(np.linalg.norm(a_p, axis=-1))), float(np.max(np.linalg.norm(a_p, axis=-1))), float(np.mean(np.linalg.norm(np.diff(a_p, axis=1), axis=-1)))
        v_rate, jc_fde, b_score = compute_advanced_metrics(g_samples, g_t_traj, g_p_traj)
        a_keer, njer, kpc = calculate_kinematic_fidelity_adaptive(g_p_traj, g_t_traj)
        step_metrics = []
        for h in range(horizon):
            p_v = (1 - min(1.0, h / 15.0)) * (g_p_traj[:, h, :2] if h == 0 else g_p_traj[:, h, :2] - g_p_traj[:, h - (2 if h >= 2 else h), :2]) + min(1.0, h / 15.0) * (g_p_traj[:, h, :2] - g_p_traj[:, 0, :2])
            t_v = (1 - min(1.0, h / 15.0)) * (g_t_traj[:, h, :2] if h == 0 else g_t_traj[:, h, :2] - g_t_traj[:, h - (2 if h >= 2 else h), :2]) + min(1.0, h / 15.0) * (g_t_traj[:, h, :2] - g_t_traj[:, 0, :2])
            n_p, n_t = np.linalg.norm(p_v, axis=1), np.linalg.norm(t_v, axis=1); valid = n_t > 0.05
            var_h = (g_std_real[:, h, :] + 1e-8) ** 2
            step_metrics.append({
                'step': h + 1, 'dist_error_m': float(np.mean(g_err[:, h])), 'step_mse': float(np.mean(g_err[:, h] ** 2)), 'step_rmse': float(np.sqrt(np.mean(g_err[:, h] ** 2))),
                'dist_acc_pct': float(np.mean(g_err[:, h] < min(0.25 + h * 0.15, 1.5)) * 100),
                'dir_acc_pct': float(np.mean(np.arccos(np.clip(np.sum(p_v[valid] * t_v[valid], axis=1) / (n_p[valid] * n_t[valid] + 1e-8), -1, 1)) < dir_thresh) * 100) if valid.sum() > 0 else 0.0,
                'nll': float(np.mean(0.5 * np.log(2*np.pi*var_h) + ((g_t_traj[:, h, :] - g_p_traj[:, h, :])**2) / (2*var_h))),
                'pic_95': float(np.mean((g_t_traj[:, h, :] >= g_p_traj[:, h, :] - 1.96*g_std_real[:, h, :]) & (g_t_traj[:, h, :] <= g_p_traj[:, h, :] + 1.96*g_std_real[:, h, :])) * 100.0),
                'miw_95': float(np.mean(2 * 1.96 * g_std_real[:, h, :]))
            })
        group_results[key] = {
            'name': group_names[key], 'n_samples': len(indices), 'ade': float(np.mean(g_err)), 'fde': float(np.mean(g_err[:, -1])),
            'min_ade': float(np.mean(np.min(np.mean(err_samples, axis=2), axis=0))), 'min_fde': float(np.mean(np.min(err_samples[:, :, -1], axis=0))),
            'mean_accel': mean_accel, 'max_accel': max_accel, 'mean_jerk': mean_jerk, 'A_KEER': a_keer, 'NJER': njer, 'KPC': kpc,
            'violations_rate': v_rate, 'jc_min_fde': jc_fde, 'brier_score': b_score, 'step_metrics': step_metrics
        }
    return group_results

class ScaleAttentionGate(nn.Module):
    def __init__(self, d_model, n_scales):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(d_model * n_scales, d_model), nn.Tanh(), nn.Linear(d_model, n_scales))
    def forward(self, scale_feats_list):
        combined = torch.cat(scale_feats_list, dim=-1)
        weights = torch.softmax(self.attention_net(combined).float(), dim=-1).to(combined.dtype)
        return torch.sum(torch.stack(scale_feats_list, dim=1) * weights.unsqueeze(-1), dim=1), weights

class KinematicProjectionLayer(nn.Module):
    def __init__(self, d_model, drop=0.1):
        super().__init__()
        self.global_proj = nn.Sequential(nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.xy_proj = nn.Sequential(nn.Linear(2, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d_model // 2, d_model // 2))
        self.z_proj = nn.Sequential(nn.Linear(1, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d_model // 2, d_model // 2))
        self.fusion_net = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Linear(d_model, d_model))
        nn.init.normal_(self.fusion_net[-1].weight, std=0.02); nn.init.zeros_(self.fusion_net[-1].bias)
    def forward(self, delta):
        feat_global = self.global_proj(delta)
        return feat_global + self.fusion_net(torch.cat([feat_global, torch.cat([self.xy_proj(delta[:, :2]), self.z_proj(delta[:, 2:3])], dim=-1)], dim=-1))

class EnhancedMambaBlock(nn.Module):
    def __init__(self, d_model, use_multiscale_conv=True, use_channel_attn=True):
        super().__init__()
        self.mamba = get_mamba2()(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
        self.convs = nn.ModuleList([nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model) for k in ((3, 5, 7) if use_multiscale_conv else (3,))])
        self.ca = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, d_model), nn.Sigmoid()) if use_channel_attn else None
        self.drop = nn.Dropout(0.15)
    def forward(self, x):
        xm = self.mamba(self.norm(x))
        xc = sum(c(self.norm(xm + x).transpose(1, 2)) for c in self.convs) / len(self.convs)
        xc = xc.transpose(1, 2)
        return self.drop(F.silu((xc * self.ca(xc) if self.ca is not None else xc) + (xm + x))) * 0.8 + x * 0.2

class ParallelTemporalDecoder(nn.Module):
    def __init__(self, d_model=128, horizon=20, n_heads=4, dropout=0.15, use_self_attn=True):
        super().__init__()
        self.horizon, self.use_self_attn = horizon, use_self_attn
        self.time_queries = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.time_pos_embed = nn.Embedding(horizon, d_model); nn.init.normal_(self.time_pos_embed.weight, std=0.02)
        self.self_attn, self.norm0 = (nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout), nn.LayerNorm(d_model)) if use_self_attn else (None, nn.Identity())
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.delta_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 3))
        nn.init.normal_(self.delta_head[-1].weight, std=0.02); nn.init.zeros_(self.delta_head[-1].bias)
    def forward(self, context_seq, intent_bias=None, current_horizon=None):
        H = current_horizon if current_horizon is not None else self.horizon
        queries = (self.time_queries[:H] + self.time_pos_embed(torch.arange(H, device=context_seq.device))).unsqueeze(0).expand(context_seq.size(0), -1, -1)
        if intent_bias is not None: queries = queries + intent_bias.unsqueeze(1)
        if self.self_attn is not None: queries = self.norm0(queries + self.self_attn(queries, queries, queries)[0])
        queries = self.norm2(self.norm1(queries + self.cross_attn(queries, context_seq, context_seq)[0]) + self.ffn(self.norm1(queries + self.cross_attn(queries, context_seq, context_seq)[0])))
        return self.delta_head(queries)

class PhysicsGatedPredictor(nn.Module):
    def __init__(self, d_model, cond_dim=16, hidden_dim=128, horizon=20, drop_res=0.15, use_kpl=True, use_magnitude_head=True, use_intent_alpha=True):
        super().__init__()
        self.horizon, self.use_kpl, self.use_magnitude_head, self.use_intent_alpha = horizon, use_kpl, use_magnitude_head, use_intent_alpha
        self.res_pred = nn.Sequential(nn.Linear(d_model + cond_dim + 16, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(drop_res), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 6))
        self.ch_net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))
        self.time_embed = nn.Embedding(horizon, 16); nn.init.normal_(self.time_embed.weight, std=0.02)
        self.gru = nn.GRU(input_size=9, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gru_init = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.Tanh())
        self.residual_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(drop_res), nn.Linear(hidden_dim // 2, 3))
        nn.init.zeros_(self.residual_proj[-1].weight); nn.init.zeros_(self.residual_proj[-1].bias)
        if use_kpl: self.kpl = KinematicProjectionLayer(d_model, drop=drop_res); self.kpl_mix = nn.Parameter(torch.tensor(-2.0))
        if use_magnitude_head:
            self.magnitude_head = nn.Sequential(nn.Linear(hidden_dim + d_model, hidden_dim // 2), nn.GELU(), nn.Dropout(drop_res), nn.Linear(hidden_dim // 2, 1))
            nn.init.zeros_(self.magnitude_head[-1].weight); nn.init.constant_(self.magnitude_head[-1].bias, 0.2)
            self.high_mag_boost = nn.Parameter(torch.tensor(0.3))
        self.alpha_per_intent = nn.Parameter(torch.tensor([-1.5, -0.4, 0.62])) if use_intent_alpha else nn.Parameter(torch.tensor(0.0))
        self.time_log_var_bias = nn.Parameter((torch.log(torch.arange(1, horizon + 1, dtype=torch.float32)) - 1.0).unsqueeze(1).repeat(1, 3).contiguous())

    def init_gru_state(self, hc): return self.gru_init(hc).unsqueeze(0)

    def forward(self, ch_prev, npd, npd_prev, t, h_gru, iw, mac, intent_bias, step_idx=0):
        ch_enriched = ch_prev + torch.sigmoid(self.kpl_mix) * self.kpl(npd) if self.use_kpl else ch_prev
        out_6d = self.res_pred(torch.cat([ch_enriched, torch.cat([npd, mac, iw, self.time_embed(torch.tensor(step_idx, device=npd.device)).unsqueeze(0).expand(npd.size(0), -1)], dim=-1)], dim=-1))
        gru_out, h_gru_new = self.gru(torch.cat([npd, npd - npd_prev, iw], dim=-1).unsqueeze(1), h_gru)
        dir_residual = self.residual_proj(gru_out.squeeze(1))
        mag_factor = torch.exp(torch.clamp(self.magnitude_head(torch.cat([gru_out.squeeze(1), ch_enriched], dim=-1)) + iw[:, 2:3] * self.high_mag_boost, min=-1.5 + 0.5 * (1.0 - iw[:, 2:3]), max=1.5 + 1.5 * iw[:, 2:3])) if self.use_magnitude_head else 1.0
        alpha_per_b = (iw * (torch.sigmoid(self.alpha_per_intent) * 0.5)).sum(-1, keepdim=True) * (1.0 + 1.5 * iw[:, 2:3]) if self.use_intent_alpha else torch.sigmoid(self.alpha_per_intent).expand(npd.size(0), 1) * 0.5
        action = out_6d[:, :3] + alpha_per_b * dir_residual
        log_var = torch.clamp(out_6d[:, 3:] + self.time_log_var_bias[step_idx] + ((iw * torch.tensor([0.0, 0.3, 1.5], device=npd.device)).sum(-1, keepdim=True) if self.use_intent_alpha else 0.0), min=2.0 * torch.log(torch.linspace(0.015, 0.12, self.horizon, device=npd.device)[step_idx]), max=3.0 + 0.18 * step_idx + 1.5 * iw[:, 2:3].mean())
        return torch.cat([action[:, :2] * mag_factor, action[:, 2:3]], dim=-1), self.ch_net(ch_enriched), log_var, h_gru_new


# ==========================================================
# 3. 意图感知预测器主干 (修复版)
# ==========================================================
class IntentAwarePredictor(nn.Module):
    """
    v3 混合架构 + 消融开关版
    ablation_config: {'emam': bool, 'dtp': bool, 'parallel': bool, 'pgd': bool}
    """

    def __init__(self, feature_dim, d_model=128, horizon=10,
                 scales=(5, 15, 30, 50), d_scaler_scale=None,
                 ablation_config=None):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.scales = sorted(scales)

        # ✨ 默认全开 = Ours-Full
        self.ablation = ablation_config or {
            'emam': True, 'dtp': True, 'parallel': True, 'pgd': True,
        }
        print(f"  🔧 当前消融配置: {self.ablation}")

        # MC 采样状态
        self.mc_sampling_active = False
        self.mc_latent_noise_scale = 0.3
        self.mc_step_noise_base = 0.04
        self.mc_step_noise_growth = 4.5

        # ============ ① 编码器 (EMam-SE 消融点) ============
        self.proj = nn.Linear(feature_dim, d_model)
        if self.ablation.get('emam', True):
            self.encoder = EnhancedMambaBlock(
                d_model,
                use_multiscale_conv=self.ablation.get('emam_multiscale_conv', True),
                use_channel_attn=self.ablation.get('emam_channel_attn', True),
            )
            self._encoder_kind = 'mamba'
        else:
            self.encoder = nn.GRU(d_model, d_model, batch_first=True)
            self._encoder_kind = 'gru'

        # ============ ② 多尺度池化 (IA-DTP 消融点) ============
        if self.ablation.get('dtp', True):
            self.pyr = lambda x: [
                x[:, -1, :] - x[:, max(0, x.shape[1] - s), :]
                for s in self.scales
            ]
            if self.ablation.get('dtp_attention_gate', True):
                self.sg = ScaleAttentionGate(d_model, len(self.scales))
                self._dtp_mode = 'gate'
            else:
                # 微观消融:用均值融合替代可学习注意力
                self.sg = None
                self._dtp_mode = 'mean'

        self.global_dropout = nn.Dropout(DROPOUT_BASE * 1.5)

        # ============ 公共意图分类(始终保留)============
        self.i_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(DROPOUT_HEAVY),
            nn.Linear(d_model // 2, 3),
        )
        self.i_bias = nn.Embedding(3, d_model)
        self.glob_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())

        # ============ ③ 并行解码器 (Parallel 消融点) ============
        if self.ablation.get('parallel', True):
            self.parallel_decoder = ParallelTemporalDecoder(
                d_model=d_model, horizon=horizon, n_heads=4, dropout=0.15,
                use_self_attn=self.ablation.get('parallel_self_attn', True),
            )
        self.global_mix_alpha = nn.Parameter(torch.tensor(-0.85))

        # ============ ④ AR-GRU refiner (UA-PGD 消融点) ============
        if self.ablation.get('pgd', True):
            self.phy = PhysicsGatedPredictor(
                d_model, horizon=horizon,
                use_kpl=self.ablation.get('pgd_kpl', True),
                use_magnitude_head=self.ablation.get('pgd_magnitude_head', True),
                use_intent_alpha=self.ablation.get('pgd_intent_alpha', True),
            )
        else:
            # 退化为最朴素的 GRUCell + 两个 MLP 头
            self.vanilla_gru = nn.GRUCell(d_model + 3, d_model)
            self.vanilla_act_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2),
                nn.GELU(), nn.Linear(d_model // 2, 3),
            )
            self.vanilla_lv_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 3),
            )
            # 时间增长先验
            self.vanilla_lv_time_bias = nn.Parameter(
                torch.linspace(-1.0, 1.0, horizon).unsqueeze(-1).repeat(1, 3).contiguous())

        # ============ 平滑器(始终保留)============
        self.sm = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(64, 3, kernel_size=3, padding=1),
        )

        # 物理约束
        if d_scaler_scale is not None:
            scale = torch.tensor(d_scaler_scale, dtype=torch.float32)
            self.register_buffer("max_vxy_norm", torch.tensor(5.0) / scale[:2].max())
            self.register_buffer("max_vz_norm", torch.tensor(2.0) / scale[2])
        else:
            self.register_buffer("max_vxy_norm", torch.tensor(20.0))
            self.register_buffer("max_vz_norm", torch.tensor(10.0))

        self.output_scale = nn.Parameter(torch.ones(3) * 1.5)

        # 转向 & Goal anchor(始终保留)
        self.turn_rate_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d_model // 2, horizon),
        )
        self.K_anchors = 3
        self.goal_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2),
            nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d_model // 2, self.K_anchors * 3 + self.K_anchors),
        )
        nn.init.normal_(self.goal_head[-1].weight, std=0.01)
        nn.init.zeros_(self.goal_head[-1].bias)
        self.goal_correction_strength = 0.0

        # 同方差消融
        if self.ablation.get('unc_mode', 'hetero') == 'homo':
            self.homo_log_var = nn.Parameter(torch.zeros(1, horizon, 3))

    # ------------------------------------------------------------------
    def forward(self, x_dict, tau=1.0, intent_targets=None,
                target_deltas=None, tf_mask=None, current_horizon=None,
                return_aux=False):
        seq, pd = x_dict["seq"], x_dict["last_delta"]
        B, dev = seq.size(0), seq.device

        # 保护边界，防止传入的 current_horizon 超出初始化时的 horizon
        H = current_horizon if current_horizon is not None else self.horizon
        H = min(H, self.horizon)

        # ① 编码 (EMam-SE 消融)
        seq_proj = self.proj(seq)
        if self._encoder_kind == 'mamba':
            xm = self.encoder(seq_proj)
        else:
            xm, _ = self.encoder(seq_proj)

        # ② 多尺度特征 (IA-DTP 消融)
        if self.ablation.get('dtp', True):
            scale_feats = self.pyr(xm)
            if self._dtp_mode == 'gate':
                hc_raw = self.sg(scale_feats)[0]
            else:
                # 均值融合 baseline
                hc_raw = torch.stack(scale_feats, dim=1).mean(dim=1)
        else:
            hc_raw = xm[:, -1, :]

        # MC 多样性源 ①:latent noise
        diversity_mode = self.training or self.mc_sampling_active
        if diversity_mode:
            if self.mc_sampling_active:
                il_pre = self.i_head(hc_raw)
                high_prob = torch.softmax(il_pre, -1)[:, 2:3]
                noise_scale = self.mc_latent_noise_scale * (1.0 + 1.5 * high_prob)
            else:
                noise_scale = self.mc_latent_noise_scale
            hc = self.global_dropout(hc_raw + torch.randn_like(hc_raw) * noise_scale)
        else:
            hc = self.global_dropout(hc_raw)

        # 意图
        il = self.i_head(hc)
        if self.training and intent_targets is not None:
            iw = F.one_hot(intent_targets, 3).float()
        elif self.mc_sampling_active:
            probs = torch.softmax(il / tau, -1)
            sampled = torch.distributions.Categorical(probs=probs.float()).sample()
            iw = F.one_hot(sampled, 3).float()
        else:
            iw = torch.softmax(il / tau, -1)

        # 获取辅助转向率预测
        turn_rates_pred = self.turn_rate_head(hc)

        intent_bias = iw @ self.i_bias.weight
        h0 = self.glob_proj(hc + intent_bias)

        # ③ 并行解码器 (Parallel 消融)
        if self.ablation.get('parallel', True):
            deltas_global = self.parallel_decoder(
                context_seq=xm, intent_bias=intent_bias, current_horizon=H)
        else:
            deltas_global = None

        # ④ AR 状态初始化 (UA-PGD 消融)
        if self.ablation.get('pgd', True):
            h_gru = self.phy.init_gru_state(hc)
        else:
            h_gru = torch.zeros(B, self.d_model, device=dev)

        prev_pd = seq[:, -2, :3].detach() if seq.shape[1] >= 2 else torch.zeros_like(pd)
        cur_pd = pd
        ch = hc
        ps_list, lv_list = [], []
        buf = seq[:, -15:, 0:3].detach()

        global_mix = torch.sigmoid(self.global_mix_alpha)
        high_w = iw[:, 2:3]
        local_global_mix = global_mix * (1.0 - 0.5 * high_w)

        for t in range(H):
            if t > 0 and t % 5 == 0:
                refresh_w = min(0.3, 0.1 + 0.05 * (t / 5))
                ch = (1 - refresh_w) * ch + refresh_w * h0

            npd = cur_pd
            sum_5 = buf[:, -5:].sum(1)
            sum_15 = buf[:, -15:].sum(1)
            dir_5 = F.normalize(sum_5[:, :2], dim=-1, eps=1e-8)
            dir_15 = F.normalize(sum_15[:, :2], dim=-1, eps=1e-8)
            mac = torch.cat([sum_5, sum_15, dir_5, dir_15], dim=-1)

            # AR 步进 (UA-PGD 消融)
            if self.ablation.get('pgd', True):
                act_ar, ch, lv, h_gru = self.phy(
                    ch, npd, prev_pd, t, h_gru, iw, mac, intent_bias, step_idx=t)
            else:
                h_gru = self.vanilla_gru(torch.cat([npd, ch], dim=-1), h_gru)
                act_ar = self.vanilla_act_head(h_gru)
                lv = self.vanilla_lv_head(h_gru) + self.vanilla_lv_time_bias[t]
                lv = torch.clamp(lv, min=-4.0, max=4.0)
                ch = h_gru

            # 融合 (Parallel 消融)
            if self.ablation.get('parallel', True) and deltas_global is not None:
                act = (1.0 - local_global_mix) * act_ar + local_global_mix * deltas_global[:, t, :]
            else:
                act = act_ar

            act = act * self.output_scale.clamp(0.1, 2.0)
            act_xy = act[:, :2]
            act_z = act[:, 2:3]

            # ✨ 微观消融 1: 惯性剥夺策略
            cur_dir = F.normalize(cur_pd[:, :2], dim=-1, eps=1e-8)
            act_dir = F.normalize(act_xy, dim=-1, eps=1e-8)
            w_high_t = iw[:, 2]
            base_blend = max(0.0, 0.4 - t * 0.025)

            inertia_mode = self.ablation.get('inertia_mode', 'sqrt')
            if inertia_mode == 'fixed':
                blend_scalar = torch.full_like(w_high_t, base_blend)
            elif inertia_mode == 'linear':
                blend_scalar = base_blend * (1.0 - w_high_t)
            else:  # 'sqrt' (Ours)
                blend_scalar = base_blend * (1.0 - torch.sqrt(w_high_t + 1e-8))

            blend = blend_scalar.unsqueeze(1)
            blended = F.normalize(blend * cur_dir + (1.0 - blend) * act_dir, dim=-1, eps=1e-8)
            act_xy = blended * torch.norm(act_xy, dim=-1, keepdim=True).clamp(min=1e-8)

            act_xy = torch.clamp(act_xy, -6.0, 6.0)
            act_z = torch.clamp(act_z, -3.0, 3.0)
            pure = torch.cat([act_xy, act_z], dim=-1)

            # ✨ 微观消融 2: 不确定性建模模式
            unc_mode = self.ablation.get('unc_mode', 'hetero')
            if unc_mode == 'none':
                lv = torch.zeros_like(lv) - 2.0  # 极小方差，等价于确定性预测
            elif unc_mode == 'homo':
                lv = self.homo_log_var[:, t, :].expand(B, -1)  # 全局同方差参数

            # MC 多样性源 ③: AR step noise
            if self.mc_sampling_active:
                high_w_step = iw[:, 2:3]
                intent_noise_boost = 1.0 + 1.2 * high_w_step
                noise_scale = self.mc_step_noise_base * (
                        1.0 + (t / max(H - 1, 1)) * self.mc_step_noise_growth)
                pure = pure + torch.randn_like(pure) * noise_scale * intent_noise_boost

            ps_list.append(pure)
            lv_list.append(lv)

            if self.training and tf_mask is not None and target_deltas is not None:
                cur_pd_new = torch.where(tf_mask[:, t:t + 1], target_deltas[:, t], pure.detach())
            else:
                cur_pd_new = pure.detach()

            prev_pd = cur_pd.detach()
            cur_pd = cur_pd_new
            buf = torch.cat([buf[:, 1:], pure.unsqueeze(1).detach()], dim=1)

        ps = torch.stack(ps_list, dim=1)
        lvs = torch.stack(lv_list, dim=1)

        # 平滑器
        ps_detached = ps.detach()
        sm_residual = self.sm(ps_detached.transpose(1, 2)).transpose(1, 2)
        low_w_2d = iw[:, 0].unsqueeze(1).unsqueeze(2)
        mid_w_2d = iw[:, 1].unsqueeze(1).unsqueeze(2)
        high_w_2d = iw[:, 2].unsqueeze(1).unsqueeze(2)
        smooth_weight = 1.0 * low_w_2d + 0.6 * mid_w_2d + 0.25 * high_w_2d
        final_ps = ps + smooth_weight * sm_residual

        # 安全裁剪 turn_rates
        safe_turn_len = min(H, turn_rates_pred.shape[1])
        turn_rates_pred_trimmed = turn_rates_pred[:, :safe_turn_len]

        # Goal anchor
        goal_raw = self.goal_head(hc)
        goals_xyz = goal_raw[:, :self.K_anchors * 3].view(B, self.K_anchors, 3)
        goal_logits = goal_raw[:, self.K_anchors * 3:]
        goal_probs = F.softmax(goal_logits, -1)

        if self.mc_sampling_active and self.goal_correction_strength > 0.01:
            high_w_scalar = iw[:, 2:3]
            effective_strength = self.goal_correction_strength * (1.0 + high_w_scalar)
            goal_idx = torch.distributions.Categorical(probs=goal_probs.float()).sample()
            selected_goal = torch.gather(goals_xyz, 1,
                                         goal_idx.view(B, 1, 1).expand(-1, 1, 3)).squeeze(1)
            final_endpoint = torch.cumsum(final_ps, dim=1)[:, -1, :]
            base_correction = 0.15 * effective_strength
            endpoint_correction = (selected_goal - final_endpoint) * base_correction
            correction_per_step = endpoint_correction.unsqueeze(1) / max(H // 2, 1)
            correction_mask = torch.zeros(1, H, 1, device=dev)
            correction_mask[:, H // 2:, :] = 1.0
            final_ps = final_ps + correction_per_step * correction_mask

        # 🚀 修复点：结构化返回逻辑，消除复杂三元推导
        if return_aux:
            aux_info = {
                'deltas_global': deltas_global if deltas_global is not None else torch.zeros(B, H, 3, device=dev),
                'global_mix_value': global_mix.detach(),
                'sm_residual': sm_residual,
                'ps_raw': ps,
                'intent_logits': il.detach(),
                'goals_xyz': goals_xyz,
                'goal_logits': goal_logits,
            }
            return final_ps, lvs, il, turn_rates_pred_trimmed, aux_info

        return final_ps, lvs, il, turn_rates_pred_trimmed

    def generate_probabilistic_trajectories(self, x_dict, num_samples=20):
        def _enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        was_training, prev_mc = self.training, self.mc_sampling_active
        self.eval()
        self.apply(_enable_dropout)
        self.mc_sampling_active = True
        all_pred = []
        try:
            with torch.no_grad():
                for _ in range(num_samples):
                    # 获取均值轨迹
                    pred = self.forward(x_dict)[0]
                    all_pred.append(pred)
        finally:
            self.mc_sampling_active = prev_mc
            if was_training:
                self.train()
            else:
                self.eval()

        return torch.stack(all_pred, dim=0)

_Mamba2 = None

def get_mamba2():
    global _Mamba2
    if _Mamba2 is None:
        mamba_main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba-main")
        if mamba_main_dir not in sys.path:
            sys.path.append(mamba_main_dir)
        module_m = importlib.import_module("mamba_ssm.modules.mamba2")
        _Mamba2 = getattr(module_m, "Mamba2")
    return _Mamba2

def make_ablation_config(**overrides):
    """
    生成完整 ablation_config。只覆盖你想关掉的子开关。
    主开关 False 时,所属子开关自动失效。
    """
    default = {
        # ===== 宏观主开关(整个模块) =====
        'emam':     True,
        'dtp':      True,
        'pgd':      True,

        # ===== EMam-SE / IA-DTP / Parallel 内部子开关 =====
        'emam_multiscale_conv': True,
        'emam_channel_attn':    True,
        'dtp_attention_gate':   True,
        'parallel_self_attn':   True,

        # ===== UA-PGD 内部子开关 (结构层) =====
        'pgd_parallel':         True,
        'pgd_kpl':              True,
        'pgd_magnitude_head':   True,
        'pgd_intent_alpha':     True,

        # ✨ 新增：UA-PGD 内部子开关 (数学机理层)
        'inertia_mode': 'sqrt',      # 可选: 'sqrt'(Ours), 'linear', 'fixed'
        'unc_mode': 'hetero',        # 可选: 'hetero'(Ours), 'homo', 'none'
        'sample_mode': 'kinematic'   # 可选: 'kinematic'(Ours), 'unconstrained'
    }
    default.update(overrides)
    return default

def compute_group_score_v2(group_res, weight=1.0):
    """
    分组评分 v2.1:针对距离精度优化
    """
    if not group_res or 'step_metrics' not in group_res:
        return 0.0

    ade = group_res.get('ade', 0)
    fde = group_res.get('fde', 0)
    min_fde = group_res.get('min_fde', fde)

    steps = group_res['step_metrics']
    H_actual = len(steps)

    s1 = max(1, H_actual // 5)
    s2 = max(s1 + 1, H_actual * 3 // 5)
    short_steps = steps[:s1]
    mid_steps = steps[s1:s2]
    long_steps = steps[s2:]

    def _avg(lst, key):
        return float(np.mean([s[key] for s in lst])) if lst else 0.0

    # ✨ A. 距离精度(权重大幅提升,目标提高)
    short_dist = _avg(short_steps, 'dist_acc_pct')
    mid_dist = _avg(mid_steps, 'dist_acc_pct')
    long_dist = _avg(long_steps, 'dist_acc_pct')

    dist_cost = (
        max(0, 98 - short_dist) * 0.08     # 目标 98%
        + max(0, 85 - mid_dist) * 0.18      # 目标 85%
        + max(0, 65 - long_dist) * 0.30     # 目标 65%(原 55%)
    )

    # ✨ B. 方向精度(平稳/普通要求高,急剧不强制)
    short_dir = _avg(short_steps, 'dir_acc_pct')
    mid_dir = _avg(mid_steps, 'dir_acc_pct')
    long_dir = _avg(long_steps, 'dir_acc_pct')

    dir_cost = (
        max(0, 90 - short_dir) * 0.10
        + max(0, 70 - mid_dir) * 0.18
        + max(0, 55 - long_dir) * 0.22
    )

    # C. 概率校准
    short_pic = _avg(short_steps, 'pic_95')
    mid_pic = _avg(mid_steps, 'pic_95')
    long_pic = _avg(long_steps, 'pic_95')

    pic_cost = (
        abs(short_pic - 92.0) * 0.03
        + abs(mid_pic - 88.0) * 0.05
        + abs(long_pic - 82.0) * 0.08
    )

    long_nll = _avg(long_steps, 'nll')
    nll_cost = max(0, long_nll - 2.0) * 0.30

    long_miw = _avg(long_steps, 'miw_95')
    miw_cost = max(0, long_miw - 2.8) * 0.50

    # ✨ D. 基础误差(加权 ADE)
    base_cost = 1.2 * ade + 0.6 * fde + 0.3 * min_fde   # ✨ 提权

    return weight * (
        base_cost + dist_cost + dir_cost + pic_cost + nll_cost + miw_cost
    )

def get_curriculum_horizon(progress, max_horizon=25):
    if progress < 0.08: return min(8, max_horizon)
    if progress < 0.20: return min(12, max_horizon)
    if progress < 0.40: return min(16, max_horizon)
    if progress < 0.65: return min(20, max_horizon)
    return max_horizon

def compute_rpe(y_pred_traj, y_true_traj, delta=5):
    """
    RPE：每隔 delta 步计算相对位移误差
    衡量局部运动预测质量，不受累积误差影响
    delta=5 → 每5步的相对运动误差
    """
    N, H, D = y_pred_traj.shape
    rpe_list = []
    for t in range(delta, H):
        pred_rel = y_pred_traj[:, t, :] - y_pred_traj[:, t-delta, :]
        true_rel = y_true_traj[:, t, :] - y_true_traj[:, t-delta, :]
        rpe = np.linalg.norm(pred_rel - true_rel, axis=-1)
        rpe_list.append(float(np.mean(rpe)))
    return float(np.mean(rpe_list))

def _direction_decay_diag(group_results, verbose=True):
    """
    计算各场景方向准确率的衰减情况

    返回字典: {group_key: (dir_t5, dir_t15, dir_t25, decay)}
    """
    diag = {}
    name_map = {'low': '平稳', 'mid': '普通', 'high': '急剧'}

    for key in ['low', 'mid', 'high']:
        if key not in group_results:
            continue
        steps = group_results[key]['step_metrics']
        dir_t5 = next((s['dir_acc_pct'] for s in steps if s['step'] == 5), 0.0)
        dir_t15 = next((s['dir_acc_pct'] for s in steps if s['step'] == 15), 0.0)
        dir_t25 = next((s['dir_acc_pct'] for s in steps if s['step'] == 25), 0.0)
        decay = dir_t5 - dir_t25
        diag[key] = {
            't5': dir_t5, 't15': dir_t15, 't25': dir_t25, 'decay': decay
        }

    if verbose:
        print(f"  🧭 方向衰减监控:")
        # 健康标准(对应平稳/普通/急剧)
        healthy_decay = {'low': 20.0, 'mid': 25.0, 'high': 20.0}
        for key in ['low', 'mid', 'high']:
            if key not in diag:
                continue
            d = diag[key]
            limit = healthy_decay[key]
            if d['decay'] < limit:
                status = '✓'
            elif d['decay'] < limit + 10:
                status = '⚠️'
            else:
                status = '❌'
            print(f"     [{name_map[key]}] T+5={d['t5']:.1f}% "
                  f"T+15={d['t15']:.1f}% T+25={d['t25']:.1f}% "
                  f"(衰减={d['decay']:.1f}pt {status})")

    return diag

def print_maneuver_group_results(group_results, epoch):
    print(f"\n{'=' * 110}\n  Epoch {epoch} ── 分机动强度场景：概率与确定性分步评估\n{'=' * 110}")
    for key in ['low', 'mid', 'high']:
        if key not in group_results: continue
        g = group_results[key]
        print(f"\n【{g['name']}】 样本: {g['n_samples']} | ADE: {g['ade']:.4f}m | FDE: {g['fde']:.4f}m")

        # 🚀 修复点 1：表头严格补充 MSE 和 RMSE，总计 9 列
        print(
            f"  {'步骤':<4} | {'距误(m)':<8} | {'MSE':<10} | {'RMSE':<8} | {'距准(%)':<8} | {'向准(%)':<8} | {'NLL':<10} | {'PIC(%)':<8} | {'MIW(m)':<8}")
        print(f"  {'-' * 105}")

        for s in g['step_metrics']:
            # 🚀 修复点 2：NLL 在早期经常发生数值爆炸，用科学计数法保护排版
            nll_val = s['nll']
            nll_str = f"{nll_val:<10.2f}" if abs(nll_val) < 99999.0 else f"{nll_val:<10.2e}"

            # 🚀 修复点 3：数据行严格与表头对齐，注意字典的键名必须是 step_mse 和 step_rmse
            print(
                f"  T+{s['step']:<2} | {s['dist_error_m']:<8.4f} | {s['step_mse']:<10.4f} | {s['step_rmse']:<8.4f} | {s['dist_acc_pct']:<8.2f} | "
                f"{s['dir_acc_pct']:<8.2f} | {nll_str} | {s['pic_95']:<8.1f} | {s['miw_95']:<8.4f}")


def save_maneuver_group_results(group_results, epoch, exp_name, save_dir, is_new_experiment):
    out_dir = os.path.join(save_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    summary_rows, detail_rows = [], []

    for key, g in group_results.items():
        summary_rows.append({
            'epoch': epoch, 'group_key': key, 'group_name': g['name'],
            'n_samples': g['n_samples'], 'ade': g['ade'], 'fde': g['fde']
        })
        for s in g['step_metrics']:
            detail_rows.append({
                'epoch': epoch, 'group_key': key, 'group_name': g['name'], 'n_samples': g['n_samples'],
                'step': s['step'],
                'dist_error_m': s['dist_error_m'],
                'step_MSE': s['step_mse'],  # 🚀 确保键名为 step_mse
                'step_RMSE': s['step_rmse'],  # 🚀 确保键名为 step_rmse
                'dist_acc_pct': s['dist_acc_pct'],
                'dir_acc_pct': s['dir_acc_pct'],
                'NLL': s['nll'],
                'PIC_95': s['pic_95'],
                'MIW_95': s['miw_95']
            })

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, f"epoch_{epoch:03d}_group_summary.csv"), index=False)
    pd.DataFrame(detail_rows).to_csv(os.path.join(out_dir, f"epoch_{epoch:03d}_step_detail.csv"), index=False)

    all_path = os.path.join(out_dir, "all_epochs_summary.csv")
    if is_new_experiment and os.path.exists(all_path): os.remove(all_path)
    pd.DataFrame(summary_rows).to_csv(all_path, mode='a', header=not os.path.exists(all_path), index=False)

def _direction_decay_diag(group_results, verbose=True):
    """
    计算各场景方向准确率的衰减情况

    返回字典: {group_key: (dir_t5, dir_t15, dir_t25, decay)}
    """
    diag = {}
    name_map = {'low': '平稳', 'mid': '普通', 'high': '急剧'}

    for key in ['low', 'mid', 'high']:
        if key not in group_results:
            continue
        steps = group_results[key]['step_metrics']
        dir_t5 = next((s['dir_acc_pct'] for s in steps if s['step'] == 5), 0.0)
        dir_t15 = next((s['dir_acc_pct'] for s in steps if s['step'] == 15), 0.0)
        dir_t25 = next((s['dir_acc_pct'] for s in steps if s['step'] == 25), 0.0)
        decay = dir_t5 - dir_t25
        diag[key] = {
            't5': dir_t5, 't15': dir_t15, 't25': dir_t25, 'decay': decay
        }

    if verbose:
        print(f"  🧭 方向衰减监控:")
        # 健康标准(对应平稳/普通/急剧)
        healthy_decay = {'low': 20.0, 'mid': 25.0, 'high': 20.0}
        for key in ['low', 'mid', 'high']:
            if key not in diag:
                continue
            d = diag[key]
            limit = healthy_decay[key]
            if d['decay'] < limit:
                status = '✓'
            elif d['decay'] < limit + 10:
                status = '⚠️'
            else:
                status = '❌'
            print(f"     [{name_map[key]}] T+5={d['t5']:.1f}% "
                  f"T+15={d['t15']:.1f}% T+25={d['t25']:.1f}% "
                  f"(衰减={d['decay']:.1f}pt {status})")

    return diag


def _print_target_progress(group_results, mratio, pic_95):
    """打印各项指标距离目标的差距"""
    print(f"\n  🎯 目标达成度报告:")

    targets = [
        # (指标名, 当前值, 目标值, 是否越大越好)
        ('mratio (健康<0.85)', mratio, 0.85, False),
        ('全局 PIC (≈90%)', pic_95, 90.0, None),  # 双向
    ]

    for key, name in [('low', '平稳'), ('mid', '普通'), ('high', '急剧')]:
        if key not in group_results:
            continue
        g = group_results[key]
        steps = g['step_metrics']
        long_dir = np.mean([s['dir_acc_pct'] for s in steps[-5:]])
        long_dist = np.mean([s['dist_acc_pct'] for s in steps[-5:]])
        long_pic = np.mean([s['pic_95'] for s in steps[-5:]])
        long_nll = np.mean([s['nll'] for s in steps[-5:]])
        print(f"     [{name}] T+21~T+25: "
              f"dist={long_dist:.1f}% | dir={long_dir:.1f}% | "
              f"PIC={long_pic:.1f}% | NLL={long_nll:.2f}")

def get_curriculum_loader(dataset, batch_size, epoch, total_epochs, original_labels, **kwargs):
    lbls = np.array(original_labels)
    w = np.ones_like(lbls, dtype=np.float64)
    prog = epoch / total_epochs

    # ✨ 急剧权重峰值从 3.0 → 5.0
    high_weight_curve = 1.0 / (1.0 + np.exp(-15 * (prog - 0.15)))
    high_weight = 0.05 + high_weight_curve * 4.95   # ✨ 0.05 → 5.0

    cw = 1.0 / np.bincount(lbls)
    cw[2] = cw[2] * high_weight
    for i in range(len(cw)):
        w[lbls == i] = cw[i]

    kwargs.pop('shuffle', None)
    kwargs.pop('sampler', None)
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=WeightedRandomSampler(
                          torch.from_numpy(w).double(), len(lbls), replacement=True),
                      **kwargs)

def run_maneuver_eval_on_best(model, val_loader, scaler_target,
                              group_indices, config, device, exp_name, save_dir):
    print(f"\n⏳ 启动 MC Dropout 推理引擎（修复版）...")

    NUM_SAMPLES = 20

    def _enable_dropout_only(m):
        if isinstance(m, nn.Dropout):
            m.train()

    all_p_mean, all_t_traj, all_p_samples = [], [], []
    all_intent_preds, all_intent_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            bx = {"seq": batch["seq"].to(device, non_blocking=True),
                  "last_delta": batch["last_delta"].to(device, non_blocking=True)}
            by = batch["target"].to(device, non_blocking=True)

            # 1. 确定性推理
            model.eval()
            with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                pred_delta_mean, _, intent_logits, _ = model(bx)

            B_b, H_b, D_b = pred_delta_mean.shape

            p_mean_real_delta = to_real(pred_delta_mean, scaler_target)
            p_mean_real_traj = np.cumsum(p_mean_real_delta, axis=1)
            true_traj_np = np.cumsum(to_real(by.cpu().numpy(), scaler_target), axis=1)

            # 2. MC Dropout 采样
            # 2. MC Dropout 采样(三源多样性版)
            model.apply(_enable_dropout_only)
            model.mc_sampling_active = True  # ✨

            mc_samples = []
            try:
                with torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                    for k in range(NUM_SAMPLES):
                        pred_k, _, _, _ = model(bx)
                        mc_samples.append(pred_k.float())
            finally:
                model.mc_sampling_active = False  # ✨
                model.eval()

            samples_delta = torch.stack(mc_samples, dim=0).cpu().numpy()
            p_samples_real = np.zeros_like(samples_delta)
            for k in range(NUM_SAMPLES):
                p_samples_real[k] = np.cumsum(
                    to_real(samples_delta[k], scaler_target), axis=1)

            all_p_mean.append(p_mean_real_traj)
            all_t_traj.append(true_traj_np)
            all_p_samples.append(p_samples_real)

            intent_pred_class = torch.argmax(intent_logits, dim=-1).cpu().numpy()
            all_intent_preds.append(intent_pred_class)
            all_intent_targets.append(batch["intent"].numpy())

    yp_mean = np.concatenate(all_p_mean, axis=0)
    yt = np.concatenate(all_t_traj, axis=0)
    yp_samples = np.concatenate(all_p_samples, axis=1)
    intents_p = np.concatenate(all_intent_preds, axis=0)
    intents_t = np.concatenate(all_intent_targets, axis=0)

    em = np.linalg.norm(yp_mean - yt, axis=2)
    delta_samples = yp_samples - yp_mean[np.newaxis]
    std_traj_real = np.std(delta_samples, axis=0)
    std_floor = np.linspace(0.01, 0.05, std_traj_real.shape[1]).reshape(1, -1, 1)
    std_traj_real = np.maximum(std_traj_real, std_floor)

    print(f"\n  ├─ 采样诊断: std T+1={std_traj_real[:,0,:].mean():.4f}, "
          f"std T+{yp_samples.shape[2]}={std_traj_real[:,-1,:].mean():.4f}")

    min_ade, min_fde, nll, pic_95, miw_95 = compute_probabilistic_metrics(
        y_true=yt, y_samples=yp_samples,
        y_pred_mean=yp_mean, std_real=std_traj_real)

    mratio = min_ade / max(np.mean(em), 0.01)

    print("\n" + "=" * 50)
    print(f" 🏆 不确定性生成多模态指标 (K={NUM_SAMPLES}，MC Dropout)")
    print("=" * 50)
    print(f"  ├─ Best-of-K ADE (minADE):  {min_ade:.4f} m")
    print(f"  ├─ Best-of-K FDE (minFDE):  {min_fde:.4f} m")
    print(f"  ├─ 🎯 minADE/ADE 比:        {mratio:.3f} (健康<1.0)")
    print(f"  ├─ 高斯负对数似然 (NLL):   {nll:.4f}")
    print(f"  ├─ 95% 区间覆盖率 (PIC):   {pic_95:.2f} %")
    print(f"  └─ 95% 平均区间宽度 (MIW): {miw_95:.4f} m")
    print("=" * 50)

    out_dir = os.path.join(save_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{
        "minADE_20": min_ade, "minFDE_20": min_fde,
        "NLL": nll, "PIC_95": pic_95, "MIW_95": miw_95,
        "mratio": mratio,
    }]).to_csv(os.path.join(out_dir, "probabilistic_metrics_summary.csv"), index=False)

    save_path = os.path.join(out_dir, "detailed_probabilistic_predictions.npz")
    np.savez(save_path, y_pred_mean=yp_mean, y_true=yt,
             y_samples=yp_samples, std_real=std_traj_real,
             intent_pred=intents_p, intent_true=intents_t)
    print(f"  💾 概率轨迹数据已保存 → {save_path}")

    results = evaluate_by_fixed_groups(
        group_indices=group_indices, err_matrix=em,
        y_p_traj=yp_mean, y_t_traj=yt,
        y_samples=yp_samples, std_real=std_traj_real,
        horizon=config["horizon"])
    print_maneuver_group_results(results, "最优模型 MC Dropout 评估")
    return results


# ========== 全局指标 key 命名映射 (放在 train_and_eval 函数上方) ==========
_STEP_KEY_MAP = {
    'dist_error_m': 'DistErr', 'step_mse': 'MSE', 'step_rmse': 'RMSE',
    'dist_acc_pct': 'DistAcc', 'dir_acc_pct': 'DirAcc',
    'nll': 'NLL', 'pic_95': 'PIC', 'miw_95': 'MIW',
}
_GROUP_KEY_MAP = {
    'ade': 'ADE', 'fde': 'FDE', 'min_ade': 'minADE', 'min_fde': 'minFDE',
    'mean_accel': 'MeanAccel', 'max_accel': 'MaxAccel', 'mean_jerk': 'MeanJerk',
    'A_KEER': 'A_KEER', 'NJER': 'NJER', 'KPC': 'KPC',
    'violations_rate': 'Violations', 'jc_min_fde': 'JC_minFDE', 'brier_score': 'Brier',
}

def apply_reachable_domain(samples, start_pos,
                           max_step_len=5.0,
                           max_step_acc=2.5,
                           max_height=150.0,
                           min_height=1.0):
    """
    时空可达域截断（物理坐标空间）
    参数放宽说明：
      max_step_len=5.0  → 对应 ~50m/s 极限速度（dt=0.1s）
      max_step_acc=2.5  → 对应 ~25m/s² 极限加速度
      z 轴爬升率放宽到 ±1.0m/step，允许急剧机动的垂直运动
    """
    K, B, H, _ = samples.shape

    # ✅ 修复：clone() 确保独立 tensor，不用 expand（expand 是视图，赋值会出问题）
    curr_pos   = start_pos.view(1, B, 3).expand(K, -1, -1).clone()
    prev_delta = torch.zeros(K, B, 3, device=samples.device)
    clamped_samples = torch.zeros_like(samples)

    for t in range(H):
        raw_delta = samples[:, :, t, :] - curr_pos

        # 1. 加速度约束（惯性连续性）
        acc      = raw_delta - prev_delta
        acc_norm = torch.norm(acc, dim=-1, keepdim=True)
        acc      = acc * torch.clamp(max_step_acc / (acc_norm + 1e-8), max=1.0)
        real_delta = prev_delta + acc

        # 2. xy 速度上限
        vel_xy = torch.norm(real_delta[..., :2], dim=-1, keepdim=True)
        real_delta[..., :2] = real_delta[..., :2] * torch.clamp(
            max_step_len / (vel_xy + 1e-8), max=1.0)

        # ✅ 修复：z 轴放宽到 ±1.0m/step，支持急剧机动垂直方向探索
        real_delta[..., 2] = torch.clamp(real_delta[..., 2], min=-1.0, max=1.0)

        curr_pos = curr_pos + real_delta
        curr_pos[..., 2] = torch.clamp(curr_pos[..., 2], min=min_height, max=max_height)

        clamped_samples[:, :, t, :] = curr_pos
        prev_delta = real_delta.clone()   # ✅ clone 防止原地修改影响下一步

    return clamped_samples

def run_maneuver_eval_on_best(model, val_loader, scaler_target, group_indices, config, device, exp_name, save_dir):
    """返回 (group_results, global_probs) 供外部提取字典"""
    print(f"\n⏳ 启动最优模型综合推理引擎 (含微观采样消融)...")
    NUM_SAMPLES = 20

    all_p_mean, all_t_traj, all_p_samples = [], [], []
    all_intent_preds, all_intent_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            bx = {"seq": batch["seq"].to(device, non_blocking=True),
                  "last_delta": batch["last_delta"].to(device, non_blocking=True)}
            by = batch["target"].to(device, non_blocking=True)

            model.eval()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                pred_delta_mean, _, intent_logits, _ = model(bx)
                mc_samples_delta = model.generate_probabilistic_trajectories(bx, num_samples=NUM_SAMPLES)

            B, H, D = pred_delta_mean.shape

            p_mean_real_delta = to_real(pred_delta_mean, scaler_target)
            p_mean_real_traj = np.cumsum(p_mean_real_delta, axis=1)
            start_phys = to_real(bx["seq"][:, -1, 0:3], scaler_target)
            start_pt = torch.from_numpy(start_phys).to(device)
            true_traj_np = np.cumsum(to_real(by.cpu().numpy(), scaler_target), axis=1)

            raw_samples_list = []
            for k in range(NUM_SAMPLES):
                delta_phys = torch.from_numpy(to_real(mc_samples_delta[k], scaler_target)).to(device)
                raw_samples_list.append(start_pt.view(B, 1, 3) + torch.cumsum(delta_phys, dim=1))
            raw_samples = torch.stack(raw_samples_list, dim=0)

            sample_mode = config.get("ablation", {}).get("sample_mode", "kinematic")
            if sample_mode == 'unconstrained':
                sampled_phys_trajs = raw_samples
            else:
                sampled_phys_trajs = apply_reachable_domain(
                    raw_samples, start_pt,
                    max_step_len=8.0, max_step_acc=4.0,
                    max_height=150.0, min_height=2.0)

            p_samples_real = (sampled_phys_trajs - start_pt.view(1, B, 1, 3)).cpu().numpy()

            all_p_mean.append(p_mean_real_traj)
            all_t_traj.append(true_traj_np)
            all_p_samples.append(p_samples_real)

            intent_pred_class = torch.argmax(intent_logits, dim=-1).cpu().numpy()
            all_intent_preds.append(intent_pred_class)
            all_intent_targets.append(batch["intent"].numpy())

    yp_mean = np.concatenate(all_p_mean, axis=0)
    yt = np.concatenate(all_t_traj, axis=0)
    yp_samples = np.concatenate(all_p_samples, axis=1)

    em = np.linalg.norm(yp_mean - yt, axis=2)
    std_traj_real = np.std(yp_samples - yp_mean[np.newaxis], axis=0)
    std_floor = np.linspace(0.01, 0.05, std_traj_real.shape[1]).reshape(1, -1, 1)
    std_traj_real = np.maximum(std_traj_real, std_floor)

    min_ade, min_fde, nll, pic_95, miw_95 = compute_probabilistic_metrics(
        yt, yp_samples, yp_mean, std_traj_real)

    print(f"  ├─ minADE={min_ade:.4f} minFDE={min_fde:.4f} NLL={nll:.4f} PIC={pic_95:.2f}% MIW={miw_95:.4f}m")

    results = evaluate_by_fixed_groups(
        group_indices=group_indices, err_matrix=em,
        y_p_traj=yp_mean, y_t_traj=yt, y_samples=yp_samples,
        std_real=std_traj_real, horizon=config["horizon"])

    print_maneuver_group_results(results, "最优模型评估")

    intent_pred_all = np.concatenate(all_intent_preds)
    intent_true_all = np.concatenate(all_intent_targets)
    intent_acc = float(np.mean(intent_pred_all == intent_true_all))
    try:
        from sklearn.metrics import f1_score
        intent_f1 = float(f1_score(intent_true_all, intent_pred_all, average='macro'))
    except Exception:
        intent_f1 = float('nan')

    global_probs = {
        "minADE": float(min_ade), "minFDE": float(min_fde),
        "NLL": float(nll), "PIC95": float(pic_95), "MIW95": float(miw_95),
        "RPE5": compute_rpe(yp_mean, yt, delta=5),
        "RPE10": compute_rpe(yp_mean, yt, delta=10),
        "Intent_Acc": intent_acc, "Intent_F1": intent_f1,
    }
    return results, global_probs

def save_best_maneuver_results(group_results, exp_name, save_dir):
    out_dir = os.path.join(save_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for key, g in group_results.items():
        for s in g['step_metrics']:
            rows.append({
                'group_key': key, 'group_name': g['name'], 'n_samples': g['n_samples'],
                'overall_ade': g['ade'], 'overall_fde': g['fde'],
                'group_min_ade': g.get('min_ade', 0.0), 'group_min_fde': g.get('min_fde', 0.0),
                'mean_accel': g.get('mean_accel', 0.0), 'max_accel': g.get('max_accel', 0.0),
                'mean_jerk': g.get('mean_jerk', 0.0),
                'step': s['step'],
                'dist_error_m': s['dist_error_m'],
                'step_MSE': s['step_mse'],  # 🚀 确保键名为 step_mse
                'step_RMSE': s['step_rmse'],  # 🚀 确保键名为 step_rmse
                'dist_acc_pct': s['dist_acc_pct'],
                'dir_acc_pct': s['dir_acc_pct'],
                'NLL': s['nll'],
                'PIC_95': s['pic_95'],
                'MIW_95': s['miw_95'],
                'violations_rate': g.get('violations_rate', 0.0),
                'jc_min_fde': g.get('jc_min_fde', 0.0),
                'brier_score': g.get('brier_score', 0.0),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "best_epoch_maneuver_detail.csv")
    df.to_csv(path, index=False)
    return path

def train_and_eval(config, global_dataset=None):
    """
    修改为支持 global_dataset 传参，并返回包含全指标的 metrics_dict
    """
    exp_name = config.get("exp_name", f"exp_{int(time.time())}")
    save_dir = config.get("maneuver_save_dir", "ablation_results")

    # ---- 1. 数据(支持共享，极大地节省预处理时间) ----
    if global_dataset is None:
        (train_seqs, train_indices, val_seqs, val_indices,
         scaler_target, f_dim, max_s, horizon) = prepare_data_from_cache(config)
    else:
        train_seqs = global_dataset["train_seqs"]
        train_indices = global_dataset["train_indices"]
        val_seqs = global_dataset["val_seqs"]
        val_indices = global_dataset["val_indices"]
        scaler_target = global_dataset["scaler_target"]
        f_dim = global_dataset["feat_dim"]
        max_s = global_dataset["max_s"]
        horizon = global_dataset["horizon"]

    optimal_workers = config.get("num_workers", 8)
    val_loader = DataLoader(
        OnTheFlyUAVDataset(val_seqs, val_indices, max_s, horizon, is_training=False),
        batch_size=config["batch_size"], shuffle=False,
        num_workers=optimal_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 2. 构建模型(接收 ablation_config 进行消融) ----
    model = IntentAwarePredictor(
        feature_dim=f_dim,
        d_model=config.get("d_model", 128),
        horizon=horizon,
        scales=config["scales"],
        d_scaler_scale=scaler_target.scale_,
        ablation_config=config.get("ablation", None),
    ).to(device)

    # ---- 3. 硬件指标评估 ----
    params_m, flops_g, latency_ms = measure_hardware_metrics(
        model, f_dim, max(config["scales"]), device, exp_name, save_dir)

    # ---- 4. 分组(支持复用共享数据) ----
    if global_dataset is not None and "group_indices" in global_dataset:
        group_indices = global_dataset["group_indices"]
    else:
        group_indices, _, _ = prebuild_maneuver_groups(
            val_loader=val_loader, scaler=scaler_target, config=config,
            force_rebuild=config.get("force_rebuild_groups", False))

    # ---- 5. 优化器参数分组 ----
    unc_params = [p for n, p in model.named_parameters() if 'time_log_var_bias' in n or 'homo_log_var' in n]
    parallel_decoder_params = [p for n, p in model.named_parameters() if
                               'parallel_decoder' in n or 'global_mix_alpha' in n or 'goal_head' in n]
    output_scale_params = [model.output_scale]
    phy_special_params = [p for n, p in model.named_parameters() if
                          'alpha_per_intent' in n or 'magnitude_head' in n or 'phy.kpl' in n or 'phy.kpl_mix' in n or 'phy.res_pred' in n or 'phy.ch_net' in n]
    other_params = [p for n, p in model.named_parameters() if not (
                ('time_log_var_bias' in n) or ('homo_log_var' in n) or ('output_scale' in n) or (
                    'parallel_decoder' in n) or ('global_mix_alpha' in n) or ('alpha_per_intent' in n) or (
                            'magnitude_head' in n) or ('phy.kpl' in n) or ('phy.res_pred' in n) or (
                            'phy.ch_net' in n) or ('goal_head' in n))]

    lr = config["learning_rate"]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': 2e-2},
        {'params': parallel_decoder_params, 'lr': lr * 2.0, 'weight_decay': 5e-3},
        {'params': phy_special_params, 'lr': lr * 1.5, 'weight_decay': 0.0},
        {'params': unc_params, 'lr': lr * 3.0, 'weight_decay': 0.0},
        {'params': output_scale_params, 'lr': lr * 0.3, 'weight_decay': 0.0},
    ], lr=lr)

    if getattr(model, "use_parallel", False) and hasattr(model, "global_mix_alpha"):
        fusion_monitor = FusionHealthMonitor(model=model, optimizer=optimizer, verbose=False)
    else:
        fusion_monitor = NullFusionMonitor()
    amp_scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    warmup_epochs = 2
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=config["epochs"] - warmup_epochs, eta_min=1e-6)
    ], milestones=[warmup_epochs])

    ema = ModelEMA(model, decay=0.999, warmup_steps=2000)
    best_composite_score = float('inf')
    epoch_logs = []
    current_curriculum_stage = -1
    train_loader = None
    NUM_SAMPLES = 20

    # ---- 6. 训练主循环 ----
    for epoch in range(config["epochs"]):
        progress = epoch / max(config["epochs"], 1)
        model.goal_correction_strength = min(1.0, max(0.0, (progress - 0.30) / 0.30))
        target_stage = 0 if progress < 0.1 else 1 if progress < 0.3 else 2

        if target_stage != current_curriculum_stage:
            if train_loader is not None:
                del train_loader
                gc.collect()
            train_loader = get_curriculum_loader(
                dataset=OnTheFlyUAVDataset(train_seqs, train_indices, max_s, horizon, is_training=True),
                batch_size=config["batch_size"], epoch=epoch, total_epochs=config["epochs"],
                original_labels=[item["label"] for item in train_indices],
                num_workers=optimal_workers, pin_memory=True
            )
            current_curriculum_stage = target_stage

        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            bx = {"seq": batch["seq"].to(device), "last_delta": batch["last_delta"].to(device)}
            by = batch["target"].to(device)
            last_pose = bx["seq"][:, -1, 3:6]
            intent_targets = batch["intent"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                current_tau = max(1.0, 5.0 - 4.0 * progress)
                is_high = (intent_targets == 2).float()
                batch_tf_probs = is_high * max(0.05, 0.8 - progress * 1.0) + (1.0 - is_high) * max(0.05,
                                                                                                   0.5 - progress * 0.8)
                tf_mask = torch.rand(bx["seq"].size(0), horizon, device=device) < batch_tf_probs.unsqueeze(1)

                pred, log_var, intent_logits, turn_rates_pred, aux_info = model(
                    bx, tau=current_tau, intent_targets=intent_targets,
                    target_deltas=by, tf_mask=tf_mask, current_horizon=horizon, return_aux=True
                )

                loss_main = uav_loss_v20(
                    pred, log_var, by, last_pose,
                    is_training=True, intent_logits=intent_logits, intent_targets=intent_targets,
                    epoch=epoch, total_epochs=config["epochs"], turn_rates_pred=turn_rates_pred,
                    current_horizon=horizon, aux_info=aux_info
                )
                loss_aux = fusion_monitor.compute_aux_loss(by, aux_info)
                loss = loss_main + loss_aux

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            with torch.no_grad():
                model.output_scale.data.clamp_(0.5, 1.5)
            ema.update(model)
            total_train_loss += loss.item()

        # ---- 阶段验证 ----
        if (epoch + 1) % config.get("maneuver_eval_interval", 5) == 0 or epoch == config["epochs"] - 1:
            model.eval()
            ema.apply_to(model)
            all_p_mean, all_t_traj, all_p_samples = [], [], []
            total_val_loss, val_batch_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    bx = {"seq": batch["seq"].to(device, non_blocking=True),
                          "last_delta": batch["last_delta"].to(device, non_blocking=True)}
                    by = batch["target"].to(device, non_blocking=True)
                    last_pose_val = bx["seq"][:, -1, 3:6]
                    intent_targets = batch["intent"].to(device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                        pred_v, log_var_v, intent_logits_v, turn_rates_v = model(bx)
                        val_loss = uav_loss_v20(
                            pred_v, log_var_v, by, last_pose_val, is_training=False,
                            intent_logits=intent_logits_v, intent_targets=intent_targets,
                            epoch=epoch, total_epochs=config["epochs"], turn_rates_pred=turn_rates_v
                        )
                        total_val_loss += val_loss.item()
                        val_batch_count += 1
                        mc_samples_delta = model.generate_probabilistic_trajectories(bx, num_samples=NUM_SAMPLES)

                    B, H, D = pred_v.shape
                    p_mean_real_delta = to_real(pred_v, scaler_target)
                    p_mean_real_traj = np.cumsum(p_mean_real_delta, axis=1)
                    start_phys = to_real(bx["seq"][:, -1, 0:3], scaler_target)
                    start_pt = torch.from_numpy(start_phys).to(device)

                    raw_samples_list = []
                    for k in range(NUM_SAMPLES):
                        delta_phys = torch.from_numpy(to_real(mc_samples_delta[k], scaler_target)).to(device)
                        raw_samples_list.append(start_pt.view(B, 1, 3) + torch.cumsum(delta_phys, dim=1))
                    raw_samples = torch.stack(raw_samples_list, dim=0)

                    if config.get("ablation", {}).get("sample_mode", "kinematic") == 'unconstrained':
                        sampled_phys_trajs = raw_samples
                    else:
                        sampled_phys_trajs = apply_reachable_domain(
                            raw_samples, start_pt, max_step_len=8.0, max_step_acc=4.0, max_height=150.0, min_height=2.0)

                    p_samples_real = (sampled_phys_trajs - start_pt.view(1, B, 1, 3)).cpu().numpy()
                    all_p_mean.append(p_mean_real_traj)
                    all_t_traj.append(np.cumsum(to_real(by.cpu().numpy(), scaler_target), axis=1))
                    all_p_samples.append(p_samples_real)

            yp_mean = np.concatenate(all_p_mean, axis=0)
            yt = np.concatenate(all_t_traj, axis=0)
            yp_samples = np.concatenate(all_p_samples, axis=1)
            em = np.linalg.norm(yp_mean - yt, axis=2)
            ade_mean = float(np.mean(em))

            std_traj_real = np.clip(np.std(yp_samples - yp_mean[np.newaxis], axis=0), 0.05, None)
            min_ade, min_fde, nll, pic_95, miw_95 = compute_probabilistic_metrics(yt, yp_samples, yp_mean,
                                                                                  std_traj_real)
            global_dist_acc, global_dir_acc = calculate_global_accuracies(yp_mean, yt, horizon)
            a_keer, njer, kpc = calculate_kinematic_fidelity_adaptive(yp_mean, yt)

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / max(1, val_batch_count)

            group_results = evaluate_by_fixed_groups(group_indices, em, yp_mean, yt, yp_samples, std_traj_real, horizon,
                                                     dir_thresh_deg=15.0)
            composite_score = min_ade + 0.5 * ade_mean + 0.1 * nll + max(0, 80.0 - pic_95) * 0.05

            print(
                f"[{exp_name}] Epoch {epoch + 1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | ADE: {ade_mean:.4f} | Score: {composite_score:.4f}")

            if composite_score < best_composite_score:
                best_composite_score = composite_score
                torch.save(model.state_dict(), f"best_{exp_name}.pth")
            ema.restore(model)

        fusion_monitor.update(epoch)
        scheduler.step()

    # ---- 7. 终极推理与指标组装 ----
    if not os.path.exists(f"best_{exp_name}.pth"):
        torch.save(model.state_dict(), f"best_{exp_name}.pth")
    model.load_state_dict(torch.load(f"best_{exp_name}.pth", weights_only=True, map_location=device))

    best_group_results, global_probs = run_maneuver_eval_on_best(
        model, val_loader, scaler_target, group_indices, config, device, exp_name, save_dir)
    save_best_maneuver_results(best_group_results, exp_name=exp_name, save_dir=save_dir)

    metrics_dict = {
        "composite_score": best_composite_score,
        "Params_M": params_m, "FLOPs_G": flops_g, "Latency_ms": latency_ms,
        "GLOBAL_minADE": global_probs["minADE"], "GLOBAL_minFDE": global_probs["minFDE"],
        "GLOBAL_NLL": global_probs["NLL"], "GLOBAL_PIC95": global_probs["PIC95"],
        "GLOBAL_MIW95": global_probs["MIW95"], "GLOBAL_RPE5": global_probs["RPE5"],
        "GLOBAL_RPE10": global_probs["RPE10"], "Intent_Acc": global_probs["Intent_Acc"],
        "Intent_F1": global_probs["Intent_F1"],
    }

    valid_groups = []
    for g_key in ['low', 'mid', 'high']:
        g = best_group_results.get(g_key, {})
        if not g: continue
        valid_groups.append(g)
        p = f"{g_key.upper()}_"
        for raw_k, pretty_k in _GROUP_KEY_MAP.items():
            metrics_dict[p + pretty_k] = g.get(raw_k, float('nan'))
        steps = g.get('step_metrics', [])
        for t_keep in [5, 10, 15, 20, 25]:
            if t_keep <= len(steps):
                s = steps[t_keep - 1]
                for raw_k, pretty_k in _STEP_KEY_MAP.items():
                    metrics_dict[f"{p}T{t_keep}_{pretty_k}"] = s.get(raw_k, float('nan'))
        if steps:
            metrics_dict[p + "Mean_NLL"] = float(np.mean([s['nll'] for s in steps]))
            metrics_dict[p + "Mean_PIC"] = float(np.mean([s['pic_95'] for s in steps]))
            metrics_dict[p + "Mean_MIW"] = float(np.mean([s['miw_95'] for s in steps]))
            metrics_dict[p + "Mean_DistAcc"] = float(np.mean([s['dist_acc_pct'] for s in steps]))
            metrics_dict[p + "Mean_DirAcc"] = float(np.mean([s['dir_acc_pct'] for s in steps]))

    if total_samples := sum(g['n_samples'] for g in valid_groups):
        for raw_k in ['ade', 'fde']:
            metrics_dict[f"GLOBAL_{_GROUP_KEY_MAP[raw_k]}"] = sum(
                g.get(raw_k, 0) * g['n_samples'] for g in valid_groups) / total_samples

    return metrics_dict

# =====================================================================
# Final Paper-Ladder Ablation Protocol
#
# Formal variants:
#   1. EMam-SE
#   2. EMam-SE+IA-DTP
#   3. Ours
#
# Paper line:
#   Heteroscedastic probabilistic prediction
#   + interval reliability / compactness
#   + kinematic consistency
#
# This block should be the ONLY ablation/main block after the base modules.
# =====================================================================

import hashlib
import json
import random
import re
from collections import OrderedDict

# Preserve the original full multi-term loss before redefining the fair wrapper.
_legacy_uav_loss_v20 = uav_loss_v20


# =====================================================================
# 1. Reproducibility and utilities
# =====================================================================

def seed_everything(seed, deterministic=True):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_variant_name(name):
    return re.sub(r"[^\w\-]+", "_", str(name)).strip("_")


def _as_float(x, default=np.nan):
    try:
        value = float(x)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _ids_digest(ids):
    joined = "\n".join(sorted(map(str, ids))).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def _protocol_cache_dir(config):
    cache_dir = config.get(
        "protocol_cache_dir",
        os.path.join(config["maneuver_save_dir"], "shared_protocol_cache"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _first_existing_column(frame_or_row, candidates):
    columns = frame_or_row.index if isinstance(frame_or_row, pd.Series) else frame_or_row.columns
    for column in candidates:
        if column in columns:
            return column
    return None


# =====================================================================
# 2. Fixed data protocol: 70 / 10 / 20 split, train-only thresholds
# =====================================================================

def _load_or_extract_all_flights(config):
    cache_dir = _protocol_cache_dir(config)
    cache_file = os.path.join(cache_dir, "raw_flights_31d_v1.npz")

    if os.path.exists(cache_file) and not config.get("force_rebuild_features", False):
        try:
            cache = np.load(cache_file, allow_pickle=True)
            flights = {key: cache[key] for key in cache.files}

            if not flights:
                raise RuntimeError("feature cache is empty")

            sample = next(iter(flights.values()))
            if sample.ndim != 2 or sample.shape[-1] != 31:
                raise RuntimeError(f"unexpected feature shape: {sample.shape}")

            print(f"✅ 已加载原始特征缓存：{len(flights)} 条轨迹")
            return flights

        except Exception as exc:
            print(f"⚠️ 原始特征缓存不可用，重新提取：{exc}")

    max_s = max(config["scales"])
    min_length = max_s + int(config["horizon"])

    tasks = []
    seen_ids = set()

    for parquet_file in sorted(glob.glob(os.path.join(config["data_dir"], "*.parquet"))):
        print(f"  ├─ 读取：{os.path.basename(parquet_file)}")
        try:
            frame = pd.read_parquet(parquet_file)
            log_col = "logs" if "logs" in frame.columns else "log"
            unique_frame = frame.drop_duplicates("id")[["id", log_col]]

            for fid, log_string in zip(unique_frame["id"].values, unique_frame[log_col].values):
                fid = str(fid)
                if fid not in seen_ids:
                    tasks.append((fid, log_string, min_length))
                    seen_ids.add(fid)

            del frame, unique_frame

        except Exception as exc:
            print(f"  ⚠️ 文件读取失败：{exc}")

    if not tasks:
        raise RuntimeError("没有找到可用于特征提取的轨迹任务。")

    cpu_total = os.cpu_count() or 4
    workers = min(max(cpu_total - 2, 2), int(config.get("feature_workers", 32)))
    chunk_size = max(1, len(tasks) // max(workers * 8, 1))

    print(f"⏳ CPU 特征提取：任务={len(tasks)}，workers={workers}")

    flights = {}
    failed = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=100,
    ) as executor:
        for result in tqdm(
            executor.map(process_single_flight, tasks, chunksize=chunk_size),
            total=len(tasks),
            desc="feature extraction",
            ncols=90,
        ):
            if result is not None and result[0] is not None:
                flights[result[0]] = result[1]
            else:
                failed += 1

    if not flights:
        raise RuntimeError("特征提取后没有有效轨迹。")

    np.savez_compressed(cache_file, **flights)
    print(f"✅ 原始特征缓存已保存：{cache_file}；成功={len(flights)}，跳过={failed}")
    return flights


def _load_or_create_split_manifest(all_ids, config):
    cache_dir = _protocol_cache_dir(config)
    split_seed = int(config.get("split_seed", 42))
    manifest_path = os.path.join(cache_dir, f"split_70_10_20_seed{split_seed}.json")

    all_ids = sorted(map(str, all_ids))

    if os.path.exists(manifest_path) and not config.get("force_rebuild_split", False):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        manifest_ids = (
            manifest.get("train_ids", [])
            + manifest.get("val_ids", [])
            + manifest.get("test_ids", [])
        )

        if set(manifest_ids) == set(all_ids) and len(manifest_ids) == len(all_ids):
            print(f"✅ 复用固定划分：{manifest_path}")
            return manifest

        print("⚠️ 既有划分文件与当前轨迹 ID 不一致，重新划分。")

    train_ids, temp_ids = train_test_split(
        all_ids,
        test_size=0.30,
        random_state=split_seed,
        shuffle=True,
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=2.0 / 3.0,
        random_state=split_seed,
        shuffle=True,
    )

    manifest = {
        "protocol": "ID-level 70/10/20",
        "split_seed": split_seed,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "all_ids_digest": _ids_digest(all_ids),
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(
        "✅ 创建固定 ID 划分："
        f"train={len(train_ids)}，val={len(val_ids)}，test={len(test_ids)}"
    )

    return manifest


def _feature_scale_mask(feature_dim):
    mask = np.ones(feature_dim, dtype=bool)

    mask[0:3] = False
    mask[12:18] = False

    for idx in (21, 22, 25, 26):
        if idx < feature_dim:
            mask[idx] = False

    return mask


def compute_future_maneuver_intensity(future_deltas):
    future_deltas = np.asarray(future_deltas, dtype=np.float32)

    if future_deltas.ndim != 2 or len(future_deltas) == 0:
        return 0.0

    displacement = np.linalg.norm(future_deltas, axis=-1)

    if float(np.mean(displacement)) <= 1e-8:
        return 0.0

    linear_change = (
        float(np.mean(np.linalg.norm(np.diff(future_deltas, axis=0), axis=-1)))
        if len(future_deltas) >= 2
        else 0.0
    )

    speed_std = float(np.std(displacement))

    return 0.7 * linear_change + 0.3 * speed_std


def build_window_indices(raw_sequences, max_s, horizon, stride):
    indices = []
    min_length = max_s + horizon

    for seq_idx, seq in enumerate(raw_sequences):
        if len(seq) < min_length:
            continue

        for start in range(0, len(seq) - min_length + 1, int(stride)):
            future_deltas = seq[start + max_s : start + max_s + horizon, :3]
            intensity = compute_future_maneuver_intensity(future_deltas)

            if intensity <= 0.0:
                continue

            indices.append(
                {
                    "seq_idx": int(seq_idx),
                    "start": int(start),
                    "intensity": float(intensity),
                    "label": -1,
                }
            )

    if not indices:
        raise RuntimeError("没有构建到有效窗口，请检查轨迹长度、stride 和预测步长。")

    return indices


def assign_intent_labels(indices, q25, q75):
    for item in indices:
        value = float(item["intensity"])
        item["label"] = 0 if value < q25 else (1 if value < q75 else 2)
    return indices


def build_fixed_groups_from_labels(indices):
    labels = np.asarray([item["label"] for item in indices], dtype=np.int64)

    return {
        "low": np.where(labels == 0)[0],
        "mid": np.where(labels == 1)[0],
        "high": np.where(labels == 2)[0],
    }


def _fit_or_load_scalers(train_raw, feature_dim, train_ids, config):
    cache_dir = _protocol_cache_dir(config)
    split_digest = _ids_digest(train_ids)
    scaler_path = os.path.join(cache_dir, f"scalers_train_{split_digest}.pkl")

    scale_mask = _feature_scale_mask(feature_dim)

    if os.path.exists(scaler_path) and not config.get("force_rebuild_scalers", False):
        d_scaler, f_scaler = joblib.load(scaler_path)
        print(f"✅ 复用训练集 Scaler：{scaler_path}")
        return d_scaler, f_scaler, scale_mask

    rng = np.random.default_rng(int(config.get("split_seed", 42)))
    target_samples = int(config.get("scaler_frame_samples", 50000))

    sampled_rows = []
    collected = 0

    for seq in train_raw:
        if collected >= target_samples:
            break

        n_take = min(50, len(seq), target_samples - collected)

        if n_take <= 0:
            continue

        selected = rng.choice(len(seq), size=n_take, replace=False)
        sampled_rows.append(seq[selected])
        collected += n_take

    if not sampled_rows:
        raise RuntimeError("训练集没有可用于拟合 Scaler 的样本。")

    frame_sample = np.vstack(sampled_rows)
    del sampled_rows

    deltas = frame_sample[:, :3]
    valid = np.linalg.norm(deltas, axis=1) > 1e-3
    delta_valid = deltas[valid]

    if len(delta_valid) == 0:
        delta_valid = deltas

    xy_std = max(float(np.std(delta_valid[:, :2])), 0.05)
    z_std = max(float(np.std(delta_valid[:, 2:3])), 0.01)

    d_scaler = StandardScaler(with_mean=False)
    d_scaler.fit(delta_valid)
    d_scaler.scale_ = np.asarray([xy_std, xy_std, z_std], dtype=np.float64)

    clipped = frame_sample.copy()

    if feature_dim >= 21:
        clipped[:, 18:21] = np.clip(clipped[:, 18:21], -10, 10)

    if feature_dim >= 12:
        clipped[:, 9:12] = np.clip(clipped[:, 9:12], -50, 50)

    f_scaler = StandardScaler().fit(
        np.clip(clipped[:, scale_mask], -500, 500)
    )

    f_scaler.scale_ = np.maximum(f_scaler.scale_, 1e-3)

    joblib.dump((d_scaler, f_scaler), scaler_path)

    print(
        f"✅ 用训练集拟合 Scaler：xy_scale={xy_std:.4f}, z_scale={z_std:.4f}"
    )

    return d_scaler, f_scaler, scale_mask


def _normalise_sequences(raw_sequences, d_scaler, f_scaler, scale_mask):
    normalised = []

    for seq in tqdm(raw_sequences, desc="normalising", ncols=80):
        scaled = seq.astype(np.float32, copy=True)

        scaled[:, :3] = d_scaler.transform(scaled[:, :3])

        if scaled.shape[1] >= 21:
            scaled[:, 18:21] = np.clip(scaled[:, 18:21], -10, 10)

        if scaled.shape[1] >= 12:
            scaled[:, 9:12] = np.clip(scaled[:, 9:12], -50, 50)

        scaled[:, scale_mask] = np.clip(
            f_scaler.transform(np.clip(scaled[:, scale_mask], -500, 500)),
            -10,
            10,
        )

        normalised.append(scaled)

    return normalised


def _infer_physical_limits(train_raw, dt, quantile=99.5, margin=1.2):
    speed_values = []
    vz_values = []
    accel_values = []
    jerk_values = []

    for seq in train_raw:
        delta = np.asarray(seq[:, :3], dtype=np.float64)

        if len(delta) < 4:
            continue

        velocity = delta / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt

        speed_values.append(np.linalg.norm(velocity[:, :2], axis=-1))
        vz_values.append(np.abs(velocity[:, 2]))

        if len(acceleration):
            accel_values.append(np.linalg.norm(acceleration, axis=-1))

        if len(jerk):
            jerk_values.append(np.linalg.norm(jerk, axis=-1))

    def q_or_default(chunks, default):
        if not chunks:
            return float(default)

        values = np.concatenate(chunks)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return float(default)

        return float(np.percentile(values, quantile) * margin)

    return {
        "sample_interval_s": float(dt),
        "max_xy_speed_mps": q_or_default(speed_values, 10.0),
        "max_vertical_speed_mps": q_or_default(vz_values, 5.0),
        "max_accel_mps2": q_or_default(accel_values, 10.0),
        "max_jerk_mps3": q_or_default(jerk_values, 30.0),
        "quantile": float(quantile),
        "margin": float(margin),
        "source": "training-data robust envelope",
    }


def prepare_shared_dataset(config):
    all_flights = _load_or_extract_all_flights(config)
    manifest = _load_or_create_split_manifest(all_flights.keys(), config)

    train_ids = manifest["train_ids"]
    val_ids = manifest["val_ids"]
    test_ids = manifest["test_ids"]

    train_raw = [all_flights[fid] for fid in train_ids]
    val_raw = [all_flights[fid] for fid in val_ids]
    test_raw = [all_flights[fid] for fid in test_ids]

    feature_dim = next(iter(all_flights.values())).shape[-1]

    del all_flights
    gc.collect()

    max_s = max(config["scales"])
    horizon = int(config["horizon"])

    train_indices = build_window_indices(
        train_raw,
        max_s=max_s,
        horizon=horizon,
        stride=int(config.get("train_stride", 5)),
    )

    val_indices = build_window_indices(
        val_raw,
        max_s=max_s,
        horizon=horizon,
        stride=int(config.get("eval_stride", 2)),
    )

    test_indices = build_window_indices(
        test_raw,
        max_s=max_s,
        horizon=horizon,
        stride=int(config.get("eval_stride", 2)),
    )

    train_intensity = np.asarray(
        [item["intensity"] for item in train_indices],
        dtype=np.float64,
    )

    q25, q75 = np.percentile(train_intensity, [25, 75])

    for indices in (train_indices, val_indices, test_indices):
        assign_intent_labels(indices, q25=q25, q75=q75)

    d_scaler, f_scaler, scale_mask = _fit_or_load_scalers(
        train_raw=train_raw,
        feature_dim=feature_dim,
        train_ids=train_ids,
        config=config,
    )

    physical_limits = _infer_physical_limits(
        train_raw=train_raw,
        dt=float(config.get("sample_interval_s", 1.0)),
        quantile=float(config.get("physical_limit_quantile", 99.5)),
        margin=float(config.get("physical_limit_margin", 1.2)),
    )

    cache_dir = _protocol_cache_dir(config)

    protocol_info = {
        "split_manifest": manifest,
        "intent_thresholds": {
            "q25_train": float(q25),
            "q75_train": float(q75),
            "definition": (
                "0=stable; 1=ordinary maneuver; 2=sharp maneuver; "
                "computed from future-window displacement variation"
            ),
        },
        "physical_limits": physical_limits,
        "max_history_steps": int(max_s),
        "prediction_horizon": int(horizon),
        "train_stride": int(config.get("train_stride", 5)),
        "eval_stride": int(config.get("eval_stride", 2)),
    }

    with open(os.path.join(cache_dir, "protocol_70_10_20.json"), "w", encoding="utf-8") as handle:
        json.dump(protocol_info, handle, ensure_ascii=False, indent=2)

    for name, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        labels = np.asarray([item["label"] for item in indices], dtype=np.int64)
        print(
            f"  ├─ {name}: windows={len(indices)}, "
            f"stable={(labels == 0).sum()}, "
            f"ordinary={(labels == 1).sum()}, "
            f"sharp={(labels == 2).sum()}"
        )

    print("⏳ 归一化训练序列...")
    train_seqs = _normalise_sequences(train_raw, d_scaler, f_scaler, scale_mask)
    del train_raw

    print("⏳ 归一化验证序列...")
    val_seqs = _normalise_sequences(val_raw, d_scaler, f_scaler, scale_mask)
    del val_raw

    print("⏳ 归一化测试序列...")
    test_seqs = _normalise_sequences(test_raw, d_scaler, f_scaler, scale_mask)
    del test_raw

    gc.collect()

    return {
        "train_seqs": train_seqs,
        "train_indices": train_indices,
        "val_seqs": val_seqs,
        "val_indices": val_indices,
        "test_seqs": test_seqs,
        "test_indices": test_indices,
        "val_groups": build_fixed_groups_from_labels(val_indices),
        "test_groups": build_fixed_groups_from_labels(test_indices),
        "scaler_target": d_scaler,
        "feat_dim": feature_dim,
        "max_s": max_s,
        "horizon": horizon,
        "intent_q25": float(q25),
        "intent_q75": float(q75),
        "physical_limits": physical_limits,
        "split_manifest": manifest,
        "protocol_info": protocol_info,
    }


# =====================================================================
# 3. Ablation switches and corrected model
# =====================================================================

def make_ablation_config(**overrides):
    config = {
        # EMam-SE
        "emam": True,
        "emam_multiscale_conv": True,
        "emam_channel_attn": True,

        # IA-DTP
        "dtp_context": True,
        "dtp_attention_gate": True,
        "intent_supervision": True,
        "intent_conditioning": True,
        "intent_weighted_loss": False,

        # Global trend decoder
        "parallel": True,
        "parallel_self_attn": True,

        # UA-PGD
        "pgd": True,
        "pgd_kpl": True,
        "pgd_magnitude_head": True,
        "pgd_intent_alpha": True,
        "adaptive_inertia": True,
        "smoother": True,
        "reachable_projection": True,

        # Uncertainty
        "unc_mode": "hetero",

        # Auxiliary heads
        "turn_rate_aux": False,
        "goal_anchor": False,
    }

    config.update(overrides)

    if not config["intent_conditioning"]:
        config["pgd_intent_alpha"] = False

    if not config["pgd"]:
        config["pgd_kpl"] = False
        config["pgd_magnitude_head"] = False
        config["pgd_intent_alpha"] = False
        config["adaptive_inertia"] = False
        config["smoother"] = False
        config["reachable_projection"] = False

    return config


class CapacityMatchedGRUEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, target_params):
        super().__init__()

        candidates = list(
            range(max(16, input_dim // 2), max(32, input_dim * 7) + 1, 16)
        )

        def estimate(hidden):
            gru_params = 3 * hidden * (input_dim + hidden + 2)
            projection_params = 0 if hidden == output_dim else hidden * output_dim + output_dim
            return gru_params + projection_params

        hidden_dim = min(candidates, key=lambda h: abs(estimate(h) - target_params))

        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(out)


class CorrectedPhysicsGatedPredictor(nn.Module):
    def __init__(
        self,
        d_model,
        cond_dim=16,
        hidden_dim=128,
        horizon=20,
        drop_res=0.15,
        use_kpl=True,
        use_magnitude_head=True,
        use_intent_alpha=True,
        use_intent_conditioning=True,
    ):
        super().__init__()

        self.horizon = horizon
        self.use_kpl = use_kpl
        self.use_magnitude_head = use_magnitude_head
        self.use_intent_alpha = use_intent_alpha
        self.use_intent_conditioning = use_intent_conditioning

        self.res_pred = nn.Sequential(
            nn.Linear(d_model + cond_dim + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT_BASE),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )

        self.ch_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        self.time_embed = nn.Embedding(horizon, 16)
        nn.init.normal_(self.time_embed.weight, std=0.02)

        self.gru = nn.GRU(9, hidden_dim, num_layers=1, batch_first=True)
        self.gru_init = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.Tanh())

        self.residual_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_BASE),
            nn.Linear(hidden_dim // 2, 3),
        )

        nn.init.zeros_(self.residual_proj[-1].weight)
        nn.init.zeros_(self.residual_proj[-1].bias)

        if self.use_kpl:
            self.kpl = KinematicProjectionLayer(d_model, drop=DROPOUT_BASE)
            self.kpl_mix = nn.Parameter(torch.tensor(-2.0))

        if self.use_magnitude_head:
            self.magnitude_head = nn.Sequential(
                nn.Linear(hidden_dim + d_model, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(DROPOUT_BASE),
                nn.Linear(hidden_dim // 2, 1),
            )

            nn.init.zeros_(self.magnitude_head[-1].weight)
            nn.init.constant_(self.magnitude_head[-1].bias, 0.2)

            self.high_mag_boost = nn.Parameter(torch.tensor(0.3))

        if self.use_intent_alpha:
            self.alpha_per_intent = nn.Parameter(torch.tensor([-1.5, -0.4, 0.62]))

        self.time_log_var_bias = nn.Parameter(
            (torch.log(torch.arange(1, horizon + 1, dtype=torch.float32)) - 1.0)
            .unsqueeze(1)
            .repeat(1, 3)
            .contiguous()
        )

    def init_gru_state(self, context):
        return self.gru_init(context).unsqueeze(0)

    def forward(
        self,
        ch_prev,
        npd,
        npd_prev,
        step,
        h_gru,
        intent_weights,
        mac,
        intent_bias=None,
    ):
        del intent_bias

        batch = npd.size(0)
        device = npd.device
        dtype = npd.dtype

        if self.use_intent_conditioning and intent_weights is not None:
            intent = intent_weights
            high_prob = intent[:, 2:3]
        else:
            intent = torch.zeros(batch, 3, device=device, dtype=dtype)
            high_prob = torch.zeros(batch, 1, device=device, dtype=dtype)

        if self.use_kpl:
            context = ch_prev + torch.sigmoid(self.kpl_mix) * self.kpl(npd)
        else:
            context = ch_prev

        time_embedding = self.time_embed(torch.tensor(step, device=device))
        time_embedding = time_embedding.unsqueeze(0).expand(batch, -1)

        auxiliary = torch.cat([npd, mac, intent], dim=-1)
        raw = self.res_pred(torch.cat([context, auxiliary, time_embedding], dim=-1))

        gru_input = torch.cat([npd, npd - npd_prev, intent], dim=-1)
        gru_out, new_h = self.gru(gru_input.unsqueeze(1), h_gru)
        gru_state = gru_out.squeeze(1)

        direction_residual = self.residual_proj(gru_state)

        if self.use_magnitude_head:
            magnitude_logits = self.magnitude_head(torch.cat([gru_state, context], dim=-1))
            lower = -1.5 + 0.5 * (1.0 - high_prob)
            upper = 1.5 + 1.5 * high_prob

            magnitude = torch.exp(
                torch.clamp(
                    magnitude_logits + high_prob * self.high_mag_boost,
                    min=lower,
                    max=upper,
                )
            )
        else:
            magnitude = 1.0

        if self.use_intent_alpha and self.use_intent_conditioning:
            alpha = (
                intent * (torch.sigmoid(self.alpha_per_intent) * 0.5)
            ).sum(dim=-1, keepdim=True)
            alpha = alpha * (1.0 + 1.5 * high_prob)
        else:
            alpha = torch.full((batch, 1), 0.25, device=device, dtype=dtype)

        action = raw[:, :3] + alpha * direction_residual

        if self.use_intent_conditioning:
            var_bias = (
                intent * torch.tensor([0.0, 0.3, 1.5], device=device, dtype=dtype)
            ).sum(dim=-1, keepdim=True)
        else:
            var_bias = 0.0

        min_log_var = 2.0 * torch.log(
            torch.linspace(0.015, 0.12, self.horizon, device=device)[step]
        )

        max_log_var = 3.0 + 0.18 * step + 1.5 * high_prob.mean()

        log_var = torch.clamp(
            raw[:, 3:] + self.time_log_var_bias[step] + var_bias,
            min=min_log_var,
            max=max_log_var,
        )

        action = torch.cat([action[:, :2] * magnitude, action[:, 2:3]], dim=-1)

        return action, self.ch_net(context), log_var, new_h


class IntentAwarePredictor(nn.Module):
    def __init__(
        self,
        feature_dim,
        d_model=192,
        horizon=20,
        scales=(5, 10, 20, 30, 45),
        d_scaler_scale=None,
        ablation_config=None,
    ):
        super().__init__()

        self.horizon = int(horizon)
        self.d_model = int(d_model)
        self.scales = sorted(scales)
        self.ablation = make_ablation_config(**(ablation_config or {}))

        print("  🔧 ablation config:", self.ablation)

        self.mc_sampling_active = False
        self.mc_latent_noise_scale = 0.30
        self.mc_step_noise_base = 0.04
        self.mc_step_noise_growth = 4.5

        self.proj = nn.Linear(feature_dim, d_model)

        if self.ablation["emam"]:
            self.encoder = EnhancedMambaBlock(
                d_model,
                use_multiscale_conv=self.ablation["emam_multiscale_conv"],
                use_channel_attn=self.ablation["emam_channel_attn"],
            )
            self._encoder_kind = "mamba"
        else:
            reference = EnhancedMambaBlock(
                d_model,
                use_multiscale_conv=True,
                use_channel_attn=True,
            )
            reference_params = sum(p.numel() for p in reference.parameters())
            del reference

            self.encoder = CapacityMatchedGRUEncoder(d_model, d_model, reference_params)
            self._encoder_kind = "matched_gru"

            print(f"  ├─ EMam-SE replacement GRU hidden size={self.encoder.hidden_dim}")

        self.use_dtp_context = self.ablation["dtp_context"]
        self.use_intent_supervision = self.ablation["intent_supervision"]
        self.use_intent_conditioning = self.ablation["intent_conditioning"]
        self.use_intent_head = self.use_intent_supervision or self.use_intent_conditioning

        if self.use_dtp_context:
            self.pyr = lambda x: [
                x[:, -1, :] - x[:, max(0, x.shape[1] - scale), :]
                for scale in self.scales
            ]

            if self.ablation["dtp_attention_gate"]:
                self.sg = ScaleAttentionGate(d_model, len(self.scales))
                self._dtp_mode = "gate"
            else:
                self.sg = None
                self._dtp_mode = "mean"

        self.global_dropout = nn.Dropout(DROPOUT_BASE * 1.5)

        if self.use_intent_head:
            self.i_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(DROPOUT_HEAVY),
                nn.Linear(d_model // 2, 3),
            )
        else:
            self.i_head = None

        if self.use_intent_conditioning:
            self.i_bias = nn.Embedding(3, d_model)
        else:
            self.register_buffer("neutral_intent_bias", torch.zeros(1, d_model))

        self.glob_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.use_parallel = self.ablation["parallel"]

        if self.use_parallel:
            self.parallel_decoder = ParallelTemporalDecoder(
                d_model=d_model,
                horizon=horizon,
                n_heads=4,
                dropout=DROPOUT_BASE,
                use_self_attn=self.ablation["parallel_self_attn"],
            )
            self.global_mix_alpha = nn.Parameter(torch.tensor(-0.85))

        self.use_pgd = self.ablation["pgd"]

        if self.use_pgd:
            self.phy = CorrectedPhysicsGatedPredictor(
                d_model=d_model,
                horizon=horizon,
                use_kpl=self.ablation["pgd_kpl"],
                use_magnitude_head=self.ablation["pgd_magnitude_head"],
                use_intent_alpha=self.ablation["pgd_intent_alpha"],
                use_intent_conditioning=self.use_intent_conditioning,
            )
        else:
            self.vanilla_gru = nn.GRUCell(d_model + 3, d_model)
            self.vanilla_act_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 3),
            )
            self.vanilla_lv_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 3),
            )
            self.vanilla_lv_time_bias = nn.Parameter(
                torch.linspace(-1.0, 1.0, horizon)
                .unsqueeze(-1)
                .repeat(1, 3)
                .contiguous()
            )

        self.use_smoother = self.ablation["smoother"]

        if self.use_smoother:
            self.sm = nn.Sequential(
                nn.Conv1d(3, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(64, 3, kernel_size=3, padding=1),
            )

        self.output_scale = nn.Parameter(torch.ones(3) * 1.5)

        self.use_turn_rate_aux = bool(self.ablation.get("turn_rate_aux", False))

        if self.use_turn_rate_aux:
            self.turn_rate_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model // 2, horizon),
            )
        else:
            self.turn_rate_head = None

        self.use_goal_anchor = bool(self.ablation.get("goal_anchor", False))
        self.goal_correction_strength = 0.0

        if self.use_goal_anchor:
            self.K_anchors = 3
            self.goal_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(DROPOUT_BASE),
                nn.Linear(d_model // 2, self.K_anchors * 3 + self.K_anchors),
            )

            nn.init.normal_(self.goal_head[-1].weight, std=0.01)
            nn.init.zeros_(self.goal_head[-1].bias)
        else:
            self.K_anchors = 0
            self.goal_head = None

        if self.ablation["unc_mode"] == "homo":
            self.homo_log_var = nn.Parameter(torch.zeros(1, horizon, 3))

    def forward(
        self,
        x_dict,
        tau=1.0,
        intent_targets=None,
        target_deltas=None,
        tf_mask=None,
        current_horizon=None,
        return_aux=False,
    ):
        # Future labels are not allowed to enter the decoder.
        del intent_targets

        seq = x_dict["seq"]
        previous_delta = x_dict["last_delta"]

        batch = seq.size(0)
        device = seq.device

        horizon = min(
            self.horizon,
            current_horizon if current_horizon is not None else self.horizon,
        )

        projected = self.proj(seq)
        encoded = self.encoder(projected)

        if self.use_dtp_context:
            scale_features = self.pyr(encoded)
            if self._dtp_mode == "gate":
                context_raw = self.sg(scale_features)[0]
            else:
                context_raw = torch.stack(scale_features, dim=1).mean(dim=1)
        else:
            context_raw = encoded[:, -1, :]

        if self.training or self.mc_sampling_active:
            context = self.global_dropout(
                context_raw + torch.randn_like(context_raw) * self.mc_latent_noise_scale
            )
        else:
            context = self.global_dropout(context_raw)

        if self.i_head is not None:
            intent_logits = self.i_head(context)
            intent_probabilities = torch.softmax(intent_logits / tau, dim=-1)
        else:
            intent_logits = None
            intent_probabilities = torch.zeros(batch, 3, device=device, dtype=context.dtype)

        if self.use_intent_conditioning:
            if self.mc_sampling_active:
                sampled_class = torch.distributions.Categorical(
                    probs=intent_probabilities.float()
                ).sample()
                intent_weights = F.one_hot(sampled_class, num_classes=3).to(context.dtype)
            else:
                intent_weights = intent_probabilities

            intent_bias = intent_weights @ self.i_bias.weight
        else:
            intent_weights = torch.zeros(batch, 3, device=device, dtype=context.dtype)
            intent_bias = self.neutral_intent_bias.expand(batch, -1)

        global_context = self.glob_proj(context + intent_bias)

        turn_rates = self.turn_rate_head(context) if self.use_turn_rate_aux else None

        if self.use_parallel:
            global_deltas = self.parallel_decoder(
                context_seq=encoded,
                intent_bias=intent_bias,
                current_horizon=horizon,
            )
            global_mix = torch.sigmoid(self.global_mix_alpha)
        else:
            global_deltas = None
            global_mix = torch.zeros((), device=device)

        if self.use_pgd:
            hidden_state = self.phy.init_gru_state(context)
        else:
            hidden_state = torch.zeros(batch, self.d_model, device=device, dtype=context.dtype)

        pre_previous_delta = (
            seq[:, -2, :3].detach()
            if seq.shape[1] >= 2
            else torch.zeros_like(previous_delta)
        )

        current_delta = previous_delta
        ar_context = context
        buffer = seq[:, -15:, :3].detach()

        outputs = []
        logvars = []

        high_probability = intent_weights[:, 2:3]
        local_global_mix = global_mix * (1.0 - 0.5 * high_probability)

        for step in range(horizon):
            if step > 0 and step % 5 == 0:
                refresh = min(0.3, 0.1 + 0.05 * (step / 5.0))
                ar_context = (1.0 - refresh) * ar_context + refresh * global_context

            sum5 = buffer[:, -5:].sum(dim=1)
            sum15 = buffer[:, -15:].sum(dim=1)

            direction5 = F.normalize(sum5[:, :2], dim=-1, eps=1e-8)
            direction15 = F.normalize(sum15[:, :2], dim=-1, eps=1e-8)

            macro_features = torch.cat([sum5, sum15, direction5, direction15], dim=-1)

            if self.use_pgd:
                ar_action, ar_context, log_var, hidden_state = self.phy(
                    ch_prev=ar_context,
                    npd=current_delta,
                    npd_prev=pre_previous_delta,
                    step=step,
                    h_gru=hidden_state,
                    intent_weights=intent_weights,
                    mac=macro_features,
                    intent_bias=intent_bias,
                )
            else:
                hidden_state = self.vanilla_gru(
                    torch.cat([current_delta, ar_context], dim=-1),
                    hidden_state,
                )
                ar_action = self.vanilla_act_head(hidden_state)
                log_var = torch.clamp(
                    self.vanilla_lv_head(hidden_state) + self.vanilla_lv_time_bias[step],
                    min=-4.0,
                    max=4.0,
                )
                ar_context = hidden_state

            if global_deltas is not None:
                action = (1.0 - local_global_mix) * ar_action + local_global_mix * global_deltas[:, step, :]
            else:
                action = ar_action

            action = action * self.output_scale.clamp(0.1, 2.0)

            xy_action = action[:, :2]
            z_action = action[:, 2:3]

            if self.ablation["adaptive_inertia"]:
                current_direction = F.normalize(current_delta[:, :2], dim=-1, eps=1e-8)
                action_direction = F.normalize(xy_action, dim=-1, eps=1e-8)

                blend = max(0.0, 0.4 - step * 0.025) * (
                    1.0 - torch.sqrt(high_probability + 1e-8)
                )

                mixed_direction = F.normalize(
                    blend * current_direction + (1.0 - blend) * action_direction,
                    dim=-1,
                    eps=1e-8,
                )

                xy_action = mixed_direction * torch.norm(
                    xy_action,
                    dim=-1,
                    keepdim=True,
                ).clamp(min=1e-8)

            action = torch.cat(
                [
                    torch.clamp(xy_action, -6.0, 6.0),
                    torch.clamp(z_action, -3.0, 3.0),
                ],
                dim=-1,
            )

            if self.ablation["unc_mode"] == "none":
                log_var = torch.full_like(log_var, -2.0)
            elif self.ablation["unc_mode"] == "homo":
                log_var = self.homo_log_var[:, step, :].expand(batch, -1)

            if self.mc_sampling_active:
                noise_scale = self.mc_step_noise_base * (
                    1.0 + (step / max(horizon - 1, 1)) * self.mc_step_noise_growth
                )
                action = action + torch.randn_like(action) * noise_scale * (
                    1.0 + 1.2 * high_probability
                )

            outputs.append(action)
            logvars.append(log_var)

            if self.training and target_deltas is not None and tf_mask is not None:
                next_delta = torch.where(
                    tf_mask[:, step : step + 1],
                    target_deltas[:, step],
                    action.detach(),
                )
            else:
                next_delta = action.detach()

            pre_previous_delta = current_delta.detach()
            current_delta = next_delta
            buffer = torch.cat([buffer[:, 1:], action.detach().unsqueeze(1)], dim=1)

        raw_output = torch.stack(outputs, dim=1)
        logvars = torch.stack(logvars, dim=1)

        if self.use_smoother:
            smoothing_residual = self.sm(raw_output.detach().transpose(1, 2)).transpose(1, 2)

            if self.use_intent_conditioning:
                smoothing_weight = (
                    intent_weights[:, 0:1].view(batch, 1, 1)
                    + 0.6 * intent_weights[:, 1:2].view(batch, 1, 1)
                    + 0.25 * intent_weights[:, 2:3].view(batch, 1, 1)
                )
            else:
                smoothing_weight = 0.6

            output = raw_output + smoothing_weight * smoothing_residual
        else:
            smoothing_residual = torch.zeros_like(raw_output)
            output = raw_output

        if turn_rates is not None:
            turn_rates = turn_rates[:, :horizon]

        if return_aux:
            aux = {
                "deltas_global": (
                    global_deltas
                    if global_deltas is not None
                    else torch.zeros(batch, horizon, 3, device=device)
                ),
                "global_mix_value": global_mix.detach(),
                "sm_residual": smoothing_residual,
                "ps_raw": raw_output,
                "intent_weights": intent_weights.detach(),
            }
            return output, logvars, intent_logits, turn_rates, aux

        return output, logvars, intent_logits, turn_rates

    def generate_probabilistic_trajectories(self, x_dict, num_samples=100):
        def enable_dropout(module):
            if isinstance(module, nn.Dropout):
                module.train()

        was_training = self.training
        was_mc = self.mc_sampling_active

        self.eval()
        self.apply(enable_dropout)
        self.mc_sampling_active = True

        samples = []

        try:
            with torch.no_grad():
                for _ in range(int(num_samples)):
                    samples.append(self.forward(x_dict)[0])
        finally:
            self.mc_sampling_active = was_mc
            self.train(was_training)

        return torch.stack(samples, dim=0)


# =====================================================================
# 4. Fair loss wrapper
# =====================================================================

def uav_loss_v20(
    pred,
    log_var,
    by,
    last_pose,
    device=None,
    intent_logits=None,
    intent_targets=None,
    epoch=0,
    total_epochs=500,
    turn_rates_pred=None,
    is_training=True,
    current_horizon=None,
    aux_info=None,
    use_intent_supervision=True,
    use_intent_weighted_loss=False,
):
    trajectory_labels = intent_targets if use_intent_weighted_loss else None

    base_loss = _legacy_uav_loss_v20(
        pred=pred,
        log_var=log_var,
        by=by,
        last_pose=last_pose,
        device=device,
        intent_logits=None,
        intent_targets=trajectory_labels,
        epoch=epoch,
        total_epochs=total_epochs,
        turn_rates_pred=turn_rates_pred,
        is_training=is_training,
        current_horizon=current_horizon,
        aux_info=aux_info,
    )

    if use_intent_supervision and intent_logits is not None and intent_targets is not None:
        class_weight = torch.tensor([1.0, 1.5, 3.0], device=pred.device)
        base_loss = base_loss + 0.5 * F.cross_entropy(
            intent_logits,
            intent_targets,
            weight=class_weight,
        )

    return base_loss


# =====================================================================
# 5. Loaders, physical metrics, probability metrics
# =====================================================================

class NullFusionMonitor:
    def compute_aux_loss(self, target, aux_info):
        del aux_info
        return torch.zeros((), device=target.device)

    def update(self, epoch):
        del epoch

    def status_string(self):
        return "disabled"


def get_curriculum_loader(
    dataset,
    batch_size,
    epoch,
    total_epochs,
    original_labels,
    seed,
    num_workers=0,
    pin_memory=True,
):
    labels = np.asarray(original_labels, dtype=np.int64)
    progress = epoch / max(total_epochs, 1)

    sharp_curve = 1.0 / (1.0 + np.exp(-15.0 * (progress - 0.15)))
    sharp_multiplier = 0.05 + 4.95 * sharp_curve

    class_count = np.bincount(labels, minlength=3)
    class_weight = 1.0 / np.maximum(class_count, 1)
    class_weight[2] *= sharp_multiplier

    sample_weight = class_weight[labels].astype(np.float64)

    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1009 * int(epoch))

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weight).double(),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=False,
    )


def _make_eval_loader(sequences, indices, max_s, horizon, batch_size, workers, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        OnTheFlyUAVDataset(sequences, indices, max_s, horizon, is_training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
        persistent_workers=False,
    )


def _make_optimizer(model, learning_rate):
    named_parameters = list(model.named_parameters())
    assigned = set()
    groups = []

    def add_group(keywords, lr, weight_decay):
        params = []

        for name, parameter in named_parameters:
            if id(parameter) in assigned:
                continue

            if any(keyword in name for keyword in keywords):
                params.append(parameter)
                assigned.add(id(parameter))

        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})

    add_group(["time_log_var_bias", "homo_log_var"], learning_rate * 3.0, 0.0)
    add_group(["parallel_decoder", "global_mix_alpha"], learning_rate * 2.0, 5e-3)
    add_group(
        ["phy.", "vanilla_gru", "vanilla_act_head", "vanilla_lv_head", "vanilla_lv_time_bias"],
        learning_rate * 1.5,
        0.0,
    )
    add_group(["output_scale"], learning_rate * 0.3, 0.0)

    remaining = [
        parameter for _, parameter in named_parameters
        if id(parameter) not in assigned
    ]

    if remaining:
        groups.append({"params": remaining, "lr": learning_rate, "weight_decay": 2e-2})

    return torch.optim.AdamW(groups)


def _physical_limits(config):
    limits = config["physical_limits"]
    dt = float(limits["sample_interval_s"])

    return {
        "dt": dt,
        "max_xy_speed_mps": float(limits["max_xy_speed_mps"]),
        "max_vertical_speed_mps": float(limits["max_vertical_speed_mps"]),
        "max_accel_mps2": float(limits["max_accel_mps2"]),
        "max_jerk_mps3": float(limits["max_jerk_mps3"]),
    }


def apply_reachable_domain(samples, limits, initial_delta=None):
    """Project sampled relative positions into the reachable domain.

    samples has shape [K, B, H, 3] and contains positions relative to the
    last observed point in physical units. The first acceleration constraint
    must be measured from the last observed displacement, not from zero.
    Initialising it with zero incorrectly assumes that every UAV is stationary
    at the prediction boundary and collapses the first few Monte Carlo
    intervals, especially for manoeuvring tracks.

    initial_delta is the last observed physical displacement with shape
    [B, 3], or an already expanded [K, B, 3] tensor. None is retained only
    for backwards compatibility with callers that do not have a history.
    """
    limits = _physical_limits({"physical_limits": limits})

    dt = limits["dt"]
    max_xy_step = limits["max_xy_speed_mps"] * dt
    max_z_step = limits["max_vertical_speed_mps"] * dt
    max_delta_change = limits["max_accel_mps2"] * (dt ** 2)

    k, batch, horizon, _ = samples.shape

    current = torch.zeros(
        k, batch, 3, device=samples.device, dtype=samples.dtype
    )
    if initial_delta is None:
        previous_delta = torch.zeros_like(current)
    else:
        previous_delta = torch.as_tensor(
            initial_delta, device=samples.device, dtype=samples.dtype
        )
        if previous_delta.ndim == 1:
            previous_delta = previous_delta.unsqueeze(0)
        if previous_delta.shape == (batch, 3):
            previous_delta = previous_delta.unsqueeze(0).expand(
                k, -1, -1
            )
        elif previous_delta.shape == (1, batch, 3):
            previous_delta = previous_delta.expand(k, -1, -1)
        elif previous_delta.shape != (k, batch, 3):
            raise ValueError(
                "initial_delta must have shape [3], [B,3], [1,B,3], "
                f"or [K,B,3], got {tuple(previous_delta.shape)}"
            )
        previous_delta = previous_delta.clone()
    projected = torch.zeros_like(samples)

    for step in range(horizon):
        raw_delta = samples[:, :, step, :] - current
        delta_change = raw_delta - previous_delta

        change_norm = torch.norm(delta_change, dim=-1, keepdim=True)

        delta = previous_delta + delta_change * torch.clamp(
            max_delta_change / (change_norm + 1e-8),
            max=1.0,
        )

        xy_norm = torch.norm(delta[..., :2], dim=-1, keepdim=True)

        delta[..., :2] = delta[..., :2] * torch.clamp(
            max_xy_step / (xy_norm + 1e-8),
            max=1.0,
        )

        delta[..., 2:3] = torch.clamp(delta[..., 2:3], -max_z_step, max_z_step)

        current = current + delta
        projected[:, :, step, :] = current
        previous_delta = delta

    return projected


def _mc_deltas_to_trajectories(
    mc_delta,
    scaler_target,
    physical_limits,
    use_projection,
    device,
    initial_delta=None,
):
    """Convert normalised sampled displacements to physical trajectories.

    When projection is enabled, initial_delta must be the final observed
    normalised displacement. It is transformed with the same target scaler
    before being passed to apply_reachable_domain.
    """
    trajectories = []

    for sample_idx in range(mc_delta.shape[0]):
        delta_physical = torch.from_numpy(
            to_real(mc_delta[sample_idx], scaler_target)
        ).to(device)

        trajectories.append(torch.cumsum(delta_physical, dim=1))

    trajectories = torch.stack(trajectories, dim=0)

    if use_projection:
        initial_delta_physical = None
        if initial_delta is not None:
            initial_delta_physical = torch.from_numpy(
                to_real(initial_delta, scaler_target)
            ).to(device=device, dtype=trajectories.dtype)
        trajectories = apply_reachable_domain(
            trajectories,
            physical_limits,
            initial_delta=initial_delta_physical,
        )

    return trajectories.cpu().numpy()


def _delta_nll_physical(pred_delta, log_var, true_delta, scaler_target):
    mean = to_real(pred_delta, scaler_target)
    target = to_real(true_delta, scaler_target)

    raw_log_var = log_var.detach().cpu().float().numpy()

    scale_sq = np.asarray(scaler_target.scale_, dtype=np.float64) ** 2

    variance = np.exp(np.clip(raw_log_var, -12.0, 12.0)) * scale_sq.reshape(1, 1, 3)

    nll = 0.5 * (
        np.log(2.0 * np.pi * variance)
        + ((target - mean) ** 2) / np.maximum(variance, 1e-10)
    )

    return np.mean(nll, axis=-1)


def _energy_score(samples, target, pair_count=32):
    k = samples.shape[0]

    first = np.linalg.norm(
        (samples - target[np.newaxis]).reshape(k, target.shape[0], -1),
        axis=-1,
    ).mean(axis=0)

    pairs = max(1, min(int(pair_count), k * max(k - 1, 1)))
    second = np.zeros(target.shape[0], dtype=np.float64)

    for idx in range(pairs):
        i = idx % k
        j = (idx * 37 + 13) % k

        if i == j:
            j = (j + 1) % k

        second += np.linalg.norm(
            (samples[i] - samples[j]).reshape(target.shape[0], -1),
            axis=-1,
        )

    second /= pairs

    return float(np.mean(first - 0.5 * second))


def compute_probabilistic_metrics(y_true, y_samples, delta_nll, pair_count=32):
    sample_error = np.linalg.norm(y_samples - y_true[np.newaxis], axis=-1)

    min_ade = float(np.mean(np.min(np.mean(sample_error, axis=2), axis=0)))
    min_fde = float(np.mean(np.min(sample_error[:, :, -1], axis=0)))

    lower = np.quantile(y_samples, 0.025, axis=0)
    upper = np.quantile(y_samples, 0.975, axis=0)

    mpic95 = float(np.mean((y_true >= lower) & (y_true <= upper)) * 100.0)
    miw95 = float(np.mean(upper - lower))

    energy = _energy_score(y_samples, y_true, pair_count=pair_count)

    return {
        "minADE": min_ade,
        "minFDE": min_fde,
        "EnergyScore": energy,
        "mPIC95": mpic95,
        "MIW95": miw95,
        "DeltaNLL": float(np.mean(delta_nll)),
    }


def _direction_accuracy(pred_traj, true_traj, step, threshold_deg):
    if step == 0:
        pred_vector = pred_traj[:, step, :2]
        true_vector = true_traj[:, step, :2]
    else:
        lookback = 2 if step >= 2 else step

        local_pred = pred_traj[:, step, :2] - pred_traj[:, step - lookback, :2]
        local_true = true_traj[:, step, :2] - true_traj[:, step - lookback, :2]

        macro_pred = pred_traj[:, step, :2] - pred_traj[:, 0, :2]
        macro_true = true_traj[:, step, :2] - true_traj[:, 0, :2]

        macro_weight = min(1.0, step / 15.0)

        pred_vector = (1.0 - macro_weight) * local_pred + macro_weight * macro_pred
        true_vector = (1.0 - macro_weight) * local_true + macro_weight * macro_true

    pred_norm = np.linalg.norm(pred_vector, axis=1)
    true_norm = np.linalg.norm(true_vector, axis=1)

    valid = true_norm > 0.05

    if not np.any(valid):
        return 0.0

    cosine = np.sum(pred_vector[valid] * true_vector[valid], axis=1) / (
        pred_norm[valid] * true_norm[valid] + 1e-8
    )

    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return float(np.mean(angle <= np.deg2rad(threshold_deg)) * 100.0)


def compute_kinematic_metrics(y_pred, y_true, physical_limits):
    dt = float(physical_limits["sample_interval_s"])
    max_accel = float(physical_limits["max_accel_mps2"])
    max_jerk = float(physical_limits["max_jerk_mps3"])

    result = {
        "Ev": np.nan,
        "Ea": np.nan,
        "Ej": np.nan,
        "A_KEER": np.nan,
        "NJER": np.nan,
        "KPC": np.nan,
        "ViolationRate": np.nan,
    }

    if y_pred.shape[1] < 2:
        return result

    velocity_pred = np.diff(y_pred, axis=1) / dt
    velocity_true = np.diff(y_true, axis=1) / dt

    result["Ev"] = float(np.mean(np.linalg.norm(velocity_pred - velocity_true, axis=-1)))

    if y_pred.shape[1] < 3:
        return result

    acceleration_pred = np.diff(velocity_pred, axis=1) / dt
    acceleration_true = np.diff(velocity_true, axis=1) / dt

    acceleration_pred_norm = np.linalg.norm(acceleration_pred, axis=-1)
    acceleration_true_norm = np.linalg.norm(acceleration_true, axis=-1)

    result["Ea"] = float(np.mean(np.linalg.norm(acceleration_pred - acceleration_true, axis=-1)))

    local_envelope = 1.10 * np.maximum(
        np.max(acceleration_true_norm, axis=1, keepdims=True),
        1e-6,
    )

    result["A_KEER"] = float(np.mean(acceleration_pred_norm > local_envelope) * 100.0)

    if y_pred.shape[1] < 4:
        result["ViolationRate"] = float(np.mean(acceleration_pred_norm > max_accel) * 100.0)
        return result

    jerk_pred = np.diff(acceleration_pred, axis=1) / dt
    jerk_true = np.diff(acceleration_true, axis=1) / dt

    jerk_pred_norm = np.linalg.norm(jerk_pred, axis=-1)
    jerk_true_norm = np.linalg.norm(jerk_true, axis=-1)

    result["Ej"] = float(np.mean(np.linalg.norm(jerk_pred - jerk_true, axis=-1)))

    pred_energy = np.sum(jerk_pred_norm ** 2, axis=1)
    true_energy = np.sum(jerk_true_norm ** 2, axis=1)

    result["NJER"] = float(np.mean(pred_energy / np.maximum(true_energy, 1e-8)))

    correlations = []

    for pred_profile, true_profile in zip(acceleration_pred_norm, acceleration_true_norm):
        if np.std(pred_profile) < 1e-8 or np.std(true_profile) < 1e-8:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(pred_profile, true_profile)[0, 1]))

    result["KPC"] = float(np.mean(correlations))

    violation = acceleration_pred_norm > max_accel

    if jerk_pred_norm.size:
        jerk_violation = jerk_pred_norm > max_jerk
        violation = violation[:, : jerk_violation.shape[1]] | jerk_violation

    result["ViolationRate"] = float(np.mean(violation) * 100.0)

    return result


def compute_candidate_violation_rate(y_samples, physical_limits):
    dt = float(physical_limits["sample_interval_s"])

    if y_samples.shape[2] < 3:
        return np.nan

    velocity = np.diff(y_samples, axis=2) / dt
    acceleration = np.diff(velocity, axis=2) / dt

    accel_norm = np.linalg.norm(acceleration, axis=-1)

    violation = accel_norm > float(physical_limits["max_accel_mps2"])

    if y_samples.shape[2] >= 4:
        jerk = np.diff(acceleration, axis=2) / dt
        jerk_norm = np.linalg.norm(jerk, axis=-1)
        violation = violation[:, :, : jerk_norm.shape[2]] | (
            jerk_norm > float(physical_limits["max_jerk_mps3"])
        )

    return float(np.mean(violation) * 100.0)


def evaluate_by_fixed_groups(
    group_indices,
    y_pred,
    y_true,
    y_samples,
    delta_nll,
    config,
):
    names = {"low": "稳定飞行", "mid": "一般机动", "high": "剧烈机动"}

    horizon = y_pred.shape[1]

    floor = float(config.get("distance_accuracy_floor_m", 0.25))
    step_gain = float(config.get("distance_accuracy_ratio", 0.15))
    cap = float(config.get("distance_accuracy_cap_m", 1.5))

    direction_threshold = float(config.get("direction_threshold_deg", 25.0))
    pair_count = int(config.get("energy_score_pairs", 32))
    limits = config["physical_limits"]

    output = {}

    for key, indices in group_indices.items():
        if len(indices) == 0:
            continue

        pred = y_pred[indices]
        truth = y_true[indices]
        samples = y_samples[:, indices]
        nll = delta_nll[indices]

        error = np.linalg.norm(pred - truth, axis=-1)
        sample_error = np.linalg.norm(samples - truth[np.newaxis], axis=-1)

        lower = np.quantile(samples, 0.025, axis=0)
        upper = np.quantile(samples, 0.975, axis=0)

        kin = compute_kinematic_metrics(pred, truth, limits)

        step_metrics = []

        for step in range(horizon):
            distance_threshold = min(floor + step_gain * step, cap)

            step_metrics.append(
                {
                    "step": step + 1,
                    "dist_error_m": float(np.mean(error[:, step])),
                    "step_mse": float(np.mean(error[:, step] ** 2)),
                    "step_rmse": float(np.sqrt(np.mean(error[:, step] ** 2))),
                    "dist_acc_pct": float(
                        np.mean(error[:, step] < distance_threshold) * 100.0
                    ),
                    "dir_acc_pct": _direction_accuracy(
                        pred,
                        truth,
                        step,
                        direction_threshold,
                    ),
                    "nll": float(np.mean(nll[:, step])),
                    "mpic_95": float(
                        np.mean(
                            (truth[:, step] >= lower[:, step])
                            & (truth[:, step] <= upper[:, step])
                        )
                        * 100.0
                    ),
                    "miw_95": float(np.mean(upper[:, step] - lower[:, step])),
                }
            )

        output[key] = {
            "name": names[key],
            "n_samples": int(len(indices)),
            "ade": float(np.mean(error)),
            "fde": float(np.mean(error[:, -1])),
            "min_ade": float(np.mean(np.min(np.mean(sample_error, axis=2), axis=0))),
            "min_fde": float(np.mean(np.min(sample_error[:, :, -1], axis=0))),
            "EnergyScore": _energy_score(samples, truth, pair_count=pair_count),
            "mPIC95": float(np.mean((truth >= lower) & (truth <= upper)) * 100.0),
            "MIW95": float(np.mean(upper - lower)),
            "DeltaNLL": float(np.mean(nll)),
            "Ev": kin["Ev"],
            "Ea": kin["Ea"],
            "Ej": kin["Ej"],
            "A_KEER": kin["A_KEER"],
            "NJER": kin["NJER"],
            "KPC": kin["KPC"],
            "violations_rate": kin["ViolationRate"],
            "candidate_violations_rate": compute_candidate_violation_rate(samples, limits),
            "step_metrics": step_metrics,
        }

    return output


# =====================================================================
# 6. Train, validation selection, test reporting
# =====================================================================

def measure_hardware_metrics(model, feature_dim, seq_len, device, exp_name, save_dir, num_iterations=50):
    model.eval()

    dummy = {
        "seq": torch.randn(1, seq_len, feature_dim, device=device),
        "last_delta": torch.randn(1, 3, device=device),
    }

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_g = np.nan

    if profile is not None:
        try:
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            flops_g = 2.0 * macs / 1e9
        except Exception:
            pass

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        measurements = []

        for _ in range(num_iterations):
            start.record()
            _ = model(dummy)
            end.record()
            torch.cuda.synchronize()
            measurements.append(start.elapsed_time(end))

        latency_ms = float(np.mean(measurements))
    else:
        timings = []

        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(dummy)
            timings.append((time.perf_counter() - start_time) * 1000.0)

        latency_ms = float(np.mean(timings))

    row = {
        "Model": exp_name,
        "Params_M": params_m,
        "FLOPs_G": flops_g,
        "Latency_ms_K1": latency_ms,
    }

    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame([row]).to_csv(
        os.path.join(save_dir, "hardware_metrics.csv"),
        index=False,
    )

    return params_m, flops_g, latency_ms


def print_maneuver_group_results(group_results, split_name):
    print("\n" + "=" * 108)
    print(f"  {split_name}：固定机动类别分场景结果")
    print("=" * 108)

    for key in ("low", "mid", "high"):
        if key not in group_results:
            continue

        group = group_results[key]

        print(
            f"\n【{group['name']}】N={group['n_samples']} | "
            f"ADE={group['ade']:.4f} m | FDE={group['fde']:.4f} m | "
            f"ES={group['EnergyScore']:.4f} | "
            f"mPIC95={group['mPIC95']:.2f}% | MIW95={group['MIW95']:.4f} m"
        )

        print("  step | dist err | DistAcc | DirAcc | Delta-NLL | mPIC95 | MIW95")

        for metric in group["step_metrics"]:
            print(
                f"  T+{metric['step']:<2} | {metric['dist_error_m']:<8.4f} | "
                f"{metric['dist_acc_pct']:<7.2f} | {metric['dir_acc_pct']:<6.2f} | "
                f"{metric['nll']:<9.4f} | {metric['mpic_95']:<6.2f} | "
                f"{metric['miw_95']:<7.4f}"
            )


def run_maneuver_eval_on_best(
    model,
    data_loader,
    scaler_target,
    group_indices,
    config,
    device,
    run_dir,
    split_name="test",
):
    samples_k = int(config.get("final_mc_samples", 100))
    use_projection = bool(config["ablation"].get("reachable_projection", True))
    projection_status = (
        "enabled; initial state=last observed physical displacement"
        if use_projection
        else "disabled"
    )
    print(f"  reachable projection: {projection_status}")

    print(f"\n⏳ {split_name} final probabilistic evaluation: K={samples_k}")

    all_pred = []
    all_true = []
    all_samples = []
    all_delta_nll = []

    all_intent_pred = []
    all_intent_true = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            bx = {
                "seq": batch["seq"].to(device, non_blocking=True),
                "last_delta": batch["last_delta"].to(device, non_blocking=True),
            }

            target_delta = batch["target"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
                enabled=(device.type == "cuda"),
            ):
                pred_delta, log_var, intent_logits, _ = model(bx)
                mc_delta = model.generate_probabilistic_trajectories(
                    bx,
                    num_samples=samples_k,
                )

            pred_trajectory = np.cumsum(to_real(pred_delta, scaler_target), axis=1)
            true_trajectory = np.cumsum(
                to_real(target_delta.detach().cpu().numpy(), scaler_target),
                axis=1,
            )

            sampled_trajectories = _mc_deltas_to_trajectories(
                mc_delta,
                scaler_target,
                config["physical_limits"],
                use_projection=use_projection,
                device=device,
                initial_delta=bx["last_delta"],
            )

            all_pred.append(pred_trajectory)
            all_true.append(true_trajectory)
            all_samples.append(sampled_trajectories)

            all_delta_nll.append(
                _delta_nll_physical(
                    pred_delta,
                    log_var,
                    target_delta,
                    scaler_target,
                )
            )

            if intent_logits is not None:
                all_intent_pred.append(torch.argmax(intent_logits, dim=-1).cpu().numpy())
                all_intent_true.append(batch["intent"].cpu().numpy())

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    y_samples = np.concatenate(all_samples, axis=1)
    delta_nll = np.concatenate(all_delta_nll, axis=0)

    probability_metrics = compute_probabilistic_metrics(
        y_true=y_true,
        y_samples=y_samples,
        delta_nll=delta_nll,
        pair_count=int(config.get("energy_score_pairs", 32)),
    )

    group_results = evaluate_by_fixed_groups(
        group_indices=group_indices,
        y_pred=y_pred,
        y_true=y_true,
        y_samples=y_samples,
        delta_nll=delta_nll,
        config=config,
    )

    if all_intent_pred:
        intent_pred = np.concatenate(all_intent_pred)
        intent_true = np.concatenate(all_intent_true)

        intent_acc = float(np.mean(intent_pred == intent_true))

        try:
            from sklearn.metrics import f1_score
            intent_f1 = float(f1_score(intent_true, intent_pred, average="macro"))
        except Exception:
            intent_f1 = float("nan")
    else:
        intent_acc = float("nan")
        intent_f1 = float("nan")

    probability_metrics["Intent_Acc"] = intent_acc
    probability_metrics["Intent_F1"] = intent_f1

    pd.DataFrame([probability_metrics]).to_csv(
        os.path.join(run_dir, f"{split_name}_probabilistic_metrics.csv"),
        index=False,
    )

    if config.get("save_prediction_npz", False):
        np.savez_compressed(
            os.path.join(run_dir, f"{split_name}_predictions.npz"),
            y_pred_mean=y_pred,
            y_true=y_true,
            y_samples=y_samples,
            delta_nll=delta_nll,
        )

    print_maneuver_group_results(group_results, f"{split_name} set")

    return group_results, probability_metrics


_STEP_KEY_MAP = {
    "dist_error_m": "DistErr",
    "step_mse": "MSE",
    "step_rmse": "RMSE",
    "dist_acc_pct": "DistAcc",
    "dir_acc_pct": "DirAcc",
    "nll": "DeltaNLL",
    "mpic_95": "mPIC95",
    "miw_95": "MIW95",
}

_GROUP_KEY_MAP = {
    "ade": "ADE",
    "fde": "FDE",
    "min_ade": "minADE",
    "min_fde": "minFDE",
    "EnergyScore": "ES",
    "mPIC95": "mPIC95",
    "MIW95": "MIW95",
    "DeltaNLL": "DeltaNLL",
    "Ev": "Ev",
    "Ea": "Ea",
    "Ej": "Ej",
    "A_KEER": "A_KEER",
    "NJER": "NJER",
    "KPC": "KPC",
    "violations_rate": "Violations",
    "candidate_violations_rate": "CandidateViolations",
}


def save_best_maneuver_results(group_results, run_dir, split_name="test"):
    rows = []

    for group_key, group in group_results.items():
        for step in group["step_metrics"]:
            rows.append(
                {
                    "split": split_name,
                    "group_key": group_key,
                    "group_name": group["name"],
                    "n_samples": group["n_samples"],
                    "overall_ADE": group["ade"],
                    "overall_FDE": group["fde"],
                    "overall_ES": group["EnergyScore"],
                    "overall_mPIC95": group["mPIC95"],
                    "overall_MIW95": group["MIW95"],
                    "Ev": group["Ev"],
                    "Ea": group["Ea"],
                    "Ej": group["Ej"],
                    "A_KEER": group["A_KEER"],
                    "NJER": group["NJER"],
                    "KPC": group["KPC"],
                    "ViolationRate": group["violations_rate"],
                    "CandidateViolationRate": group["candidate_violations_rate"],
                    "step": step["step"],
                    "dist_error_m": step["dist_error_m"],
                    "step_MSE": step["step_mse"],
                    "step_RMSE": step["step_rmse"],
                    "dist_acc_pct": step["dist_acc_pct"],
                    "dir_acc_pct": step["dir_acc_pct"],
                    "DeltaNLL": step["nll"],
                    "mPIC95": step["mpic_95"],
                    "MIW95": step["miw_95"],
                }
            )

    path = os.path.join(run_dir, f"{split_name}_maneuver_detail.csv")
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def train_and_eval(config, global_dataset):
    seed = int(config["seed"])
    seed_everything(seed, deterministic=bool(config.get("deterministic", True)))

    run_dir = config["run_dir"]
    os.makedirs(run_dir, exist_ok=True)

    ablation = make_ablation_config(**config["ablation"])

    train_seqs = global_dataset["train_seqs"]
    train_indices = global_dataset["train_indices"]
    val_seqs = global_dataset["val_seqs"]
    val_indices = global_dataset["val_indices"]
    test_seqs = global_dataset["test_seqs"]
    test_indices = global_dataset["test_indices"]

    scaler_target = global_dataset["scaler_target"]
    feature_dim = global_dataset["feat_dim"]
    max_s = global_dataset["max_s"]
    horizon = global_dataset["horizon"]
    physical_limits = global_dataset["physical_limits"]

    config = {**config, "ablation": ablation, "physical_limits": physical_limits}

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {**config, "ablation": ablation},
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    train_workers = int(config.get("train_num_workers", 0))
    eval_workers = int(config.get("eval_num_workers", config.get("num_workers", 4)))

    val_loader = _make_eval_loader(
        val_seqs,
        val_indices,
        max_s,
        horizon,
        config["batch_size"],
        eval_workers,
        seed + 777,
    )

    test_loader = _make_eval_loader(
        test_seqs,
        test_indices,
        max_s,
        horizon,
        config["batch_size"],
        eval_workers,
        seed + 888,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IntentAwarePredictor(
        feature_dim=feature_dim,
        d_model=int(config.get("d_model", 192)),
        horizon=horizon,
        scales=config["scales"],
        d_scaler_scale=scaler_target.scale_,
        ablation_config=ablation,
    ).to(device)

    params_m, flops_g, latency_ms = measure_hardware_metrics(
        model=model,
        feature_dim=feature_dim,
        seq_len=max_s,
        device=device,
        exp_name=config["model_variant"],
        save_dir=run_dir,
    )

    optimizer = _make_optimizer(model, float(config["learning_rate"]))

    fusion_monitor = (
        FusionHealthMonitor(model=model, optimizer=optimizer, verbose=False)
        if ablation["parallel"] and hasattr(model, "global_mix_alpha")
        else NullFusionMonitor()
    )

    amp_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    warmup_epochs = min(2, max(1, int(config["epochs"]) - 1))

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(config["epochs"]) - warmup_epochs),
                eta_min=1e-6,
            ),
        ],
        milestones=[warmup_epochs],
    )

    ema = ModelEMA(model, decay=0.999, warmup_steps=2000)

    train_labels = [item["label"] for item in train_indices]

    history = []
    best_score = float("inf")
    best_epoch = -1
    checkpoint_path = os.path.join(run_dir, "best_checkpoint.pt")

    for epoch in range(int(config["epochs"])):
        progress = epoch / max(int(config["epochs"]) - 1, 1)

        model.goal_correction_strength = min(1.0, max(0.0, (progress - 0.30) / 0.30))

        train_loader = get_curriculum_loader(
            dataset=OnTheFlyUAVDataset(
                train_seqs,
                train_indices,
                max_s,
                horizon,
                is_training=True,
            ),
            batch_size=config["batch_size"],
            epoch=epoch,
            total_epochs=config["epochs"],
            original_labels=train_labels,
            seed=seed,
            num_workers=train_workers,
            pin_memory=True,
        )

        model.train()

        loss_sum = 0.0
        batches = 0

        for batch in train_loader:
            bx = {
                "seq": batch["seq"].to(device, non_blocking=True),
                "last_delta": batch["last_delta"].to(device, non_blocking=True),
            }

            target = batch["target"].to(device, non_blocking=True)
            intent_target = batch["intent"].to(device, non_blocking=True)

            last_pose = bx["seq"][:, -1, 3:6]

            optimizer.zero_grad(set_to_none=True)

            tf_probability = max(0.05, 0.55 - 0.45 * progress)
            tf_mask = torch.rand(bx["seq"].size(0), horizon, device=device) < tf_probability

            with torch.amp.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
                enabled=(device.type == "cuda"),
            ):
                temperature = max(1.0, 5.0 - 4.0 * progress)

                pred, logvar, intent_logits, turn_rates, aux = model(
                    bx,
                    tau=temperature,
                    target_deltas=target,
                    tf_mask=tf_mask,
                    current_horizon=horizon,
                    return_aux=True,
                )

                loss = uav_loss_v20(
                    pred=pred,
                    log_var=logvar,
                    by=target,
                    last_pose=last_pose,
                    intent_logits=intent_logits,
                    intent_targets=intent_target,
                    epoch=epoch,
                    total_epochs=config["epochs"],
                    turn_rates_pred=turn_rates,
                    is_training=True,
                    current_horizon=horizon,
                    aux_info=aux,
                    use_intent_supervision=ablation["intent_supervision"],
                    use_intent_weighted_loss=ablation["intent_weighted_loss"],
                )

                loss = loss + fusion_monitor.compute_aux_loss(target, aux)

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            amp_scaler.step(optimizer)
            amp_scaler.update()

            with torch.no_grad():
                model.output_scale.data.clamp_(0.5, 1.5)

            ema.update(model)

            loss_sum += float(loss.item())
            batches += 1

        evaluation_interval = int(config.get("maneuver_eval_interval", 5))

        evaluate_now = (
            (epoch + 1) % evaluation_interval == 0
            or epoch + 1 == int(config["epochs"])
        )

        if evaluate_now:
            model.eval()
            ema.apply_to(model)

            val_loss_sum = 0.0
            val_batches = 0

            predictions = []
            targets = []

            with torch.no_grad():
                for batch in val_loader:
                    bx = {
                        "seq": batch["seq"].to(device, non_blocking=True),
                        "last_delta": batch["last_delta"].to(device, non_blocking=True),
                    }

                    target = batch["target"].to(device, non_blocking=True)
                    intent_target = batch["intent"].to(device, non_blocking=True)
                    last_pose = bx["seq"][:, -1, 3:6]

                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=AMP_DTYPE,
                        enabled=(device.type == "cuda"),
                    ):
                        pred, logvar, intent_logits, turn_rates = model(bx)

                        val_loss = uav_loss_v20(
                            pred=pred,
                            log_var=logvar,
                            by=target,
                            last_pose=last_pose,
                            intent_logits=intent_logits,
                            intent_targets=intent_target,
                            epoch=epoch,
                            total_epochs=config["epochs"],
                            turn_rates_pred=turn_rates,
                            is_training=False,
                            current_horizon=horizon,
                            aux_info=None,
                            use_intent_supervision=ablation["intent_supervision"],
                            use_intent_weighted_loss=ablation["intent_weighted_loss"],
                        )

                    predictions.append(np.cumsum(to_real(pred, scaler_target), axis=1))
                    targets.append(
                        np.cumsum(
                            to_real(target.detach().cpu().numpy(), scaler_target),
                            axis=1,
                        )
                    )

                    val_loss_sum += float(val_loss.item())
                    val_batches += 1

            y_pred = np.concatenate(predictions, axis=0)
            y_true = np.concatenate(targets, axis=0)

            errors = np.linalg.norm(y_pred - y_true, axis=2)

            ade = float(np.mean(errors))
            fde = float(np.mean(errors[:, -1]))

            selection_score = ade + 0.5 * fde

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": loss_sum / max(batches, 1),
                    "val_loss": val_loss_sum / max(val_batches, 1),
                    "val_ADE": ade,
                    "val_FDE": fde,
                    "selection_score": selection_score,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            pd.DataFrame(history).to_csv(
                os.path.join(run_dir, "training_history.csv"),
                index=False,
            )

            print(
                f"[{config['model_variant']}] seed={seed} epoch={epoch + 1:03d} | "
                f"val_ADE={ade:.4f} | val_FDE={fde:.4f} | "
                f"selection={selection_score:.4f}"
            )

            if selection_score < best_score:
                best_score = selection_score
                best_epoch = epoch + 1

                torch.save(
                    {
                        "epoch": best_epoch,
                        "seed": seed,
                        "selection_score": best_score,
                        "ablation": ablation,
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )

            ema.restore(model)

        fusion_monitor.update(epoch)
        scheduler.step()

        del train_loader
        gc.collect()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    test_results, probability_metrics = run_maneuver_eval_on_best(
        model=model,
        data_loader=test_loader,
        scaler_target=scaler_target,
        group_indices=global_dataset["test_groups"],
        config=config,
        device=device,
        run_dir=run_dir,
        split_name="test",
    )

    save_best_maneuver_results(test_results, run_dir=run_dir, split_name="test")

    metrics = {
        "Model Variant": config["model_variant"],
        "Seed": seed,
        "BestEpoch": best_epoch,
        "ValidationSelectionScore": best_score,
        "Params_M": params_m,
        "FLOPs_G": flops_g,
        "Latency_ms_K1": latency_ms,
        "GLOBAL_minADE": probability_metrics["minADE"],
        "GLOBAL_minFDE": probability_metrics["minFDE"],
        "GLOBAL_ES": probability_metrics["EnergyScore"],
        "GLOBAL_NLL_Delta": probability_metrics["DeltaNLL"],
        "GLOBAL_mPIC95": probability_metrics["mPIC95"],
        "GLOBAL_mPIC95_CalErr": abs(probability_metrics["mPIC95"] - 95.0),
        "GLOBAL_MIW95": probability_metrics["MIW95"],
        "Intent_Acc": probability_metrics["Intent_Acc"],
        "Intent_F1": probability_metrics["Intent_F1"],
        "IntentEnabled": int(
            ablation["intent_supervision"] or ablation["intent_conditioning"]
        ),
        "TestSplit": "test",
    }

    valid_groups = []

    for key in ("low", "mid", "high"):
        group = test_results.get(key)

        if group is None:
            continue

        valid_groups.append(group)

        prefix = f"{key.upper()}_"

        for raw_key, output_key in _GROUP_KEY_MAP.items():
            metrics[prefix + output_key] = group.get(raw_key, np.nan)

        for keep_step in (5, 10, 15, 20, 25):
            if keep_step <= len(group["step_metrics"]):
                step = group["step_metrics"][keep_step - 1]

                for raw_key, output_key in _STEP_KEY_MAP.items():
                    metrics[f"{prefix}T{keep_step}_{output_key}"] = step.get(raw_key, np.nan)

    if valid_groups:
        total_n = sum(group["n_samples"] for group in valid_groups)

        def weighted(key):
            numerator = 0.0
            denominator = 0.0

            for group in valid_groups:
                value = group.get(key, np.nan)

                if np.isfinite(value):
                    numerator += value * group["n_samples"]
                    denominator += group["n_samples"]

            return float(numerator / max(denominator, 1)) if denominator > 0 else np.nan

        metrics["GLOBAL_ADE"] = weighted("ade")
        metrics["GLOBAL_FDE"] = weighted("fde")

        metrics["GLOBAL_Ev"] = weighted("Ev")
        metrics["GLOBAL_Ea"] = weighted("Ea")
        metrics["GLOBAL_Ej"] = weighted("Ej")

        metrics["GLOBAL_SE"] = metrics["GLOBAL_Ej"]
        metrics["GLOBAL_NJER"] = weighted("NJER")
        metrics["GLOBAL_KPC"] = weighted("KPC")

        metrics["GLOBAL_A_KEER"] = weighted("A_KEER")
        metrics["GLOBAL_NJERDev"] = float(
            sum(
                abs(group.get("NJER", 1.0) - 1.0) * group["n_samples"]
                for group in valid_groups
            )
            / max(total_n, 1)
        )

        metrics["GLOBAL_Violations"] = weighted("violations_rate")
        metrics["GLOBAL_CandidateViolations"] = weighted("candidate_violations_rate")

    pd.DataFrame([metrics]).to_csv(
        os.path.join(run_dir, "seed_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return metrics


# =====================================================================
# 7. Formal paper-ladder variants
# =====================================================================

def cfg_emam_se():
    return make_ablation_config(
        emam=True,
        emam_multiscale_conv=True,
        emam_channel_attn=True,

        dtp_context=False,
        dtp_attention_gate=False,
        intent_supervision=False,
        intent_conditioning=False,
        intent_weighted_loss=False,

        parallel=False,
        parallel_self_attn=False,

        pgd=False,
        pgd_kpl=False,
        pgd_magnitude_head=False,
        pgd_intent_alpha=False,
        adaptive_inertia=False,
        smoother=False,
        reachable_projection=False,

        unc_mode="homo",

        turn_rate_aux=False,
        goal_anchor=False,
    )


def cfg_emam_se_iadtp():
    return make_ablation_config(
        emam=True,
        emam_multiscale_conv=True,
        emam_channel_attn=True,

        dtp_context=True,
        dtp_attention_gate=True,
        intent_supervision=True,
        intent_conditioning=True,
        intent_weighted_loss=False,

        parallel=False,
        parallel_self_attn=False,

        pgd=False,
        pgd_kpl=False,
        pgd_magnitude_head=False,
        pgd_intent_alpha=False,
        adaptive_inertia=False,
        smoother=False,
        reachable_projection=False,

        unc_mode="homo",

        turn_rate_aux=False,
        goal_anchor=False,
    )


def cfg_ours():
    return make_ablation_config(
        emam=True,
        emam_multiscale_conv=True,
        emam_channel_attn=True,

        dtp_context=True,
        dtp_attention_gate=True,
        intent_supervision=True,
        intent_conditioning=True,
        intent_weighted_loss=False,

        parallel=True,
        parallel_self_attn=True,

        pgd=True,
        pgd_kpl=True,
        pgd_magnitude_head=True,
        pgd_intent_alpha=True,
        adaptive_inertia=True,
        smoother=True,
        reachable_projection=True,

        unc_mode="hetero",

        turn_rate_aux=False,
        goal_anchor=False,
    )


def validate_paper_ladder_configs(variants):
    expected = {
        "EMam-SE": {
            "emam": True,
            "dtp_context": False,
            "intent_supervision": False,
            "intent_conditioning": False,
            "parallel": False,
            "pgd": False,
            "unc_mode": "homo",
            "turn_rate_aux": False,
            "goal_anchor": False,
        },
        "EMam-SE+IA-DTP": {
            "emam": True,
            "dtp_context": True,
            "intent_supervision": True,
            "intent_conditioning": True,
            "parallel": False,
            "pgd": False,
            "unc_mode": "homo",
            "turn_rate_aux": False,
            "goal_anchor": False,
        },
        "Ours": {
            "emam": True,
            "dtp_context": True,
            "intent_supervision": True,
            "intent_conditioning": True,
            "parallel": True,
            "pgd": True,
            "pgd_kpl": True,
            "pgd_magnitude_head": True,
            "pgd_intent_alpha": True,
            "adaptive_inertia": True,
            "smoother": True,
            "reachable_projection": True,
            "unc_mode": "hetero",
            "turn_rate_aux": False,
            "goal_anchor": False,
        },
    }

    for name, checks in expected.items():
        if name not in variants:
            raise RuntimeError(f"Missing ablation variant: {name}")

        config = variants[name]

        for key, target in checks.items():
            actual = config.get(key, None)

            if actual != target:
                raise RuntimeError(
                    f"Wrong ablation config for [{name}]: "
                    f"{key}={actual}, expected {target}"
                )

    print("✅ 消融配置自检通过：EMam-SE → EMam-SE+IA-DTP → Ours")


def build_paper_ladder_variants():
    variants = OrderedDict(
        [
            ("EMam-SE", cfg_emam_se()),
            ("EMam-SE+IA-DTP", cfg_emam_se_iadtp()),
            ("Ours", cfg_ours()),
        ]
    )

    validate_paper_ladder_configs(variants)

    return variants


# =====================================================================
# 8. Summary tables
# =====================================================================

PAPER_VARIANT_ORDER = [
    "EMam-SE",
    "EMam-SE+IA-DTP",
    "Ours",
]

PAPER_METRIC_ALIASES = OrderedDict(
    {
        "ADE": ["GLOBAL_ADE"],
        "FDE": ["GLOBAL_FDE"],
        "High-FDE": ["HIGH_FDE"],

        "DeltaNLL": ["GLOBAL_NLL_Delta"],
        "PIC95": ["GLOBAL_mPIC95"],
        "MIW95": ["GLOBAL_MIW95"],

        "SE": ["GLOBAL_SE", "GLOBAL_Ej"],
        "NJER": ["GLOBAL_NJER"],
        "KPC": ["GLOBAL_KPC"],
    }
)

METRIC_PREFERENCE = {
    "ADE": "lower",
    "FDE": "lower",
    "High-FDE": "lower",
    "DeltaNLL": "lower",
    "MIW95": "lower",
    "SE": "lower",

    "PIC95": "higher",
    "KPC": "higher",

    "NJER": "close_to_one",
}


def _weighted_mean_from_groups(row, group_prefixes, metric_name):
    values = []

    for prefix in group_prefixes:
        column = f"{prefix}_{metric_name}"

        if column in row:
            value = _as_float(row[column])

            if np.isfinite(value):
                values.append(value)

    return float(np.mean(values)) if values else np.nan


def normalize_metrics_for_paper(metrics):
    metrics = dict(metrics)

    if "GLOBAL_SE" not in metrics or not np.isfinite(_as_float(metrics.get("GLOBAL_SE"))):
        if "GLOBAL_Ej" in metrics and np.isfinite(_as_float(metrics.get("GLOBAL_Ej"))):
            metrics["GLOBAL_SE"] = _as_float(metrics["GLOBAL_Ej"])
        else:
            metrics["GLOBAL_SE"] = _weighted_mean_from_groups(
                metrics,
                ["LOW", "MID", "HIGH"],
                "Ej",
            )

    if "GLOBAL_NJER" not in metrics or not np.isfinite(_as_float(metrics.get("GLOBAL_NJER"))):
        metrics["GLOBAL_NJER"] = _weighted_mean_from_groups(
            metrics,
            ["LOW", "MID", "HIGH"],
            "NJER",
        )

    if "GLOBAL_KPC" not in metrics or not np.isfinite(_as_float(metrics.get("GLOBAL_KPC"))):
        metrics["GLOBAL_KPC"] = _weighted_mean_from_groups(
            metrics,
            ["LOW", "MID", "HIGH"],
            "KPC",
        )

    if "GLOBAL_NJERDev" not in metrics or not np.isfinite(_as_float(metrics.get("GLOBAL_NJERDev"))):
        njer_value = _as_float(metrics.get("GLOBAL_NJER"))
        metrics["GLOBAL_NJERDev"] = abs(njer_value - 1.0) if np.isfinite(njer_value) else np.nan

    return metrics


def _metric_series(frame, metric_name):
    aliases = PAPER_METRIC_ALIASES[metric_name]
    column = _first_existing_column(frame, aliases)

    if column is None:
        return pd.Series([np.nan] * len(frame), index=frame.index)

    return pd.to_numeric(frame[column], errors="coerce")


def _format_mean_std(mean, std, metric_name):
    if not np.isfinite(mean):
        return ""

    if metric_name == "PIC95":
        return f"{mean:.2f} ± {std:.2f}" if np.isfinite(std) else f"{mean:.2f}"

    return f"{mean:.4f} ± {std:.4f}" if np.isfinite(std) else f"{mean:.4f}"


def save_module_composition_table(save_dir):
    table = pd.DataFrame(
        [
            {
                "Variant": "EMam-SE",
                "EMam-SE": "Yes",
                "IA-DTP": "No",
                "UA-PGD": "No",
            },
            {
                "Variant": "EMam-SE+IA-DTP",
                "EMam-SE": "Yes",
                "IA-DTP": "Yes",
                "UA-PGD": "No",
            },
            {
                "Variant": "Ours",
                "EMam-SE": "Yes",
                "IA-DTP": "Yes",
                "UA-PGD": "Yes",
            },
        ]
    )

    path = os.path.join(save_dir, "ablation_module_composition.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")

    return path


def compute_progressive_module_effects(successful, save_dir):
    comparisons = [
        ("EMam-SE", "EMam-SE+IA-DTP", "Add IA-DTP"),
        ("EMam-SE+IA-DTP", "Ours", "Add UA-PGD"),
    ]

    rows = []

    for before_name, after_name, effect_name in comparisons:
        before_df = successful[successful["Model Variant"] == before_name].copy()
        after_df = successful[successful["Model Variant"] == after_name].copy()

        if before_df.empty or after_df.empty:
            continue

        paired = before_df.merge(
            after_df,
            on="Seed",
            suffixes=("_before", "_after"),
        )

        if paired.empty:
            continue

        for metric_name, preference in METRIC_PREFERENCE.items():
            aliases = PAPER_METRIC_ALIASES[metric_name]

            before_col = None
            after_col = None

            for alias in aliases:
                if f"{alias}_before" in paired.columns:
                    before_col = f"{alias}_before"

                if f"{alias}_after" in paired.columns:
                    after_col = f"{alias}_after"

                if before_col is not None and after_col is not None:
                    break

            if before_col is None or after_col is None:
                continue

            before_val = pd.to_numeric(paired[before_col], errors="coerce")
            after_val = pd.to_numeric(paired[after_col], errors="coerce")

            valid = before_val.notna() & after_val.notna()

            if valid.sum() == 0:
                continue

            before_val = before_val[valid]
            after_val = after_val[valid]

            if preference == "lower":
                improvement = before_val - after_val
            elif preference == "higher":
                improvement = after_val - before_val
            elif preference == "close_to_one":
                improvement = (before_val - 1.0).abs() - (after_val - 1.0).abs()
            else:
                raise ValueError(f"Unknown preference: {preference}")

            rows.append(
                {
                    "Effect": effect_name,
                    "Before": before_name,
                    "After": after_name,
                    "Metric": metric_name,
                    "Preference": preference,
                    "PositiveMeansAfterBetter": True,
                    "PairedMeanImprovement": float(improvement.mean()),
                    "PairedStdImprovement": (
                        float(improvement.std(ddof=1))
                        if len(improvement) > 1
                        else np.nan
                    ),
                    "N": int(len(improvement)),
                    "WinRate": float((improvement > 0).mean()),
                }
            )

    result = pd.DataFrame(rows)

    path = os.path.join(save_dir, "progressive_module_effects.csv")
    result.to_csv(path, index=False, encoding="utf-8-sig")

    return path


def summarize_ablation_results(raw_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(save_dir, "seed_results.csv")
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    successful = raw_df.copy()

    if "Error" in successful.columns:
        successful = successful[
            successful["Error"].isna()
            | (successful["Error"].astype(str).str.strip() == "")
            | (successful["Error"].astype(str).str.lower() == "nan")
            | (successful["Error"].astype(str).str.lower() == "none")
        ].copy()

    if successful.empty:
        print("⚠️ 没有成功完成的消融实验，无法汇总。")
        return None

    normalized_rows = [
        normalize_metrics_for_paper(row.to_dict())
        for _, row in successful.iterrows()
    ]

    successful = pd.DataFrame(normalized_rows)

    successful = successful[
        successful["Model Variant"].isin(PAPER_VARIANT_ORDER)
    ].copy()

    successful["VariantOrder"] = successful["Model Variant"].map(
        {name: idx for idx, name in enumerate(PAPER_VARIANT_ORDER)}
    )

    successful = successful.sort_values(["VariantOrder", "Seed"]).reset_index(drop=True)

    norm_path = os.path.join(save_dir, "seed_results_normalized.csv")
    successful.to_csv(norm_path, index=False, encoding="utf-8-sig")

    numeric_cols = successful.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = {
        "Seed",
        "VariantOrder",
        "BestEpoch",
        "IntentEnabled",
    }

    metric_cols = [column for column in numeric_cols if column not in exclude_cols]

    summary = (
        successful.groupby("Model Variant")[metric_cols]
        .agg(["mean", "std", "count"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary["VariantOrder"] = summary["Model Variant"].map(
        {name: idx for idx, name in enumerate(PAPER_VARIANT_ORDER)}
    )

    summary = (
        summary.sort_values("VariantOrder")
        .drop(columns=["VariantOrder"])
        .reset_index(drop=True)
    )

    mean_std_path = os.path.join(save_dir, "ablation_mean_std.csv")
    summary.to_csv(mean_std_path, index=False, encoding="utf-8-sig")

    paper = pd.DataFrame({"Variant": PAPER_VARIANT_ORDER})

    for metric_name in PAPER_METRIC_ALIASES.keys():
        formatted_values = []

        for variant_name in PAPER_VARIANT_ORDER:
            sub = successful[successful["Model Variant"] == variant_name]
            series = _metric_series(sub, metric_name)

            mean = float(series.mean()) if len(series) else np.nan
            std = float(series.std(ddof=1)) if len(series) > 1 else np.nan

            formatted_values.append(_format_mean_std(mean, std, metric_name))

        paper[metric_name] = formatted_values

    paper_path = os.path.join(save_dir, "ablation_paper_table.csv")
    paper.to_csv(paper_path, index=False, encoding="utf-8-sig")

    module_path = save_module_composition_table(save_dir)
    effect_path = compute_progressive_module_effects(successful, save_dir)

    print("\n" + "=" * 80)
    print("✅ 消融实验汇总完成")
    print("=" * 80)
    print(f"  Seed results:       {raw_path}")
    print(f"  Normalized results: {norm_path}")
    print(f"  Mean/std table:     {mean_std_path}")
    print(f"  Paper table:        {paper_path}")
    print(f"  Module table:       {module_path}")
    print(f"  Progressive effect: {effect_path}")

    try:
        print("\n--- Paper table preview ---")
        print(paper.to_markdown(index=False))
    except Exception:
        print(paper)

    return summary


# =====================================================================
# 9. Runner
# =====================================================================

def load_existing_seed_metrics(run_dir):
    path = os.path.join(run_dir, "seed_metrics.csv")

    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)

        if df.empty:
            return None

        return df.iloc[-1].to_dict()

    except Exception:
        return None


def row_already_successful(rows, variant_name, seed):
    for row in rows:
        if str(row.get("Model Variant", "")) != str(variant_name):
            continue

        if int(row.get("Seed", -999999)) != int(seed):
            continue

        error = str(row.get("Error", "")).strip()

        if error == "" or error.lower() in ("nan", "none"):
            return True

    return False


def run_paper_ladder_ablation(base_config, global_shared_data, seeds):
    variants = build_paper_ladder_variants()

    root_dir = base_config.get("maneuver_save_dir", "Ablation_Paper_Ladder")
    scope = base_config.get("ablation_scope", "paper_ladder")

    save_dir = os.path.join(root_dir, scope)
    os.makedirs(save_dir, exist_ok=True)

    raw_result_path = os.path.join(save_dir, "seed_results.csv")

    all_rows = []

    if os.path.exists(raw_result_path) and base_config.get("resume", True):
        try:
            old_df = pd.read_csv(raw_result_path)

            if not old_df.empty:
                all_rows = old_df.to_dict("records")
                print(f"✅ 已加载已有结果：{raw_result_path}")

        except Exception as exc:
            print(f"⚠️ 已有结果读取失败，将重新追加记录：{exc}")

    for seed in seeds:
        for variant_name, ablation_cfg in variants.items():
            print("\n" + "=" * 96)
            print(f"🚀 Paper-ladder ablation | Variant: {variant_name} | Seed: {seed}")
            print("=" * 96)

            variant_dir = safe_variant_name(variant_name)
            run_dir = os.path.join(save_dir, variant_dir, f"seed_{int(seed)}")
            os.makedirs(run_dir, exist_ok=True)

            if base_config.get("resume", True):
                cached = load_existing_seed_metrics(run_dir)

                if cached is not None:
                    cached = normalize_metrics_for_paper(cached)
                    cached["Model Variant"] = variant_name
                    cached["Seed"] = int(seed)
                    cached["Ablation Scope"] = scope
                    cached["RunDir"] = run_dir
                    cached["Error"] = ""

                    if not row_already_successful(all_rows, variant_name, seed):
                        all_rows.append(cached)

                        pd.DataFrame(all_rows).to_csv(
                            raw_result_path,
                            index=False,
                            encoding="utf-8-sig",
                        )

                    print(f"  ⏭ 已复用完成结果：{run_dir}")
                    continue

            run_config = base_config.copy()

            run_config.update(
                {
                    "seed": int(seed),
                    "exp_name": f"{safe_variant_name(variant_name)}_seed_{int(seed)}",
                    "model_variant": variant_name,
                    "run_dir": run_dir,
                    "ablation": ablation_cfg,
                }
            )

            seed_everything(
                int(seed),
                deterministic=bool(base_config.get("deterministic", True)),
            )

            try:
                metrics = train_and_eval(
                    run_config,
                    global_dataset=global_shared_data,
                )

                metrics = normalize_metrics_for_paper(metrics)

                metrics["Model Variant"] = variant_name
                metrics["Seed"] = int(seed)
                metrics["Ablation Scope"] = scope
                metrics["RunDir"] = run_dir
                metrics["Error"] = ""

                pd.DataFrame([metrics]).to_csv(
                    os.path.join(run_dir, "seed_metrics.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

                print(f"✅ 完成：{variant_name} | seed={seed}")

            except Exception as exc:
                import traceback
                traceback.print_exc()

                metrics = {
                    "Model Variant": variant_name,
                    "Seed": int(seed),
                    "Ablation Scope": scope,
                    "RunDir": run_dir,
                    "Error": repr(exc),
                }

                print(f"❌ 失败：{variant_name} | seed={seed} | {exc}")

            all_rows.append(metrics)

            pd.DataFrame(all_rows).to_csv(
                raw_result_path,
                index=False,
                encoding="utf-8-sig",
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(float(base_config.get("inter_run_sleep_s", 2.0)))

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(raw_result_path, index=False, encoding="utf-8-sig")

    summarize_ablation_results(raw_df, save_dir)

    return raw_df


def generate_latex_tables_from_csv(save_dir):
    module_csv = os.path.join(save_dir, "ablation_module_composition.csv")
    paper_csv = os.path.join(save_dir, "ablation_paper_table.csv")

    if not os.path.exists(module_csv) or not os.path.exists(paper_csv):
        print("⚠️ CSV 表不存在，跳过 LaTeX 生成。")
        return

    module_df = pd.read_csv(module_csv)
    paper_df = pd.read_csv(paper_csv)

    def yesno(value):
        return r"\checkmark" if str(value).lower() in ("yes", "1", "true") else "--"

    module_lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Module composition of ablation variants.}",
        r"    \label{tab:ablation_module_composition}",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{4.5pt}",
        r"    \renewcommand{\arraystretch}{1.08}",
        r"    \begin{tabular}{lccc}",
        r"        \toprule",
        r"        \textbf{Variant} & \textbf{EMam-SE} & \textbf{IA-DTP} & \textbf{UA-PGD} \\",
        r"        \midrule",
    ]

    for _, row in module_df.iterrows():
        module_lines.append(
            f"        {row['Variant']} & {yesno(row['EMam-SE'])} & "
            f"{yesno(row['IA-DTP'])} & {yesno(row['UA-PGD'])} \\\\"
        )

    module_lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]

    paper_lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Ablation results of progressively adding the proposed modules.}",
        r"    \label{tab:ablation_three_variants}",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{4.0pt}",
        r"    \renewcommand{\arraystretch}{1.08}",
        r"    \begin{tabular}{l ccc ccc ccc}",
        r"        \toprule",
        r"        \multirow{2}{*}{\textbf{Variant}}",
        r"        & \multicolumn{3}{c}{\textbf{Deterministic accuracy}}",
        r"        & \multicolumn{3}{c}{\textbf{Probabilistic prediction}}",
        r"        & \multicolumn{3}{c}{\textbf{Kinematic consistency}} \\",
        r"        \cmidrule(lr){2-4}",
        r"        \cmidrule(lr){5-7}",
        r"        \cmidrule(lr){8-10}",
        r"        & ADE$\downarrow$ & FDE$\downarrow$ & High-FDE$\downarrow$",
        r"        & $\Delta$NLL$\downarrow$ & PIC$_{95}\uparrow$ & MIW$_{95}\downarrow$",
        r"        & SE$\downarrow$ & NJER$\approx 1$ & KPC$\uparrow$ \\",
        r"        \midrule",
    ]

    for _, row in paper_df.iterrows():
        variant = str(row["Variant"])

        if variant == "Ours":
            variant = r"\textit{Ours}"

        paper_lines.append(
            "        "
            + variant
            + f" & ${row['ADE']}$"
            + f" & ${row['FDE']}$"
            + f" & ${row['High-FDE']}$"
            + f" & ${row['DeltaNLL']}$"
            + f" & ${row['PIC95']}$"
            + f" & ${row['MIW95']}$"
            + f" & ${row['SE']}$"
            + f" & ${row['NJER']}$"
            + f" & ${row['KPC']}$ \\\\"
        )

    paper_lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \vspace{0.3em}",
        r"    \begin{minipage}{0.96\textwidth}",
        r"        \footnotesize",
        r"        PIC$_{95}$ and MIW$_{95}$ should be interpreted jointly. NJER is optimal when it is close to 1 rather than simply smaller.",
        r"    \end{minipage}",
        r"\end{table*}",
    ]

    module_tex = os.path.join(save_dir, "ablation_module_composition.tex")
    paper_tex = os.path.join(save_dir, "ablation_paper_table.tex")

    with open(module_tex, "w", encoding="utf-8") as handle:
        handle.write("\n".join(module_lines))

    with open(paper_tex, "w", encoding="utf-8") as handle:
        handle.write("\n".join(paper_lines))

    print(f"✅ LaTeX module table saved: {module_tex}")
    print(f"✅ LaTeX paper table saved:  {paper_tex}")


# =====================================================================
# 10. Main entry
# =====================================================================

if __name__ == "__main__":
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    mamba_main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba-main")

    if mamba_main_dir not in sys.path:
        sys.path.append(mamba_main_dir)

    try:
        importlib.import_module("mamba_ssm.modules.mamba2")
        print("✅ 动态导入 Mamba2 成功")
    except Exception as exc:
        print(f"⚠️ Mamba2 导入失败：{exc}")

    try:
        _check_mamba_cuda()
    except Exception as exc:
        print(f"⚠️ Mamba CUDA 自检失败：{exc}")

    CONFIG = {
        "data_dir": "uavdataset/",

        # 如果你的论文最终设置为 H=25，这里必须改成 25 并重新训练。
        "horizon": 20,

        "batch_size": 1024,
        "epochs": 300,
        "learning_rate": 5e-4,
        "scales": [5, 10, 20, 30, 45],
        "d_model": 192,

        # 为避免每个 epoch 重启大量 worker，训练集默认 0。
        "train_num_workers": 0,
        "eval_num_workers": 4,
        "num_workers": 4,

        "split_seed": 42,
        "protocol_cache_dir": "Shared_UAV_Protocol_70_10_20",
        "force_rebuild_features": False,
        "force_rebuild_split": False,
        "force_rebuild_scalers": False,
        "deterministic": True,

        "train_stride": 5,
        "eval_stride": 2,
        # UAV-Flow trajectory states are aligned at 5 Hz; no 5-to-1
        # downsampling is performed in this pipeline.
        "sample_interval_s": 0.2,

        "distance_accuracy_ratio": 0.15,
        "distance_accuracy_floor_m": 0.25,
        "distance_accuracy_cap_m": 1.5,

        # 若你的论文仍采用 15°，可以改回 15.0；
        # 但需要与对比实验和指标定义保持一致。
        "direction_threshold_deg": 25.0,

        "maneuver_eval_interval": 5,
        "final_mc_samples": 100,
        "energy_score_pairs": 32,
        "save_prediction_npz": False,

        "physical_limit_quantile": 99.5,
        "physical_limit_margin": 1.2,

        "maneuver_save_dir": "Ablation_Paper_Ladder_5Hz_70_10_20_5Seeds",
        "ablation_scope": "paper_ladder",

        "resume": True,
        "inter_run_sleep_s": 2.0,
    }

    seed_everything(CONFIG["split_seed"], deterministic=CONFIG["deterministic"])

    print("\n" + "=" * 96)
    print("⏳ 构建共享 Train / Val / Test 数据协议")
    print("=" * 96)

    SHARED_DATA = prepare_shared_dataset(CONFIG)

    print("\n固定测试分组：")
    print(f"  stable/low     = {len(SHARED_DATA['test_groups']['low'])}")
    print(f"  ordinary/mid   = {len(SHARED_DATA['test_groups']['mid'])}")
    print(f"  sharp/high     = {len(SHARED_DATA['test_groups']['high'])}")

    # 调试阶段建议：
    # SEEDS = [2026]
    #
    # 正式论文实验：
    SEEDS = [2026, 2036, 2046, 2056, 2066]

    raw_df = run_paper_ladder_ablation(
        base_config=CONFIG,
        global_shared_data=SHARED_DATA,
        seeds=SEEDS,
    )

    SAVE_DIR = os.path.join(CONFIG["maneuver_save_dir"], CONFIG["ablation_scope"])
    generate_latex_tables_from_csv(SAVE_DIR)

    print("\n✅ Paper-ladder ablation study finished.")
