# UAV trajectory prediction · v4

本目录提供 v4 长时域概率轨迹预测的训练、验证集校准、检查点评估及多种子汇总代码。正式 300 轮实验使用 31 维输入和冻结的三状态 SSL/HMM 教师，输出 20 步三维轨迹及其预测不确定性。该目录可放在 GitHub 仓库根目录下，内部文件保持相对位置。

## 运行环境

建议使用 Linux、Python 3.11 和支持 CUDA 的 NVIDIA GPU。优先使用原实验环境中的 PyTorch、CUDA、Mamba2、`causal-conv1d` 和序列化库版本。新环境应先安装与 CUDA 匹配的 PyTorch，以及能导入 `mamba_ssm.modules.mamba2` 的 Mamba2 运行环境，再安装数据处理与统计依赖。

```bash
python -m pip install -r requirements.txt
python -c "import torch; from mamba_ssm.modules.mamba2 import Mamba2; print(torch.__version__, torch.cuda.is_available())"
```

如果沿用原服务器的 Mamba 源码树，用 `--mamba-dir` 指向该目录，并确保对应 CUDA 扩展已安装。基线图表和卷积/线性层 FLOPs 估算可另装 `matplotlib` 与 `thop`。对自定义或融合算子应核查 FLOPs 统计覆盖范围。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `train_hmm_softstate_uav_formal_longhorizon_v4.py` | 正式 300 轮实验入口，含软状态加权和长时域预测目标 |
| `train_hmm_softstate_uav_formal_piccal_v4.py` | 验证集条件校准及采样区间修正 |
| `train_hmm_softstate_uav_formal_probcal_v4.py` | 数据加载适配及跨种子共享教师 |
| `train_hmm_softstate_uav_probcal_v4.py` | 训练、检查点选择和最终评估 |
| `uav_formal_base_v4.py`、`uav_formal_base_v2.py` | 31D 正式入口及其底层实现。保留 v2 文件名供 v4 导入 |
| `latent_state_emamse.py` | SSL 表征和三状态 HMM 教师 |
| `summarize_hmm_multiseed_results_v2.py` | 总体、分机动、分时域的五种子统计和 LaTeX 表 |
| `uav_formal_base_v4_12d.py` | 可选的 12D 位置特征前端 |
| `baselines/`、`complexity_report.py` | 独立 12D 基线实验及其复杂度 CSV 汇总 |

## 复现 31D 正式实验

从本目录运行。将原实验 Parquet 分片放在 `uavdataset/`，或通过 `--data-dir` 指定实际目录。分片需包含轨迹 `id` 与 `logs` 或 `log` 列，日志应带有 31D 特征提取所需的 `raw_logs` 和 `preprocessed_logs`。

逐项复核已报告的实验时，使用同一数据快照、固定划分、训练归一化缓存及冻结教师目录。以下命令使用原实验的目录命名。没有缓存时，程序构建特征、划分和教师，可用于从头训练。新生成的划分和教师属于新一次复现，应核对样本数并记录配置。仅加载来源可信的检查点和缓存。

```bash
python train_hmm_softstate_uav_formal_longhorizon_v4.py \
  --base-script ./uav_formal_base_v4.py \
  --data-dir ./uavdataset \
  --cache-dir ./Shared_UAV_Protocol_HMM_v2 \
  --shared-teacher-dir ./Frozen_ThreeState_Teacher_v1 \
  --output-dir ./outputs/HMM_LongHorizon_v4_Constrained_300E_5Seeds \
  --mamba-dir ./mamba-main \
  --epochs 300 --batch-size 512 --eval-batch-size 256 --workers 4 \
  --eval-interval 5 --state-weight 0.20 --state-temperature 1.0 \
  --state-confidence-floor 0.25 \
  --checkpoint-crps-tolerance 0.01 --checkpoint-brier-tolerance 0.02 \
  --calibration-samples 100 --mc-samples 100 \
  --seeds 2026 2036 2046 2056 2066 --save-predictions
```

训练与验证采用 5 Hz 实验协议，历史窗口 45 个采样点，预测 20 步（4 秒），划分种子为 42。训练器每 5 轮检查验证集候选，先筛选 CRPS 容差内的候选，再按 StateBrier 容差筛选，最后以校准偏差选择检查点。校准参数在验证集拟合，测试集用于最终评估。

复评已有权重时，在上述命令中加入 `--evaluate-only --checkpoint-root <原运行目录>`，并把 `--output-dir` 改为新的复评目录。原运行目录按 `seed_<种子>/best_checkpoint.pt` 存放权重。复评会重新执行验证集校准及测试评估。

## 汇总五个种子

```bash
python summarize_hmm_multiseed_results_v2.py \
  --result-dir ./outputs/HMM_LongHorizon_v4_Constrained_300E_5Seeds \
  --model-name Ours-v4 --expected-seeds 5 \
  --out-dir ./outputs/v4_300e_summary
```

重点查看 `global_mean_std_long.csv`、`state_mean_std_long.csv`、`stepwise_mean_std_long.csv` 和两份 `paper_table_*.tex`。汇总程序读取各 `seed_*` 子目录，适用于分批训练。结果使用种子间均值和样本标准差；点预测 ADE/FDE 与 best-of-`K` 指标分开计算。

## 12D 前端及基线实验

12D 前端仅使用位置及其派生特征，使用右对齐 Savitzky–Golay 滤波。若用同一 v4 入口训练 12D 版本，将 `--base-script` 改为 `./uav_formal_base_v4_12d.py`，并为缓存、冻结教师和输出分别指定全新的目录。31D 正式结果沿用原预处理，其中位置的 Savitzky–Golay 滤波使用居中窗口。两种前端的结果、缓存、教师和模型权重应分别保存。

`baselines/uner_compare1.6_5hz_formal_125_causal12d_trajectronpp_fp32_5seeds.py` 是独立的 12D 基线实验入口。运行前按实验环境核对脚本内的 `CONFIG`，尤其是数据、缓存、划分文件、轮次和输出目录。从本目录运行该脚本。其 `Ours_Full` 行是基线脚本自带的模型实现。正式论文比较应核对输入、测试轨迹 ID、滑窗、评价指标实现和采样口径，分机动比较还应使用同一分组标签。

可从已有基线结果导出复杂度表：

```bash
python complexity_report.py \
  --input ./Compare_5Hz_125_Causal12D_SG7P2_Seed2026_StepwiseProb/comparison_seed_results.csv \
  --out ./outputs/baseline_complexity
```

该工具汇总已有计时和参数量。若要在复杂度表加入正式 v4，请用相同硬件和批大小测量当前检查点，并按输出路径分别报告点预测和概率采样的延迟。

## 发布内容

本目录包含源代码和复现说明。五种子权重与校准文件单独提供在 `UavTrajector_v4_checkpoints_300E.zip` 中。解压后，复评命令的 `--checkpoint-root` 指向 `HMM_LongHorizon_v4_Constrained_300E_5Seeds/`。复评仍需原数据及相应缓存或重建所需的实验输入。数据、HMM/SSL 教师制品、缓存、训练日志及第三方 Mamba 源码需另行准备。
