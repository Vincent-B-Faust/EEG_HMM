# EEG_HMM

基于 EEG/EMG 的无监督睡眠阶段聚类与动力学分析流水线，支持命令行执行和 Jupyter 交互式执行。

默认情况下，文件输入的所有结果会输出到 `sessions/<输入文件名>/`，并额外生成一个可直接打开的交互式会话文件 `session_view.html`。

## 核心模块

- `preprocess()`：EEG/EMG 滤波、工频抑制、标准化
- `window_signals()`：固定时窗切分，支持 overlap
- `extract_features()`：频谱、能量、峰峰值、过零率、峰值数量、频带功率等特征
- `cluster()`：KMeans、GMM、Hierarchical 三种聚类
- `align_cluster_labels()`：不同聚类结果标签对齐
- `hmm_analysis()`：基于 feature time series 拟合 Gaussian HMM（continuous emissions），输出 hidden states、转移矩阵、持续时间、log likelihood 和 BIC
- `manifold()`：PCA / Diffusion Map / UMAP（若本地已安装则启用，否则自动回退）
- `explain()`：基于全特征的浅层决策树和阈值规则提取
- `run_pipeline()`：全流程执行、导出中间数据和图表、生成 Markdown 报告
- `run_file_pipeline()` / `run_interactive()`：兼容 notebook 参数命名的文件入口

## 输入格式

支持以下格式：

- `.edf`：自动读取通道标签，可通过 `eeg_channel` / `emg_channel` 或 CLI 参数指定通道
- `.csv`：支持带表头和无表头文件，可自动识别；也可显式指定 EEG / EMG 列
- `.mat`：支持标准 MAT 文件；若是 v7.3 大文件，需要本地额外安装 `h5py`
- `.npz` / `.npy`：保留已有支持

## 与附件 notebook 一致的参数

兼容以下参数命名：

- `filename`
- `fs`
- `use_dask`
- `k_user`
- `window_size`
- `overlap`

其中：

- `window_strategy="notebook_auto"` 会复现附件 notebook 的窗口规则：`window_size = min(len(data)/20, fs*0.1)`，`overlap = 0.25 * window_size`
- `feature_mode="legacy"` 会使用附件 notebook 的 4 个特征：`peak_to_peak`、`zero_crossing_rate`、`peak_count`、`iemg2`
- `feature_scaling="minmax"` 可与附件 notebook 的 `MinMaxScaler` 保持一致

## GPU 加速与多平台兼容

当前实现支持可选 GPU 加速，并且兼容：

- macOS Apple Silicon：通过 `PyTorch + MPS`
- Windows 11 + NVIDIA RTX：通过 `PyTorch + CUDA`
- 无 GPU 或未安装 PyTorch：自动回退到 `NumPy + CPU`

加速策略：

- `backend="auto"` 时优先选择 `CUDA`，其次 `MPS`，否则回退 `CPU`
- GPU 主要加速：批量特征提取、KMeans、GMM、Gaussian HMM、PCA、Diffusion Map
- 仍保留 CPU：滤波、层次聚类、部分评估和绘图

命令行示例：

```bash
python3 -m sleep_hmm.cli \
  --filename path/to/data.edf \
  --device auto \
  --acceleration-backend auto
```

如果你要显式指定设备：

- MacBook Air M3：`--device mps`
- Windows RTX：`--device cuda`

说明：

- 本仓库不会把 `torch` 固定进基础依赖，因为 macOS 和 Windows 的安装包不同
- 在 Windows 机器上请安装带 CUDA 支持的 PyTorch；在 Apple Silicon 上安装原生 arm64 PyTorch 即可启用 MPS

## 分析流程概述

系统默认采用以下分析流程：

1. 读取 `EDF / CSV / MAT / NPZ / NPY` 输入，并统一整理为 EEG/EMG 序列与采样率。
2. 对 EEG/EMG 进行带通滤波、工频抑制和标准化。
3. 按固定窗口切分信号，支持秒级配置和样本级配置，也支持兼容附件 notebook 的自动分窗规则。
4. 从每个窗口提取频谱、频带功率、能量、峰峰值、过零率、峰值数量等特征。
5. 对标准化特征分别进行 `KMeans`、`GMM` 和 `Hierarchical` 聚类。
6. 使用标签对齐模块统一不同聚类方法的状态编号，便于横向比较。
7. 基于窗口级 feature time series 拟合 Gaussian HMM（continuous emissions），并比较 3-state、4-state、5-state 模型的 log likelihood 与 BIC。
8. 使用 `PCA / UMAP / Diffusion Map` 构建低维流形并可视化睡眠状态轨迹。
9. 使用基于全特征的浅层决策树提取 cluster 阈值规则和特征重要性。
10. 导出中间数据、统计结果、图表和自动报告。

如果需要论文写作版的流程描述，可直接参考：

- [docs/Method_Analysis_Workflow.md](/Users/vincent.bfaust/Library/CloudStorage/OneDrive-个人/CIBR/YamanakaLab/code/EEG_HMM/docs/Method_Analysis_Workflow.md)

## 命令行执行

运行内置合成数据：

```bash
python3 -m sleep_hmm.cli --demo --output sessions --fs 128 --k-user 3 --manifold pca
```

运行 EDF：

```bash
python3 -m sleep_hmm.cli \
  --filename path/to/data.edf \
  --eeg-channel EEG \
  --emg-channel EMG \
  --output sessions \
  --k-user 3 \
  --window-strategy notebook_auto \
  --acceleration-backend auto \
  --device auto
```

运行 CSV：

```bash
python3 -m sleep_hmm.cli \
  --filename path/to/data.csv \
  --fs 2000 \
  --csv-eeg-column eeg \
  --csv-emg-column emg \
  --output sessions \
  --window-size 200 \
  --overlap 50 \
  --window-strategy samples
```

运行 MAT：

```bash
python3 -m sleep_hmm.cli \
  --filename path/to/data.mat \
  --fs 2000 \
  --mat-variable signal \
  --output sessions
```

## Jupyter 交互式执行

可直接打开：

- `examples/interactive_sleep_hmm.ipynb`

也可以在 notebook 中手动写：

```python
from sleep_hmm import run_interactive

filename = "path/to/data.edf"
fs = 2000
use_dask = True
k_user = 3
window_size = None
overlap = None

result = run_interactive(
    filename=filename,
    fs=fs,
    use_dask=use_dask,
    k_user=k_user,
    window_size=window_size,
    overlap=overlap,
    output_dir="sessions",
    window_strategy="notebook_auto",
    acceleration_backend="auto",
    acceleration_device="auto",
    eeg_channel="EEG",
    emg_channel="EMG",
)
```

## 输出内容

运行后会在输出目录生成：

- `features_raw.csv` / `features_scaled.csv`
- 每种方法的 `*_labels.csv`、`*_metrics.json`
- `gaussian_hmm_3_state_*`、`gaussian_hmm_4_state_*`、`gaussian_hmm_5_state_*`
- `hmm_model_comparison.csv`
- 每种方法的 `*_feature_importance.csv`、`*_thresholds.csv`
- `manifold_embedding.csv`
- `session_view.html`
- `report.md`
- `figures/` 下的全部可视化图

## 交互式 Session 视图

对文件输入，系统会在 `sessions/<输入文件名>/session_view.html` 生成一个自包含的交互式会话页面。打开后可查看：

- 全长 EEG / EMG 时间轴
- 更易观察的方形布局 manifold cluster 散点图
- 选中 epoch 的 EEG 频谱，以及当前 cluster 的平均频谱叠加
- 与 cluster 颜色一致的原始信号背景标注
- 同步缩放、平移和 epoch 追踪

交互方式：

- 鼠标滚轮：围绕光标位置缩放
- 在 EEG / EMG 面板中拖动：同步平移时间范围
- 点击 manifold 散点图中的任一点：在原始信号中高亮，并同步更新右侧标签信息和频谱面板
- 点击 overview 面板：将当前视窗居中到对应时间位置

## 依赖

默认实现基于：

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

可选依赖：

- `h5py`：MAT v7.3 大文件
- `umap-learn`：UMAP 降维
