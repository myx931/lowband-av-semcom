# LivePortrait 运动提取与重建基线

本阶段只验证“真实嘴部运动表示经过扰动后，冻结重建器的输出如何变化”。不训练
音频预测器，不计算预测残差，不加入 JSCC、AWGN 或可学习选择器。

## 表示定义

LivePortrait 固定版本输出 21×3 的隐式表情变形。本项目采用官方实现中嘴部相关的
索引 `[6, 12, 14, 17, 19, 20]`，并减去参考帧表情，得到 `[T, 6, 3]`，展平后
为 18 维。完整表情、旋转、平移、尺度和 canonical keypoints 同时保留，以便
冻结后端进行 oracle 重建。

当前 20 条 `s1` 数据只用于 pilot。`pilot_stats.json` 的均值和标准差不得直接用于
正式多说话人实验；正式统计必须只从 train speakers 估计。

## 源码、权重与环境

```bash
git submodule update --init --recursive

conda env create -f environments/liveportrait.yaml
conda activate liveportrait
python -m pip install -e . --no-deps

export DATA_ROOT=/root/autodl-tmp/datasets
export MODEL_ROOT=/root/autodl-tmp/models

python scripts/motion/download_liveportrait_instructions.py
huggingface-cli download KlingTeam/LivePortrait \
  --local-dir "$MODEL_ROOT/liveportrait" \
  --exclude "*.git*" "README.md" "docs"
```

下载权重前阅读官方模型卡。源码、模型、输出视频和实验结果不得提交到 Git。
配置默认使用 `cuda:0` 和半精度；CPU 仅用于自动化 fake 测试。
该独立环境使用 `requirements/liveportrait.txt` 中的最小固定运行时，不安装上游
Gradio、训练、ONNX 或人脸二次检测依赖，因为本适配器直接读取已有 256×256 裁剪。

## 命令

```bash
python scripts/motion/extract_motion.py \
  --config configs/motion/liveportrait.yaml \
  --speakers s1 --max-samples 20

python scripts/motion/validate_motion.py \
  --config configs/motion/liveportrait.yaml \
  --speakers s1 --max-samples 20

python scripts/motion/reconstruct_motion.py \
  --config configs/motion/liveportrait.yaml \
  --speakers s1 --max-samples 20

python scripts/motion/run_motion_sensitivity.py \
  --config configs/motion/liveportrait.yaml \
  --speakers s1 --max-samples 20
```

运动产物位于 `$DATA_ROOT/grid/processed/motion/liveportrait/`。实验结果位于
`outputs/motion_sensitivity/{timestamp}/`，其中包含解析配置、环境、Git commit、
逐样本 JSONL、汇总 JSON/CSV、曲线和代表性样例。重复提取时配置指纹一致则跳过；
配置变化必须显式使用 `--overwrite`。

## 固定敏感性设置

- Gaussian 标准差：`0.05, 0.1, 0.2, 0.5`；
- 均匀量化：`8, 6, 4, 3, 2 bit`，归一化范围截断至 `[-3, 3]`；
- 随机丢弃保留率：`0.75, 0.5, 0.25, 0.1`，种子 `42, 43, 44`；
- 每帧幅度稀疏化：相同保留率，仅作为 oracle 敏感性上界。

报告运动 L1、RMSE、速度误差、全脸 MAE/PSNR/SSIM、嘴部 ROI MAE、嘴部 NME
和关键点检测覆盖率。pilot 不设置必须改善的质量阈值，结果必须如实保存和报告。
汇总中的通用字段 `value` 必须结合 `parameter_name` 解读：Gaussian 使用
`noise_standard_deviation`，量化使用 `quantization_bits`，随机丢弃和幅度稀疏
使用 `keep_ratio`。曲线横轴显示这些实际参数，不把方向相反的参数统称为扰动强度。

## 真实 GPU 验收

2026-07-26 在 Python 3.10.20、PyTorch 2.3.1、CUDA 12.1 和 RTX 4080 SUPER
上完成 `s1` 的 20 条 pilot。运行引用 Git commit `2760569` 和 LivePortrait
commit `9b294b3d`，共得到 28 个条件、560 条逐样本记录、0 个失败；20 条运动
artifact 复验均有效，重复提取未改写已有文件。

代表性结果如下，均为 20 条样本的均值：

| 条件 | PSNR (dB) | SSIM | 嘴部 NME | 检测覆盖率 |
|---|---:|---:|---:|---:|
| 仅嘴部真实运动 | 24.003 | 0.7539 | 0.0489 | 1.000 |
| 静止参考脸 | 23.911 | 0.7489 | 0.0530 | 1.000 |
| 完整运动 oracle | 23.864 | 0.7368 | 0.0542 | 1.000 |
| 2-bit 量化 | 23.947 | 0.7506 | 0.0524 | 1.000 |
| 随机保留 10%，seed 42 | 23.899 | 0.7486 | 0.0533 | 1.000 |
| 幅度保留 10% | 23.919 | 0.7493 | 0.0533 | 1.000 |

本结果只说明冻结后端在当前单说话人 pilot 上对扰动的响应，不证明某种条件优于
oracle，也不外推到正式多说话人实验。完整本地输出位于忽略目录
`outputs/motion_sensitivity/20260726T060450.218238Z/`，包含 84 个代表性视频、
84 张对照图和 4 张曲线，不提交 Git。
