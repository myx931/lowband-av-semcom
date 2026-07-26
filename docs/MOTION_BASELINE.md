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
