# Sionna AWGN 与残差 JSCC 基线

## 研究目的

E5 检验 18 维音频预测残差在受限复数信道使用次数和 AWGN 下能否改善纯音频预测。
本阶段只建立可解释的最小通信基线，不实现量化、熵编码、OFDM、信道编码、衰落或
可学习残差选择器。

正式信道采用 NVIDIA [Sionna PHY](https://nvlabs.github.io/sionna/phy/) 2.0.1 的
PyTorch `AWGN` 算子。Sionna 源码采用 Apache-2.0 许可证，见
[官方仓库](https://github.com/NVlabs/sionna)。仓库中的 `NativeComplexAWGN`
仅用于无需 Sionna 的快速测试，不得生成正式论文结果。

## 冻结协议

- 输入：十说话人 E3 中由 validation L1 选出的 seed 43 GRU 残差。
- 数据隔离：train/validation/test 为 800/100/100，分别含 8/1/1 个说话人。
- 残差：`真实运动 - 音频预测运动`，再除以 train-only 运动标准差；不减均值。
- 参考帧：第 0 帧残差固定为零，不占信道使用；无效帧同样不传输。
- 编码器：`Linear(18,64) + ReLU + Linear(64,2C)`。
- 复数打包：每两个实数构成一个复数符号，因此每个有效帧使用 `C` 个复数信道
  符号和 `2C` 个实自由度。
- 功率：每条样本在所有有效帧和全部复数符号上的平均 `|x|²` 归一到 1。
- 信道：复数 AWGN；`N0` 表示每个复数维度的噪声功率，每个实部/虚部方差为
  `N0/2`，单位功率时 `N0 = 10^(-SNR/10)`。
- 解码器：`Linear(2C,64) + ReLU + Linear(64,18)`。
- 损失：有效非参考帧上的归一化残差 MSE。

正式比较 `C = 1/2/3/4`，训练 SNR 从 `[0,10] dB` 连续采样；checkpoint 仅以
validation 的 `2.5/7.5 dB` 平均 MSE 选择。test 只在模型冻结后运行
`-5/0/5/10 dB`，每个条件使用噪声种子 `42/43/44`。

`C/18` 只称为“复数信道使用次数与语义维度之比”。它没有考虑量化精度、协议头、
信道编码或时长，不能称为压缩率或真实 bitrate。

## 独立环境

Sionna 2.0.1 需要较新的 PyTorch/NumPy，因此不得安装到 MediaPipe 或
LivePortrait 环境。创建 Python 3.11 环境后安装固定依赖：

```bash
conda create -p /path/to/lowband-av-semcom-sionna python=3.11 pip
/path/to/lowband-av-semcom-sionna/bin/python \
  -m pip install --no-cache-dir -r requirements/sionna.txt
```

建议设置：

```bash
export SIONNA_PYTHON=/path/to/lowband-av-semcom-sionna/bin/python
export DATA_ROOT=/path/to/public-datasets
```

## 命令

先训练 12 个条件（4 个 `C` × 3 个模型种子）：

```bash
PYTHONPATH=src "$SIONNA_PYTHON" scripts/train/train_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e3-run-dir outputs/audio_to_motion_ten_speaker/<e3-timestamp>
```

记录命令输出中的运行目录；完成训练后独立评价 test：

```bash
PYTHONPATH=src "$SIONNA_PYTHON" scripts/eval/evaluate_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e3-run-dir outputs/audio_to_motion_ten_speaker/<e3-timestamp> \
  --run-dir outputs/residual_jscc/<e5-timestamp>
```

中断后两条命令均可加 `--resume`。配置、manifest、E3 fingerprint、统计文件或
checkpoint 哈希变化时，恢复会明确拒绝，避免混合实验。

最后从不可变的逐样本 test JSONL 生成跨模型种子的均值、标准差和 SNR 曲线：

```bash
PYTHONPATH=src "$SIONNA_PYTHON" scripts/eval/report_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --run-dir outputs/residual_jscc/<e5-timestamp>
```

## 产物与判读

运行目录包含解析配置、环境、输入来源哈希、train/validation 派生残差缓存、
逐模型 checkpoint 和 history，以及 test 逐样本 JSONL 和汇总 JSON/CSV。
这些产物全部被 Git 忽略。

评价必须同时报告：

- `prediction_only`：不发送残差；
- `full_residual_oracle`：完整真实残差上界；
- `noiseless_autoencoder`：分离表示容量损失与信道噪声损失；
- `jscc_awgn`：各 `C`、SNR、模型种子和噪声种子的正式条件。

主要指标为归一化残差 MSE，以及原始运动空间的 L1、RMSE 和速度 L1。本节数值
只来自下述已保存正式运行。

## 十说话人真实结果

正式运行目录为 `outputs/residual_jscc/20260728T060039.837712Z`。输入包含
800 条 train、100 条 validation 和 100 条 test 样本；12 个模型均完成且无
NaN。test 共保存 15,800 条有限指标：14,400 条带噪 JSCC、1,200 条无噪
autoencoder，以及各 100 条纯预测和完整残差上界。

三种子 validation 归一化残差 MSE 均值随 `C=1/2/3/4` 分别为
`0.5345/0.4348/0.3784/0.3156`。test 纯预测运动 L1 为 `0.001893`，带噪
JSCC 的三种子、三噪声种子聚合结果如下：

| 复数信道使用数 C | -5 dB | 0 dB | 5 dB | 10 dB | 无噪 autoencoder |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.001978 | 0.001708 | 0.001573 | 0.001514 | 0.001482 |
| 2 | 0.002093 | 0.001616 | 0.001410 | 0.001327 | 0.001283 |
| 3 | 0.002156 | 0.001548 | 0.001276 | 0.001171 | 0.001118 |
| 4 | 0.002269 | 0.001513 | 0.001159 | 0.001014 | 0.000939 |

在 0/5/10 dB，所有 `C` 的运动 L1 均优于不发送残差；`C=4, 10 dB` 改善约
46.4%。在训练范围外的 -5 dB，所有条件反而更差，而且 `C` 越大退化越明显。
因此下一阶段需要显式比较“低 SNR 不发送残差”的门控基线，不能假设传输总有益。

时间质量结论更谨慎：纯预测速度 L1 为 `0.000678`，`C=4` 在 5 dB 时升至
`0.001006`，在 10 dB 才回落到 `0.000658`。也就是说逐帧 MSE 能降低位置误差，
但中低 SNR 会引入抖动。因此位置误差和时间误差必须同时报告，不能根据单一运动
指标宣称视频感知质量提升。

## 冻结视频评价协议

视频评价不再根据 test 选择模型。每个信道预算只使用 validation 归一化残差
MSE 最低的模型种子：

| C | 选择种子 |
|---:|---:|
| 1 | 43 |
| 2 | 44 |
| 3 | 43 |
| 4 | 44 |

test 的信道噪声种子固定为 42。每条 s7 test 样本共重建 22 个条件：纯预测、
完整残差 oracle、每个 `C` 的无噪 autoencoder，以及每个 `C` 在
`-5/0/5/10 dB` 下的带噪结果。视频指标只代表这一条预先固定的噪声实现；运动
空间报告仍以三个噪声种子聚合为主。

由于 Sionna 与 LivePortrait 使用不同依赖环境，先在 Sionna 环境导出运动候选：

```bash
PYTHONPATH=src "$SIONNA_PYTHON" \
  scripts/eval/export_residual_jscc_reconstruction.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --run-dir outputs/residual_jscc/<e5-timestamp>
```

导出器逐样本核对候选运动指标与已冻结的 `test_metrics.jsonl`；任何不一致都会
失败。随后切换到 LivePortrait Python 3.10 环境：

```bash
PYTHONPATH=src python scripts/eval/reconstruct_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --run-dir outputs/residual_jscc/<e5-timestamp> \
  --resume --reconstruction-batch-size 48 --metric-workers 8
```

全量指标覆盖 100 条 test 样本和全部 22 个条件。只为排序首、中、末三条样本保存
媒体；媒体包含纯预测以及 `C=4` 的无噪和四个 SNR 条件，避免保存 2,200 个视频。

## 冻结视频评价结果

正式视频评价仍使用上述运行目录，输出位于其 `video_reconstruction/` 子目录。
100 条 `s7` test 样本均完成，每条包含 22 个条件，共 2,200 条指标，失败数为
0；所有条件的 MediaPipe 嘴部关键点检测覆盖率均为 100%。重复使用 `--resume`
运行时，总 JSONL 的 SHA-256 和修改时间均保持不变。

下表报告相对 lip-only oracle 的嘴部 ROI MAE 和嘴部关键点 NME。带噪结果只对应
预先固定的噪声种子 42；它们不是三个噪声种子的均值。

| C | 指标 | 纯预测 | -5 dB | 0 dB | 5 dB | 10 dB | 无噪 autoencoder |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | ROI MAE | 5.299 | 5.588 | 5.241 | 5.097 | 5.039 | 5.008 |
| 1 | NME | 0.02195 | 0.02296 | 0.02147 | 0.02077 | 0.02048 | 0.02031 |
| 2 | ROI MAE | 5.299 | 6.096 | 5.163 | 4.786 | 4.631 | 4.555 |
| 2 | NME | 0.02195 | 0.02515 | 0.02074 | 0.01904 | 0.01837 | 0.01803 |
| 3 | ROI MAE | 5.299 | 6.651 | 5.060 | 4.287 | 3.990 | 3.816 |
| 3 | NME | 0.02195 | 0.02614 | 0.01928 | 0.01624 | 0.01512 | 0.01450 |
| 4 | ROI MAE | 5.299 | 7.136 | 5.165 | 4.239 | 3.822 | 3.619 |
| 4 | NME | 0.02195 | 0.02905 | 0.01975 | 0.01601 | 0.01436 | 0.01357 |

在 0/5/10 dB，全部预算的两项嘴部指标都优于纯预测。`C=4,10 dB` 将 ROI MAE
和 NME 分别改善 27.9% 和 34.6%，已经证明运动空间改善可以转化为冻结
LivePortrait 的嘴部重建改善。无噪 `C=4` 的对应改善为 31.7% 和 38.2%，说明
10 dB 已接近但尚未达到该表示容量的无噪上限。

结论在 -5 dB 反转：`C=4` 的 ROI MAE 和 NME 分别恶化 34.7% 和 32.4%，且预算
越大退化越明显。这与运动空间结果一致，说明固定发送策略在分布外低 SNR 下不够
稳健。下一阶段应先加入基于 validation 冻结阈值的“发送 JSCC / 不发送残差”
门控基线，再研究信道感知的重要性选择器；不得根据 test 后验挑选阈值。

这些视频指标仍只覆盖一个 test 说话人、每个 `C` 各一个由 validation 选出的
模型种子，以及一个固定噪声实现。它们支持 E5 的可行性结论，不等同于真实
bitrate、复杂衰落信道或跨数据集泛化结论。
