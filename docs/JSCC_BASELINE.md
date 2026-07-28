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
但中低 SNR 会引入抖动。当前结果只完成运动空间 E5 验证，尚未完成代表条件的
LivePortrait 视频重建，不能据此宣称视频感知质量提升。
