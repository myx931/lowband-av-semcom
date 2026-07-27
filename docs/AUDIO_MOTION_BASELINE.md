# 因果音频到嘴部运动基线

本阶段回答 RQ1：仅使用音频能否预测 18 维 LivePortrait 相对嘴部运动。输入为
逐视频帧对齐的 `[T,4,80]` log-Mel，目标为 `[T,18]` 嘴部表达变化。当前
`s1/s2/s3` 开发集包含 train 99、validation 99、test 100 条有效样本，只用于
可行性验证。

## 模型与数据隔离

模型将每帧 4×80 音频展平，经 128 维投影、两层单向 GRU（hidden 256）和线性层
输出 18 维运动。它不使用说话人身份、真实历史运动或未来音频。预测在反归一化后
强制将参考帧设为零。

音频统计仅从 train 拟合；运动统计直接读取
`$DATA_ROOT/grid/processed/multispeaker/motion/liveportrait/train_stats.json`。
checkpoint 同时绑定配置、manifest、音频统计和运动统计哈希，禁止用不匹配的运行
目录恢复。

## 环境与命令

```bash
conda env create -f environments/training.yaml
conda activate lowband-av-semcom-training
python -m pip install -e . --no-deps

export DATA_ROOT=/root/autodl-tmp/datasets
python scripts/train/train_audio_to_motion.py \
  --config configs/experiment/audio_to_motion_gru.yaml

python scripts/eval/evaluate_audio_to_motion.py \
  --config configs/experiment/audio_to_motion_gru.yaml \
  --run-dir outputs/audio_to_motion/<timestamp>
```

全量重建评价使用既有 LivePortrait 环境和权重：

```bash
conda activate liveportrait
export DATA_ROOT=/root/autodl-tmp/datasets
export MODEL_ROOT=/root/autodl-tmp/models
python scripts/eval/reconstruct_audio_to_motion.py \
  --config configs/experiment/audio_to_motion_gru.yaml \
  --run-dir outputs/audio_to_motion/<timestamp> --resume
```

## 公平比较与指标

- `zero_motion`：始终使用参考脸；
- `train_mean`：除参考帧外使用 train 运动均值；
- `oracle_persistence`：用真实上一帧预测当前帧，仅为不可部署的时间基线；
- `audio_gru`：纯音频因果预测，种子为 42、43、44。

运动层报告原始 18 维空间的 L1、RMSE 和速度 L1。重建层主要与真实 lip-only
oracle 比较，次要与原始裁剪比较，并报告嘴部 NME、嘴部 ROI MAE、PSNR、SSIM 和
检测覆盖率。NME 只由重建帧上的 MediaPipe 几何关键点计算，不用于命名 18 维隐式
表达误差。

validation 仅用于 checkpoint 选择，test 每个种子只运行一次。若三个种子的
validation 平均 L1 未优于 `train_mean`，不得进入预测残差里程碑；应先检查时间
对齐并扩充训练身份。无论结果是否支持假设，都保留完整运行记录。

## 已作废的首次三说话人运行

下列运行在工程上完整闭环，但数据审计发现其音频时间轴错误，因此只保留为故障
诊断记录，不得用于回答 RQ1、通过 E4 门槛或与后续实验比较。

运行时间为 2026-07-26，代码提交为 `6df350f`，实验指纹为
`a9b0e364d9fb227d6372e5b131d8a66b15fe9386e280916b2566b3fb347ab3da`。训练使用
一张 RTX 4080 SUPER、Python 3.11.15、PyTorch 2.3.1 和 CUDA 12.1。数据审计结果
为 train `s3` 99 条、validation `s1` 99 条、test `s2` 100 条，形状、路径和
身份隔离错误均为 0。

最佳 validation epoch 分别为 seed 42 的第 6 轮、seed 43 的第 11 轮和 seed 44
的第 8 轮。原始 18 维运动空间结果如下：

| split | 方法 | L1 | RMSE | 速度 L1 |
| --- | --- | ---: | ---: | ---: |
| validation | audio GRU，三种子均值 ± 标准差 | 0.001775 ± 0.000058 | 0.002875 ± 0.000121 | 0.000647 ± 0.000004 |
| validation | train mean | 0.001780 | 0.002801 | 0.000629 |
| validation | zero motion | 0.001878 | 0.003101 | 0.000622 |
| validation | oracle persistence | 0.000613 | 0.001108 | 0.000859 |
| test | audio GRU，三种子均值 ± 标准差 | 0.002529 ± 0.000106 | 0.004587 ± 0.000152 | 0.000645 ± 0.000003 |
| test | train mean | 0.002655 | 0.004765 | 0.000630 |
| test | zero motion | 0.002149 | 0.004171 | 0.000623 |
| test | oracle persistence | 0.000615 | 0.001209 | 0.000843 |

在当时的错误输入上，validation 平均 L1 比 `train_mean` 低约 0.30%，test 平均
L1 比 `train_mean` 低约 4.77%。这些差值现已作废，不再用于 E4 门槛。test 上
GRU 比
`zero_motion` 高约 17.7%，速度误差也未优于静态基线，因此不能据此声称音频输入
产生了稳健的跨身份收益。`oracle_persistence` 使用上一帧真实运动，不可部署，
但它显示当前预测器与时间连续性上界仍有明显差距。

冻结 LivePortrait 在全部 199 条 validation/test 样本上完成评价，共产生 1,194
条“样本 × 方法/种子”记录，失败数为 0，重建帧 MediaPipe 检测覆盖率为 100%。
相对于真实 lip-only oracle，test 三种子 GRU 的嘴部 NME 均值为 `0.03079`、嘴部
ROI MAE 为 `8.2838`、PSNR 为 `36.5595 dB`、SSIM 为 `0.97156`。对应的
`train_mean` 为 `0.03306`、`8.7215`、`36.2672 dB`、`0.96997`，而
`zero_motion` 为 `0.02402`、`6.8816`、`37.5804 dB`、`0.97717`。重建结论与
运动层一致：GRU 略优于训练均值，但没有优于静止参考脸。

同一运行目录使用 `--resume` 复跑后，样本结果和汇总指标的修改时间及大小均保持
不变。checkpoint、预测、媒体和逐样本实验输出只保存在被 Git 忽略的本地
`outputs/audio_to_motion/` 下。

## 时间轴故障与修复

`audio_25k.zip` 的 WAV 是变长语音片段。本地 298 条样本的音频时长为
1.12–2.50 秒，均值 1.749 秒，而视频固定为 75/25 = 3 秒。旧特征提取器把每条
WAV 的完整 Mel 序列线性插值为 300 步，相当于逐样本改变语速，破坏了音频与嘴部
运动的对应时间。

修复后从官方 MPG 的内嵌音轨提取 16 kHz mono PCM WAV，并写入独立
`grid_multispeaker_synced.jsonl`。298 条同步 WAV 的音频/视频时长比统一为
`0.992667`；约 22 ms 的封装尾差只补零。log-Mel 以 10 ms 绝对时间戳计算，并
取每个视频帧对应的四步，不再对时间轴插值。时长比不在 `0.95..1.05` 的样本会
失败。真实 CPU 验收为 298 条、失败 0，恢复性复跑没有改写 WAV 或特征。

## 修正后的三说话人验收

修正运行时间为 2026-07-26，代码提交为 `6543533`，实验指纹为
`a60e19ada842b90c8305c6795d0b112b4983db6604c77d2ede3ed0c41d293561`。数据划分和
视觉运动目标与首次运行相同，只替换音频源和特征时间轴。三个种子的最佳 epoch
分别为 3、5、1。数据、形状、路径和身份隔离错误为 0，训练无 NaN。

原始 18 维运动空间结果如下：

| split | 方法 | L1 | RMSE | 速度 L1 |
| --- | --- | ---: | ---: | ---: |
| validation | audio GRU，三种子均值 ± 标准差 | 0.001674 ± 0.000042 | 0.002693 ± 0.000117 | 0.000639 ± 0.000015 |
| validation | train mean | 0.001780 | 0.002801 | 0.000629 |
| validation | zero motion | 0.001878 | 0.003101 | 0.000622 |
| validation | oracle persistence | 0.000613 | 0.001108 | 0.000859 |
| test | audio GRU，三种子均值 ± 标准差 | 0.002492 ± 0.000090 | 0.004411 ± 0.000142 | 0.000640 ± 0.000011 |
| test | train mean | 0.002655 | 0.004765 | 0.000630 |
| test | zero motion | 0.002149 | 0.004171 | 0.000623 |
| test | oracle persistence | 0.000615 | 0.001209 | 0.000843 |

validation 和 test L1 分别比 `train_mean` 低 5.95% 和 6.15%，因此按预先定义的
规则通过 E4 技术门槛。validation 还比 `zero_motion` 低 10.87%，但 test 比
`zero_motion` 高 15.95%；三种子平均速度误差也没有超过静态基线。与作废运行
相比，修正后的 validation/test L1 分别改善 5.67% 和 1.45%，说明同步修复有实际
影响，但没有消除跨身份退化。

冻结 LivePortrait 在全部 199 条 validation/test 样本上再次完成评价，共 1,194
条记录、0 失败，重建帧 MediaPipe 覆盖率为 100%。相对于 lip-only oracle：

| split | 方法 | 嘴部 NME | 嘴部 ROI MAE | PSNR | SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | audio GRU，三种子均值 | 0.01769 | 5.9700 | 39.6562 dB | 0.98458 |
| validation | train mean | 0.01850 | 6.8777 | 38.9921 dB | 0.98276 |
| validation | zero motion | 0.01909 | 6.5149 | 38.5372 dB | 0.98157 |
| test | audio GRU，三种子均值 | 0.03128 | 8.3545 | 36.6178 dB | 0.97160 |
| test | train mean | 0.03306 | 8.7214 | 36.2672 dB | 0.96997 |
| test | zero motion | 0.02402 | 6.8813 | 37.5804 dB | 0.97717 |

validation 上 GRU 的 NME 比 train mean 和 zero motion 分别低 4.38% 和 7.35%；
test 上比 train mean 低 5.37%，但比 zero motion 高 30.21%。这与运动层结论一致：
音频提供了可测信号，但当前单训练说话人的模型没有形成稳定的跨身份优势。训练、
预测和重建均通过恢复性复跑，匹配产物的修改时间与大小保持不变。

## 结论与下一步约束

E3 的修正闭环已经完成，预设 E4 技术门槛通过，但 RQ1 只获得有限支持。当前每个
split 仍只有一个说话人，test 未超过静止参考脸。下一步应先扩充训练说话人并保持
validation/test 身份隔离，再决定是否开展预测残差价值实验；否则 E4 可能主要测量
单说话人过拟合。

## 十说话人扩充验收

为降低单训练说话人过拟合，随后将 GRID 扩充到 `s1`–`s10`，每位说话人固定选取
100 条有效样本。固定种子 42 的身份隔离划分为：

- train：`s1/s2/s4/s5/s6/s8/s9/s10`，800 条；
- validation：`s3`，100 条；
- test：`s7`，100 条。

运行时间为 2026-07-27，实验指纹为
`b805db9937ec257f02ba878ce5b3a84bceaa17994998a5022b168609ff313c99`。
三个种子的最佳 epoch 分别为 7、10、8。数据审计确认 1,000 条输入的形状、路径
和身份隔离错误均为 0。

原始 18 维运动空间结果如下：

| split | 方法 | L1 | RMSE | 速度 L1 |
| --- | --- | ---: | ---: | ---: |
| validation | audio GRU，三种子均值 ± 标准差 | 0.001884 ± 0.000020 | 0.003177 ± 0.000042 | 0.000709 ± 0.000007 |
| validation | train mean | 0.002406 | 0.004332 | 0.000749 |
| validation | zero motion | 0.002649 | 0.004617 | 0.000749 |
| validation | oracle persistence | 0.000739 | 0.001487 | 0.000907 |
| test | audio GRU，三种子均值 ± 标准差 | 0.001978 ± 0.000065 | 0.003155 ± 0.000165 | 0.000673 ± 0.000004 |
| test | train mean | 0.002146 | 0.003832 | 0.000711 |
| test | zero motion | 0.002264 | 0.003907 | 0.000711 |
| test | oracle persistence | 0.000701 | 0.001383 | 0.000949 |

GRU 的 validation 平均 L1 比 `train_mean` 低 21.72%，test 平均 L1 低 7.82%；
相对 `zero_motion` 分别低 28.90% 和 12.61%。因此 E4 门槛不再依赖单训练说话人
结果，RQ1 在新的 test 身份上获得初步支持。不可部署的 `oracle_persistence` 仍
明显更优，说明时序建模和音频歧义仍有较大改进空间。

冻结 LivePortrait 对全部 200 条 validation/test 样本完成 1,200 条重建评价，
失败数为 0，所有重建帧的 MediaPipe 检测覆盖率为 100%。按 validation L1 选择的
最佳种子为 43；相对于 lip-only oracle：

| split | 方法 | 嘴部 NME | 嘴部 ROI MAE | PSNR | SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | audio GRU，seed 43 | 0.02262 | 6.8861 | 38.5803 dB | 0.98197 |
| validation | train mean | 0.02970 | 8.9698 | 35.9894 dB | 0.97360 |
| validation | zero motion | 0.03191 | 9.8772 | 35.3448 dB | 0.97070 |
| test | audio GRU，seed 43 | 0.02208 | 5.2993 | 40.6027 dB | 0.98745 |
| test | train mean | 0.02568 | 5.9690 | 39.3595 dB | 0.98531 |
| test | zero motion | 0.02557 | 6.1921 | 38.8873 dB | 0.98377 |

seed 43 的 test 嘴部 NME 比 `train_mean` 和 `zero_motion` 分别低约 14.01% 和
13.65%。200 个逐样本原子产物、汇总 JSON/CSV、代表性媒体和完成标记均已生成；
恢复性复跑直接复用匹配指纹的完成结果。至此十说话人 E3 工程和真实 GPU 验收
完成，可以进入无信道预测残差价值实验。
