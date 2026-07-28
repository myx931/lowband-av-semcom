# E7 完整运动与预测残差的匹配 JSCC 对照

## 研究问题

E4–E6 已证明预测残差存在价值，但此前没有直接回答 RQ2 的核心问题：在相同
复信道符号预算下，编码音频预测残差是否比编码完整 18 维嘴部运动更有效。

本实验只补齐缺失的完整运动对照。十说话人 E3 音频预测器、E5 residual JSCC、
E6 门控和 LivePortrait 全部冻结，不为形式上的重复重新训练。

## 公平控制

- 数据身份与 E5 完全相同：train/validation/test 为 800/100/100，对应
  8/1/1 个互斥说话人；完整运动直接读取 E5 不可变样本缓存中的同一 target 和
  audio prediction。
- 完整运动按既有 train-only 运动均值和标准差变换为 18 维标准化输入；音频预测
  不进入完整运动编码器，只作为共同的零传输基线。
- 两条链路使用相同的两层 18→64→`2C` 编码器和 `2C`→64→18 解码器、
  逐样本平均复功率 1、Sionna PHY 2.0.1 复 AWGN、训练 SNR 0–10 dB。
- `C=1/2/3/4`、模型种子 `42/43/44`、validation SNR `2.5/7.5 dB`、
  test SNR `-5/0/5/10 dB` 和噪声种子 `42/43/44` 与 E5 一致。
- 每个 `C` 只按 validation normalized MSE 选择一个种子。test 在模型冻结后
  评价一次，不用来选择模型、门控或超参数。
- 主要比较 always-send 表示效率；E6 residual 安全门控仍是独立的部署稳定性
  模块，不为完整运动重新拟合一套 test 导向门控。
- 两者均使用每有效帧 `C` 个复符号，即每三秒片段 74/148/222/296 个复符号。
  本实验没有数字 bitstream，不能报告 bit/s。

## 命令

Sionna Python 3.11 环境：

```bash
PYTHONPATH=src "$SIONNA_PYTHON" scripts/train/train_full_motion_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --source-e5-run-dir outputs/residual_jscc/20260728T060039.837712Z

PYTHONPATH=src "$SIONNA_PYTHON" scripts/eval/evaluate_full_motion_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --source-e5-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --run-dir outputs/full_motion_jscc/<timestamp>

PYTHONPATH=src "$SIONNA_PYTHON" \
  scripts/eval/export_full_motion_jscc_reconstruction.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --source-e5-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --run-dir outputs/full_motion_jscc/<timestamp>
```

LivePortrait Python 3.10 环境：

```bash
PYTHONPATH=src "$LIVEPORTRAIT_PYTHON" \
  scripts/eval/reconstruct_full_motion_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --run-dir outputs/full_motion_jscc/<timestamp> \
  --reconstruction-batch-size 48 --metric-workers 16
```

冻结比较：

```bash
PYTHONPATH=src python scripts/eval/compare_full_motion_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --residual-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --full-motion-run-dir outputs/full_motion_jscc/<timestamp>
```

所有阶段支持指纹保护的 `--resume`；已有完成标记时默认拒绝覆盖。

## 正式训练与运动结果

正式完整运动运行目录为
`outputs/full_motion_jscc/20260728T123819.525394Z`。12 个模型均完成且无
NaN。validation 选择的种子为 `C=1/2/3/4 → 44/42/44/42`。test 生成
15,800 条运动记录，schema/path 错误为 0。

下表只比较两种表示各自在 validation 选中的模型；每个单元聚合 100 条 `s7`
样本和 3 个名义噪声种子。正的 residual advantage 表示 residual L1 更低。

| SNR | C | residual L1 | full-motion L1 | residual advantage |
| ---: | ---: | ---: | ---: | ---: |
| -5 | 1 | 0.001974 | 0.002383 | +17.20% |
| -5 | 2 | 0.002118 | 0.002627 | +19.38% |
| -5 | 3 | 0.002190 | 0.002583 | +15.19% |
| -5 | 4 | 0.002284 | 0.002647 | +13.71% |
| 0 | 1 | 0.001706 | 0.001932 | +11.72% |
| 0 | 2 | 0.001642 | 0.001839 | +10.73% |
| 0 | 3 | 0.001554 | 0.001731 | +10.25% |
| 0 | 4 | 0.001521 | 0.001679 | +9.40% |
| 5 | 1 | 0.001574 | 0.001678 | +6.18% |
| 5 | 2 | 0.001452 | 0.001453 | +0.07% |
| 5 | 3 | 0.001259 | 0.001318 | +4.48% |
| 5 | 4 | 0.001161 | 0.001198 | +3.08% |
| 10 | 1 | 0.001514 | 0.001550 | +2.30% |
| 10 | 2 | 0.001382 | 0.001281 | -7.85% |
| 10 | 3 | 0.001143 | 0.001143 | +0.00% |
| 10 | 4 | 0.001015 | 0.000989 | -2.56% |

运动层的结论不是“残差永远更好”。残差在全部 -5/0/5 dB 点以及 10 dB 的
紧预算 `C=1` 更好；随着信道质量和预算提高，完整运动在 `C=2/4,10 dB`
略优，`C=3,10 dB` 基本相同。这支持更精确的解释：音频侧信息在较差或较紧的
信道条件下能减少必须通过视觉信道恢复的不确定性，而高 SNR、较大预算时直接
编码完整运动可以追平。

名义噪声种子在两条链路中匹配，但现有确定性噪声派生还包含各自
validation-selected 模型种子；当两个选中种子不同时，不能声称逐元素噪声实现
完全相同。比较报告因此以 100 条样本和三个名义噪声实现的聚合结果为主，不虚构
严格配对显著性。

## 视频结果

完整运动的 100 条、每条 17 条件 LivePortrait 评价正在正式运行：1 个共同
oracle 加 16 个 `C×SNR` 条件。audio prediction 直接复用 E5 的同一候选结果，
无噪声自编码器已经在运动层完整评价，因此不重复渲染。完成后将与冻结 E5
residual 视频结果按同一 `C`、SNR 和噪声种子汇总，并在此填写嘴部 ROI MAE、
NME、PSNR、SSIM 和失败数。
