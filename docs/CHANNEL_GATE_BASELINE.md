# Validation-only 信道安全门控基线

## 研究目的

E5 表明残差 JSCC 在 0/5/10 dB 可降低位置误差，但在训练范围外的 -5 dB 会比
完全不发送残差更差。E6 的第一步因此不是立即训练复杂选择器，而是建立一个不可
缺少的安全基线：只根据 validation 冻结每个信道预算的最低发送 SNR，低于该阈值
时直接使用音频预测运动。

该门控只决定“发送完整 JSCC 残差或完全不发送”，不选择 18 维残差元素，也不是
论文最终的信道感知重要性选择器。

## 冻结协议

- 源模型：E5 正式 Sionna 运行
  `outputs/residual_jscc/20260728T060039.837712Z`。
- 身份隔离：校准只用 100 条 validation `s3`；最终评价只用 100 条 test `s7`。
- 模型选择：每个 `C` 只使用 E5 validation 归一化残差 MSE 最低的模型种子，
  分别为 `C1/2/3/4 = 43/44/43/44`。
- 校准 SNR：`-4.5` 至 `9.5 dB`，间隔 1 dB；与 test 的
  `-5/0/5/10 dB` 完全错开。
- 噪声种子：校准和 test 均使用固定的 `42/43/44`。
- 主指标：原始运动空间 L1。
- 阈值规则：选择最低的 validation SNR，使该点及所有更高校准 SNR 的平均 L1
  都严格低于纯预测。这样不会因某个孤立的偶然好点提前发送。
- 低于校准网格或不存在安全后缀时，固定回退到 `prediction_only`。
- 策略落盘前，校准指纹不包含 test 指标哈希；策略冻结后才单独绑定不可变 E5
  test JSONL 的 SHA-256，并一次性评价 test。

## 命令

在独立 Python 3.11 Sionna 环境运行：

```bash
PYTHONPATH=src DATA_ROOT=/path/to/public-datasets \
  "$SIONNA_PYTHON" scripts/eval/evaluate_channel_gate.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e5-run-dir outputs/residual_jscc/20260728T060039.837712Z
```

中断后使用打印出的运行目录：

```bash
PYTHONPATH=src DATA_ROOT=/path/to/public-datasets \
  "$SIONNA_PYTHON" scripts/eval/evaluate_channel_gate.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e5-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --run-dir outputs/channel_gate/<timestamp> --resume
```

正式运行目录为 `outputs/channel_gate/20260728T093222.859499Z`。输出包括独立的
校准与 test 来源信息、validation 逐样本 JSONL、冻结策略、test 门控 JSONL、
汇总 JSON/CSV 和曲线，全部由 Git 忽略。

## 真实结果

校准共生成 18,100 条 validation 记录：100 条纯预测，以及
`4 C × 15 SNR × 3 噪声种子 × 100 样本 = 18,000` 条 JSCC 记录。四个预算都在
`-2.5` 与 `-1.5 dB` 之间越过纯预测，因此均冻结为 `-1.5 dB`：

| C | -2.5 dB 相对纯预测 L1 | -1.5 dB 相对纯预测 L1 | 冻结阈值 |
|---:|---:|---:|---:|
| 1 | -0.68% | +2.11% | -1.5 dB |
| 2 | -0.66% | +4.35% | -1.5 dB |
| 3 | -0.40% | +5.35% | -1.5 dB |
| 4 | -1.64% | +5.76% | -1.5 dB |

正值表示 L1 降低。策略冻结后，从 E5 test JSONL
`eda94cb8b790ec1f2023d3fed844b2767151aa7a71a4b83e00d8996896f997b9`
生成 4,800 条有限门控指标：

| C | -5 dB 始终发送 | -5 dB 门控 | 0 dB 门控 | 5 dB 门控 | 10 dB 门控 |
|---:|---:|---:|---:|---:|---:|
| 1 | -4.27% | 0.00% | +9.89% | +16.84% | +20.01% |
| 2 | -11.88% | 0.00% | +13.25% | +23.26% | +26.99% |
| 3 | -15.73% | 0.00% | +17.92% | +33.46% | +39.59% |
| 4 | -20.67% | 0.00% | +19.64% | +38.66% | +46.40% |

门控在 -5 dB 对所有预算使用零个复数信道符号并退回纯预测，因此消除了始终发送
造成的 4.3%–20.7% 位置 L1 退化；0/5/10 dB 则完整保留 JSCC 的位置收益。
重复运行 `--resume` 时，test JSONL 的 SHA-256 和修改时间均保持不变。

## 局限与下一步

该规则只使用全局已知 SNR，不看样本内容、残差幅度、音频或运动速度，因此不能
回答“有限预算下哪些残差最重要”。它也没有解决中等 SNR 的时间抖动：以 `C=4`
为例，0/5 dB 的速度 L1 仍分别比纯预测高 147.9%/48.9%，只有 10 dB 才降低
2.7%。下一阶段应在匹配信道预算下实现信道感知残差重要性选择，并同时优化或
约束位置和速度误差；本门控必须保留为安全对照。
