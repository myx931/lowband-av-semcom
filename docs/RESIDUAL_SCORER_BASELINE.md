# 信道感知残差选择器基线

## 研究目的

E4 证明发送端已知的真实预测残差存在稀疏价值，E5/E6 第一阶段又证明残差经过
Sionna AWGN JSCC 后需要低 SNR 安全门控。本实验进一步回答：在相同复数信道
使用次数下，学习式逐帧 Top-K 选择是否比随机、固定位置和幅值规则更有效。

本实验只训练一个 4,882 参数的小型 scorer。E3 音频预测器、E5 JSCC 编解码器和
E6 validation-only 安全门控全部冻结，不联合微调。

## 冻结协议

- 源 E5 运行：`outputs/residual_jscc/20260728T060039.837712Z`。
- 安全门控：`outputs/channel_gate/20260728T093222.859499Z`，四个预算的阈值
  均为 `-1.5 dB`。
- 数据隔离：训练使用 800 条、8 个说话人；checkpoint 选择只使用 100 条
  validation `s3`；所有模型冻结后才一次性评价 100 条 test `s7`。
- E5 模型只按原 validation 结果选择，`C=1/2/3/4` 分别使用种子
  `43/44/43/44`。scorer 的三个独立种子为 `42/43/44`。
- 每个有效非参考帧先从 18 维残差中保留 `K=2C` 维，再输入同一个冻结 E5
  `C`-复数符号 JSCC。索引开销尚未计入，`K` 和 `C` 均不能称为真实 bitrate。
- scorer 输入为归一化残差、原始尺度绝对残差、相邻帧原始尺度变化、SNR、K 和
  C；网络为 `Linear(57,64) + ReLU + Linear(64,18)`。
- 前向传播使用恰好 K 个元素的 hard Top-K；反向传播使用 softmax
  straight-through 近似。
- 训练 SNR 在 `[0,10] dB` 连续采样。checkpoint 只按与 test 错开的
  `0.5/2.5/4.5/6.5/8.5 dB` validation 网格选择。
- 损失为原始运动空间位置 L1 加 `0.5 ×` 速度 L1。
- validation 的信道噪声由冻结 E5 模型种子、噪声种子、SNR 和 batch 编号确定，
  三个 scorer 种子共享同一组噪声条件。
- test 使用 E5 已冻结的 `-5/0/5/10 dB` 和噪声种子 `42/43/44`。每个方法在
  同一样本和条件下复用完全相同的信道噪声 realization。

匹配比较包括：

- `dense_jscc`：18 维全部输入冻结 JSCC，是同 C 下的非稀疏参照；
- `raw_magnitude`：按原始运动单位的残差绝对值逐帧 Top-K；
- `normalized_magnitude`：按 train-only 标准差归一化后的绝对值 Top-K；
- `fixed_train_magnitude`：只由训练集平均原始残差幅度冻结 K 个维度；
- `random`：三个固定随机种子的逐帧 Top-K；
- `learned_scorer`：三个独立训练种子的 hard Top-K。

## 命令

在独立 Python 3.11 Sionna 环境中运行：

```bash
PYTHONPATH=src DATA_ROOT=/path/to/public-datasets \
  "$SIONNA_PYTHON" scripts/train/train_residual_scorer.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e5-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --gate-run-dir outputs/channel_gate/20260728T093222.859499Z

PYTHONPATH=src DATA_ROOT=/path/to/public-datasets \
  "$SIONNA_PYTHON" scripts/eval/evaluate_residual_scorer.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e5-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --gate-run-dir outputs/channel_gate/20260728T093222.859499Z \
  --run-dir outputs/residual_scorer/<timestamp>
```

两条命令均可用 `--run-dir ... --resume` 验证并复用匹配产物；配置、运动统计、
E5 checkpoint、validation 缓存或门控策略哈希变化时会拒绝恢复。

## 十说话人真实结果

正式运行目录为 `outputs/residual_scorer/20260728T102637.647750Z`。12 个
scorer 均完成，无 NaN。冻结 test 共生成
`100 样本 × 4 C × 4 SNR × 3 噪声种子 × 10 方法/种子条件 = 48,000`
条有限指标。dense 条件与不可变 E5 test JSONL 的四项指标最大差为 `0.0`。

下表报告原始运动空间 L1；learned 是三个 scorer 种子的均值，raw 是逐帧原始
幅值规则：

| C | K | SNR | dense | raw magnitude | learned | learned 相对 raw |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0 dB | 0.001706 | 0.001806 | 0.001745 | +3.39% |
| 1 | 2 | 5 dB | 0.001574 | 0.001693 | 0.001621 | +4.25% |
| 1 | 2 | 10 dB | 0.001514 | 0.001643 | 0.001566 | +4.69% |
| 2 | 4 | 0 dB | 0.001642 | 0.001736 | 0.001691 | +2.57% |
| 2 | 4 | 5 dB | 0.001452 | 0.001578 | 0.001522 | +3.55% |
| 2 | 4 | 10 dB | 0.001382 | 0.001526 | 0.001464 | +4.07% |
| 3 | 6 | 0 dB | 0.001554 | 0.001621 | 0.001612 | +0.54% |
| 3 | 6 | 5 dB | 0.001259 | 0.001372 | 0.001391 | -1.38% |
| 3 | 6 | 10 dB | 0.001143 | 0.001289 | 0.001324 | -2.69% |
| 4 | 8 | 0 dB | 0.001521 | 0.001534 | 0.001543 | -0.63% |
| 4 | 8 | 5 dB | 0.001161 | 0.001234 | 0.001280 | -3.79% |
| 4 | 8 | 10 dB | 0.001015 | 0.001126 | 0.001189 | -5.58% |

正值表示 learned 的 L1 更低。跨 0/5/10 dB 平均后，learned 相对 raw 在
`C=1/2` 分别改善 `4.09%/3.36%`，在 `C=3/4` 则退化
`1.05%/3.07%`。逐种子结果方向一致：K=2/4 的三个种子都优于 raw，而
K=6 在 5/10 dB、K=8 在全部三个可发送 SNR 上都不优于 raw。

learned 在所有可发送条件都优于随机选择；相对随机的 L1 改善范围为
`7.0%–14.8%`。它在 K=2/4 也稳定优于归一化幅值和固定训练维度，说明模型确实
学到了可利用的内容相关选择。但 dense 在全部可发送条件仍最好；稀疏选择没有
在保持同一 C 的同时超过完整 18 维输入。

速度指标给出更温和的结果。跨 0/5/10 dB，learned 相对 raw 的速度 L1 在
`C=1/2/3/4` 分别改善 `3.36%/0.33%/2.35%/0.83%`，但单点并不全部改善。
因此加入速度损失抑制了部分抖动，却不能据此宣称全面优于幅值规则。

-5 dB 低于冻结门控阈值，所有方法都回退到纯音频预测，L1 均为
`0.001893`，且使用零个复数信道符号。该点验证了选择器没有绕过安全门控。

选择频率之和对每个可发送条件严格等于 K。learned 的高频维度会随预算和 SNR
变化，例如 K=6 时维度 `13/3/16` 的总体选择频率接近 1；这些维度编号对应展平
后的 LivePortrait 嘴部隐式表情坐标，不应解释成人脸几何关键点重要性。

训练和评估分别再次执行 `--resume` 后，完成标记、汇总和 test JSONL 的组合
SHA-256 均保持不变。

## 结论、局限与下一步

结果对“学习式选择优于规则”只提供有条件支持：在最紧预算 K=2/4 下结论稳定，
在 K=6/8 和较高 SNR 下不成立。一个可能原因是 scorer 的 straight-through
近似与冻结、原本按 dense 输入训练的 JSCC 在较大 K 下存在优化失配；这只是
待验证解释，不能由当前结果直接断言。

test `s7` 已用于本轮一次性评价，后续不得根据这些数值选择超参数。后续已严格
限制在 validation 预留半区完成 SNR 输入与速度损失的最小消融，结果仍未使
K=6/8 超过原始幅值规则，详见
`RESIDUAL_SCORER_VALIDATION_ABLATION.md`。本轮尚未对 48,000 个条件运行
LivePortrait，也未计入 Top-K 索引传输、量化或真实比特率。
