# 冻结通信代价与率—质量报告

## 目的与口径

本报告对已经完成的 E5/E6 聚合结果做只读汇总，不重新选择模型、不重新计算
test 指标。正式产物为
`outputs/communication_report/20260728T120635.097427Z`，其报告指纹为
`de296a24c992a31ef6c049f7283742ebcd01266297c849f1ae47b310310d3d27`。
源 E5 test JSONL、门控策略、scorer 汇总和 validation 消融汇总的 SHA-256 均
保存在 `source_provenance.json`。

GRID 样本为 75 帧、25 fps，即每段 3 秒。第一帧是参考帧，不发送残差，因此每段
有 74 个有效传输帧。`C` 表示每个有效帧使用的复数信道符号数：

| C | 复符号/段 | 实自由度/段 | 复符号/秒 |
| ---: | ---: | ---: | ---: |
| 1 | 74 | 148 | 24.67 |
| 2 | 148 | 296 | 49.33 |
| 3 | 222 | 444 | 74.00 |
| 4 | 296 | 592 | 98.67 |

这是模拟复数 AWGN 的信道使用计量，不是真实数字 bitrate。当前没有调制阶数、
信道编码码率或包格式，因此报告中的 bit/s 固定为空。

## Dense 与 sparse 的公平解释

dense residual JSCC 将完整 18 维残差送入编码器；hard Top-K 在编码前保留
`K=2C` 个残差值。但二者随后都被映射为恰好 `C` 个复符号，所以：

- sparse 减少的是进入 JSCC 的非零语义维数；
- sparse 没有减少本实验实际发送的复信道符号数；
- `K/18` 不能称为信道压缩率；
- 当前 Top-K 身份通过固定的 18 维稀疏输入隐式影响编码结果，没有另行实现数字
  索引包，因此不能虚构索引 bit 数。

## 冻结 test 结果

- -5 dB 低于 validation 冻结阈值 `-1.5 dB`，四个预算全部回退
  prediction-only，实际发送 0 个复符号。
- 0/5/10 dB 下，dense residual JSCC 的运动 L1 随 `C` 增加总体下降。
- `C=1/2/3/4` 在 10 dB 相对 prediction-only 的运动 L1 改善分别为
  20.01%、26.99%、39.59% 和 46.40%。
- `C=4,10 dB` 的冻结视频评价将相对 lip-only oracle 的嘴部 NME 从
  prediction-only 的 `0.02195` 降至 `0.01436`。
- 在 3 个可发送 SNR、4 个预算和 2 个稀疏方法形成的 24 个点上，
  raw magnitude 和 learned scorer 均被相同符号速率的 dense residual JSCC
  严格支配。

最后一项不是“稀疏思想无效”：E4 已证明真实残差存在稀疏 oracle 上界。它说明
当前“先置零部分残差、再送入为 dense 输入训练的固定长度 JSCC”没有转化为更少
信道符号或更好同速率质量。继续微调 scorer 的科研收益低于补齐完整运动通信
对照。

## 未计入的公共或系统开销

- 音频侧信息链路；
- 参考脸或关键帧；
- 调制与信道编码；
- 同步、协议头和重传；
- 可独立传输的 Top-K 索引。

因此当前图只能称为“复信道符号预算—质量曲线”。下一阶段若仍使用模拟 JSCC，
继续以复符号为横轴；只有明确实现数字表示、量化、调制和编码后才能报告 bit/s。

## 复现命令

```bash
PYTHONPATH=src python scripts/eval/report_communication_cost.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e5-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --gate-run-dir outputs/channel_gate/20260728T093222.859499Z \
  --scorer-run-dir outputs/residual_scorer/20260728T102637.647750Z \
  --ablation-run-dir \
    outputs/residual_scorer_validation_ablation/20260728T112523.904356Z
```

对已有目录加 `--run-dir ... --resume` 时，源哈希和配置指纹一致才会读取完成
结果，且不会改写有效产物。

