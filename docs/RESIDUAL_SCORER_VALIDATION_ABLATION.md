# 残差选择器 validation-only 消融

## 研究问题

E6 正式 test 表明 learned scorer 在紧预算 `K=2/4` 优于原始幅值规则，但在
`K=6/8` 和中高 SNR 下更差。本消融只诊断两个最直接的原因：

1. SNR 条件输入是否没有帮助或导致过拟合；
2. `0.5 ×` 速度 L1 是否以过多位置 L1 为代价。

本实验不读取或重算 `s7` test，不扩大网络，也不尝试根据已有 test 结果搜索新
超参数。

## 冻结协议

- E5 来源固定为 `outputs/residual_jscc/20260728T060039.837712Z`。
- 只研究 `C=3/4`，对应逐帧 `K=6/8`；冻结的 E5 模型种子分别为 `43/44`。
- 800 条、8 个说话人的 train 残差用于训练 scorer。
- 100 条 validation 实际来自 `s3`。按
  `SHA-256(partition_salt:sample_id)` 排序后，前 50 条只用于 checkpoint
  calibration，后 50 条在全部 24 个 checkpoint 冻结后才首次用于 audit。
- audit 使用 `0.5/2.5/4.5/6.5/8.5 dB` 和噪声种子 `42/43/44`，不接触 E5
  的 `-5/0/5/10 dB` test 网格。
- 每个因子格训练种子 `42/43/44`。同一种子、epoch、batch 的初始化、样本顺序、
  SNR 和 AWGN realization 都相同。
- E5 JSCC 权重完全冻结；前向仍为恰好 K 维 hard Top-K。

2×2 因子格如下：

| 变体 | 使用 SNR 输入 | 速度损失权重 |
|:---|:---:|---:|
| `full` | 是 | 0.5 |
| `no_snr` | 否，输入位置固定置零 | 0.5 |
| `no_velocity` | 是 | 0 |
| `no_snr_no_velocity` | 否 | 0 |

`no_snr` 仍保留相同网络宽度和 4,882 个参数，因此差异只来自信息开关，而不是
模型容量。

## 命令

在 Python 3.11 Sionna 环境运行：

```bash
PYTHONPATH=src DATA_ROOT=/path/to/public-datasets \
  "$SIONNA_PYTHON" scripts/train/train_residual_scorer_ablation.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e5-run-dir outputs/residual_jscc/20260728T060039.837712Z

PYTHONPATH=src DATA_ROOT=/path/to/public-datasets \
  "$SIONNA_PYTHON" scripts/eval/evaluate_residual_scorer_ablation.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e5-run-dir outputs/residual_jscc/20260728T060039.837712Z \
  --run-dir outputs/residual_scorer_validation_ablation/<timestamp>
```

训练阶段只建立 validation 分区并使用 calibration 半区。audit 命令单独运行，
且不依赖 `test_metrics.jsonl` 或 test 残差缓存。两条命令均支持
`--run-dir ... --resume`。

## 正式结果

运行目录为
`outputs/residual_scorer_validation_ablation/20260728T112523.904356Z`。
24/24 个模型完成，无 NaN。预留的 50 条 `s3` audit 生成
`50 × 2 C × 5 SNR × 3 噪声 × (2 规则 + 4 变体 × 3 种子) = 21,000`
条有限指标，所有记录均标记 `test_data_accessed=false`。

下表对五个 audit SNR 取平均。learned 报告三个种子的均值和种子标准差；“相对
raw”正值表示位置 L1 更低：

| C | K | 方法 | 位置 L1 | 相对 raw | 速度 L1 | 相对 raw 速度 |
|---:|---:|:---|---:|---:|---:|---:|
| 3 | 6 | dense | 0.001445 | +7.82% | 0.001075 | +0.03% |
| 3 | 6 | raw magnitude | 0.001568 | 0.00% | 0.001076 | 0.00% |
| 3 | 6 | full | 0.001584 ± 0.000011 | -1.05% | 0.001033 ± 0.000003 | +3.92% |
| 3 | 6 | no SNR | 0.001584 ± 0.000010 | -1.01% | 0.001036 ± 0.000005 | +3.67% |
| 3 | 6 | no velocity | 0.001576 ± 0.000007 | -0.49% | 0.001047 ± 0.000013 | +2.63% |
| 3 | 6 | no SNR、no velocity | 0.001573 ± 0.000008 | -0.35% | 0.001047 ± 0.000011 | +2.69% |
| 4 | 8 | dense | 0.001362 | +5.32% | 0.001133 | -3.09% |
| 4 | 8 | raw magnitude | 0.001439 | 0.00% | 0.001099 | 0.00% |
| 4 | 8 | full | 0.001476 ± 0.000003 | -2.61% | 0.001057 ± 0.000008 | +3.87% |
| 4 | 8 | no SNR | 0.001475 ± 0.000004 | -2.53% | 0.001054 ± 0.000010 | +4.09% |
| 4 | 8 | no velocity | 0.001475 ± 0.000001 | -2.55% | 0.001061 ± 0.000003 | +3.49% |
| 4 | 8 | no SNR、no velocity | 0.001470 ± 0.000001 | -2.18% | 0.001063 ± 0.000005 | +3.35% |

去掉 SNR 对平均位置 L1 的影响绝对值不超过 `0.36%`，方向上通常是极小改善；
它不是 K=6/8 退化的主要原因。加入速度损失会把速度 L1 再降低约
`0.4%–1.3%`，但位置 L1 反而增加约 `0.06%–0.66%`，说明存在明确但很小的
位置—时间权衡。

四个格中位置最好的 `no_snr_no_velocity` 仍比 raw magnitude 差
`0.35%（K=6）/2.18%（K=8）`。这与已冻结 test 中 K=6/8 的方向一致。因此
关闭这两个因素不能挽救 learned scorer；当前证据不支持继续增加输入或损失项来
追求全面胜出。

audit 中 dense 仍比 raw 的位置 L1 低 `7.82%/5.32%`，说明相同 C 的冻结 JSCC
仍存在可利用的信息空间，但本轮不能判断差距来自 straight-through Top-K、冻结
dense-trained JSCC 与稀疏输入的失配，还是 scorer 容量。它们只能列为待验证
解释。

训练和 audit 分别再次执行 `--resume` 后，关键产物组合 SHA-256 前后完全相同。
逐行审计确认 21,000 条记录全部有限、只来自 validation `s3`，选择频率之和严格
等于 K，且没有 test 访问。

## 决策

- 保留速度损失的结论仅限“稍微改善时间误差”，不能说它改善总体位置质量。
- SNR 标量输入在当前固定 AWGN 和逐 C 独立模型中没有观察到实质收益。
- K=6/8 的可靠规则基线仍是 raw magnitude；learned scorer 只保留其 K=2/4
  的有限正结果，不再把它描述成所有预算的首选方法。
- 不再使用已经消费的 `s7` test 调整 scorer。

下一项工作应停止 scorer 超参数搜索，转向冻结方法的完整通信开销与率—质量报告；
任何新的选择器结构都必须留到新的未触碰说话人协议后再评价。
