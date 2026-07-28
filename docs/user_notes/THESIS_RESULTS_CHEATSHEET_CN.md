# 论文曲线、数据与结论速查

更新时间：2026-07-28

用途：写论文结果章节、制作答辩 PPT、寻找正式 CSV 或检查一句结论能否成立。

这份文档只引用已经冻结且有效的 E2–E7 结果，不引用作废的首次 E3 时间拉伸
实验。大体积曲线和 CSV 保留在本机 `outputs/`，不会上传 GitHub。

## 1. 最终研究主线

建议论文只围绕下面这条主线展开：

```text
音频能够预测基础嘴部运动
        ↓
真实预测错误具有稀疏上界
        ↓
正常 AWGN 条件下可以用 JSCC 传输预测残差
        ↓
极低 SNR 应安全回退为纯音频预测
        ↓
匹配符号预算时，残差在低/中 SNR 和紧预算下优于完整运动
```

推荐的最终方法名称：

> 音频预测 + dense 预测残差 JSCC + validation-only 低 SNR 安全回退。

不要把 learned hard Top-K scorer 写成最终主方法。它只在紧预算有有限正结果，
并且当前稀疏输入没有降低同一 `C` 下的实际复符号数量。

## 2. 正文建议保留的 6 张主图

### 图 1：音频预测基线

- 建议标题：十说话人身份隔离条件下的音频到嘴部运动预测性能。
- 横轴：方法或训练 epoch。
- 纵轴：运动 L1；可在子图中补充嘴部 NME。
- 本地已有图：
  - `outputs/audio_to_motion_ten_speaker/20260727T083555.175559Z/training_curves.png`
  - `outputs/audio_to_motion_ten_speaker/20260727T083555.175559Z/reconstruction/metric_comparison.png`
- 一句话结论：GRU 在 test 上比 train mean 和 zero motion 的运动 L1 分别低
  7.82% 和 12.61%，证明音频对未见说话人的嘴部运动具有可利用预测能力。
- 边界：不可部署的 oracle persistence 仍明显更好，不能宣称音频已经完全解释
  嘴部运动。

关键数据：

| test 方法 | 运动 L1 ↓ | 嘴部 NME ↓ | 嘴部 ROI MAE ↓ |
| --- | ---: | ---: | ---: |
| audio GRU，三种子运动均值 | 0.001978 ± 0.000065 | — | — |
| audio GRU，validation 选中 seed 43 | 0.001893 | 0.02208 | 5.2993 |
| train mean | 0.002146 | 0.02568 | 5.9690 |
| zero motion | 0.002264 | 0.02557 | 6.1921 |
| oracle persistence | 0.000701 | — | — |

### 图 2：真实残差的 oracle 稀疏率—质量曲线

- 建议标题：无信道条件下真实预测残差的固定预算上界。
- 横轴：每帧保留残差维数 `K`，推荐显示 `0/1/2/4/6/9/12/18`。
- 纵轴：运动 L1；可增加嘴部 ROI MAE 或 NME 子图。
- 本地已有图：
  `outputs/residual_baseline/20260727T161813.688662Z/plots/rate_quality_test.png`
- 一句话结论：真实幅度 Top-K 在 K=4/18 时已保留约 82.3% 原始残差能量，
  将运动 L1 从 0.001893 降至 0.000785，并明显优于同预算随机选择。
- 边界：这是发送端看见真实残差后的 oracle 上界，不是可部署码率结果。

关键数据：

| K/18 | raw Top-K 运动 L1 ↓ | raw energy | 嘴部 MAE ↓ | 嘴部 NME ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0，纯预测 | 0.001893 | 0% | 5.299 | 0.02195 |
| 1 | 0.001464 | 47.4% | — | — |
| 2 | 0.001189 | 64.1% | 3.214 | 0.01184 |
| 4 | 0.000785 | 82.3% | 2.100 | 0.00789 |
| 6 | 0.000501 | 91.4% | 1.380 | 0.00516 |
| 9 | 0.000223 | 97.5% | 0.688 | 0.00250 |
| 12 | 0.000064 | 99.6% | — | — |
| 18 | 约 0 | 100% | 0 | 0 |

K=4 时相对纯预测，嘴部 ROI MAE/NME 分别改善 60.4%/64.0%；相对同预算
随机选择分别改善 56.7%/58.7%。

### 图 3：残差 JSCC 的运动质量—SNR 曲线

- 建议标题：不同复符号预算下残差 JSCC 的运动恢复性能。
- 横轴：SNR，`-5/0/5/10 dB`。
- 纵轴：原始运动空间 L1。
- 曲线：`C=1/2/3/4`，另画纯音频 prediction-only 水平线。
- 本地已有图：
  `outputs/residual_jscc/20260728T060039.837712Z/plots/motion_l1_vs_snr.png`
- 一句话结论：0/5/10 dB 下全部预算均优于纯音频预测；-5 dB 时全部退化，
  说明训练范围外强噪声残差会污染接收端预测。
- 边界：图中 `C` 是每有效帧复符号数，不是 bit/s。

三种子均值：

| C | -5 dB | 0 dB | 5 dB | 10 dB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.001978 | 0.001708 | 0.001573 | 0.001514 |
| 2 | 0.002093 | 0.001616 | 0.001410 | 0.001327 |
| 3 | 0.002156 | 0.001548 | 0.001276 | 0.001171 |
| 4 | 0.002269 | 0.001513 | 0.001159 | 0.001014 |

纯音频 prediction-only L1 为 0.001893。

### 图 4：残差 JSCC 的嘴部视频质量—SNR 曲线

- 建议标题：冻结 LivePortrait 重建下的嘴部质量随信道条件变化。
- 横轴：SNR。
- 纵轴：嘴部 NME；ROI MAE 可作为第二子图。
- 本地已有图：
  - `outputs/residual_jscc/20260728T060039.837712Z/video_reconstruction/plots/mouth_nme_vs_snr.png`
  - `outputs/residual_jscc/20260728T060039.837712Z/video_reconstruction/plots/mouth_mae_vs_snr.png`
- 一句话结论：运动空间收益能够转化到视频嘴部质量；C=4、10 dB 时 NME
  相对纯预测改善 34.6%，但 C=4、-5 dB 时恶化 32.4%。
- 边界：指标相对 lip-only oracle 重建，不代表完全恢复原始真实视频纹理。

嘴部 NME：

| C | prediction-only | -5 dB | 0 dB | 5 dB | 10 dB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.02195 | 0.02296 | 0.02147 | 0.02077 | 0.02048 |
| 2 | 0.02195 | 0.02515 | 0.02074 | 0.01904 | 0.01837 |
| 3 | 0.02195 | 0.02614 | 0.01928 | 0.01624 | 0.01512 |
| 4 | 0.02195 | 0.02905 | 0.01975 | 0.01601 | 0.01436 |

### 图 5：低 SNR 安全门控

- 建议标题：validation-only SNR 门控对最差信道性能的保护。
- 横轴：SNR。
- 纵轴：相对纯音频预测的运动 L1 改善百分比。
- 本地已有图：
  - `outputs/channel_gate/20260728T093222.859499Z/plots/validation_gate_calibration.png`
  - `outputs/channel_gate/20260728T093222.859499Z/plots/test_gate_l1_vs_snr.png`
- 一句话结论：validation 冻结阈值为 -1.5 dB；test 的 -5 dB 条件自动不发送，
  将 4.3%–20.7% 的退化变为 0%，同时完整保留 0/5/10 dB 收益。
- 边界：门控只使用全局 SNR，不是内容感知选择器。

正值表示比纯预测更好：

| C | -5 dB 始终发送 | -5 dB 门控 | 0 dB | 5 dB | 10 dB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -4.27% | 0.00% | +9.89% | +16.84% | +20.01% |
| 2 | -11.88% | 0.00% | +13.25% | +23.26% | +26.99% |
| 3 | -15.73% | 0.00% | +17.92% | +33.46% | +39.59% |
| 4 | -20.67% | 0.00% | +19.64% | +38.66% | +46.40% |

### 图 6：完整运动与预测残差的匹配对照

- 建议标题：匹配复符号预算下完整运动与预测残差 JSCC 的性能比较。
- 横轴：复符号预算 `C` 或复符号/段。
- 纵轴：运动 L1、嘴部 NME；按 SNR 分面。
- 本地已有图：
  `outputs/full_motion_vs_residual/20260728T151133.243833Z/plots/full_motion_vs_residual.png`
- 一句话结论：residual 在运动 L1 的 16 组中胜 14 组，在嘴部 NME 中胜
  13 组；优势集中在低/中 SNR 和紧预算，高 SNR、大预算时完整运动追平。
- 这是 RQ2 最重要的主图。

residual 相对 full-motion 的运动 L1 优势：

| SNR | C=1 | C=2 | C=3 | C=4 |
| ---: | ---: | ---: | ---: | ---: |
| -5 dB | +17.20% | +19.38% | +15.19% | +13.71% |
| 0 dB | +11.72% | +10.73% | +10.25% | +9.40% |
| 5 dB | +6.18% | +0.07% | +4.48% | +3.08% |
| 10 dB | +2.30% | -7.85% | 约 0% | -2.56% |

residual 相对 full-motion 的嘴部 NME 优势：

| SNR | C=1 | C=2 | C=3 | C=4 |
| ---: | ---: | ---: | ---: | ---: |
| -5 dB | +18.56% | +16.51% | +14.91% | +11.24% |
| 0 dB | +14.11% | +7.18% | +9.94% | +6.60% |
| 5 dB | +11.26% | +0.14% | +6.61% | -0.34% |
| 10 dB | +9.89% | -3.24% | +5.19% | -5.24% |

## 3. 通信预算速查

GRID 每条样本 75 帧、25 fps、3 秒；第一帧是参考帧，因此 74 帧参与传输。

| C | 复符号/段 | 复符号/秒 | 实自由度/段 |
| ---: | ---: | ---: | ---: |
| 1 | 74 | 24.67 | 148 |
| 2 | 148 | 49.33 | 296 |
| 3 | 222 | 74.00 | 444 |
| 4 | 296 | 98.67 | 592 |

论文必须写“复符号/段”或“复符号/秒”，不能写成 bit/s。当前未实现量化、
调制编码、协议头、参考脸、音频链路和 Top-K 索引的完整开销。

## 4. 适合放到附录的曲线

### A1：LivePortrait 运动敏感性

本地曲线：

- `outputs/motion_sensitivity/20260726T060450.218238Z/plots/gaussian_psnr.png`
- `outputs/motion_sensitivity/20260726T060450.218238Z/plots/quantization_psnr.png`
- `outputs/motion_sensitivity/20260726T060450.218238Z/plots/random_dropout_psnr.png`
- `outputs/motion_sensitivity/20260726T060450.218238Z/plots/magnitude_sparsity_psnr.png`

横轴方向必须分别解释：

- Gaussian 横轴是噪声标准差：越大扰动越强，通常越差；
- quantization 横轴是 bit 数：越大表示量化越精细，通常越好；
- dropout 横轴是保留比例：越大保留信息越多，通常越好；
- magnitude sparsity 横轴也是保留比例：越大保留信息越多，通常越好。

所以除了 Gaussian 外，其余曲线与横轴正相关并不反直觉。它们的横轴不是“扰动
强度”，而是“精度或保留量”。E2 仅为单说话人 pilot，适合证明评价闭环有效，
不宜作为论文主结果。

### A2：速度误差曲线

本地曲线：
`outputs/residual_jscc/20260728T060039.837712Z/plots/velocity_l1_vs_snr.png`

结论：位置 L1 变好不保证时序更平滑。C=4 在 0/5 dB 的速度 L1 比纯预测分别
高 147.9%/48.9%，10 dB 才降低约 2.7%。这是论文讨论“信道噪声可能造成嘴部
抖动”的主要证据。

### A3：learned scorer

本地曲线：
`outputs/residual_scorer/20260728T102637.647750Z/plots/c_*_position_velocity.png`

| 预算 | learned 相对 raw 的平均位置 L1 |
| --- | ---: |
| C=1，K=2 | +4.09% |
| C=2，K=4 | +3.36% |
| C=3，K=6 | -1.05% |
| C=4，K=8 | -3.07% |

结论：learned scorer 只在紧预算获得有限支持；dense residual 在全部可发送
条件仍最好。

### A4：scorer validation-only 消融

本地曲线：
`outputs/residual_scorer_validation_ablation/20260728T112523.904356Z/plots/`

结论：去掉 SNR 输入或速度损失都不能让 K=6/8 稳定超过 raw magnitude；最好的
消融格仍差 0.35%/2.18%。因此停止继续用已有 test 调 scorer 是合理决策。

## 5. 已有图怎样分配到论文

建议正文：

1. 图 1：音频预测基线；
2. 图 2：oracle 残差率—质量；
3. 图 3：残差 JSCC 运动 L1—SNR；
4. 图 4：残差 JSCC 嘴部 NME—SNR；
5. 图 5：低 SNR 安全门控；
6. 图 6：full-motion 与 residual 匹配对照。

建议附录或消融章节：

1. E2 四类敏感性曲线；
2. velocity L1 曲线；
3. scorer 四预算位置—速度曲线；
4. scorer validation-only 2×2 消融；
5. communication report 中按 SNR 分面的 dense/sparse 率—质量图。

答辩 PPT 最少保留图 1、图 3、图 5、图 6。图 6 是核心贡献图，图 5 是安全性
补充，图 2 用于解释为什么一开始研究残差。

## 6. 可以直接使用的结论句

### 音频预测

在十说话人、8/1/1 身份隔离划分上，因果音频 GRU 的 test 运动 L1 相对
train mean 和 zero motion 分别降低 7.82% 和 12.61%，表明音频对未见说话人的
低维嘴部运动具有可利用预测能力。

### 残差稀疏上界

在无信道 oracle 条件下，每帧仅保留 4/18 个真实幅度最大的残差值即可保留约
82.3% 残差能量，并将嘴部 ROI MAE/NME 相对纯预测降低 60.4%/64.0%；但该结果
尚未计入索引、量化或信道开销。

### 信道可行性

在 Sionna 复数 AWGN 中，dense residual JSCC 在 0/5/10 dB 的全部测试预算下
均改善纯音频预测；训练范围外 -5 dB 则产生退化，说明带噪语义补充需要安全
回退机制。

### 信道自适应

仅使用 validation 冻结的 -1.5 dB 门控，在 test 的 -5 dB 条件将
4.3%–20.7% 的始终发送退化消除为零，同时不牺牲 0/5/10 dB 的既有收益。

### 完整运动公平对照

在匹配网络规模、Sionna AWGN 和复符号预算下，预测残差在运动 L1 的 16 个条件
中胜出 14 个，在嘴部 NME 中胜出 13 个；优势主要位于低/中 SNR 与紧预算，
高 SNR、大预算时完整运动可追平或局部胜出。

### 稀疏选择负结果

虽然真实残差具有显著 oracle 稀疏性，但当前 hard Top-K 后仍映射为固定 `C`
个复符号，24 个可发送 sparse 点全部被同速率 dense residual JSCC 支配；
语义维数减少不能被直接解释为通信资源减少。

## 7. 不能写的结论

- 不能称当前结果为真实 bit/s 或完整数字通信系统；
- 不能声称 residual 在所有 SNR、预算和指标都优于 full-motion；
- 不能声称 learned scorer 全面优于简单规则；
- 不能把 oracle Top-K 当成已部署选择器；
- 不能声称 GRID 单数据集已经证明真实会议或跨数据集泛化；
- 不能使用已作废的首次 E3 时间拉伸结果；
- 不能根据已经消费的 `s7` test 再选择超参数。

## 8. 正式数据索引

| 内容 | 冻结数据 |
| --- | --- |
| 十说话人音频预测 | `outputs/audio_to_motion_ten_speaker/20260727T083555.175559Z/summary.csv` |
| oracle 残差曲线 | `outputs/residual_baseline/20260727T161813.688662Z/summary.csv` |
| residual JSCC 运动 | `outputs/residual_jscc/20260728T060039.837712Z/report_summary.csv` |
| residual JSCC 视频 | `outputs/residual_jscc/20260728T060039.837712Z/video_reconstruction/summary.csv` |
| 安全门控 | `outputs/channel_gate/20260728T093222.859499Z/test_summary.csv` |
| learned scorer | `outputs/residual_scorer/20260728T102637.647750Z/evaluation_summary.csv` |
| scorer 消融 | `outputs/residual_scorer_validation_ablation/20260728T112523.904356Z/audit_summary.csv` |
| 通信代价 | `outputs/communication_report/20260728T120635.097427Z/` |
| full-motion JSCC | `outputs/full_motion_jscc/20260728T123819.525394Z/evaluation_summary.csv` |
| E7 最终比较 | `outputs/full_motion_vs_residual/20260728T151133.243833Z/` |

## 9. 下一步计划：E8 论文证据包

E0–E7 核心仿真已经完成。下一阶段不重新训练模型，建立只读、可追溯的论文证据包。

### 必须完成

1. 从上述冻结 CSV/JSON 生成统一风格的 6 张正文图和附录图；
2. 生成一张总表，统一 prediction-only、full-motion、dense residual 和
   gated residual 的方法命名、复符号预算与指标；
3. 对 E7 逐样本差值计算 bootstrap 置信区间，以样本为统计单位，不能把三个噪声
   实现当作 300 个独立说话样本；
4. 选取固定排序的少量代表性样本，生成低/中/高 SNR 对照图；不重新挑“最好看”
   的样本；
5. 保存源文件 SHA-256、绘图配置、Git commit、环境、图和表；
6. 起草论文“实验设置、结果与讨论”中文章节。

### 可选但不阻塞论文

- 统计模型参数量、推理时间和峰值显存；
- 增加 paired effect distribution 或箱线图；
- 对代表性视频做小规模人工观察记录；
- 在论文 future work 中讨论数字 bitstream、衰落信道和跨数据集。

### 明确不做

- 不为小数点优势重训 E3/E5/E7；
- 不继续使用现有 `s7` test 调 scorer；
- 不把 scope 扩大到大型模型、扩散模型或完整协议栈；
- 不在没有新未触碰说话人协议时尝试新的选择器并报告为 test 改进。

### E8 完成标准

- 正文 6 张图均能追溯到冻结源哈希；
- 主表中的所有数值可由脚本重新生成；
- 置信区间的统计单位和噪声聚合方式明确；
- 图注同时写结果、条件和边界；
- Git 中只提交小型脚本、配置和文档，不提交原始输出或重建媒体。
