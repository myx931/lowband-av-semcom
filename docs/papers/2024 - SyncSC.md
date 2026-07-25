---
title: "Synchronous Multi-Modal Semantic Communication System With Packet-Level Coding"
short_title: "SyncSC"
authors: ["Yun Tian", "Jingkai Ying", "Zhijin Qin", "Ye Jin", "Xiaoming Tao"]
year: 2025
preprint_year: 2024
venue: "IEEE Transactions on Wireless Communications, 24(5), 3684–3697"
doi: "10.1109/TWC.2025.3534995"
url: "https://arxiv.org/abs/2408.04535"
tags: [literature-note, primary-research-paper, multimodal-semantic-communication, 3dmm-semantic-representation, packet-level-semantic-fec, generative-reconstruction]
evidence_grade: "A-（正式期刊；完整系统实验，但生成链路较重）"
verified_on: "2026-07-25"
---

# SyncSC 精读

## 核验后的定位

该工作 2024 年先发布预印本，正式版本发表于 2025 年 IEEE Transactions on Wireless Communications，DOI 为 [10.1109/TWC.2025.3534995](https://doi.org/10.1109/TWC.2025.3534995)。它解决两类问题：多模态语义在语义域与时间域的同步，以及 RTP 式分组网络中的语义级丢包恢复。[原文 §I–II](https://arxiv.org/html/2408.04535)

## 传输语义与系统链路

- 视频发送端提取每帧 3DMM 系数；论文报告每帧仅传 16 个浮点数。
- 语音发送端先做 ASR，传文本而非高保真波形。
- 视频语义经 MAE/Transformer 风格的 PacSC 包级编码，以 `Dropout2d` 模拟整包擦除。
- 文本先随机交织并经 Huffman-RS；丢词由微调 BERT 的 TextPC 预测。
- 接收端用共享参考图像、参考语音、3DMM 和带时间戳文本生成视频与语音；时间戳承担跨模态对齐。[原文 §II–III](https://arxiv.org/html/2408.04535)

## 关键机制

PacSC 的思路是把相邻帧 3DMM 序列视作具有冗余的语义块，学习在随机包擦除后恢复原语义。视频包编码的总损失由 Huber 重建项与轻量 GAN 分布项组成；TextPC 采用丢失词分类的交叉熵。[原文 Eq. 12、16–20](https://arxiv.org/html/2408.04535)

同步机制并非对音频波形和唇形直接做对比学习，而是给文本块和视频帧共享时间戳，再由 visual-guided speech synthesis 联合利用文本、参考语音与表情语义。[原文 §II-B、§III-D](https://arxiv.org/html/2408.04535)

## 实验设计与结果

- VoxCeleb：17,913 个训练视频、514 个测试视频，约 500 名说话人；用于视频语义编码、包编码和图像生成。
- Chem：9,734 个训练片段、1,500 个测试片段；用于有文本对齐的视觉引导语音合成。
- 源编码基线：H.264、H.265、AV1、FOM；音频基线：AAC、Opus；包级基线：Reed-Solomon。[原文 §IV-A](https://arxiv.org/html/2408.04535)
- 同为 0.0039 bpp 时，FOM 为 SSIM 0.710 / LPIPS 0.243，SyncSC 视频语义编码为 0.745 / 0.192。相较 H.264、H.265、AV1，其 SSIM 较低，但 LPIPS接近，带宽分别节省约 84%、67%、49%。[原文 Table I、§IV-B](https://arxiv.org/html/2408.04535)
- 语音侧仅 55.37 bps，UTMOS 2.878 接近 AAC/Opus，但 PESQ 1.104 明显低于 AAC 2.346 和 Opus 3.591。这说明“语义可懂/自然度”与波形保真是不同目标。[原文 Table II](https://arxiv.org/html/2408.04535)
- PacSC 在训练包丢失率 0.4、测试不同丢失率时表现平滑；传统视频码流一旦 RS 无法修复会出现解码崩溃或马赛克。TextPC 在丢失率高于 0.5 时给 RS 带来约 0.1 的 BLEU 增益。[原文 §IV-E](https://arxiv.org/html/2408.04535)
- 系统端到端延迟为秒级；视觉引导语音合成约 0.942 秒，是主要瓶颈。PacSC 本身仅 0.18M 参数、约 0.013 秒。[原文 Table V](https://arxiv.org/html/2408.04535)

## 证据质量与局限

- 优点：正式同行评议期刊；明确区分源语义压缩、时间同步与包级鲁棒性；有传统编码、语义编码和丢包对照。
- 限制：将语音压成文本会丢失韵律、情绪与强调，不符合本课题“保留高保真音频”的目标；共享参考图像/语音的成本未计入 bpp；系统秒级延迟；不同数据集承担视频和语音实验，端到端跨模态结论需谨慎外推。
- 本轮未定位到作者公开的完整代码仓库，故可复现性低于 Wav2Lip；应记录为“未找到”，不能断言代码不存在。

## 对课题的可复用与不可照搬

可复用：

- 3DMM 表情/嘴部系数作为结构化传输语义；
- 包擦除而非仅 AWGN/Rayleigh 的测试；
- PacSC 作为可选保护层；
- 语义、时间戳和丢包率三维评价。

不可照搬：

- 把音频替换为文本；
- 把完整 16 维语义等保护，而不区分可预测与不可预测分量；
- 把共享参考知识的成本排除后直接宣称端到端带宽优势。

## 对本课题的直接启示

在 3DMM 路径下，可令 \(z_t\) 只包含嘴部/表情相关系数，并以音频预测 \(\tilde z_t\)。先验证残差稀疏化，再在包擦除场景叠加 PacSC。这样可以分清两个增益来源：

1. 残差选择减少了多少语义负载；
2. PacSC 在给定冗余下恢复了多少丢失语义。

二者必须分别做消融，避免把“少传”和“多加冗余”混为一个结论。

## AI 辅助说明

本笔记由 AI 辅助检索、核验与归纳；关键结果均回指原始论文。正式写作前应对照 IEEE 版本复核最终排版页码。
