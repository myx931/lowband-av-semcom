# Initial GitHub issue backlog

远端 Issue 尚未自动创建。本文件是等价、可审查的初始任务清单；创建远端 Issue 时逐项复制并关联依赖。

## 1. Bootstrap research repository

- 研究目标：建立可复现、可维护的研究仓库。
- 输入：项目任务说明和空 GitHub 仓库。
- 输出：Milestone 0/1 目录、配置、文档、测试和质量工具。
- 依赖：无。
- 验收标准：lint、unit、smoke 通过；分支提交不含数据、权重或输出。
- 风险：过早实现后续模块导致范围膨胀。
- 暂不包含：训练、数据下载、第三方模型集成。

## 2. Complete literature matrix for core papers

- 研究目标：建立经原文核对的相关工作证据表。
- 输入：论文原文、补充材料和官方代码。
- 输出：完整矩阵与逐篇阅读笔记。
- 依赖：Issue 1。
- 验收标准：每个结论可追溯到原文；未知项明确标记；许可证已核对。
- 风险：使用二手摘要、混淆任务设置或夸大结论。
- 暂不包含：复现所有论文或进行元分析。

## 3. Prepare GRID subset preprocessing pipeline

- 研究目标：建立可断点恢复、说话人隔离的数据管线。
- 输入：用户手动获取的 GRID 小子集。
- 输出：音频特征、人脸裁剪、关键点、manifest 和失败日志。
- 依赖：Issue 1；本地 `DATA_ROOT`。
- 验收标准：manifest 字段完整；重复运行跳过有效输出；split 无说话人交叉。
- 风险：许可、媒体解码、时间对齐和磁盘占用。
- 暂不包含：自动下载完整 GRID 或支持大量数据集。

## 4. Integrate motion extraction and reconstruction baseline

- 研究目标：用真实运动和参考人脸闭合重建链路。
- 输入：GRID 处理样本、选定第三方模型和合法权重。
- 输出：统一适配器、配置、重建样例和 mock/集成测试。
- 依赖：Issues 2、3。
- 验收标准：真实样本可重建；缺失依赖错误清晰；来源和版本固定。
- 风险：许可证、权重缺失、版本/运动表示不兼容。
- 暂不包含：训练大型生成器或声称未测试模型已可用。

## 5. Run motion perturbation sensitivity experiment

- 研究目标：测量运动表示误差对重建质量的影响。
- 输入：Issue 4 的运动表示与重建器。
- 输出：高斯噪声、均匀量化、随机丢弃、固定稀疏化的曲线和样例。
- 依赖：Issue 4。
- 验收标准：匹配扰动强度、固定种子、结果配置和逐样本指标齐全。
- 风险：运动代理指标与感知质量不一致。
- 暂不包含：音频预测、信道或可学习选择器。

## 6. Train audio-to-mouth-motion baseline

- 研究目标：验证音频能否预测低维嘴部运动。
- 输入：对齐的 log-Mel 与真实运动、speaker-isolated split。
- 输出：平均嘴型、上一帧和轻量 GRU 基线及评价。
- 依赖：Issues 3、4。
- 验收标准：报告 L1/L2、NME、速度误差和多种子结果；保存完整运行元数据。
- 风险：对齐错误、身份泄漏、过拟合受控语句。
- 暂不包含：大型 Transformer、视频扩散或同步损失堆叠。

## 7. Evaluate oracle prediction residual

- 研究目标：在无信道条件下验证预测残差是否值得传输。
- 输入：Issue 6 预测运动与真实运动。
- 输出：完整运动、纯预测、完整残差的匹配评价。
- 依赖：Issues 5、6。
- 验收标准：残差定义与单元测试明确；同一重建器下给出率—质量基准点。
- 风险：oracle 设置被误解为可部署方法。
- 暂不包含：AWGN、JSCC 或学习式选择。

## 8. Implement fixed and magnitude-based Top-K residual selection

- 研究目标：比较简单稀疏残差策略。
- 输入：Issue 7 的残差张量和预算。
- 输出：随机、固定位置、固定比例、幅度 Top-K 及预算统计。
- 依赖：Issue 7。
- 验收标准：相同 K/比例、公平计入选择信息、确定性测试通过。
- 风险：忽略索引开销使码率比较失真。
- 暂不包含：可学习打分和信道感知。

## 9. Add AWGN residual transmission baseline

- 研究目标：评估低维残差在简单信道下的鲁棒性。
- 输入：Issue 8 的选择残差和预算。
- 输出：MLP JSCC、功率归一化、AWGN 与 -5/0/5/10 dB 结果。
- 依赖：Issue 8。
- 验收标准：训练/测试 SNR 分离；功率与形状测试通过；报告全 SNR 曲线。
- 风险：归一化或 SNR 定义错误、单 SNR 过拟合。
- 暂不包含：调制、LDPC、OFDM 或复杂衰落信道。

## 10. Implement channel-aware learnable residual selector

- 研究目标：验证信道与预算条件化的重要性选择是否优于规则基线。
- 输入：残差、音频摘要、SNR、预算和可选运动动态。
- 输出：小型 scorer、Top-K/可微门控、可关闭损失和完整消融。
- 依赖：Issues 6、8、9。
- 验收标准：匹配预算比较；每个输入/损失可关闭；跨 SNR 稳定性有统计报告。
- 风险：训练不稳定、复杂度过高或不优于幅度规则。
- 暂不包含：强化学习、扩散模型或完整协议栈。
- 状态：validation-only 全局 SNR 安全门控和 4,882 参数 hard Top-K scorer
  均已完成。12 个模型、48,000 条 test 指标和匹配预算规则比较已验收；
  learned 只在紧预算 `K=2/4` 稳定优于原始幅值，`K=6/8` 未通过该比较。
  下一步只在 train/validation 上做最小输入与损失消融，避免重复利用 `s7`
  test 调参。
