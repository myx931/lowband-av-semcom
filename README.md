# lowband-av-semcom

面向低带宽视频会议的音频辅助嘴部运动语义传输与生成式重建研究原型。

本仓库服务于硕士毕业论文研究。核心假设是：音频能够解释一部分基础嘴部运动，因此发送端可以优先传输音频无法预测、且对同步与重建重要的嘴部运动残差，而不是传输完整运动表示。

## 当前状态

当前完成 Milestone 0（仓库初始化）、Milestone 1（研究设计文档）以及 Milestone 2
的 GRID 小子集数据管线。真实 `s1` pilot 已用 20 条样本完成预处理验收。
Milestone 3 的冻结 LivePortrait 运动提取与重建敏感性基线已完成真实 GPU
验收；训练模型、残差传输与信道实验尚未实现，不宣称已经跑通。

研究路线：

1. 在 GRID 等公开数据上建立可复现的数据管线。
2. 提取嘴部运动表示并验证重建基线。
3. 训练轻量音频到嘴部运动预测器。
4. 比较完整运动、纯音频预测和稀疏预测残差的码率—质量权衡。
5. 在 AWGN 信道上加入轻量 JSCC。
6. 在基线稳定后研究信道感知的残差重要性选择器。

## 环境

- 推荐 Python 3.11，兼容目标为 Python 3.10–3.11。
- 深度学习框架：PyTorch。
- 数据根目录：环境变量 `DATA_ROOT`，不得指向仓库内的受版本控制目录。
- 本地路径配置：复制 `configs/paths.example.yaml` 后按需修改；不要提交含机器绝对路径的个人配置。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make setup
cp configs/paths.example.yaml configs/paths.local.yaml
export DATA_ROOT=/path/to/public-datasets
make test
```

## 常用命令

```bash
make lint       # Ruff 静态检查与格式检查
make format     # Ruff 自动格式化
make test       # 全部测试
make smoke      # 快速导入与配置测试
make clean      # 删除本地测试和构建缓存
```

## 配置与输出

实验配置采用 YAML。基础模板位于 `configs/experiment/baseline.yaml`，路径示例位于 `configs/paths.example.yaml`。代码不得写死本地绝对路径。

未来每次实验统一写入：

```text
outputs/{experiment_name}/{timestamp}/
```

并保存完整配置、Git commit、随机种子、Python/CUDA 环境、日志、最佳模型引用、评价 JSON 和示例重建视频引用。`outputs/` 中除说明文件外均被 Git 忽略。

## 文档导航

- [项目规范](docs/PROJECT_SPEC.md)
- [研究问题](docs/RESEARCH_QUESTIONS.md)
- [实验计划](docs/EXPERIMENT_PLAN.md)
- [文献矩阵](docs/LITERATURE_MATRIX.md)
- [风险登记](docs/RISK_REGISTER.md)
- [GitHub 任务清单](docs/ISSUE_BACKLOG.md)
- [数据说明](data/README.md)
- [运动与重建基线](docs/MOTION_BASELINE.md)
- [第三方依赖说明](third_party/README.md)

## 研究边界

当前不训练大型生成模型，不自建数据集，不使用强化学习、扩散视频模型或多模态大模型，不实现完整传统通信协议栈。第一阶段信道只考虑 AWGN。任何未阅读的论文、未运行的模型和未获得的指标都必须明确标为 `TODO`。

## 许可证与引用

仓库暂未选择开源许可证。公开发布或复用第三方代码、数据、模型前，必须分别核对其许可证和引用要求。
