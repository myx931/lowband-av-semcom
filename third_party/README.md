# Third-party integrations

第三方运动提取或视频重建模型统一在此处登记，但不直接复制来源不明的代码。

优先接入方式：

1. 固定 commit 的 Git submodule；
2. 官方 Python 包；
3. 记录来源、版本、许可证和安装步骤的独立本地安装。

任何适配器在代码、配置或权重缺失时都必须给出明确错误，不得假装集成已跑通。

## LivePortrait

- 官方来源：<https://github.com/KlingAIResearch/LivePortrait>
- 固定 commit：`9b294b3d0536135442ea73cb01e6cb3ca7029dd3`
- 代码许可证：MIT
- 官方权重：<https://huggingface.co/KlingTeam/LivePortrait>
- 模型卡许可证：MIT

源码以 Git submodule 形式固定在 `third_party/LivePortrait`。权重不得进入仓库，
统一保存在 `$MODEL_ROOT/liveportrait`。本项目直接读取已有 256×256 人脸裁剪，
不调用 LivePortrait 的 InsightFace 检测器。

```bash
git submodule update --init --recursive
```

权重准备和独立环境见[运动基线说明](../docs/MOTION_BASELINE.md)。FOMM、Wav2Lip
和 SadTalker 当前未集成。
