# Scripts

命令行入口按阶段组织：

- `data/`：GRID 下载说明、子集发现、音频特征、关键点、裁剪与验证；
- `baselines/`：运动提取、重建和扰动基线；
- `train/`：训练入口；
- `eval/`：独立评价入口；
- `experiments/`：论文实验编排与绘图。

脚本入口只负责参数解析和调用 `av_semcom` 包，不在脚本中实现核心逻辑。GRID 脚本
共享 `--config`、`--speakers`、`--max-samples`、`--resume/--no-resume` 和
`--overwrite` 参数；完整使用顺序见 `data/README.md`。

运动提取与重建实验入口位于 `motion/`。真实 LivePortrait 命令必须在独立 GPU
环境运行；CPU smoke 使用注入式 fake 后端，不加载第三方权重。
