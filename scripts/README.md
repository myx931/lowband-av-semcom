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

`data/extract_grid_synced_audio.py` 从官方 MPG 原子提取同步 PCM 音轨，并生成独立
manifest。`audio_25k` 的变长 WAV 不得再作为 75 帧视频的直接对齐输入。

运动提取与重建实验入口位于 `motion/`。真实 LivePortrait 命令必须在独立 GPU
环境运行；CPU smoke 使用注入式 fake 后端，不加载第三方权重。

E3 的因果音频到运动训练入口为 `train/train_audio_to_motion.py`。运动层独立验证和
冻结 LivePortrait 重建评价分别位于 `eval/evaluate_audio_to_motion.py` 和
`eval/reconstruct_audio_to_motion.py`。训练与重建使用同一个被 Git 忽略的运行
目录，但分别在 Python 3.11 CUDA 训练环境和 Python 3.10 LivePortrait 环境执行。
