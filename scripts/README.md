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

`data/extract_grid_frames.py` 从官方 MPG 原子生成固定 75 帧 JPG 序列；
`data/extract_grid_synced_audio.py` 从同一 MPG 原子提取同步 PCM 音轨，并生成独立
manifest。`audio_25k` 的变长 WAV 不得再作为 75 帧视频的直接对齐输入。

运动提取与重建实验入口位于 `motion/`。真实 LivePortrait 命令必须在独立 GPU
环境运行；CPU smoke 使用注入式 fake 后端，不加载第三方权重。

E3 的因果音频到运动训练入口为 `train/train_audio_to_motion.py`。运动层独立验证和
冻结 LivePortrait 重建评价分别位于 `eval/evaluate_audio_to_motion.py` 和
`eval/reconstruct_audio_to_motion.py`。训练与重建使用同一个被 Git 忽略的运行
目录，但分别在 Python 3.11 CUDA 训练环境和 Python 3.10 LivePortrait 环境执行。

长时间重建任务可在另一个终端只读查看，不会修改实验产物：

```bash
PYTHONPATH=src python scripts/eval/show_reconstruction_progress.py \
  --run-dir outputs/audio_to_motion_ten_speaker/<timestamp>
```

增加 `--watch 10` 每 10 秒刷新；增加 `--json` 输出机器可读快照。

E4 先在 Python 3.11 环境运行不需要 GPU 的残差分析：

```bash
PYTHONPATH=src python scripts/eval/analyze_prediction_residuals.py \
  --config configs/experiment/residual_baseline_ten_speaker.yaml \
  --e3-run-dir outputs/audio_to_motion_ten_speaker/<timestamp>
```

随后在 LivePortrait Python 3.10 环境运行冻结重建评价：

```bash
PYTHONPATH=src python scripts/eval/reconstruct_prediction_residuals.py \
  --config configs/experiment/residual_baseline_ten_speaker.yaml \
  --e3-run-dir outputs/audio_to_motion_ten_speaker/<timestamp> \
  --run-dir outputs/residual_baseline/<timestamp> \
  --resume --reconstruction-batch-size 56 --metric-workers 8
```

横轴 `K` 表示每个有效非参考帧保留的残差元素数，不是经过编码和信道传输后的
真实码率。实验同时报告原始幅度、训练集归一化幅度和随机固定预算选择。正式
十说话人运行应产生 200 个样本文件和 4,600 条重建指标，完成标记中的
`sample_count/result_count` 应为 `200/4600`，失败数应为 0；匹配完成标记的
`--resume` 不得改写已有产物。

E5 在独立 Python 3.11 Sionna 环境中训练和评价复数 AWGN 残差 JSCC：

```bash
PYTHONPATH=src "$SIONNA_PYTHON" scripts/train/train_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e3-run-dir outputs/audio_to_motion_ten_speaker/<timestamp>

PYTHONPATH=src "$SIONNA_PYTHON" scripts/eval/evaluate_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --e3-run-dir outputs/audio_to_motion_ten_speaker/<timestamp> \
  --run-dir outputs/residual_jscc/<timestamp>

PYTHONPATH=src "$SIONNA_PYTHON" scripts/eval/report_residual_jscc.py \
  --config configs/experiment/residual_jscc_ten_speaker.yaml \
  --run-dir outputs/residual_jscc/<timestamp>
```

正式配置必须使用 `channel.backend: sionna`。`C` 表示每个有效非参考帧的复数
信道符号数，不是比特率；详细冻结协议见 `docs/JSCC_BASELINE.md`。
