# Data

本目录不存放数据集。默认数据根目录由环境变量 `DATA_ROOT` 指定：

```bash
export DATA_ROOT=/path/to/datasets
```

## GRID

首选数据集为 GRID Audio-Visual Sentence Corpus。权威数据记录为
[Zenodo 3625687](https://zenodo.org/records/3625687)，引用 DOI：
`10.5281/zenodo.3625687`。下载前请自行阅读记录中的许可和使用条件。

仓库不会自动下载数据。首轮 `s1` pilot 需要手动获取：

| 文件 | 大小 | MD5 |
|---|---:|---|
| `s1.zip` | 约 423.5 MB | `cbd6556668f061b5c3681bc722659b39` |
| `audio_25k.zip` | 约 2.6 GB | `4b3ac37b1a258f55d1eebe657de491a9` |

只查看下载说明（不会产生网络写入）：

```bash
python scripts/data/download_grid_instructions.py --speakers s1
```

Zenodo 的 `s1.zip` 实际包含 `s1/*.mpg`，并不是已经拆好的 JPG。先保留原始
MPG，再用 FFmpeg 以原始 25 fps 拆帧；`audio_25k.zip` 中的音频位于
`audio_25k/s1/*.wav`。整理后的输入为：

```text
$DATA_ROOT/
└── grid/
    ├── raw/
    │   ├── video_mpg/
    │   │   └── s1/
    │   │       └── <utterance_id>.mpg
    │   ├── video/
    │   │   └── s1/
    │   │       └── <utterance_id>/
    │   │           └── *.jpg
    │   └── audio/
    │       └── s1/
    │           └── <utterance_id>.wav
    ├── processed/
    │   ├── audio_features/
    │   ├── landmarks/
    │   └── face_crops/
    └── manifests/
        └── failures/
```

例如，为一个视频拆帧：

```bash
mkdir -p "$DATA_ROOT/grid/raw/video/s1/bbaf2n"
ffmpeg -nostdin -loglevel error \
  -i "$DATA_ROOT/grid/raw/video_mpg/s1/bbaf2n.mpg" \
  -q:v 2 "$DATA_ROOT/grid/raw/video/s1/bbaf2n/%06d.jpg"
```

首轮 pilot 只需对按文件名排序后的最多 20 个视频执行同样操作。每个 GRID
视频应产生 75 张 JPG。FFmpeg 仅用于从官方 MPG 拆帧；后续五阶段直接读取
JPG 和 WAV。

进入正式说话人隔离实验前，最小扩展为 `s1/s2/s3`。已有 `audio_25k.zip`
同时包含这三个说话人的音频，因此只需额外手动下载两个视频压缩包：

| 文件 | 大小 | MD5 |
|---|---:|---|
| `s2.zip` | 约 394.6 MB | `36e513652d9abec68c721221ede557df` |
| `s3.zip` | 约 394.1 MB | `b854132feecda313f0a0c6145131d693` |

```bash
python scripts/data/download_grid_instructions.py --speakers s2 s3
```

这三个说话人只满足管线的最小身份隔离条件；正式模型结果仍应明确说明说话人数
较少，并在算力允许时再扩大说话人覆盖。

多说话人开发子集使用独立配置
[`configs/data/grid_multispeaker.yaml`](../configs/data/grid_multispeaker.yaml)，不会覆盖
已经验收的 `s1 pilot` manifest 和处理产物。默认每位说话人取文件名排序后的前
100 个有效配对；固定种子 42 下，`s3` 为 train、`s1` 为 validation、`s2` 为
test。该划分仅用于验证 E3 音频到运动基线的实现和身份隔离，不能据此宣称跨说话人
泛化已经充分验证。

三个说话人的 JPG 序列整理完成后，运行：

```bash
python scripts/data/prepare_grid_subset.py \
  --config configs/data/grid_multispeaker.yaml
python scripts/data/extract_audio_features.py \
  --config configs/data/grid_multispeaker.yaml
python scripts/data/extract_landmarks.py \
  --config configs/data/grid_multispeaker.yaml
python scripts/data/extract_face_crops.py \
  --config configs/data/grid_multispeaker.yaml
python scripts/data/validate_dataset.py \
  --config configs/data/grid_multispeaker.yaml --require-processed
```

多说话人产物写入 `$DATA_ROOT/grid/processed/multispeaker/`，失败记录写入
`$DATA_ROOT/grid/manifests/failures_multispeaker/`。

## 小子集处理

```bash
export DATA_ROOT=/path/to/datasets

python scripts/data/prepare_grid_subset.py \
  --config configs/data/grid.yaml --speakers s1 --max-samples 20
python scripts/data/extract_audio_features.py --config configs/data/grid.yaml
python scripts/data/extract_landmarks.py --config configs/data/grid.yaml
python scripts/data/extract_face_crops.py --config configs/data/grid.yaml
python scripts/data/validate_dataset.py \
  --config configs/data/grid.yaml --require-processed
```

manifest 使用 JSONL，路径均相对 `DATA_ROOT`。必填字段为 `sample_id`、`speaker_id`、
`video_path`、`audio_path`、`fps`、`sample_rate`、`frame_count`、`split`，处理后追加
音频特征、关键点和裁剪路径。

完成运动提取里程碑后还会追加可选 `motion_path`，指向
`grid/processed/motion/<backend>/` 下的低维运动产物。运动配置和命令见
[LivePortrait 运动基线](../docs/MOTION_BASELINE.md)。

单个实际可用说话人只能标记为 `pilot`，不得用于正式实验结论。正式
train/validation/test 至少需要三个实际存在且成功配对的说话人，并按说话人隔离。

音频产物为与视频帧对齐的 `[T, 4, 80]` log-Mel；关键点为 MediaPipe 的 40 个嘴唇点
`[T, 40, 3]`，同时保存检测掩码和人脸框；裁剪为 `[T, 256, 256, 3]` 的压缩 NumPy
数组。MediaPipe 是首轮工程基线，不代表论文最终运动表示。

数据准备必须可断点恢复，默认跳过已有输出，并记录失败样本及原因。不要将生成的 manifest（可能暴露本机路径）、人脸裁剪、音频特征或关键点提交到 Git。

每个产物都有配置指纹 sidecar。配置一致时重复执行会跳过；配置变化时程序拒绝覆盖，
必须显式传入 `--overwrite`。`--no-resume` 可用于要求输出目录为空的严格运行。
