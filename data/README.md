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

`audio_25k.zip`（约 2.6 GB，MD5
`4b3ac37b1a258f55d1eebe657de491a9`）是可选的独立语音资料，不再作为本项目的
全视频同步音频源。本地审计发现其中 298 条 `s1/s2/s3` WAV 时长为
1.12–2.50 秒，而对应视频固定为 3 秒；将这些 WAV 直接插值到 75 帧会错误地改变
语速和时间轴。

只查看下载说明（不会产生网络写入）：

```bash
python scripts/data/download_grid_instructions.py --speakers s1
```

Zenodo 的 `s1.zip` 实际包含 `s1/*.mpg`，并不是已经拆好的 JPG。MPG 同时包含
25 fps 视频和从 0 秒开始的音轨。先保留原始 MPG，再用 FFmpeg 拆出 JPG 和同步
PCM WAV。整理后的输入为：

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
    │   └── audio_synced/
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

每个 GRID 视频应产生 75 张 JPG。同步 WAV 应从同一个 MPG 的音轨提取，禁止把
变长的 `audio_25k` WAV 拉伸到三秒。仓库命令会原子提取 WAV、记录来源 MPG 和
时长，并将音频与既有视觉产物组合到独立 manifest：

```bash
python scripts/data/extract_grid_synced_audio.py \
  --config configs/data/grid_multispeaker_synced.yaml
```

随后 log-Mel 使用 10 ms 绝对时间步：音频尾部不足三秒时只补零，超过三秒时只
截尾，不进行时间插值。音频/视频时长比不在 `0.95..1.05` 时直接记为失败。

进入正式说话人隔离实验前，最小扩展为 `s1/s2/s3`，需要额外手动下载两个包含
同步音轨的视频压缩包：

| 文件 | 大小 | MD5 |
|---|---:|---|
| `s2.zip` | 约 394.6 MB | `36e513652d9abec68c721221ede557df` |
| `s3.zip` | 约 394.1 MB | `b854132feecda313f0a0c6145131d693` |

```bash
python scripts/data/download_grid_instructions.py --speakers s2 s3
```

这三个说话人只满足管线的最小身份隔离条件；正式模型结果仍应明确说明说话人数
较少，并在算力允许时再扩大说话人覆盖。

修正后的多说话人开发子集使用独立配置
[`configs/data/grid_multispeaker_synced.yaml`](../configs/data/grid_multispeaker_synced.yaml)，
不会覆盖旧 manifest、旧特征或 `s1 pilot` 产物。默认每位说话人取文件名排序后的
前 100 个有效样本；固定种子 42 下，`s3` 为 train、`s1` 为 validation、`s2`
为 test。该划分仅用于验证 E3 音频到运动基线的实现和身份隔离，不能据此宣称跨
说话人泛化已经充分验证。

三个说话人的 JPG 序列和 MPG 整理完成后，修复已有三说话人 manifest 时运行：

```bash
python scripts/data/extract_grid_synced_audio.py \
  --config configs/data/grid_multispeaker_synced.yaml
python scripts/data/extract_audio_features.py \
  --config configs/data/grid_multispeaker_synced.yaml
python scripts/data/validate_dataset.py \
  --config configs/data/grid_multispeaker_synced.yaml --require-processed
```

同步音频和特征分别写入 `$DATA_ROOT/grid/raw/audio_synced/` 与
`$DATA_ROOT/grid/processed/multispeaker_synced/`。关键点、裁剪和运动来自同一
视频时间轴，因此新 manifest 安全复用既有视觉产物；失败记录写入
`$DATA_ROOT/grid/manifests/failures_multispeaker_synced/`。

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
