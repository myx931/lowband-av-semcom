# Data

本目录不存放数据集。默认数据根目录由环境变量 `DATA_ROOT` 指定：

```bash
export DATA_ROOT=/path/to/datasets
```

## GRID

首选数据集为公开的 GRID Audio-Visual Sentence Corpus。请从数据集官方发布渠道阅读许可和访问条件，并由研究者手动获取所需说话人的视频/音频；仓库不会自动下载完整数据集。

建议的本地布局（尚未由代码强制）：

```text
$DATA_ROOT/
└── grid/
    ├── raw/
    ├── interim/
    ├── processed/
    └── manifests/
```

Milestone 2 将提供小规模子集处理脚本、按说话人隔离的数据划分和 `manifest.jsonl`/`manifest.csv`。清单至少包含 `sample_id`、`speaker_id`、`video_path`、`audio_path`、`fps`、`sample_rate`、`frame_count`、`split`。

数据准备必须可断点恢复，默认跳过已有输出，并记录失败样本及原因。不要将生成的 manifest（可能暴露本机路径）、人脸裁剪、音频特征或关键点提交到 Git。
