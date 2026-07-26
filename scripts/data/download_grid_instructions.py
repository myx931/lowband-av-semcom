"""Print verified GRID download instructions without downloading data."""

from __future__ import annotations

from av_semcom.data.cli import build_data_parser

ZENODO_RECORD = "https://zenodo.org/records/3625687"
AUDIO_URL = f"{ZENODO_RECORD}/files/audio_25k.zip?download=1"
VIDEO_ARCHIVES = {
    "s1": ("423.5 MB", "cbd6556668f061b5c3681bc722659b39"),
    "s2": ("394.6 MB", "36e513652d9abec68c721221ede557df"),
    "s3": ("394.1 MB", "b854132feecda313f0a0c6145131d693"),
}


def main() -> int:
    """Print authoritative URLs, checksums, and the expected local layout."""

    parser = build_data_parser("Show manual GRID download instructions.")
    args = parser.parse_args()
    speakers = args.speakers or ["s1"]
    unsupported = sorted(set(speakers) - VIDEO_ARCHIVES.keys())
    if unsupported:
        parser.error("verified archive metadata is unavailable for: " + ", ".join(unsupported))
    print("GRID must be downloaded manually. This script performs no network writes.")
    print(f"Dataset record and terms: {ZENODO_RECORD}")
    print(f"Requested speakers: {', '.join(speakers)}")
    for speaker in speakers:
        size, checksum = VIDEO_ARCHIVES[speaker]
        url = f"{ZENODO_RECORD}/files/{speaker}.zip?download=1"
        print(f"{speaker} video archive (contains {speaker}/*.mpg): {url}")
        print(f"  size: approximately {size}")
        print(f"  md5: {checksum}")
    print(f"Optional endpointed 25 kHz audio archive: {AUDIO_URL}")
    print("  size: approximately 2.6 GB")
    print("  md5: 4b3ac37b1a258f55d1eebe657de491a9")
    print("  do not use these variable-duration WAV files as full-video aligned audio")
    print("Expected extracted layout:")
    for speaker in speakers:
        print(f"  $DATA_ROOT/grid/raw/video_mpg/{speaker}/<utterance_id>.mpg")
        print(f"  $DATA_ROOT/grid/raw/video/{speaker}/<utterance_id>/*.jpg")
        print(f"  $DATA_ROOT/grid/raw/audio_synced/{speaker}/<utterance_id>.wav")
    print("Use FFmpeg to extract both JPG frames and the synchronized MPG audio track.")
    print("Cite DOI: 10.5281/zenodo.3625687")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
