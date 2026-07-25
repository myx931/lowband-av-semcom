"""Print verified GRID download instructions without downloading data."""

from __future__ import annotations

from av_semcom.data.cli import build_data_parser

ZENODO_RECORD = "https://zenodo.org/records/3625687"
S1_URL = f"{ZENODO_RECORD}/files/s1.zip?download=1"
AUDIO_URL = f"{ZENODO_RECORD}/files/audio_25k.zip?download=1"


def main() -> int:
    """Print authoritative URLs, checksums, and the expected local layout."""

    parser = build_data_parser("Show manual GRID download instructions.")
    args = parser.parse_args()
    speakers = args.speakers or ["s1"]
    print("GRID must be downloaded manually. This script performs no network writes.")
    print(f"Dataset record and terms: {ZENODO_RECORD}")
    print(f"Requested pilot speakers: {', '.join(speakers)}")
    print(f"s1 video frames: {S1_URL}")
    print("  size: approximately 423.5 MB")
    print("  md5: cbd6556668f061b5c3681bc722659b39")
    print(f"25 kHz audio archive: {AUDIO_URL}")
    print("  size: approximately 2.6 GB")
    print("  md5: 4b3ac37b1a258f55d1eebe657de491a9")
    print("Expected extracted layout:")
    print("  $DATA_ROOT/grid/raw/video/s1/<utterance_id>/*.jpg")
    print("  $DATA_ROOT/grid/raw/audio/s1/<utterance_id>.wav")
    print("Cite DOI: 10.5281/zenodo.3625687")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
