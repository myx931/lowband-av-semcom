"""Print verified GRID download instructions without downloading data."""

from __future__ import annotations

from av_semcom.data.cli import build_data_parser

ZENODO_RECORD = "https://zenodo.org/records/3625687"
AUDIO_URL = f"{ZENODO_RECORD}/files/audio_25k.zip?download=1"
VIDEO_ARCHIVES = {
    "s1": ("423.5 MB", "cbd6556668f061b5c3681bc722659b39"),
    "s2": ("394.6 MB", "36e513652d9abec68c721221ede557df"),
    "s3": ("394.1 MB", "b854132feecda313f0a0c6145131d693"),
    "s4": ("491.9 MB", "1bb4543ca0a27fe76e2874845468a016"),
    "s5": ("407.2 MB", "895d17182889324dd2453aff0ae49083"),
    "s6": ("423.3 MB", "e82a4330653ed81d00c0d2738431e6e7"),
    "s7": ("384.1 MB", "ff31aaddf10bbe345aa1a8434b205fd5"),
    "s8": ("412.9 MB", "af52367c91e96c20cabb0820046dbd73"),
    "s9": ("390.2 MB", "602221e579190b7cb393b4b86a4228bc"),
    "s10": ("429.5 MB", "628b4df6915b379e2c050512f661fa04"),
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
