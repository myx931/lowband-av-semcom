"""Print official LivePortrait source and weight setup without downloading."""

from __future__ import annotations


def main() -> int:
    print("This script performs no downloads.")
    print("Pinned source: https://github.com/KlingAIResearch/LivePortrait")
    print("Pinned commit: 9b294b3d0536135442ea73cb01e6cb3ca7029dd3")
    print("Official weights: https://huggingface.co/KlingTeam/LivePortrait")
    print("Code and model-card license: MIT")
    print("Expected local command:")
    print(
        "  huggingface-cli download KlingTeam/LivePortrait "
        '--local-dir "$MODEL_ROOT/liveportrait" '
        '--exclude "*.git*" "README.md" "docs"'
    )
    print("Weights and generated media must remain outside Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
