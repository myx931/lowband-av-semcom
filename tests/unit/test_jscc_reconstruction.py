from pathlib import Path

from av_semcom.models.jscc.reconstruction import _write_plots


def test_empty_reconstruction_summary_does_not_plot(tmp_path: Path) -> None:
    _write_plots(tmp_path / "plots", {"groups": []})

    assert not (tmp_path / "plots").exists()
