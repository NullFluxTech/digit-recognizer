"""Tests for the milestone-5 tools: rasterizer, sample names, drawing window, scorer."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from digits.inference import next_sample_path, parse_sample_name, top_k
from draw import stamp_segment

MODEL = Path(__file__).resolve().parent.parent / "models" / "mlp.npz"


# --- rasterizer ----------------------------------------------------------------------------

def test_stamp_point_is_a_disc():
    ink = np.zeros((21, 21), np.float32)
    stamp_segment(ink, (10, 10), (10, 10), radius=3)
    assert ink[10, 10] == 1 and ink[10, 13] == 1 and ink[13, 10] == 1   # distance 3: inside
    assert ink[10, 14] == 0 and ink[12, 13] == 0                       # distance 4 and 3.6: outside
    assert ink.sum() == 29                                             # lattice points with x^2+y^2 <= 9


def test_stamp_segment_has_constant_width_and_round_ends():
    ink = np.zeros((40, 60), np.float32)
    stamp_segment(ink, (20, 10), (20, 50), radius=4)
    assert np.all(ink[16:25, 10:51] == 1)                              # 9 px thick band along the segment
    assert ink[15, 30] == 0 and ink[25, 30] == 0
    assert ink[20, 54] == 1 and ink[20, 55] == 0                       # round cap reaches exactly r past the end


def test_stamp_segment_clips_at_the_border():
    ink = np.zeros((10, 10), np.float32)
    stamp_segment(ink, (-5, -5), (2, 2), radius=2)                     # starts outside: must not crash
    assert ink[0, 0] == 1 and ink[2, 2] == 1 and ink[9, 9] == 0


# --- sample names ----------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [("7_03.png", (7, 3)), ("0_1.png", (0, 1)),
                                            ("12_01.png", None), ("7-03.png", None), ("7_03.jpg", None)])
def test_parse_sample_name(name, expected):
    assert parse_sample_name(name) == expected


def test_next_sample_path_fills_numbering(tmp_path):
    assert next_sample_path(4, tmp_path).name == "4_01.png"
    for n in (1, 2, 4):
        (tmp_path / f"4_{n:02d}.png").touch()
    (tmp_path / "5_03.png").touch()
    assert next_sample_path(4, tmp_path).name == "4_03.png"            # first free number
    assert next_sample_path(5, tmp_path).name == "5_01.png"


def test_load_samples_preprocesses_and_labels(tmp_path):
    from digits.inference import load_samples

    for name in ("3_02.png", "1_01.png"):
        ink = np.zeros((120, 90), np.float32)
        stamp_segment(ink, (15, 45), (105, 45), radius=6)
        Image.fromarray(np.round((1 - ink) * 255).astype(np.uint8)).save(tmp_path / name)
    (tmp_path / "scan.png").touch()                                    # skipped: wrong name
    X, y, names = load_samples(tmp_path, verbose=False)
    assert X.shape == (2, 784) and X.dtype == np.float32
    assert names == ["1_01.png", "3_02.png"] and y.tolist() == [1, 3]   # sorted by name
    X0, y0, n0 = load_samples(tmp_path / "missing", verbose=False)
    assert X0.shape == (0, 784) and len(y0) == 0 and n0 == []


def test_top_k_orders_by_probability():
    p = np.array([0.1, 0.5, 0.05, 0.3, 0.05])
    assert top_k(p, 3) == [(1, 0.5), (3, 0.3), (0, 0.1)]


# --- drawing window (headless) ------------------------------------------------------------------

@pytest.mark.skipif(not MODEL.exists(), reason="needs models/mlp.npz")
def test_draw_app_predicts_and_saves(tmp_path):
    tk = pytest.importorskip("tkinter")
    from digits.inference import load_model
    from draw import DrawApp

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    try:
        app = DrawApp(root, load_model(MODEL), samples_dir=tmp_path)
        app.paint((40, 140), (240, 140))                    # a vertical stroke: a "1"
        app.update_prediction()
        assert app.guess_labels[0].cget("text").startswith("1")
        app.label.set("1")
        path = app.save()
        assert path is not None and path.name == "1_01.png"
        saved = np.asarray(Image.open(path))
        assert saved.shape == (280, 280) and saved[0, 0] == 255 and saved[140, 140] == 0  # paper style
        assert app.ink.max() == 0                           # canvas cleared after saving
        assert app.save() is None                           # empty canvas is not saved
    finally:
        root.destroy()


# --- scorer end to end -----------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL.exists(), reason="needs models/mlp.npz")
def test_score_samples_end_to_end(tmp_path, monkeypatch, capsys):
    import score_samples

    for name, rows in [("1_01.png", (40, 240)), ("1_02.png", (60, 220))]:
        ink = np.zeros((280, 280), np.float32)
        stamp_segment(ink, (rows[0], 140), (rows[1], 140), radius=10)
        Image.fromarray(np.round((1 - ink) * 255).astype(np.uint8)).save(tmp_path / name)
    (tmp_path / "notes.png").touch()                        # wrong name pattern: skipped, not fatal
    gallery = tmp_path / "reports" / "scored.png"          # never write into the real reports/
    monkeypatch.setattr(sys, "argv", ["score_samples.py", "--dir", str(tmp_path), "--out", str(gallery),
                                      "--model", str(MODEL)])
    score_samples.main()
    printed = capsys.readouterr().out
    assert "skipping notes.png" in printed
    assert "2 of 2 correct" in printed
    assert gallery.exists()
