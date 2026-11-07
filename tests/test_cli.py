"""Command-line tests: exit codes, error messages and what lands on disk.

``main`` returns an exit code rather than calling ``sys.exit``, so every case
here runs in-process against a BIDS tree built in a temporary directory. The
sections are tiny synthetic images, so the suite needs no data and carries
none.

What these cannot cover is packaging — that the wheel installs, that the
console script resolves, and that ``--qc`` on an install without the ``qc``
extra exits with the hint. matplotlib is present whenever the suite runs, so
testing that here would only exercise a mock. It belongs in a real install.
"""
import json

import cv2
import numpy as np
import pytest

from cleanslide.cli import main

PIXEL_SIZE_UM = 31.75
SIDECAR = {"PixelSize": [PIXEL_SIZE_UM, PIXEL_SIZE_UM],
           "PixelSizeUnits": "um", "SampleStaining": "LFB"}


def write_section(micr, name, sidecar=SIDECAR, image=True):
    """One section plus its sidecar; either may be skipped to make it invalid."""
    micr.mkdir(parents=True, exist_ok=True)
    if image:
        rgb = np.full((120, 120, 3), 250, np.uint8)
        # A dark blob, so there is something for the mask to find.
        cv2.circle(rgb, (60, 60), 40, (120, 100, 140), -1)
        cv2.imwrite(str(micr / f"{name}_BF.tif"), rgb)
    if sidecar is not None:
        (micr / f"{name}_BF.json").write_text(json.dumps(sidecar))


@pytest.fixture
def dataset(tmp_path):
    """A two-subject dataset that cleans without error."""
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB")
    write_section(tmp_path / "sub-02" / "micr", "sub-02_sample-B1_stain-HE")
    return tmp_path


def run(dataset, tmp_path, *extra):
    out = tmp_path / "derivative"
    code = main([str(dataset), str(out), "participant", *extra])
    return code, out


# ── the happy path ───────────────────────────────────────────────────────────

def test_clean_run_writes_every_declared_output(dataset, tmp_path):
    code, out = run(dataset, tmp_path)

    assert code == 0
    micr = out / "sub-01" / "micr"
    stem = "sub-01_sample-A1_stain-LFB"
    for name in (f"{stem}_desc-clean_BF.tif",
                 f"{stem}_desc-tissue_mask.tif",
                 f"{stem}_desc-cleanrgb_BF.tif",
                 f"{stem}_desc-clean_BF.json"):
        assert (micr / name).exists(), name
    # BIDS requires this of a derivative, and a consumer reads it first.
    assert (out / "dataset_description.json").exists()


def test_sidecar_carries_the_scale_and_the_schema_version(dataset, tmp_path):
    _, out = run(dataset, tmp_path)
    meta = json.loads((out / "sub-01" / "micr" /
                       "sub-01_sample-A1_stain-LFB_desc-clean_BF.json").read_text())

    assert meta["PixelSize"] == [PIXEL_SIZE_UM, PIXEL_SIZE_UM]
    assert meta["PixelSizeUnits"] == "um"
    assert meta["SampleStaining"] == "LFB"
    assert meta["OutputFormatVersion"] >= 1
    assert meta["GeneratedBy"][0]["Name"] == "cleanslide"
    # Everything the run estimated, so a result can be judged without rerunning.
    assert "mask_area_mm2" in meta["Measurements"]
    assert meta["Parameters"]["mask_blur_um"] > 0


def test_participant_label_restricts_the_run(dataset, tmp_path):
    code, out = run(dataset, tmp_path, "--participant-label", "01")

    assert code == 0
    assert (out / "sub-01").is_dir()
    assert not (out / "sub-02").exists()


def test_sigmoid_enhancement_is_selectable(dataset, tmp_path):
    code, out = run(dataset, tmp_path, "--enhance", "sigmoid")
    meta = json.loads((out / "sub-01" / "micr" /
                       "sub-01_sample-A1_stain-LFB_desc-clean_BF.json").read_text())

    assert code == 0
    assert meta["Measurements"]["enhance"] == "sigmoid"


def test_qc_panels_are_written_beside_the_derivative(dataset, tmp_path):
    pytest.importorskip("matplotlib")
    code, out = run(dataset, tmp_path, "--qc")

    assert code == 0
    qc = out / "qc"
    assert (qc / "sub-01_sample-A1_stain-LFB_qc.png").exists()
    assert (qc / "sub-01_sample-A1_stain-LFB_outputs.png").exists()


def test_qc_dir_can_be_redirected(dataset, tmp_path):
    pytest.importorskip("matplotlib")
    panels = tmp_path / "panels"
    code = main([str(dataset), str(tmp_path / "derivative"), "participant",
                 "--qc", "--qc-dir", str(panels)])

    assert code == 0
    assert list(panels.glob("*_qc.png"))


# ── setup failures: exit 2, before anything is processed ─────────────────────

def test_missing_dataset_is_rejected(tmp_path):
    assert main([str(tmp_path / "absent"), str(tmp_path / "out"),
                 "participant"]) == 2


def test_dataset_without_subjects_is_rejected(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("no sub- directories here")

    code = main([str(tmp_path), str(tmp_path / "out"), "participant"])

    assert code == 2
    assert "no microscopy sections" in capsys.readouterr().err


def test_section_without_a_sidecar_is_rejected(tmp_path, capsys):
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar=None)

    code = main([str(tmp_path), str(tmp_path / "out"), "participant"])

    assert code == 2
    # The scale cannot be guessed, so this must fail rather than assume one.
    assert "PixelSize" in capsys.readouterr().err


@pytest.mark.parametrize("sidecar, expected", [
    ({"PixelSizeUnits": "um"}, "PixelSize"),
    ({"PixelSize": [4.0, 4.0]}, "PixelSize"),
    ({"PixelSize": [4.0, 4.0], "PixelSizeUnits": "furlong"}, "mm, um or nm"),
    ({"PixelSize": [4.0, 8.0], "PixelSizeUnits": "um"}, "anisotropic"),
])
def test_unusable_pixel_size_is_rejected(tmp_path, capsys, sidecar, expected):
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar=sidecar)

    code = main([str(tmp_path), str(tmp_path / "out"), "participant"])

    assert code == 2
    assert expected in capsys.readouterr().err


# ── per-section failure: exit 1, the rest of the run still completes ─────────

def test_one_unreadable_section_does_not_abandon_the_others(dataset, tmp_path,
                                                            capsys):
    micr = dataset / "sub-02" / "micr"
    (micr / "sub-02_sample-B1_stain-HE_BF.tif").write_bytes(b"not a TIFF")

    code, out = run(dataset, tmp_path)

    assert code == 1
    assert "FAILED" in capsys.readouterr().err
    # The healthy subject is still published.
    assert (out / "sub-01" / "micr" /
            "sub-01_sample-A1_stain-LFB_desc-clean_BF.tif").exists()


# ── entities ─────────────────────────────────────────────────────────────────

def test_a_section_without_a_stain_entity_is_processed(tmp_path):
    # The stain is metadata, not an input: its absence must not stop a run.
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1",
                  sidecar={"PixelSize": [PIXEL_SIZE_UM, PIXEL_SIZE_UM],
                           "PixelSizeUnits": "um"})
    out = tmp_path / "derivative"

    code = main([str(tmp_path), str(out), "participant"])

    assert code == 0
    # The output simply loses the entity too, rather than inventing one.
    assert (out / "sub-01" / "micr" / "sub-01_sample-A1_desc-clean_BF.tif").exists()
