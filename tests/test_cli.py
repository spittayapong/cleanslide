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
from cleanslide.params import CleanParams

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


# ── where the stain entity comes from ────────────────────────────────────────

def stain_of(out, subject="sub-01", sample="sample-A1"):
    """The stain entity on the written output, or None if it carries none."""
    produced = sorted((out / subject / "micr")
                      .glob(f"{subject}_{sample}*_desc-clean_BF.tif"))
    assert len(produced) == 1, produced
    entities = dict(part.split("-", 1) for part in produced[0].name.split("_")
                    if "-" in part)
    return entities.get("stain")


def test_filename_entity_wins_over_the_sidecar(tmp_path):
    # Both present and disagreeing: BIDS names a file by its entities.
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar={**SIDECAR, "SampleStaining": "HE"})
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant"]) == 0
    assert stain_of(out) == "LFB"


def test_sidecar_supplies_the_stain_when_the_filename_omits_it(tmp_path):
    # Otherwise the derivative would silently drop what the input recorded.
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1",
                  sidecar={**SIDECAR, "SampleStaining": "LFB-PAS"})
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant"]) == 0
    # Entity values must be alphanumeric, so the hyphen goes.
    assert stain_of(out) == "LFBPAS"
    # The sidecar keeps the original spelling.
    meta = json.loads((out / "sub-01" / "micr" /
                       "sub-01_sample-A1_stain-LFBPAS_desc-clean_BF.json").read_text())
    assert meta["SampleStaining"] == "LFB-PAS"


def test_no_stain_anywhere_leaves_the_entity_off(tmp_path):
    # The stain is metadata, not an input: its absence must not stop a run,
    # and nothing is invented to fill the gap.
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1",
                  sidecar={"PixelSize": [PIXEL_SIZE_UM, PIXEL_SIZE_UM],
                           "PixelSizeUnits": "um"})
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant"]) == 0
    assert stain_of(out) is None
    assert (out / "sub-01" / "micr" / "sub-01_sample-A1_desc-clean_BF.tif").exists()


# ── the component size threshold ─────────────────────────────────────────────

def test_min_component_mm2_is_adjustable(tmp_path):
    # Detached tissue can be far smaller than the debris the default rejects,
    # so the threshold has to be reachable without editing the package.
    micr = tmp_path / "sub-01" / "micr"
    micr.mkdir(parents=True)
    # 71 mm2 and 11 mm2 at this pixel size, both wide enough to survive the
    # 1.1 mm opening that runs before any component filter.
    rgb = np.full((600, 600, 3), 250, np.uint8)
    cv2.circle(rgb, (200, 300), 150, (120, 100, 140), -1)    # the section
    cv2.circle(rgb, (480, 300), 60, (120, 100, 140), -1)     # a detached piece
    cv2.imwrite(str(micr / "sub-01_sample-A1_stain-LFB_BF.tif"), rgb)
    (micr / "sub-01_sample-A1_stain-LFB_BF.json").write_text(json.dumps(SIDECAR))

    def components(out_dir, *extra):
        assert main([str(tmp_path), str(out_dir), "participant", *extra]) == 0
        meta = json.loads((out_dir / "sub-01" / "micr" /
                           "sub-01_sample-A1_stain-LFB_desc-clean_BF.json").read_text())
        return meta["Measurements"]["mask_components"]

    strict = components(tmp_path / "strict", "--min-component-mm2", "50")
    keep_all = components(tmp_path / "keep_all", "--min-component-mm2", "0")

    assert strict == 1          # 11 mm2 is below the threshold, so it goes
    assert keep_all == 2        # and is kept when the threshold allows it


# ── where the background threshold comes from ────────────────────────────────

def threshold_used(out_dir):
    meta = json.loads((out_dir / "sub-01" / "micr" /
                       "sub-01_sample-A1_stain-LFB_desc-clean_BF.json").read_text())
    return meta["Measurements"]["background_threshold"], meta["Parameters"]


def test_default_applies_when_the_sidecar_says_nothing(tmp_path):
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB")
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant"]) == 0
    used, _ = threshold_used(out)
    assert used == CleanParams().background_threshold


def test_sidecar_overrides_the_default(tmp_path):
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar={**SIDECAR, "BackgroundThreshold": 212})
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant"]) == 0
    used, params = threshold_used(out)
    assert used == 212
    # Recorded as asked for as well as used, so a reader can tell a declared
    # level from the estimator happening to land on the same number.
    assert params["background_threshold"] == 212


def test_command_line_overrides_the_sidecar(tmp_path):
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar={**SIDECAR, "BackgroundThreshold": 212})
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant",
                 "--background-threshold", "228"]) == 0
    used, _ = threshold_used(out)
    assert used == 228


def test_auto_restores_the_per_section_estimate(tmp_path):
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar={**SIDECAR, "BackgroundThreshold": 212})
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant",
                 "--background-threshold", "auto"]) == 0
    used, params = threshold_used(out)
    assert params["background_threshold"] is None
    assert used != 212, "the estimate replaced the level the sidecar asked for"


def test_a_sidecar_can_ask_for_estimation(tmp_path):
    # How a stain that no fixed level suits travels with its own data.
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar={**SIDECAR, "BackgroundThreshold": "auto"})
    out = tmp_path / "derivative"

    assert main([str(tmp_path), str(out), "participant"]) == 0
    _, params = threshold_used(out)
    assert params["background_threshold"] is None


@pytest.mark.parametrize("value", [300, -1, "nonsense", True])
def test_an_unusable_sidecar_threshold_is_rejected(tmp_path, capsys, value):
    write_section(tmp_path / "sub-01" / "micr", "sub-01_sample-A1_stain-LFB",
                  sidecar={**SIDECAR, "BackgroundThreshold": value})

    code = main([str(tmp_path), str(tmp_path / "out"), "participant"])

    assert code == 2
    assert "BackgroundThreshold" in capsys.readouterr().err
