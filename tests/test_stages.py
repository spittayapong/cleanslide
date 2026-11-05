"""Unit tests for the cleaning stages, on synthetic sections.

These use generated images rather than histology so they run anywhere and carry
no data. Behaviour against real sections is covered by the regression suite.
"""
import cv2
import numpy as np
import pytest

from cleanslide.params import CleanParams, um_to_px
from cleanslide import stages

PIXEL_SIZE_UM = 31.75


@pytest.fixture
def section():
    """A bright slide with a tissue blob, a round bubble and specks.

    The tissue has two intensity bands and mild noise rather than being a flat
    colour: a uniform blob has no percentile spread and no top-hat response, so
    the normalisation and dust stages would correctly do nothing and the tests
    would prove nothing.
    """
    rgb = np.full((600, 600, 3), 250, np.uint8)
    cv2.ellipse(rgb, (300, 300), (200, 240), 0, 0, 360, (150, 130, 170), -1)
    # An inner band standing in for white matter against cortex.
    cv2.ellipse(rgb, (300, 300), (110, 140), 0, 0, 360, (90, 70, 120), -1)

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 6, rgb.shape)
    rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # A bubble: a round hole in the tissue, ~3 mm across at 31.75 um/px.
    radius = int(round(3000.0 * 0.5 / PIXEL_SIZE_UM))
    cv2.circle(rgb, (260, 260), radius, (250, 250, 250), -1)

    for _ in range(40):
        y, x = rng.integers(200, 400, size=2)
        cv2.circle(rgb, (int(x), int(y)), 2, (245, 245, 245), -1)
    return rgb


def test_um_to_px_rounds_and_forces_odd():
    assert um_to_px(1000.0, 31.75) == 31
    assert um_to_px(1000.0, 31.75, odd=True) == 31
    assert um_to_px(635.0, 31.75, odd=True) == 21
    assert um_to_px(1.0, 1000.0) == 1, "never returns zero"
    with pytest.raises(ValueError):
        um_to_px(100.0, 0.0)


def test_params_reject_impossible_values():
    with pytest.raises(ValueError):
        CleanParams(bubble_min_diameter_um=5000, bubble_max_diameter_um=1000)
    with pytest.raises(ValueError):
        CleanParams(bubble_min_circularity=1.5)


def test_background_threshold_tracks_illumination(section):
    """The estimate must move with the background, which is why it replaced the
    hard-coded 252 and 230 of the two prototypes."""
    gray = cv2.cvtColor(section, cv2.COLOR_RGB2GRAY)
    bright = stages.estimate_background_threshold(gray)

    dimmer = np.clip(gray.astype(np.int16) - 30, 0, 255).astype(np.uint8)
    assert stages.estimate_background_threshold(dimmer) < bright


def test_tissue_mask_finds_the_blob_and_excludes_background(section):
    mask, info = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())

    assert mask.shape == section.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[300, 300] == 255, "centre of the blob is tissue"
    assert mask[10, 10] == 0, "slide corner is background"
    assert 0 < info["mask_area_px"] < mask.size


def test_bubble_detected_by_size_and_roundness(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    bubbles, info = stages.find_bubbles(mask, PIXEL_SIZE_UM, CleanParams())

    assert info["bubbles_kept"] >= 1
    assert bubbles[260, 260] == 255

    # The returned tissue mask must still contain the hole: filling is detection
    # only, which is the distinction between this and the dropped fill_holes.
    assert mask[260, 260] == 0


def test_bubble_rejected_when_outside_the_size_band(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    tight = CleanParams(bubble_min_diameter_um=100.0, bubble_max_diameter_um=200.0)
    _, info = stages.find_bubbles(mask, PIXEL_SIZE_UM, tight)
    assert info["bubbles_kept"] == 0


def test_masked_rgb_keeps_colour_and_blacks_out_background(section):
    """The colour output is the stain reference: the inverted, normalised
    grayscale no longer shows which stain produced the contrast."""
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    colour = stages.to_masked_rgb(section, mask)

    assert colour.shape == section.shape
    assert colour[10, 10].max() == 0, "background is black"
    assert np.array_equal(colour[300, 300], section[300, 300]), "tissue unchanged"


def test_pipeline_returns_all_three_outputs(section):
    from cleanslide import clean_section

    result = clean_section(section, PIXEL_SIZE_UM)
    assert result.clean.ndim == 2, "grayscale"
    assert result.mask.ndim == 2 and set(np.unique(result.mask)) <= {0, 255}
    assert result.rgb.ndim == 3 and result.rgb.shape[2] == 3, "colour"
    assert result.rgb.shape[:2] == result.clean.shape == result.mask.shape


def test_to_gray_inverts_inside_and_zeroes_outside(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)

    assert gray.dtype == np.uint8
    assert gray[10, 10] == 0, "background is zero"
    assert gray[300, 300] > 128, "dark tissue becomes bright after inversion"


def test_remove_dust_reduces_speck_response(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)
    cleaned, info = stages.remove_dust(gray, PIXEL_SIZE_UM, CleanParams())

    assert cleaned.shape == gray.shape
    assert info["dust_threshold"] >= 5.0
    assert cleaned.std() <= gray.std() + 1e-6


def test_normalise_stretches_tissue_and_keeps_background_zero(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)
    out, info = stages.normalise_intensity(gray, mask, CleanParams())

    inside = mask > 0
    assert out[~inside].max() == 0, "background must stay zero"
    assert np.ptp(out[inside]) > np.ptp(gray[inside]), "tissue range must widen"
    assert info["normalise_high"] > info["normalise_low"]


def test_normalise_is_scale_invariant(section):
    """A pale stain and a dark one must end at the same brightness: the whole
    reason the stage exists."""
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)

    pale = (gray.astype(np.float32) * 0.4).astype(np.uint8)
    pale[mask == 0] = 0

    bright, _ = stages.normalise_intensity(gray, mask, CleanParams())
    lifted, _ = stages.normalise_intensity(pale, mask, CleanParams())

    assert abs(int(bright[mask > 0].mean()) - int(lifted[mask > 0].mean())) < 12


def test_dust_threshold_uses_tissue_pixels_only(section):
    """With a large zero background the whole-frame median and MAD collapse to
    zero, dropping the threshold to its floor and inpainting real tissue."""
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)

    _, masked = stages.remove_dust(gray, PIXEL_SIZE_UM, CleanParams(), mask)
    _, whole = stages.remove_dust(gray, PIXEL_SIZE_UM, CleanParams())

    assert masked["dust_threshold"] > whole["dust_threshold"]
    assert masked["dust_area_px"] < whole["dust_area_px"]


def test_suppress_folds_keeps_shape_and_background(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)
    opened = stages.suppress_folds(gray, mask, PIXEL_SIZE_UM, CleanParams())

    assert opened.shape == gray.shape
    assert opened.dtype == np.uint8
    assert opened[10, 10] == 0


def test_clahe_leaves_background_untouched(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)
    out = stages.enhance_clahe(gray, mask, PIXEL_SIZE_UM, CleanParams())

    assert out[mask == 0].max() == 0, "equalisation must not touch background"


def test_sigmoid_estimates_its_own_cutoff(section):
    mask, _ = stages.tissue_mask(section, PIXEL_SIZE_UM, CleanParams())
    gray = stages.to_gray(section, mask)
    out, info = stages.enhance_sigmoid(gray, mask, CleanParams())

    assert out.dtype == np.uint8
    assert 0.0 < info["sigmoid_cutoff"] < 1.0


def test_no_anti_alias_blur_is_applied():
    """Pre-smoothing for a downstream resampling step belongs to that step. The
    package must not soften the saved image for one assumed target resolution."""
    assert not hasattr(stages, "presmooth")
    assert "target_pixel_size_um" not in CleanParams().__dict__


@pytest.mark.parametrize("pixel_size_um", [4.0, 31.75])
def test_parameters_transfer_across_resolution(pixel_size_um):
    """The same um parameters must describe the same physical size at either
    scanner resolution - the property that lets one config span both datasets."""
    params = CleanParams()
    blur_um = um_to_px(params.mask_blur_um, pixel_size_um) * pixel_size_um
    assert abs(blur_um - params.mask_blur_um) < pixel_size_um
