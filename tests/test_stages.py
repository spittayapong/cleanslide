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
    colour: a uniform blob has no percentile spread, so the normalisation would
    correctly do nothing and the tests would prove nothing.
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


@pytest.fixture
def section_with_chart():
    """A slide carrying a printed calibration chart at the bottom edge.

    Two sections in the real set have one; it is as dark as tissue, so a
    grayscale mask includes it. The gap between tissue and chart is wide
    compared with the mask blur, as it is on a real slide — otherwise the two
    merge into a single component and the question does not arise.
    """
    rgb = np.full((900, 600, 3), 250, np.uint8)
    cv2.ellipse(rgb, (300, 280), (200, 220), 0, 0, 360, (150, 130, 170), -1)
    cv2.ellipse(rgb, (300, 280), (110, 130), 0, 0, 360, (90, 70, 120), -1)

    rng = np.random.default_rng(1)
    rgb = np.clip(rgb.astype(np.float32) + rng.normal(0, 6, rgb.shape),
                  0, 255).astype(np.uint8)

    # Spanning the full slide width and running to the bottom edge, as a
    # printed chart does: that contact with the frame is what identifies it.
    cv2.rectangle(rgb, (0, 760), (599, 810), (200, 40, 60), -1)    # colour bar
    cv2.rectangle(rgb, (0, 820), (599, 899), (40, 40, 40), -1)     # grey wedge
    return rgb


def test_calibration_chart_is_dropped_from_the_mask(section_with_chart):
    chart_point = (840, 300)          # (row, col) inside the printed bars

    with_drop, info = stages.tissue_mask(section_with_chart, PIXEL_SIZE_UM,
                                         CleanParams())
    without, _ = stages.tissue_mask(
        section_with_chart, PIXEL_SIZE_UM,
        CleanParams(mask_drop_border_components=False, mask_max_extent=1.5,
                    mask_min_component_mm2=0.0))

    assert without[chart_point] == 255, "chart is masked as tissue without the fix"
    assert with_drop[chart_point] == 0, "chart must be dropped"
    assert info["border_components_dropped"] >= 1
    assert with_drop[280, 300] == 255, "the section itself must survive"
    assert info["mask_area_px"] < int((without > 0).sum())


def gray_for(mask, *wedges):
    """A grayscale to go with a synthetic mask: flat tissue, step-wedge bars.

    The printed test needs grey levels as well as shape, because a chart is a
    wedge spanning many levels while tissue is comparatively uniform. Each
    wedge is an ``(x0, y0, x1, y1)`` box filled with a full black-to-white
    ramp, which is what a calibration strip looks like.
    """
    gray = np.full(mask.shape, 235, np.uint8)
    gray[mask > 0] = 120
    for x0, y0, x1, y1 in wedges:
        ramp = np.linspace(0, 255, x1 - x0, dtype=np.uint8)
        gray[y0:y1, x0:x1] = np.tile(ramp, (y1 - y0, 1))
    return gray


def test_rectangular_component_is_dropped_but_irregular_tissue_is_kept():
    """Printed matter fills its bounding box; tissue does not. Measured on the
    affected section: brain 0.648, temporal lobes 0.51 and 0.57, chart 0.885."""
    mask = np.zeros((600, 600), np.uint8)
    cv2.ellipse(mask, (300, 180), (200, 140), 0, 0, 360, 255, -1)   # irregular
    cv2.rectangle(mask, (100, 420), (500, 520), 255, -1)            # printed bar

    gray = gray_for(mask, (100, 420, 500, 520))

    out, dropped = stages.drop_rectangular_components(mask, gray, max_extent=0.80)

    assert dropped == 1
    assert out[470, 300] == 0, "the rectangle is dropped"
    assert out[180, 300] == 255, "the ellipse is kept"


def test_rectangular_component_is_dropped_even_when_it_is_the_largest():
    """On one real slide the chart bar outweighs either hemisphere, so rank
    cannot be used to identify tissue."""
    mask = np.zeros((900, 900), np.uint8)
    cv2.rectangle(mask, (50, 600), (850, 880), 255, -1)             # big, printed
    cv2.ellipse(mask, (300, 280), (150, 200), 0, 0, 360, 255, -1)   # smaller, tissue
    cv2.ellipse(mask, (600, 280), (140, 190), 0, 0, 360, 255, -1)

    areas = [int((mask[600:880, 50:850] > 0).sum())]
    gray = gray_for(mask, (50, 600, 850, 880))
    out, dropped = stages.drop_rectangular_components(mask, gray, max_extent=0.80)

    assert areas[0] > 0
    assert dropped == 1
    assert out[740, 450] == 0, "the printed bar is dropped despite being largest"
    assert out[280, 300] == 255 and out[280, 600] == 255, "both lobes survive"


def test_mask_is_never_emptied_by_the_rectangularity_test():
    """If every component looks printed, the test is more likely wrong than the
    slide empty, so the largest survives."""
    mask = np.zeros((600, 600), np.uint8)
    cv2.rectangle(mask, (50, 50), (550, 250), 255, -1)
    cv2.rectangle(mask, (50, 350), (400, 500), 255, -1)
    gray = gray_for(mask, (50, 50, 550, 250), (50, 350, 400, 500))

    out, dropped = stages.drop_rectangular_components(mask, gray, max_extent=0.80)

    assert dropped == 1, "only the smaller one goes"
    assert out[150, 300] == 255, "the largest is kept as a fallback"
    assert (out > 0).any(), "the mask is never emptied"


def test_elongation_is_required_as_well_as_rectangularity():
    """A compact section can reach extent 0.759 while a chart bar sits at 0.778,
    so extent alone cannot separate them. Only printed bars are also elongated."""
    mask = np.zeros((900, 900), np.uint8)
    cv2.rectangle(mask, (40, 100), (860, 290), 255, -1)    # extent 1.0, aspect 4.3
    cv2.rectangle(mask, (300, 450), (700, 820), 255, -1)   # extent 1.0, aspect 1.08

    gray = gray_for(mask, (40, 100, 860, 290), (300, 450, 700, 820))

    out, dropped = stages.drop_rectangular_components(mask, gray,
                                                      max_extent=0.70,
                                                      min_aspect=1.7)
    assert dropped == 1
    assert out[200, 450] == 0, "the elongated bar goes"
    assert out[600, 500] == 255, "the compact block stays"


def test_small_components_are_removed_by_physical_area():
    mask = np.zeros((800, 800), np.uint8)
    cv2.ellipse(mask, (400, 400), (250, 300), 0, 0, 360, 255, -1)   # a section
    cv2.rectangle(mask, (20, 20), (60, 44), 255, -1)                # a label fragment

    out, dropped = stages.drop_small_components(mask, PIXEL_SIZE_UM,
                                                min_area_mm2=100.0)
    assert dropped == 1
    assert out[32, 40] == 0
    assert out[400, 400] == 255


def test_small_component_filter_never_empties_the_mask():
    mask = np.zeros((400, 400), np.uint8)
    cv2.rectangle(mask, (10, 10), (40, 40), 255, -1)
    cv2.rectangle(mask, (100, 100), (118, 118), 255, -1)

    out, dropped = stages.drop_small_components(mask, PIXEL_SIZE_UM,
                                                min_area_mm2=10000.0)
    assert (out > 0).any(), "the largest survives even when all look too small"
    assert out[25, 25] == 255


def test_small_component_filter_can_be_disabled():
    mask = np.zeros((400, 400), np.uint8)
    cv2.rectangle(mask, (10, 10), (40, 40), 255, -1)
    out, dropped = stages.drop_small_components(mask, PIXEL_SIZE_UM, 0.0)
    assert dropped == 0 and np.array_equal(out, mask)


def test_rectangularity_filter_can_be_disabled():
    mask = np.zeros((600, 600), np.uint8)
    cv2.ellipse(mask, (300, 180), (200, 140), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (100, 420), (500, 520), 255, -1)

    gray = gray_for(mask, (100, 420, 500, 520))
    out, dropped = stages.drop_rectangular_components(mask, gray, max_extent=1.5)
    assert dropped == 0 and np.array_equal(out, mask)


def test_small_debris_is_not_judged_on_rectangularity():
    """A tiny speck fills its bounding box too; the area floor keeps the test
    from being applied where it is meaningless."""
    mask = np.zeros((600, 600), np.uint8)
    cv2.ellipse(mask, (300, 300), (200, 150), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (10, 10), (18, 18), 255, -1)                # tiny square
    gray = gray_for(mask, (10, 10, 18, 18))

    out, dropped = stages.drop_rectangular_components(mask, gray, max_extent=0.80)
    assert dropped == 0, "below the area fraction, so not tested"
    assert out[14, 14] == 255


def test_largest_component_is_kept_even_when_it_touches_the_border():
    """A section whose own tissue runs off the frame must not be deleted."""
    mask = np.zeros((400, 400), np.uint8)
    mask[100:400, 100:300] = 255          # large, touches the bottom edge
    mask[0:20, 0:20] = 255                # small speck at the corner

    out, dropped = stages.drop_border_components(mask)

    assert out[350, 200] == 255, "the large border-touching region is kept"
    assert out[10, 10] == 0, "the corner speck is dropped"
    assert dropped == 1


def test_tissue_bbox_tracks_the_tissue_with_a_margin():
    mask = np.zeros((600, 600), np.uint8)
    mask[200:400, 150:350] = 255

    tight = stages.tissue_bbox(mask, PIXEL_SIZE_UM, margin_um=0)
    assert tight == (200, 400, 150, 350)

    margin_px = um_to_px(500.0, PIXEL_SIZE_UM)
    y0, y1, x0, x1 = stages.tissue_bbox(mask, PIXEL_SIZE_UM, margin_um=500.0)
    assert (y0, y1, x0, x1) == (200 - margin_px, 400 + margin_px,
                                150 - margin_px, 350 + margin_px)

    assert stages.tissue_bbox(np.zeros((10, 10), np.uint8), PIXEL_SIZE_UM, 0) is None


def test_bbox_is_clipped_to_the_image(section):
    mask = np.zeros((100, 100), np.uint8)
    mask[0:100, 0:100] = 255
    assert stages.tissue_bbox(mask, PIXEL_SIZE_UM, margin_um=5000.0) == (0, 100, 0, 100)


def test_cropping_shrinks_the_outputs_and_records_the_origin(section):
    from cleanslide import clean_section

    cropped = clean_section(section, PIXEL_SIZE_UM)
    full = clean_section(section, PIXEL_SIZE_UM, CleanParams(crop_to_tissue=False))

    assert cropped.clean.shape < full.clean.shape
    assert cropped.clean.shape == cropped.mask.shape == cropped.rgb.shape[:2]
    # Without the origin a consumer cannot map a cropped coordinate back to the
    # scan it came from, which registration needs.
    assert cropped.info["crop_origin_xy"][0] >= 0
    assert cropped.info["source_shape_hw"] == list(section.shape[:2])
    assert cropped.info["output_shape_hw"] == list(cropped.mask.shape)
    assert full.info["source_shape_hw"] == full.info["output_shape_hw"]
    assert "crop_origin_xy" not in full.info


def test_cropping_does_not_change_the_tissue_itself(section):
    """The crop must move the window, not alter what is inside it."""
    from cleanslide import clean_section

    cropped = clean_section(section, PIXEL_SIZE_UM)
    full = clean_section(section, PIXEL_SIZE_UM, CleanParams(crop_to_tissue=False))

    x0, y0 = cropped.info["crop_origin_xy"]
    h, w = cropped.info["output_shape_hw"]
    assert np.array_equal(cropped.mask, full.mask[y0:y0 + h, x0:x0 + w])


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


def test_mask_edge_smoothing_regularises_without_moving_the_region():
    """A ragged outline reads badly once the mask is applied to the tissue, but
    smoothing must not shift where the tissue is."""
    rng = np.random.default_rng(3)
    mask = np.zeros((600, 600), np.uint8)
    cv2.circle(mask, (300, 300), 200, 255, -1)
    # Chew the boundary the way the component filters do.
    noise = rng.random(mask.shape) < 0.25
    ring = cv2.dilate(mask, np.ones((9, 9), np.uint8)) - cv2.erode(mask, np.ones((9, 9), np.uint8))
    mask[(ring > 0) & noise] = 0

    out = stages.smooth_mask_edge(mask, PIXEL_SIZE_UM, 150.0)

    def perimeter(m):
        c, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
        return sum(cv2.arcLength(x, True) for x in c)

    assert perimeter(out) < perimeter(mask), "the outline must get smoother"
    assert abs(int((out > 0).sum()) - int((mask > 0).sum())) < 0.06 * (mask > 0).sum()
    assert out[300, 300] == 255 and out[10, 10] == 0


def test_mask_edge_smoothing_can_be_disabled():
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 120, 255, -1)
    assert np.array_equal(stages.smooth_mask_edge(mask, PIXEL_SIZE_UM, 0.0), mask)


def test_kept_stages_open_with_the_untouched_input(section):
    """The QC panel leads with the section as it arrived.

    Uncropped, unmasked and identical to what was passed in, so the panels
    after it can be read as changes to a visible starting point.
    """
    from cleanslide import clean_section
    from cleanslide.qc import PANEL_ORDER

    result = clean_section(section, PIXEL_SIZE_UM, keep_stages=True)

    assert next(iter(result.intermediates)) == "input"
    assert PANEL_ORDER[0] == "input"
    assert np.array_equal(result.intermediates["input"], section)
    # It is the source frame, not a stage output: the rest are cropped.
    assert result.intermediates["input"].shape[:2] == section.shape[:2]
    assert result.intermediates["tissue_mask"].shape < section.shape[:2]


def test_input_is_not_kept_without_keep_stages(section):
    from cleanslide import clean_section

    assert clean_section(section, PIXEL_SIZE_UM).intermediates == {}


def test_no_dust_stage_exists():
    """Dust removal was removed: it inpainted 5-6% of the tissue and the loss
    was visible in the middle of sections."""
    from cleanslide.pipeline import STAGE_ORDER
    assert not hasattr(stages, "remove_dust")
    assert "remove_dust" not in STAGE_ORDER
    assert not any(f.startswith("dust") for f in CleanParams().__dict__)
