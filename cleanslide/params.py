"""Parameters for the cleaning stages.

Every size is a physical quantity in micrometres and is converted to pixels from
the section's own pixel size. Values that depend on the scan rather than on the tissue
— illumination, noise level, stain darkness — are not here: they are estimated per section.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CleanParams:
    """Physical sizes for every stage. Override individually, e.g.
    ``CleanParams(bubble_max_diameter_um=6000)``."""

    # ── Tissue mask ──────────────────────────────────────────────────────────
    # Grey level above which a pixel is slide rather than tissue. A fixed level
    # gives a tighter mask than the estimate does, but it only holds for a
    # given scanner: measured backgrounds across two datasets run from 241 to
    # 254, and a level well below a section's own background starts taking the
    # tissue with it. Set to None to estimate it per section instead, which is
    # what a section whose scale or illumination differs will need.
    background_threshold: int | None = 240
    # Smoothing applied to the coarse foreground before Otsu. Large on purpose:
    # it is bridging gaps across tissue, not preserving detail.
    mask_blur_um: float = 1200.0
    # Morphological opening that removes speckle outside the section. Too large
    # and thin real tissue is lost.
    mask_open_um: float = 1100.0
    # Drop mask components that touch the image edge: printed calibration
    # charts and slide labels sit there and are as dark as tissue. The largest
    # component is always kept, so a section whose own tissue runs off the
    # frame survives.
    mask_drop_border_components: bool = True
    # Drop substantial mask components that nearly fill their bounding box and
    # are elongated. Printed charts, scale bars and label blocks are filled
    # rectangles by construction; tissue is rarely both. Both tests must pass,
    # because extent alone also flags compact sections. Set the extent above
    # 1.0 to disable.
    mask_max_extent: float = 0.70
    mask_min_aspect: float = 1.7
    # And the grey levels inside must vary more than tissue ever does. Shape
    # alone deleted whole brain sections: a section cut as a long cortical
    # strip is geometrically a bar, and two such strips were discarded over
    # 0.006 of extent. These charts are step wedges and colour bars, so one
    # component spans many grey levels while tissue is comparatively uniform.
    mask_printed_max_sd: float = 45.0
    # Only components at least this fraction of the largest are tested for
    # rectangularity, so ordinary small debris is judged on its own terms.
    mask_extent_min_area_fraction: float = 0.05
    # Components smaller than this are never tissue sections: they are slide
    # labels, grey-wedge fragments and debris. Set to 0 to keep everything.
    mask_min_component_mm2: float = 100.0

    # Smooth the finished mask boundary. The component filters leave an outline
    # that follows individual pixels, which shows as a ragged edge on the
    # masked tissue; blurring the binary mask and re-thresholding at the half
    # level regularises it without moving the region. Small enough to keep
    # folia tips and gyral crowns.
    mask_smooth_um: float = 150.0

    # ── Cropping ─────────────────────────────────────────────────────────────
    # Trim the outputs to the tissue's bounding box. Most of a slide is empty
    # glass, so this is both the useful region and a real saving. The crop
    # origin is recorded in the sidecar, so coordinates map back to the source
    # scan. Applied after the mask is finished, because the border filter needs
    # the real image edge.
    crop_to_tissue: bool = True
    # Context kept around the tissue. Also keeps it off the crop edge, so the
    # border diagnostics stay meaningful.
    crop_margin_um: float = 500.0

    # ── Bubble removal ───────────────────────────────────────────────────────
    # Mounting-medium bubbles are round holes in the tissue. Diameter bounds are
    # deliberately loose because circularity is the discriminating test.
    bubble_min_diameter_um: float = 2500.0
    bubble_max_diameter_um: float = 4500.0
    # 1.0 is a perfect circle. Folds and tears score far below this.
    bubble_min_circularity: float = 0.7
    # Grown slightly before inpainting so the bubble rim goes too.
    bubble_dilate_um: float = 100.0

    # ── Fold suppression ─────────────────────────────────────────────────────
    # Grayscale opening deletes bright structures narrower than this: thin folds
    # and the cerebellar molecular layer. Wide white matter survives intact.
    fold_open_um: float = 1000.0
    # Morphology runs on a decimated copy; 4x is ~16x faster and the structuring
    # element is far larger than the decimation.
    fold_decimate: int = 4

    # ── Intensity ────────────────────────────────────────────────────────────
    # CLAHE tiles sized in um rather than as a fixed grid, so one value means
    # the same physical tile at any scan resolution.
    clahe_tile_um: float = 12000.0
    clahe_clip_limit: float = 1.0
    # Stretch the tissue to the full 0-255 range before enhancement. Without it
    # output brightness tracks how dark the stain happens to be, so a pale
    # section and a dark one enhance differently. Percentiles rather than
    # min/max, so a few extreme pixels cannot set the scale.
    normalise: bool = True
    normalise_low_percentile: float = 1.0
    normalise_high_percentile: float = 99.0
    # Sigmoid steepness. Its centre is estimated per section, not fixed.
    sigmoid_gain: float = 30.0
    # "clahe" (local equalisation) or "sigmoid" (S-curve about an estimated
    # white-matter/cortex boundary).
    enhance: str = "clahe"

    # ── Stages that are off by default ───────────────────────────────────────
    # Fold suppression deletes bright structures narrower than fold_open_um,
    # which merges the cerebellar molecular and granular layers into one band.
    # Run unconditionally it also removes fine structure from cerebral
    # sections, so it is off until asked for.
    suppress_folds: bool = False

    def __post_init__(self):
        if self.bubble_min_diameter_um >= self.bubble_max_diameter_um:
            raise ValueError("bubble_min_diameter_um must be below the maximum")
        if not 0.0 < self.bubble_min_circularity <= 1.0:
            raise ValueError("bubble_min_circularity must lie in (0, 1]")
        if self.fold_decimate < 1:
            raise ValueError("fold_decimate must be at least 1")
        if self.enhance not in ("clahe", "sigmoid"):
            raise ValueError(f"enhance must be 'clahe' or 'sigmoid', got {self.enhance!r}")
        if not 0.0 <= self.normalise_low_percentile < self.normalise_high_percentile <= 100.0:
            raise ValueError("normalise percentiles must satisfy 0 <= low < high <= 100")


def um_to_px(um: float, pixel_size_um: float, odd: bool = False) -> int:
    """Convert a physical size to pixels, never returning less than one.

    ``odd=True`` is required by OpenCV kernels and structuring elements.
    """
    if pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
    n = max(1, int(round(um / pixel_size_um)))
    if odd and n % 2 == 0:
        n += 1
    return n
