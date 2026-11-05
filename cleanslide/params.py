"""Parameters for the cleaning stages.

Every size is a physical quantity in micrometres and is converted to pixels from
the section's own pixel size. Values that depend on the scan rather than on the tissue 
— illumination, noise level, stain darkness — are not here: they are estimated per section.

Defaults come from the two prototype datasets.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CleanParams:
    """Physical sizes for every stage. Override individually, e.g.
    ``CleanParams(bubble_max_diameter_um=6000)``."""

    # ── Tissue mask ──────────────────────────────────────────────────────────
    # Smoothing applied to the coarse foreground before Otsu. Large on purpose:
    # it is bridging gaps across tissue, not preserving detail.
    mask_blur_um: float = 1200.0
    # Morphological opening that removes speckle outside the section. Too large
    # and thin real tissue is lost; the prototypes settled near 1.1 mm.
    mask_open_um: float = 1100.0

    # ── Bubble removal ───────────────────────────────────────────────────────
    # Mounting-medium bubbles are round holes in the tissue. Diameter bounds are
    # deliberately loose because circularity is the discriminating test.
    bubble_min_diameter_um: float = 2500.0
    bubble_max_diameter_um: float = 4500.0
    # 1.0 is a perfect circle. Folds and tears score far below this.
    bubble_min_circularity: float = 0.7
    # Grown slightly before inpainting so the bubble rim goes too.
    bubble_dilate_um: float = 100.0

    # ── Dust removal ─────────────────────────────────────────────────────────
    # Top-hat structuring element: bright specks smaller than this are detected.
    dust_tophat_um: float = 290.0
    # Detection threshold as a multiple of the MAD of the top-hat response, so it
    # adapts to each section's noise instead of assuming an absolute grey level.
    dust_mad_k: float = 3.0
    dust_inpaint_radius_um: float = 95.0

    # ── Fold suppression ─────────────────────────────────────────────────────
    # Grayscale opening deletes bright structures narrower than this: thin folds
    # and the cerebellar molecular layer. Wide white matter survives intact.
    fold_open_um: float = 1000.0
    # Morphology runs on a decimated copy; 4x is ~16x faster and the structuring
    # element is far larger than the decimation.
    fold_decimate: int = 4

    # ── Intensity ────────────────────────────────────────────────────────────
    # CLAHE tiles sized in um rather than as a fixed grid. The prototypes both
    # used tileGridSize=(5,5), which silently meant 22 mm tiles on one dataset
    # and 11.7 mm on the other.
    clahe_tile_um: float = 12000.0
    clahe_clip_limit: float = 1.0
    # Stretch the tissue to the full 0-255 range before enhancement. Without it
    # output brightness tracks how dark the stain happens to be: a pale LFB-PAS
    # section inverts to roughly 42-71, about 12% of the range, while a brown
    # DAB-TBB section inverts far brighter. Percentiles rather than min/max so a
    # few extreme pixels cannot set the scale.
    normalise: bool = True
    normalise_low_percentile: float = 1.0
    normalise_high_percentile: float = 99.0
    # Sigmoid steepness. Its centre is estimated per section, not fixed.
    sigmoid_gain: float = 30.0
    # "clahe" (local equalisation) or "sigmoid" (S-curve about an estimated
    # white-matter/cortex boundary). The prototypes used CLAHE everywhere except
    # cerebellar H&E.
    enhance: str = "clahe"

    # ── Stages that are off by default ───────────────────────────────────────
    # Fold suppression deletes bright structures narrower than fold_open_um. In
    # the prototypes it was applied only to cerebellar sections, to merge the
    # molecular and granular layers into one band; run unconditionally it also
    # removes fine structure from cerebral sections. Off until asked for.
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
