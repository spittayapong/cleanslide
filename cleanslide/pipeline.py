"""The cleaning pipeline: one section in, a cleaned section and its mask out.

This is the layer a notebook calls directly. It needs no files and no BIDS
dataset — an array and a pixel size are enough:

    from cleanslide import clean_section
    result = clean_section(rgb, pixel_size_um=4.0)

The command line walks a BIDS dataset and calls this for each section, so the
demo and the batch run exercise the same code.
"""
from dataclasses import dataclass, field

import numpy as np

from . import stages
from .params import CleanParams

# Order matters: artefacts are removed
# from the colour image first, so inpainting has real colour to work from, and
# intensity processing runs last on the repaired grayscale.
STAGE_ORDER = ("tissue_mask", "remove_bubbles", "to_masked_rgb", "to_gray",
               "normalise", "remove_dust", "suppress_folds", "enhance")


@dataclass
class CleanResult:
    """Cleaned section, its tissue mask, and what the run measured."""

    # Grayscale, inverted and normalised: what the registration step consumes.
    clean: np.ndarray
    # Binary tissue mask, 0 or 255. Bubbles stay as holes in it.
    mask: np.ndarray
    # The colour section with bubbles inpainted and the background blacked out.
    # Keeps the stain visible, which `clean` no longer does once inverted.
    rgb: np.ndarray
    pixel_size_um: float
    info: dict = field(default_factory=dict)
    # Intermediate images, populated only when keep_stages=True. Used by the QC
    # panel and the regression suite; skipped otherwise to save memory, which
    # matters on the 14k x 12k sections.
    intermediates: dict = field(default_factory=dict)


def clean_section(rgb: np.ndarray, pixel_size_um: float,
                  params: CleanParams | None = None,
                  keep_stages: bool = False) -> CleanResult:
    """Clean one histology section.

    Args:
        rgb: the section as an RGB uint8 array.
        pixel_size_um: physical size of one pixel. Every parameter is converted
            through it, so this must be right — a wrong value does not fail, it
            silently rescales every filter.
        params: sizes and switches; defaults are in :class:`CleanParams`.
        keep_stages: also return the image after each stage.

    Returns:
        A :class:`CleanResult` carrying three images: ``clean`` (8-bit
        grayscale, inverted so tissue is bright on a zero background),
        ``mask`` (0 or 255) and ``rgb`` (the colour section, bubbles
        inpainted, background black).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {rgb.dtype}")
    if pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")

    params = params or CleanParams()
    info: dict = {"pixel_size_um": pixel_size_um}
    kept: dict = {}

    def record(name, image):
        if keep_stages:
            kept[name] = image.copy()

    mask, mask_info = stages.tissue_mask(rgb, pixel_size_um, params)
    info.update(mask_info)
    record("tissue_mask", mask)

    bubbles, bubble_info = stages.find_bubbles(mask, pixel_size_um, params)
    info.update(bubble_info)
    repaired = stages.inpaint(rgb, bubbles, pixel_size_um,
                              params.bubble_dilate_um)
    record("remove_bubbles", repaired)

    colour = stages.to_masked_rgb(repaired, mask)
    record("to_masked_rgb", colour)

    gray = stages.to_gray(repaired, mask)
    record("to_gray", gray)

    # First, so that every threshold downstream works on the same scale. A pale
    # stain inverts into a narrow band (roughly 42-71 of 255 on LFB-PAS), where
    # the top-hat response and its MAD are both tiny and the dust threshold
    # collapses to its floor, inpainting ordinary tissue as specks.
    if params.normalise:
        gray, normalise_info = stages.normalise_intensity(gray, mask, params)
        info.update(normalise_info)
        record("normalise", gray)

    gray, dust_info = stages.remove_dust(gray, pixel_size_um, params, mask)
    info.update(dust_info)
    record("remove_dust", gray)

    if params.suppress_folds:
        gray = stages.suppress_folds(gray, mask, pixel_size_um, params)
        record("suppress_folds", gray)

    if params.enhance == "sigmoid":
        gray, enhance_info = stages.enhance_sigmoid(gray, mask, params)
        info.update(enhance_info)
    else:
        gray = stages.enhance_clahe(gray, mask, pixel_size_um, params)
    info["enhance"] = params.enhance
    record("enhance", gray)

    info.update(quality_metrics(gray, mask, pixel_size_um))
    return CleanResult(clean=gray, mask=mask, rgb=colour,
                       pixel_size_um=pixel_size_um, info=info,
                       intermediates=kept)


def quality_metrics(clean: np.ndarray, mask: np.ndarray,
                    pixel_size_um: float) -> dict:
    """Scalars that are bad when the result is bad, with no ground truth.

    None of these prove the cleaning is good. They catch the failures that do
    happen: a mask that shattered into speckle, one that swallowed the whole
    frame, or an intensity stage that flattened the tissue contrast it exists to
    create.
    """
    import cv2

    binary = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    area_px = int(binary.sum())
    mm2_per_px = (pixel_size_um * 1e-3) ** 2

    metrics = {
        "mask_area_mm2": round(area_px * mm2_per_px, 3),
        "mask_fraction": round(area_px / binary.size, 4),
        # One section should be one component, or a few. Dozens means speckle
        # survived the morphological opening.
        "mask_components": int(count - 1),
    }

    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        metrics["mask_largest_fraction"] = round(
            float(stats[largest, cv2.CC_STAT_AREA]) / max(area_px, 1), 4)

    # Tissue running off the edge of the frame usually means the section was
    # cropped through, or the background estimate caught the whole slide.
    border = np.concatenate([binary[0, :], binary[-1, :],
                             binary[:, 0], binary[:, -1]])
    metrics["mask_border_fraction"] = round(float(border.mean()), 4)

    inside = clean[mask > 0]
    if inside.size:
        # Otsu's between-class variance inside the tissue: how separable white
        # matter and cortex are, which is what the intensity stage is for.
        level, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low, high = inside[inside <= level], inside[inside > level]
        if low.size and high.size:
            w0, w1 = low.size / inside.size, high.size / inside.size
            metrics["tissue_separability"] = round(
                float(w0 * w1 * (low.mean() - high.mean()) ** 2), 2)
        metrics["tissue_mean"] = round(float(inside.mean()), 2)
        metrics["tissue_std"] = round(float(inside.std()), 2)

    return metrics
