"""The cleaning stages.

Each takes arrays in and returns arrays out. Every size argument is converted 
from micrometres using the section's own pixel size.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from .params import CleanParams, um_to_px

def estimate_background_threshold(gray: np.ndarray, shoulder: float = 0.1) -> int:
    """Find the grey level separating glass background from tissue.

    The slide background is the dominant bright peak in the histogram. Walk down
    from its mode until the counts fall to ``shoulder`` of the peak; that is the
    foot of the background distribution and everything below it is tissue.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    bright = hist[128:]
    if bright.max() <= 0:
        return 128
    mode = 128 + int(np.argmax(bright))
    floor_count = hist[mode] * shoulder
    level = mode
    while level > 1 and hist[level] > floor_count:
        level -= 1
    return int(level)


def drop_border_components(mask: np.ndarray):
    """Remove mask components touching the image edge, keeping the largest.

    Printed calibration charts, grey step wedges and handwritten slide labels
    sit at the edge of a scan and are as dark as tissue, so a grayscale mask
    includes them. They are not tissue, and leaving them in corrupts more than
    the mask: the area measurement counts printed ink, and the intensity
    normalisation takes its endpoints from a wedge that spans pure white to
    pure black.

    The largest component is never dropped, so a section whose own tissue runs
    off the frame is kept rather than deleted.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask, 0

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    touching = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])).tolist())
    touching -= {0, largest}
    if not touching:
        return mask, 0

    out = mask.copy()
    out[np.isin(labels, list(touching))] = 0
    return out, len(touching)


def drop_small_components(mask: np.ndarray, pixel_size_um: float,
                          min_area_mm2: float):
    """Remove components too small to be a piece of a tissue section.

    Slide labels, grey-wedge fragments and debris are far smaller than any
    piece of a section, so the two separate cleanly by area.
    """
    if min_area_mm2 <= 0:
        return mask, 0

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask, 0

    mm2_per_px = (pixel_size_um * 1e-3) ** 2
    small = [label for label in range(1, count)
             if stats[label, cv2.CC_STAT_AREA] * mm2_per_px < min_area_mm2]

    if len(small) == count - 1:
        # Everything is small, so the threshold is wrong for this section
        # rather than the section being empty. Keep the largest.
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        small = [label for label in small if label != largest]
    if not small:
        return mask, 0

    out = mask.copy()
    out[np.isin(labels, small)] = 0
    return out, len(small)


def drop_rectangular_components(mask: np.ndarray, gray: np.ndarray,
                                max_extent: float, min_aspect: float = 1.7,
                                min_area_fraction: float = 0.05,
                                max_sd: float = 45.0):
    """Remove substantial mask components that are printed matter, not tissue.

    Three tests, all of which must agree: the component nearly fills its
    bounding box, it is elongated, and the grey levels inside it vary more
    than tissue ever does.

    Shape alone is not enough, and relying on it deleted whole brain sections.
    A section cut as a long cortical strip is geometrically a bar.

    However variance is what separates them, because these charts are grey step wedges
    and colour bars: one component spans many levels, so its standard
    deviation is large, while tissue is comparatively uniform in grey.

    Only components at least ``min_area_fraction`` of the largest are tested, so
    small debris is not judged by measures that are noisy at that size.
    """
    if max_extent > 1.0:
        return mask, 0

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    threshold_area = areas.max() * min_area_fraction

    printed = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < threshold_area:
            continue
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        box = width * height
        if not box:
            continue
        aspect = max(width / height, height / width)
        if not (area / box > max_extent and aspect > min_aspect):
            continue
        # Shape says bar; only the grey levels inside can say print. Last,
        # because it is the expensive test and most components never reach it.
        if float(gray[labels == label].std()) > max_sd:
            printed.append(label)

    if len(printed) == count - 1:
        # Everything looks printed, which is far more likely to mean the test is
        # wrong for this section than that it contains no tissue.
        printed = [label for label in printed if label != largest]

    if not printed:
        return mask, 0

    out = mask.copy()
    out[np.isin(labels, printed)] = 0
    return out, len(printed)


def smooth_mask_edge(mask: np.ndarray, pixel_size_um: float,
                     smooth_um: float) -> np.ndarray:
    """Regularise the mask boundary without moving the region.

    Blur the binary mask and re-threshold at the half level: the outline is
    pulled smooth while anything solidly inside or outside stays where it is.
    The component filters leave a boundary that follows individual pixels,
    which reads as a ragged edge once the mask is applied to the tissue.
    """
    if smooth_um <= 0:
        return mask
    kernel = um_to_px(smooth_um, pixel_size_um, odd=True)
    if kernel < 3:
        return mask
    blurred = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return out


def tissue_mask(rgb: np.ndarray, pixel_size_um: float, params: CleanParams):
    """Binary tissue mask (0 or 255) plus the values estimated along the way."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    threshold = (estimate_background_threshold(gray)
                 if params.background_threshold is None
                 else int(params.background_threshold))
    _, coarse = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Blur then Otsu: the blur bridges gaps inside tissue so Otsu sees one
    # foreground region rather than speckle. The kernel is large on purpose.
    blur_px = um_to_px(params.mask_blur_um, pixel_size_um, odd=True)
    blurred = cv2.GaussianBlur(coarse, (blur_px, blur_px), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    open_px = um_to_px(params.mask_open_um, pixel_size_um)
    kernel = np.ones((open_px, open_px), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    dropped = 0
    if params.mask_drop_border_components:
        mask, dropped = drop_border_components(mask)

    # After the border pass, so a chart that both touches the edge and is
    # rectangular is only counted once.
    mask, printed = drop_rectangular_components(
        mask, gray, params.mask_max_extent, params.mask_min_aspect,
        params.mask_extent_min_area_fraction, params.mask_printed_max_sd)

    # Last, so the printed bars are gone before the size test runs and cannot
    # affect what counts as large.
    mask, small = drop_small_components(mask, pixel_size_um,
                                        params.mask_min_component_mm2)

    info = {
        "background_threshold": threshold,
        "mask_blur_px": blur_px,
        "mask_open_px": open_px,
        "border_components_dropped": dropped,
        "printed_components_dropped": printed,
        "small_components_dropped": small,
        "mask_area_px": int((mask > 0).sum()),
    }
    return mask, info


def tissue_bbox(mask: np.ndarray, pixel_size_um: float, margin_um: float):
    """Bounding box of the tissue as ``(y0, y1, x0, x1)``, or None if empty.

    ``margin_um`` keeps context around the section and, as a side effect, keeps
    the tissue off the crop edge so the border diagnostics stay meaningful.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size or not cols.size:
        return None

    margin = um_to_px(margin_um, pixel_size_um) if margin_um > 0 else 0
    height, width = mask.shape[:2]
    return (max(0, int(rows[0]) - margin), min(height, int(rows[-1]) + 1 + margin),
            max(0, int(cols[0]) - margin), min(width, int(cols[-1]) + 1 + margin))


def find_bubbles(mask: np.ndarray, pixel_size_um: float, params: CleanParams):
    """Locate round mounting-medium bubbles as holes enclosed by tissue.

    The hole filling here is a detection step only — the tissue mask returned by
    :func:`tissue_mask` is never modified.
    """
    tissue = mask > 0
    structure = ndi.generate_binary_structure(2, 2)
    closed = ndi.binary_closing(tissue, structure=structure)
    # Pad so a hole touching the border is still enclosed for the fill.
    padded = np.pad(closed, 1, mode="constant")
    filled = ndi.binary_fill_holes(padded, structure=structure)[1:-1, 1:-1]

    holes = (filled & ~tissue).astype(np.uint8) * 255

    min_px = um_to_px(params.bubble_min_diameter_um, pixel_size_um)
    max_px = um_to_px(params.bubble_max_diameter_um, pixel_size_um)
    min_area = np.pi / 4.0 * min_px ** 2
    max_area = np.pi / 4.0 * max_px ** 2

    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = np.zeros_like(holes)
    kept = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        if circularity < params.bubble_min_circularity:
            continue
        cv2.drawContours(bubbles, [contour], -1, 255, -1)
        kept += 1

    if kept:
        dilate_px = um_to_px(params.bubble_dilate_um, pixel_size_um, odd=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        bubbles = cv2.dilate(bubbles, kernel, iterations=1)

    info = {
        "bubbles_found": len(contours),
        "bubbles_kept": kept,
        "bubble_area_px": int((bubbles > 0).sum()),
    }
    return bubbles, info


def inpaint(image: np.ndarray, defect_mask: np.ndarray, pixel_size_um: float,
            radius_um: float) -> np.ndarray:
    """Fill the masked region from its surroundings."""
    if not defect_mask.any():
        return image
    radius = um_to_px(radius_um, pixel_size_um)
    return cv2.inpaint(image, defect_mask, radius, cv2.INPAINT_TELEA)


def to_gray(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grayscale, inverted inside the tissue and zero outside it.

    The inversion is the point of this stage, not a convenience. These sections
    exist to be registered to a T1 MRI, so the cleaned image should look as much
    like a T1 as possible: in a T1 white matter is bright, grey matter is darker
    and the background is dark. Stained histology is the other way round —
    myelin takes up the stain and reads dark, while the glass background is
    near-white. Inverting restores the T1 ordering, so the registration's
    similarity measure compares like with like instead of having to bridge an
    inverted contrast.

    Only the tissue is inverted. The background is set to zero rather than
    inverted to white, which also matches a skull-stripped T1.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = np.zeros_like(gray)
    inside = mask > 0
    out[inside] = 255 - gray[inside]
    return out


def to_masked_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The colour section with everything outside the tissue set to black.

    Kept alongside the grayscale output as the colour reference: it still shows
    the stain, which the inverted, normalised grayscale no longer does. Bubbles
    are already inpainted at this point.
    """
    out = np.zeros_like(rgb)
    inside = mask > 0
    out[inside] = rgb[inside]
    return out


def normalise_intensity(gray: np.ndarray, mask: np.ndarray, params: CleanParams):
    """Stretch the tissue to the full range, leaving the background at zero.

    Output brightness otherwise tracks the stain's density rather than the
    tissue: inverting a pale stain gives a dark, compressed image while a dark
    stain gives a bright one, so the same settings produce different-looking
    output per stain. Percentile endpoints keep a handful of extreme pixels from
    setting the scale.
    """
    inside = mask > 0
    if not inside.any():
        return gray, {"normalise_low": None, "normalise_high": None}

    values = gray[inside]
    low, high = np.percentile(values, [params.normalise_low_percentile,
                                       params.normalise_high_percentile])
    if high <= low:
        return gray, {"normalise_low": float(low), "normalise_high": float(high)}

    scaled = (gray.astype(np.float32) - low) * (255.0 / (high - low))
    out = np.zeros_like(gray)
    out[inside] = np.clip(scaled[inside], 0, 255).astype(np.uint8)

    return out, {"normalise_low": round(float(low), 2),
                 "normalise_high": round(float(high), 2)}


def suppress_folds(gray: np.ndarray, mask: np.ndarray, pixel_size_um: float,
                   params: CleanParams) -> np.ndarray:
    """Delete bright structures narrower than ``fold_open_um``.

    A grayscale opening removes thin bright laminae — tissue folds, and the
    cerebellar molecular layer — while leaving wide white matter at full
    brightness. A pointwise curve cannot do this: the molecular layer is brighter
    than white matter, so no monotonic map can reorder them.
    """
    size = um_to_px(params.fold_open_um, pixel_size_um)
    decimate = max(1, params.fold_decimate)

    source = gray.astype(np.float32).copy()
    # Background bright during the opening keeps the erosion off the tissue edge.
    source[mask == 0] = 255.0

    small = cv2.resize(source, None, fx=1.0 / decimate, fy=1.0 / decimate,
                       interpolation=cv2.INTER_AREA)
    k = max(3, size // decimate) | 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, element)

    opened = cv2.resize(small, (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    opened[mask == 0] = 0.0
    return np.clip(opened, 0, 255).astype(np.uint8)


def enhance_clahe(gray: np.ndarray, mask: np.ndarray, pixel_size_um: float,
                  params: CleanParams) -> np.ndarray:
    """Local contrast equalisation, applied only inside the tissue.

    Tiles are sized in micrometres rather than as a fixed grid, so a tile
    covers the same physical area whatever the scan resolution.
    """
    tile_px = um_to_px(params.clahe_tile_um, pixel_size_um)
    height, width = gray.shape[:2]
    grid = (max(1, width // tile_px), max(1, height // tile_px))

    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip_limit, tileGridSize=grid)
    equalised = clahe.apply(gray)

    out = gray.copy()
    inside = mask > 0
    out[inside] = equalised[inside]
    return out


def enhance_sigmoid(gray: np.ndarray, mask: np.ndarray, params: CleanParams):
    """S-curve contrast with its centre estimated from the tissue histogram.

    A fixed centre has to be hand-tuned per section. Otsu inside the mask
    finds the same boundary — between white matter and cortex — on any
    section, without tuning.
    """
    inside = mask > 0
    if not inside.any():
        return gray, {"sigmoid_cutoff": None}

    values = gray[inside]
    cutoff_level, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = float(cutoff_level) / 255.0

    normalised = gray.astype(np.float32) / 255.0
    curved = 1.0 / (1.0 + np.exp(-params.sigmoid_gain * (normalised - cutoff)))

    out = np.zeros_like(gray)
    out[inside] = np.clip(curved[inside] * 255.0, 0, 255).astype(np.uint8)
    return out, {"sigmoid_cutoff": round(cutoff, 4)}
