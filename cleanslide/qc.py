"""Stage-by-stage QC panel for one cleaned section.

Shows what each stage did rather than only the endpoint, so a result that looks
plausible but was produced by the wrong stage is still visible. Images are
downsampled for display; a 14k x 12k section would otherwise dominate the file.
"""
import numpy as np

from .pipeline import STAGE_ORDER

QC_EXTRA_HINT = ("the QC panel needs matplotlib, which is not part of the core "
                 "install: pip install 'cleanslide[qc]'")


def _pyplot():
    """Import matplotlib on demand.

    It is an optional dependency — only the QC panel uses it, and a batch run
    does not — so importing it at module scope would make it required in
    practice for anyone who imports this module.
    """
    try:
        import matplotlib
    except ImportError as error:                    # pragma: no cover
        raise ImportError(QC_EXTRA_HINT) from error
    matplotlib.use("Agg")                           # no display on a compute node
    import matplotlib.pyplot as plt
    return plt

_CAPTIONS = {
    "tissue_mask": "tissue mask",
    "remove_bubbles": "bubbles inpainted (RGB)",
    "to_masked_rgb": "colour output, background removed",
    "to_gray": "grayscale, inverted in mask",
    "remove_dust": "dust removed",
    "normalise": "tissue stretched to full range",
    "suppress_folds": "folds suppressed",
    "enhance": "intensity enhanced",
}


def require_matplotlib() -> None:
    """Raise early if the optional QC dependency is missing."""
    _pyplot()


def _thumbnail(image: np.ndarray, longest: int = 700) -> np.ndarray:
    scale = longest / max(image.shape[:2])
    if scale >= 1.0:
        return image
    import cv2
    return cv2.resize(image, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def stage_panel(result, title: str, out_path: str) -> str:
    """Render every retained stage as one PNG.

    Requires a result produced with ``keep_stages=True``.
    """
    if not result.intermediates:
        raise ValueError("no intermediates; run clean_section(keep_stages=True)")

    plt = _pyplot()
    present = [name for name in STAGE_ORDER if name in result.intermediates]
    columns = min(4, len(present))
    rows = (len(present) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.4 * rows),
                             squeeze=False)
    fig.suptitle(title, fontsize=13)

    for axis, name in zip(axes.flat, present):
        image = _thumbnail(result.intermediates[name])
        axis.imshow(image, cmap=None if image.ndim == 3 else "gray",
                    vmin=None if image.ndim == 3 else 0,
                    vmax=None if image.ndim == 3 else 255)
        axis.set_title(_CAPTIONS.get(name, name), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes.flat[len(present):]:
        axis.axis("off")

    # The measured values belong next to the pictures: a mask that looks fine
    # but shattered into 40 components is only visible in the numbers.
    summary = "   ".join(
        f"{key}={result.info[key]}" for key in
        ("background_threshold", "bubbles_kept", "mask_components",
         "mask_area_mm2", "tissue_separability") if key in result.info)
    fig.text(0.5, 0.005, summary, ha="center", fontsize=9, family="monospace")

    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out_path, dpi=100, format="png")
    plt.close(fig)
    return out_path
