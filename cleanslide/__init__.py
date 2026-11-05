"""Artifact removal and tissue masking for histology sections.

One configurable pipeline for any dataset and any stain: fixed processing stages,
parameters expressed in micrometres and converted from each section's own pixel
size, so a single configuration spans scanners an order of magnitude apart.

    from cleanslide import clean_section
    result = clean_section(rgb, pixel_size_um=4.0)

Outputs TIFF images with a JSON sidecar carrying the pixel size and the
parameters actually used.
"""
from .params import CleanParams, um_to_px
from .pipeline import CleanResult, clean_section, quality_metrics

__all__ = [
    "CleanParams",
    "CleanResult",
    "clean_section",
    "quality_metrics",
    "um_to_px",
    "__version__",
    "OUTPUT_FORMAT_VERSION",
]

__version__ = "0.1.0.dev0"

# Version of the output layout and JSON sidecar schema. Readers should check it.
OUTPUT_FORMAT_VERSION = 1
