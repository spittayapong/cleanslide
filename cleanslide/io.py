"""Reading sections from a BIDS microscopy dataset and writing the derivative.

The dataset is the single source of metadata. Subject and sample come from the
filename entities and the pixel size from the sidecar's required ``PixelSize``.
"""
import json
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

# Micrometres per unit, for the three units BIDS permits for PixelSize.
_UNIT_TO_UM = {"mm": 1000.0, "um": 1.0, "nm": 0.001}

_ENTITY = re.compile(r"(?P<key>[a-zA-Z]+)-(?P<value>[a-zA-Z0-9]+)")


@dataclass(frozen=True)
class Section:
    """One microscopy section and the metadata the pipeline needs."""

    path: str
    subject: str
    sample: str
    stain: str | None
    pixel_size_um: float
    staining: str | None = None

    @property
    def name(self) -> str:
        parts = [f"sub-{self.subject}", f"sample-{self.sample}"]
        if self.stain:
            parts.append(f"stain-{self.stain}")
        return "_".join(parts)


def _entities(filename: str) -> dict:
    return {m.group("key"): m.group("value") for m in _ENTITY.finditer(filename)}


def read_pixel_size_um(sidecar_path: str) -> tuple[float, dict]:
    """Read PixelSize from a BIDS sidecar and convert it to micrometres."""
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(
            f"no sidecar for {os.path.basename(sidecar_path)}; PixelSize is "
            "required and cannot be inferred from the image")
    with open(sidecar_path) as handle:
        meta = json.load(handle)

    if "PixelSize" not in meta or "PixelSizeUnits" not in meta:
        raise ValueError(
            f"{sidecar_path}: PixelSize and PixelSizeUnits are required by the "
            "BIDS microscopy specification and are what every um parameter is "
            "converted through")

    units = meta["PixelSizeUnits"]
    if units not in _UNIT_TO_UM:
        raise ValueError(f"{sidecar_path}: PixelSizeUnits must be mm, um or nm")

    size = meta["PixelSize"]
    # A 2- or 3-number array; anisotropic pixels would need resampling first.
    if abs(float(size[0]) - float(size[1])) > 1e-9:
        raise ValueError(
            f"{sidecar_path}: anisotropic PixelSize {size[:2]} is not supported")
    return float(size[0]) * _UNIT_TO_UM[units], meta


def find_sections(bids_dir: str, participant_labels: list[str] | None = None,
                  suffix: str = "BF") -> list[Section]:
    """Collect every microscopy section in a BIDS dataset."""
    sections = []
    for subject_dir in sorted(os.listdir(bids_dir)):
        if not subject_dir.startswith("sub-"):
            continue
        label = subject_dir[4:]
        if participant_labels and label not in participant_labels:
            continue

        micr = os.path.join(bids_dir, subject_dir, "micr")
        if not os.path.isdir(micr):
            continue

        for filename in sorted(os.listdir(micr)):
            stem, extension = os.path.splitext(filename)
            if extension.lower() not in (".tif", ".tiff", ".png"):
                continue
            if not stem.endswith(f"_{suffix}"):
                continue

            entities = _entities(stem)
            pixel_size_um, meta = read_pixel_size_um(
                os.path.join(micr, stem + ".json"))
            sections.append(Section(
                path=os.path.join(micr, filename),
                subject=entities.get("sub", label),
                sample=entities.get("sample", stem),
                stain=entities.get("stain"),
                pixel_size_um=pixel_size_um,
                staining=meta.get("SampleStaining"),
            ))
    return sections


def read_image(section: Section) -> np.ndarray:
    """Load a section as RGB uint8."""
    image = cv2.imread(section.path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read {section.path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def derivative_dir(out_dir: str, section: Section) -> str:
    return os.path.join(out_dir, f"sub-{section.subject}", "micr")


def write_result(result, section: Section, out_dir: str, params,
                 version: str, format_version: int) -> dict:
    """Write the cleaned section, its mask and a sidecar as a BIDS derivative.

    ``desc-clean`` and ``desc-tissue`` distinguish the outputs from their source,
    which carries the same entities and suffix.
    """
    directory = derivative_dir(out_dir, section)
    os.makedirs(directory, exist_ok=True)

    dpi = 25400.0 / result.pixel_size_um
    written = {}
    # Three outputs: the grayscale
    # the registration step consumes, the binary mask, and the colour section as
    # the stain reference the inverted grayscale can no longer show.
    for desc, suffix, image in (("clean", "BF", result.clean),
                                ("tissue", "mask", result.mask),
                                ("cleanrgb", "BF", result.rgb)):
        if image is None:
            continue
        # OpenCV writes BGR; the colour image is held as RGB.
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(directory, f"{section.name}_desc-{desc}_{suffix}.tif")
        # resolution tags so the file opens correctly in Fiji or QuPath alone;
        # the sidecar stays authoritative.
        cv2.imwrite(path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 8,
                                  cv2.IMWRITE_TIFF_XDPI, int(round(dpi)),
                                  cv2.IMWRITE_TIFF_YDPI, int(round(dpi))])
        written[desc] = path

    sidecar = os.path.join(directory, f"{section.name}_desc-clean_BF.json")
    with open(sidecar, "w") as handle:
        json.dump({
            "PixelSize": [result.pixel_size_um, result.pixel_size_um],
            "PixelSizeUnits": "um",
            "SampleStaining": section.staining,
            "Sources": [os.path.relpath(section.path, os.path.dirname(out_dir))],
            "GeneratedBy": [{"Name": "cleanslide", "Version": version}],
            "OutputFormatVersion": format_version,
            "Parameters": vars(params),
            "Measurements": result.info,
        }, handle, indent=2, default=str)
    written["sidecar"] = sidecar
    return written


def write_dataset_description(out_dir: str, version: str) -> str:
    """The dataset_description.json BIDS requires of every derivative."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset_description.json")
    with open(path, "w") as handle:
        json.dump({
            "Name": "cleanslide: artifact removal and tissue masking",
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "cleanslide", "Version": version}],
        }, handle, indent=2)
    return path
