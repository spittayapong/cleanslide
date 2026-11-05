# cleanslide

Artifact removal and tissue masking for histology sections, producing registration-ready
images. One configurable pipeline for any dataset and any stain: the processing stages are
fixed, the parameters come from a per-stain YAML preset, and the per-section geometry comes
from a CSV.

> Status: in development. Nothing here is released yet.

## What it does

Per section, in order:

| Stage | What it removes or produces |
|---|---|
| `geometry` | crop, pad, rotate, flip the raw scan |
| `tissue_mask` | threshold → blur → Otsu → morphological open |
| `remove_bubbles` | round mounting-medium bubbles, filtered by area and circularity, then inpainted |
| `remove_dust` | bright specks, via top-hat detection and inpainting |
| `suppress_folds` | tissue folds and thin bright laminae, via grayscale opening |
| `intensity` | invert, then CLAHE inside the mask or a sigmoid contrast curve |
| `presmooth` | anti-alias Gaussian ahead of downsampling *(transitional, see below)* |

Every size parameter is given in µm or mm and converted to pixels from the section's own
pixel size, so one preset transfers between datasets scanned at different resolutions.

## Outputs

Per section, written under `<out>/<subject_id>/<stain>/`:

| File | Format |
|---|---|
| `<subject>_<num>_gray.tif` | 8-bit grayscale TIFF, deflate — inverted, background 0 |
| `<subject>_<num>_bimask.tif` | 8-bit TIFF, deflate — 0/255 |
| `<subject>_<num>_rgb.tif` | 8-bit RGB TIFF, deflate — optional |
| `<subject>_<num>_qc.png` | stage-by-stage QC panel |
| `<subject>_<num>.json` | sidecar: `format_version`, pixel size, shape, resolved stage parameters, source file, version |

The `.json` sidecar is authoritative for the pixel size. The TIFF resolution tags carry the
same value so the files open correctly in Fiji and QuPath on their own, but readers should
prefer the sidecar.

**Why the grayscale output is inverted.** These sections are registered to a T1 MRI, so the
cleaned image is made to resemble a T1 as closely as possible. In a T1, white matter is
bright, grey matter darker and the background dark. Stained histology is the opposite:
myelin takes up the stain and reads dark, while the glass background is near-white.
Inverting restores the T1 ordering — and the background is set to 0 rather than inverted to
white, matching a skull-stripped T1 — so the registration's similarity measure compares
like with like instead of having to bridge an inverted contrast.

## Usage

```bash
cleanslide preview --config configs/lfb.yaml --sections sections.csv --section <id>
cleanslide run     --config configs/lfb.yaml --sections sections.csv --out out/
```

`preview` processes one section and writes its QC panel, for tuning parameters. `run`
batches the CSV.

## Development

```bash
git config core.hooksPath .githooks   # once per clone: blocks internal IDs and absolute paths
```

Everything under `local/` is git-ignored scratch: raw histology, the test bench and internal
section lists. Demo notebooks and tests use public NextBrain data only.

## Notes

- **`presmooth` is transitional.** The anti-alias Gaussian belongs in the downstream
  registration step, which is the only part that knows the target resolution. It stays here
  until `histo2mri` implements it, so that cleaned output does not alias when downsampled.
- `cleanslide` has no dependency on any registration package, and registration packages
  should read its TIFF + JSON output rather than importing it.
