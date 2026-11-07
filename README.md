# cleanslide

Artifact removal and tissue masking for histology sections, producing
registration-ready images. One pipeline for any dataset and any stain: the
processing stages are fixed, every parameter is a physical size in micrometres,
and the per-section scale is read from the dataset rather than configured.

Input is a [BIDS microscopy](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/microscopy.html)
dataset; output is a BIDS derivative.

> **Status: in development, nothing released yet.** The output layout and the
> sidecar schema are settled and versioned by `OUTPUT_FORMAT_VERSION`, so a
> downstream package can rely on them. Everything else may still move.

## Example

One section in, three outputs out:

![input section and the three cleaned outputs](docs/example_section.png)

NextBrain subject P57-16, block P3.1, section 09 — Luxol Fast Blue at 4 µm/px.
Note that the outputs are cropped to the tissue while the input is not; the crop
origin is recorded in the sidecar so coordinates map back to the source scan.
See [Data and attribution](#data-and-attribution).

## Installation

Not on PyPI yet. Install from the repository:

```bash
pip install git+https://github.com/spittayapong/cleanslide.git
```

Python 3.10 or newer. The core install pulls in numpy, opencv-python-headless
and scipy — nothing that needs a display, so it runs on a compute node.

Two optional extras:

| Extra | Adds | For |
|---|---|---|
| `cleanslide[qc]` | matplotlib | the QC panels, `--qc` |
| `cleanslide[test]` | pytest | running the test suite |

`--qc` without matplotlib exits with status 2 and a message naming the extra,
before any section is processed, rather than failing after an hour of work.

## Input

A BIDS microscopy dataset. A minimal one looks like this:

```
dataset/
├── dataset_description.json
├── participants.tsv
├── samples.tsv
├── sub-01/
│   └── micr/
│       ├── sub-01_sample-A1_stain-LFB_BF.tif
│       ├── sub-01_sample-A1_stain-LFB_BF.json
│       ├── sub-01_sample-A2_stain-LFB_BF.tif
│       └── sub-01_sample-A2_stain-LFB_BF.json
└── sub-02/
    └── micr/
        ├── sub-02_sample-B1_BF.tif        # no stain entity: also fine
        └── sub-02_sample-B1_BF.json
```

What the pipeline reads, and from where:

| What | Source | Required |
|---|---|---|
| subject | the `sub-<label>` entity | yes |
| section | the `sample-<label>` entity | yes |
| image | a file ending `_BF`, with a `<same stem>.json` beside it | yes |
| **pixel size** | sidecar `PixelSize` + `PixelSizeUnits` | **yes — no default** |
| stain | `stain-<label>` entity, and sidecar `SampleStaining` | no — both optional |

The stain is read from both places and required in neither. The label is
copied to the output filename and sidecar, but no stage reads it, so the same
parameters run on H&E, LFB-PAS and DAB-TBB. Drop the `stain-` entity and the
outputs simply lose that entity too; leave `SampleStaining` out of the sidecar
and it comes through as null.

Accepted image formats are `.tif`, `.tiff` and `.png`. **Not JPEG** — BIDS does
not permit it for microscopy. Convert before building the dataset; re-wrapping a
JPEG as TIFF does not restore what the JPEG lost, but it does make the dataset
valid.

Labels must be alphanumeric, which is worth knowing when your own identifiers
contain dots or hyphens: `P3.1` section `09` has to become something like
`sample-P3109`, and that cannot be split again. Record the original form in
`samples.tsv` so it is not lost.

### Pixel size is mandatory, and it matters more than anything else here

Every filter in the pipeline is specified as a physical size — a 1.1 mm opening,
a 2.5–4.5 mm bubble, a 12 mm CLAHE tile — and converted to pixels through the
section's own pixel size. That is what lets one parameter set work on data
scanned at 4 µm/px and at 31.75 µm/px.

It also means a wrong pixel size **does not raise an error**. It silently
rescales every filter at once, so the pipeline runs to completion and removes
the wrong objects. A section declared at 4 µm/px that is really 31.75 µm/px gets
filters eight times too small, and the bubble detector looks for 2.5 mm bubbles
in a window a third of a millimetre wide.

`cleanslide` therefore refuses a section whose scale it cannot read, rather than
assuming one. If `PixelSize` is missing from a sidecar, that section is an error.

## What it does

Per section, in order:

| Stage | What it removes or produces |
|---|---|
| `tissue_mask` | threshold → blur → Otsu → morphological open, then drop printed charts, labels and debris; crop to the tissue |
| `remove_bubbles` | round mounting-medium bubbles, filtered by area and circularity, then inpainted |
| `smooth_mask` | regularise the mask outline so the masked tissue has no ragged pixel border |
| `to_masked_rgb` | the colour section with the background blacked out |
| `to_inverted_gray` | invert, so tissue is bright and the background 0 |
| `normalise` | stretch the tissue to the full range, by percentile |
| `suppress_folds` | tissue folds and thin bright laminae, via grayscale opening *(off by default)* |
| `enhance` | CLAHE inside the mask, or a sigmoid contrast curve |

## Usage

A BIDS App, so the three positional arguments are the convention:

```bash
cleanslide <bids_dir> <output_dir> participant
```

Options:

| Option | Effect |
|---|---|
| `--participant-label LABEL [...]` | subjects to process, without the `sub-` prefix; default is all of them |
| `--qc` | also write QC panels |
| `--qc-dir DIR` | where panels go; default `<output_dir>/qc` |
| `--enhance {clahe,sigmoid}` | intensity stage, default `clahe` |
| `--suppress-folds` | delete bright structures narrower than the fold size; intended for cerebellar sections |

For example, two subjects with QC:

```bash
cleanslide data/bids derivatives/cleanslide participant \
    --participant-label 01 02 --qc
```

## Outputs

A BIDS derivative. Per section, under `<output_dir>/sub-<label>/micr/`:

| File | What |
|---|---|
| `…_desc-clean_BF.tif` | 8-bit grayscale, deflate — inverted, stretched, background 0. What a registration step consumes. |
| `…_desc-tissue_mask.tif` | 8-bit, deflate — 0 or 255. Bubbles stay as holes. |
| `…_desc-cleanrgb_BF.tif` | 8-bit RGB, deflate — the colour section with the background removed |
| `…_desc-clean_BF.json` | the sidecar |

Plus `<output_dir>/dataset_description.json`, which BIDS requires of a
derivative.

The sidecar carries `PixelSize`, `PixelSizeUnits`, `SampleStaining`, `Sources`,
`GeneratedBy`, `OutputFormatVersion`, the fully resolved `Parameters`, and
`Measurements` — everything the run estimated or counted, including the crop
origin, the mask area in mm², how many components were dropped and why, and how
many bubbles were found against how many were kept.

The sidecar is authoritative for the pixel size. The TIFF resolution tags carry
the same value so a file opens correctly in Fiji or QuPath on its own, but a
reader should prefer the sidecar.

`OUTPUT_FORMAT_VERSION` versions that schema. `cleanslide` owns the layout and
the schema; a consumer reads the TIFFs and the JSON and should not import this
package.

### QC

`--qc` writes two PNGs per section:

- `<section>_outputs.png` — the three outputs side by side
- `<section>_qc.png` — every stage in order

The stage panel is how you answer "is the mask right on my data?", which is the
first question a new dataset raises. Look at it before trusting a batch run.
[`notebook/demo_cleaning_one_section.ipynb`](notebook/demo_cleaning_one_section.ipynb)
walks through the same thing interactively on one section.

### Why the grayscale output is inverted

Not cosmetic. These sections are registered to a T1 MRI, so the cleaned image is
made to resemble a T1 as closely as possible.

| | white matter | grey matter | background |
|---|---|---|---|
| T1 MRI | bright | darker | dark |
| stained histology | **dark** (myelin takes the stain) | lighter | **near-white** (glass) |
| after inversion | bright | darker | dark, set to 0 |

Inverting restores the T1 ordering, and the background is set to 0 rather than
inverted to white, which matches a skull-stripped T1. The registration's
similarity measure then compares like with like instead of having to bridge an
inverted contrast.

## Known limitations

- **The mask includes background between gyri and folia.** The blur that forms
  the mask has to be wide enough to bridge gaps inside tissue, and that same
  width spans a sulcus. Sharpening it on brightness does not work, because pale
  tissue and glass overlap in grey. A colour-based refinement was tried and
  reverted: it cleared the sulci on most sections but removed real tissue from a
  faint DAB section, where unstained tissue is as colourless as glass.
- **Bubble detection is conservative.** Circularity is the discriminating test,
  so a bubble that has burst, merged or sits at the tissue edge is not removed.
  On some sections every candidate is rejected and the stage does nothing;
  `Measurements` records `bubbles_found` against `bubbles_kept` so you can see
  when that happened.
- **No anti-aliasing before downsampling.** `cleanslide` does not pre-smooth,
  because it does not know your target resolution. A consumer that downsamples
  must apply its own anti-alias filter.

## Data and attribution

The example images and the demo notebook use NextBrain:

> Iglesias Gonzalez, Eugenio; Casamitjana, Adria; Atzeni, Alessia; Billot,
> Benjamin; Thomas, David; Blackburn, Emily; et al. (2023). *Registered
> histology, MRI, and manual annotations of over 300 brain regions in 5 human
> hemispheres (data from ERC Starting Grant 677697 "BUNGEE-TOOLS")*. University
> College London. Dataset. <https://doi.org/10.5522/04/24243835.v1>

Released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The cleaned outputs shown here were produced from that data by `cleanslide`.

## License

The code is [MIT](LICENSE) — use it, change it, redistribute it, including
commercially; keep the copyright notice; no warranty.

That covers this package. The example data is a separate matter: NextBrain is
CC0, as above, and nothing in this repository places any condition on the
sections *you* process with it.

## Development

```bash
git clone https://github.com/spittayapong/cleanslide.git
cd cleanslide
pip install -e '.[qc,test]'
pytest
```

The tests run on synthetic sections, so they need no data and carry none.
Anything under `local/` is git-ignored scratch. Demo notebooks and tests use
public NextBrain data only.
