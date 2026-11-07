"""Command line entry point, following the BIDS App convention.

    cleanslide <bids_dir> <output_dir> participant [--participant-label ...]

Everything the pipeline needs — subject, sample, pixel size — comes from the
dataset, so there is nothing to configure per section and no spreadsheet to
keep in step with the data. The stain label is copied to the output filename
and sidecar but no stage reads it, so the same parameters run on every stain.
"""
import argparse
import json
import os
import sys
import traceback

from . import OUTPUT_FORMAT_VERSION, __version__, io, qc
from .params import CleanParams
from .pipeline import clean_section


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cleanslide",
        description="Artifact removal and tissue masking for histology sections.")
    parser.add_argument("bids_dir", help="BIDS microscopy dataset to read")
    parser.add_argument("output_dir", help="where the derivative is written")
    parser.add_argument("analysis_level", choices=["participant"],
                        help="only participant level is defined")
    parser.add_argument("--participant-label", nargs="+", metavar="LABEL",
                        help="subject labels without the sub- prefix; "
                             "default is every subject in the dataset")
    parser.add_argument("--qc", action="store_true",
                        help="also write a stage-by-stage QC panel per section")
    parser.add_argument("--qc-dir", metavar="DIR",
                        help="where QC panels go; default is <output_dir>/qc")
    parser.add_argument("--enhance", choices=["clahe", "sigmoid"], default="clahe",
                        help="intensity stage (default: clahe)")
    parser.add_argument("--suppress-folds", action="store_true",
                        help="delete bright structures narrower than the fold "
                             "size; intended for cerebellar sections")
    parser.add_argument("--version", action="version",
                        version=f"cleanslide {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.bids_dir):
        print(f"error: no such dataset: {args.bids_dir}", file=sys.stderr)
        return 2

    params = CleanParams(
        enhance=args.enhance,
        suppress_folds=args.suppress_folds,
    )

    try:
        sections = io.find_sections(args.bids_dir, args.participant_label)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not sections:
        print(f"error: no microscopy sections found in {args.bids_dir}",
              file=sys.stderr)
        return 2

    if args.qc:
        # matplotlib is only imported when a panel is drawn, which first happens
        # after a section has already been cleaned. Check it up front so a core
        # install fails immediately instead of part-way through a run.
        try:
            qc.require_matplotlib()
        except ImportError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    io.write_dataset_description(args.output_dir, __version__)
    qc_dir = args.qc_dir or os.path.join(args.output_dir, "qc")

    failures = 0
    for index, section in enumerate(sections, 1):
        print(f"[{index}/{len(sections)}] {section.name} "
              f"({section.pixel_size_um} um/px)", flush=True)
        try:
            rgb = io.read_image(section)
            result = clean_section(rgb, section.pixel_size_um, params,
                                   keep_stages=args.qc)
            written = io.write_result(result, section, args.output_dir, params,
                                      __version__, OUTPUT_FORMAT_VERSION)
            if args.qc:
                os.makedirs(qc_dir, exist_ok=True)
                # The three published outputs, then the stage breakdown. The
                # first is what a consumer receives; the second is why.
                qc.output_panel(result, section.name,
                                os.path.join(qc_dir, f"{section.name}_outputs.png"))
                qc.stage_panel(result, section.name,
                               os.path.join(qc_dir, f"{section.name}_qc.png"))
            print(f"    -> {os.path.basename(written['clean'])}  "
                  + json.dumps({k: result.info[k] for k in
                                ("mask_area_mm2", "mask_components",
                                 "bubbles_kept") if k in result.info}),
                  flush=True)
        except Exception:                            # noqa: BLE001
            # One bad section must not abandon the rest of the run; the
            # traceback goes to stderr and the exit code reflects the failure.
            failures += 1
            print(f"    FAILED {section.name}", file=sys.stderr)
            traceback.print_exc()

    print(f"\n{len(sections) - failures}/{len(sections)} sections cleaned")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
