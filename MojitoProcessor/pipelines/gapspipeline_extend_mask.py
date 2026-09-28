"""
Segment-first gap-aware pipeline (v2).

Unlike :mod:`~MojitoProcessor.pipelines.gapspipeline`, which processes the
entire dataset as one block and then identifies contaminated regions, this
pipeline **segments the raw data first** and then processes each clean segment
independently:

1. Load raw L1 data.
2. Extract contiguous clean segments directly from the raw data using a
   **binary** gap mask (no tapering, no smoothing).
3. For each segment: filter → optional downsample → trim (absolute, based on
   filter settling time) → optional window.
4. Discard any segment that becomes too short after trimming.
5. Write to HDF5 (optional).

**Why v2 is better than v1:**

* **Less data loss** — each gap edge loses only the filter settling time
  (~1.7 h/end for order-2 Butterworth at 5 × 10⁻⁵ Hz) rather than the taper
  zone + filter ringing (~7 h/end in v1).
* **No cross-contamination** — each segment is filtered independently; there
  is no mechanism for gap-edge ringing to leak into neighbouring segments.
* **No extended-mask computation** — :func:`~MojitoProcessor.gaps.compute_extended_mask`
  is not needed and is not called.

The v1 pipeline (:func:`~MojitoProcessor.pipelines.gapspipeline.gapspipeline`)
is **unchanged and fully supported**.  Both pipelines can be used side-by-side.

Can be run as a script::

    python -m MojitoProcessor.pipelines.gapspipeline_extend_mask path/to/data.h5 \\
        --mask-path binary_mask.npy \\
        --output processed_v2.h5 \\
        --target-fs 4.0
"""

import argparse
import logging
import pathlib
from typing import Dict, List, Optional

import numpy as np

from ..gaps import compute_settling_trim_samples, extract_clean_segments
from ..io.read import load_file
from ..io.write import write
from ..process.sigprocess import KAISER_BETA_DEFAULT, SignalProcessor

__all__ = ["gapspipeline_extend_mask"]

logger = logging.getLogger(__name__)


def gapspipeline_extend_mask(
    path: str | pathlib.Path,
    binary_mask: np.ndarray,
    channels: Optional[List[str]] = None,
    *,
    load_days: Optional[float] = None,
    filter_kwargs: Optional[dict] = None,
    downsample_kwargs: Optional[dict] = None,
    trim_settling_factor: float = 3.0,
    window_kwargs: Optional[dict] = None,
    min_clean_hours: float = 8.0,
    max_segment_days: Optional[float] = None,
    output_path: Optional[str | pathlib.Path] = None,
) -> Dict[str, SignalProcessor]:
    """
    Segment raw LISA L1 data first, then process each segment independently.

    Parameters
    ----------
    path : str or Path
        Path to the MojitoL1 ``.h5`` input file.
    binary_mask : ndarray
        1-D integer or boolean array at the **raw** sampling rate.  Values
        must be 0 (gap) or 1 (data present).  Typically obtained from
        ``GapMaskGenerator.generate_mask()``.  **Do not pass a smoothed/tapered
        mask** — use a binary mask only; the function validates this.
    channels : list of str, optional
        TDI channels to process.  Default ``['X', 'Y', 'Z']``.
    load_days : float, optional
        Load only the first *load_days* days from the file.  ``None`` loads
        everything.  If used, *binary_mask* must be pre-sliced to the same
        length as the loaded data.
    filter_kwargs : dict, optional
        Filter parameters.  Keys: ``"highpass_cutoff"`` (Hz, required),
        ``"lowpass_cutoff"`` (Hz, optional), ``"order"`` (int, default 2).
    downsample_kwargs : dict, optional
        Keys: ``"target_fs"`` (Hz), ``"kaiser_window"`` (float, default 31.0).
        Omit to keep the raw sampling rate.
    trim_settling_factor : float, optional
        Number of filter time-constants to trim from each end of each segment.
        ``k=3`` reduces the residual transient to :math:`e^{-3} \\approx 5\\%`;
        ``k=5`` to :math:`< 1\\%`.  Default ``3.0``.
    window_kwargs : dict, optional
        Window applied to each segment after trimming.  Keys: ``"window"``
        (str, default ``"tukey"``), ``"alpha"`` (float).  Omit to skip.
    min_clean_hours : float, optional
        Minimum raw duration (hours) a clean stretch must have to be kept.
        Default ``8.0``.
    max_segment_days : float, optional
        Maximum segment duration (days).  Longer clean runs are split
        greedily.  ``None`` applies no upper limit.
    output_path : str or Path, optional
        If given, write processed segments and raw auxiliary data to this
        HDF5 file.

    Returns
    -------
    processed_segments : dict of SignalProcessor
        Contiguous clean segments keyed by ``"segment0"``, ``"segment1"``,
        etc., ready for FFT / Whittle analysis.

    Raises
    ------
    ValueError
        If *binary_mask* contains values other than 0 and 1, if its length
        does not match the loaded data, or if *filter_kwargs* is missing
        ``"highpass_cutoff"``.
    """
    _fkw = filter_kwargs or {}
    _dkw = downsample_kwargs or {}

    if channels is None:
        channels = ["X", "Y", "Z"]

    # ── Validate binary mask ─────────────────────────────────────────────────
    unique_vals = np.unique(np.asarray(binary_mask))
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"binary_mask must contain only 0 and 1, found: {unique_vals[:10]}. "
            "Did you accidentally pass a smoothed/tapered mask? "
            "Use GapMaskGenerator.generate_mask() for a binary mask."
        )

    # ── Step 1: load raw data ─────────────────────────────────────────────────
    logger.info("Loading %s", path)
    data = load_file(path, load_days=load_days)
    fs_raw: float = data["fs"]
    t0_raw: float = float(data["t_tdi"][0])

    # ── Step 2: build raw SignalProcessor ────────────────────────────────────
    laser_frequency = float(data["metadata"]["laser_frequency"])
    missing = [ch for ch in channels if ch not in data["tdis"]]
    if missing:
        raise ValueError(
            f"Channels {missing} not found in data. "
            f"Available: {list(data['tdis'].keys())}"
        )
    raw_channels = {ch: data["tdis"][ch] / laser_frequency for ch in channels}
    raw_sp = SignalProcessor(raw_channels, fs=fs_raw, t0=t0_raw)
    logger.info(
        "Raw data: %d samples @ %.4g Hz (%.2f days), channels=%s",
        raw_sp.N,
        raw_sp.fs,
        raw_sp.T / 86400,
        channels,
    )

    # Free large TDI arrays — write() only needs orbits/ltts/noise/metadata
    del data["tdis"], data["t_tdi"]

    # ── Step 3: validate mask length, then segment at raw rate ───────────────
    if len(binary_mask) != raw_sp.N:
        raise ValueError(
            f"binary_mask length ({len(binary_mask):,}) does not match "
            f"raw data length ({raw_sp.N:,}). "
            "If using load_days, pre-slice binary_mask to the same length."
        )

    logger.info(
        "Segmenting raw data (gap fraction: %.4f%%, min_clean_hours=%.1f h)",
        (1.0 - float(np.mean(binary_mask))) * 100,
        min_clean_hours,
    )
    raw_segments = extract_clean_segments(
        raw_sp,
        binary_mask,
        min_clean_hours=min_clean_hours,
        max_segment_days=max_segment_days,
    )
    del raw_sp
    logger.info("Found %d raw clean segment(s)", len(raw_segments))

    # ── Step 4: compute absolute trim from filter settling time ──────────────
    target_fs = _dkw.get("target_fs", None)
    effective_fs = target_fs if target_fs is not None else fs_raw
    kaiser_window = _dkw.get("kaiser_window", KAISER_BETA_DEFAULT)

    if _fkw.get("highpass_cutoff") is not None:
        trim_n = compute_settling_trim_samples(
            _fkw, effective_fs, k=trim_settling_factor
        )
        trim_hours = trim_n / effective_fs / 3600
        logger.info(
            "Filter settling trim: %d samples per end (%.2f h) at %.4g Hz "
            "(k=%.1f, fc=%.2e Hz, order=%d)",
            trim_n,
            trim_hours,
            effective_fs,
            trim_settling_factor,
            _fkw["highpass_cutoff"],
            int(_fkw.get("order", 2)),
        )
        # Warn if trim exceeds 25% of the minimum segment
        min_seg_samples = int(min_clean_hours * 3600 * effective_fs)
        if trim_n > min_seg_samples // 4:
            logger.warning(
                "trim_n (%d samples = %.2f h) exceeds 25%% of min_clean_hours "
                "(%.1f h = %d samples). Short segments will lose a large fraction "
                "of data. Consider increasing min_clean_hours.",
                trim_n,
                trim_hours,
                min_clean_hours,
                min_seg_samples,
            )
    else:
        trim_n = 0
        logger.info("No highpass_cutoff specified — skipping settling trim")

    highpass = _fkw.get("highpass_cutoff")
    lowpass = _fkw.get("lowpass_cutoff", None)
    order = int(_fkw.get("order", 2))

    # ── Step 5: process each segment independently ───────────────────────────
    processed_segments: Dict[str, SignalProcessor] = {}
    seg_names = list(raw_segments.keys())

    for name in seg_names:
        seg = raw_segments.pop(name)  # free raw segment from dict as we go

        # Filter
        if highpass is not None:
            seg.filter(low=highpass, high=lowpass, order=order, zero_phase=True)
        elif lowpass is not None:
            seg.filter(high=lowpass, order=order, zero_phase=True)

        # Downsample
        if target_fs is not None and target_fs != fs_raw:
            seg.downsample(target_fs, window=("kaiser", kaiser_window))

        # Trim based on filter settling time
        if trim_n > 0:
            if 2 * trim_n >= seg.N:
                logger.warning(
                    "Skipping %s: too short (%d samples = %.2f h) after "
                    "filtering — trim_n=%d would remove all data. "
                    "Consider reducing trim_settling_factor or increasing "
                    "min_clean_hours.",
                    name,
                    seg.N,
                    seg.N / effective_fs / 3600,
                    trim_n,
                )
                continue
            seg.trim(n_samples=trim_n)

        # Window
        if window_kwargs:
            window = window_kwargs.get("window", "tukey")
            alpha = window_kwargs.get("alpha", 0.025)
            seg.apply_window(window=window, alpha=alpha)

        processed_segments[name] = seg
        logger.info(
            "  %s: N=%d, fs=%.4g Hz, t0=%.4f d, duration=%.2f h",
            name,
            seg.N,
            seg.fs,
            seg.t0 / 86400 if seg.t0 is not None else float("nan"),
            seg.T / 3600,
        )

    logger.info(
        "Retained %d / %d segment(s) after processing",
        len(processed_segments),
        len(seg_names),
    )

    # ── Step 6: optional HDF5 write ──────────────────────────────────────────
    if output_path is not None:
        write(
            output_path,
            processed_segments,
            raw_data=data,
            filter_kwargs=filter_kwargs,
            downsample_kwargs=downsample_kwargs,
            trim_kwargs={
                "n_samples_per_end": trim_n,
                "settling_factor": trim_settling_factor,
            },
            window_kwargs=window_kwargs,
        )
        logger.info("Written to %s", output_path)

    return processed_segments


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Segment-first gap-aware pipeline (v2). "
            "Segments raw data on a binary mask then processes each segment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=pathlib.Path, help="Path to the MojitoL1 .h5 file")
    p.add_argument(
        "--mask-path",
        type=pathlib.Path,
        required=True,
        metavar="NPY",
        help="Path to a .npy file containing the binary gap mask (0/1)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output .h5 path for processed segments (optional)",
    )
    p.add_argument("--load-days", type=float, default=None, metavar="DAYS")
    p.add_argument("--channels", nargs="+", default=None, metavar="CH")
    p.add_argument("--target-fs", type=float, default=None, metavar="HZ")
    p.add_argument("--highpass", type=float, default=5e-5, metavar="HZ")
    p.add_argument("--lowpass", type=float, default=None, metavar="HZ")
    p.add_argument("--filter-order", type=int, default=2)
    p.add_argument(
        "--trim-settling-factor",
        type=float,
        default=3.0,
        metavar="K",
        help="Time constants to trim from each segment end",
    )
    p.add_argument("--window", type=str, default="tukey")
    p.add_argument("--window-alpha", type=float, default=0.025, metavar="ALPHA")
    p.add_argument("--min-clean-hours", type=float, default=8.0, metavar="HOURS")
    p.add_argument("--max-segment-days", type=float, default=None, metavar="DAYS")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    args = _build_parser().parse_args()

    binary_mask = np.load(args.mask_path)

    filter_kwargs = {"highpass_cutoff": args.highpass, "order": args.filter_order}
    if args.lowpass is not None:
        filter_kwargs["lowpass_cutoff"] = args.lowpass

    downsample_kwargs = (
        {"target_fs": args.target_fs} if args.target_fs is not None else None
    )

    segments = gapspipeline_extend_mask(
        args.input,
        binary_mask,
        channels=args.channels,
        load_days=args.load_days,
        filter_kwargs=filter_kwargs,
        downsample_kwargs=downsample_kwargs,
        trim_settling_factor=args.trim_settling_factor,
        window_kwargs={"window": args.window, "alpha": args.window_alpha},
        min_clean_hours=args.min_clean_hours,
        max_segment_days=args.max_segment_days,
        output_path=args.output,
    )

    print(f"\nExtracted {len(segments)} clean segment(s):")
    for name, sp in segments.items():
        print(f"  {name}: {sp}")
