"""
Clean-segment extraction from gapped data.

:func:`extract_clean_segments` splits a processed
:class:`~MojitoProcessor.process.sigprocess.SignalProcessor` at gap
boundaries defined by a binary mask and returns the contiguous clean
stretches as independent ``SignalProcessor`` objects in a dict keyed by
``"segment0"``, ``"segment1"``, etc. — the same format returned by
:func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
"""

from typing import Dict, Optional

import numpy as np

from ..process.sigprocess import SignalProcessor

__all__ = ["extract_clean_segments"]


def extract_clean_segments(
    sp: SignalProcessor,
    mask: np.ndarray,
    *,
    min_clean_hours: float = 8.0,
    max_segment_days: Optional[float] = None,
) -> Dict[str, SignalProcessor]:
    """Extract contiguous clean segments from a processed SignalProcessor.

    Identifies contiguous runs of ``1`` in *mask*, discards runs shorter than
    *min_clean_hours*, and returns each remaining stretch as its own
    :class:`SignalProcessor`.  If *max_segment_days* is set, any clean run
    longer than that limit is split greedily from the front into chunks of at
    most *max_segment_days* days; the final remainder is kept if it meets
    *min_clean_hours*, otherwise it is discarded.

    Parameters
    ----------
    sp : SignalProcessor
        Processed multi-channel time series (e.g. ``segment0`` from
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`).
    mask : array_like, shape (sp.N,)
        Binary mask at the same sampling rate as *sp*.  Typically the
        ``extended_mask`` returned by :func:`compute_extended_mask`.
        Values must be 0 or 1 (int or float).
    min_clean_hours : float, optional
        Minimum duration (hours) for a clean stretch (or sub-chunk after
        splitting) to be kept.  Default 8.0.
    max_segment_days : float, optional
        Maximum duration (days) of any returned segment.  Clean runs longer
        than this are split into chunks of exactly *max_segment_days* from the
        front, with the final remainder kept if it meets *min_clean_hours*.
        ``None`` (default) applies no upper limit.

    Returns
    -------
    segments : dict of SignalProcessor
        ``{"segment0": sp0, "segment1": sp1, ...}`` where each value is a
        new :class:`SignalProcessor` containing only the clean data for that
        stretch.  The ``t0`` attribute of each segment is set so that
        ``sp_i.t`` gives the correct absolute timestamps.

    Raises
    ------
    ValueError
        If *mask* length does not match ``sp.N``.
    """
    mask = np.asarray(mask, dtype=int)
    if mask.shape != (sp.N,):
        raise ValueError(f"mask length ({len(mask)}) does not match sp.N ({sp.N})")

    # Find contiguous runs of 1
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    min_samples = int(min_clean_hours * 3600 * sp.fs)
    max_samples = (
        int(max_segment_days * 86400 * sp.fs) if max_segment_days is not None else None
    )

    segments: Dict[str, SignalProcessor] = {}
    seg_idx = 0

    for s, e in zip(starts, ends):
        if (e - s) < min_samples:
            continue

        # Split long runs into sub-chunks if max_segment_days is set
        if max_samples is None or (e - s) <= max_samples:
            sub_ranges = [(s, e)]
        else:
            sub_ranges = []
            pos = s
            while pos < e:
                sub_ranges.append((pos, min(pos + max_samples, e)))
                pos += max_samples

        for sub_s, sub_e in sub_ranges:
            if (sub_e - sub_s) < min_samples:
                continue
            chunk_data = {ch: sp.data[ch][sub_s:sub_e] for ch in sp.channels}
            chunk_t0 = sp.t[sub_s] if sp.t0 is not None else float(sub_s * sp.dt)
            seg = SignalProcessor(chunk_data, fs=sp.fs, t0=chunk_t0)
            segments[f"segment{seg_idx}"] = seg
            seg_idx += 1

    return segments
