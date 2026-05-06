"""
Extended gap mask computation.

Butterworth filters leak energy into the data on either side of every gap.
:func:`compute_extended_mask` quantifies this leakage by running the
complement of the gap mask (``1 − mask``) through the *same* filter that the
processing pipeline applied to the data, and marks any region where the
leakage exceeds a configurable threshold as excluded.  Short clean stretches
between excluded regions are merged via binary closing so that the result is
a smooth, conservative mask.

The function returns **both** the final binary mask and the raw filter-leakage
array so the caller can inspect the contamination level before committing to a
threshold.
"""

import math
from typing import Tuple

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import binary_closing

__all__ = ["compute_extended_mask"]


def compute_extended_mask(
    smoothed_mask: np.ndarray,
    sp,
    filter_kwargs: dict,
    downsample_kwargs: dict,
    trim_kwargs: dict,
    *,
    fs_raw: float,
    contamination_threshold: float = 1e-4,
    min_clean_hours: float = 12.0,
    mask_fs: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an extended binary mask that accounts for Butterworth filter leakage.

    Mirrors the downsampling and trimming steps applied to the data, then
    filters ``1 − smoothed_mask`` with the same Butterworth filter used in the
    pipeline.  Samples where the absolute leakage exceeds
    *contamination_threshold* are flagged as contaminated and excluded from the
    mask alongside the original gap samples.  Short clean intervals (shorter
    than *min_clean_hours*) between excluded regions are merged via binary
    closing.

    Parameters
    ----------
    smoothed_mask : ndarray
        1-D float array at the **raw** (pre-downsample) sampling rate.  Values
        in ``[0, 1]``:  ``1`` = data present, ``0`` = gap.  Typically the
        output of ``lisagap.GapWindowGenerator``.
    sp : SignalProcessor
        Processed segment (post-downsample, post-trim) whose length ``sp.N``
        defines the target output length.  The segment may represent only part
        of the full dataset (e.g. segment0 of many).
    filter_kwargs : dict
        Must contain ``"highpass_cutoff"`` (Hz).  Optionally ``"lowpass_cutoff"``
        and ``"order"``.  Must match the kwargs passed to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    downsample_kwargs : dict
        Must contain ``"target_fs"`` (Hz).  Must match the kwargs passed to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    trim_kwargs : dict
        Must contain ``"fraction"`` (trimmed fraction from each end).  Must
        match the kwargs passed to
        :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
    fs_raw : float
        Sampling frequency (Hz) of *smoothed_mask* — the **raw** rate before
        downsampling.  Pass ``data["fs"]`` from the output of
        :func:`~MojitoProcessor.io.read.load_file`.
    contamination_threshold : float, optional
        Absolute amplitude of the filtered gap indicator above which a sample
        is considered contaminated.  Default: ``1e-4``.
    min_clean_hours : float, optional
        Short clean gaps (in hours) between excluded regions that are shorter
        than this value will be merged into the surrounding excluded region via
        binary closing.  Default: ``12.0``.
    mask_fs : float or None, optional
        Internal sampling rate (Hz) at which the mask extension is computed.
        If ``None`` (default), uses ``target_fs`` from *downsample_kwargs*.
        Set to a value lower than ``target_fs`` to reduce memory use when the
        data is kept at a high sampling rate — the contamination zones span
        hours, so a rate of 0.2 Hz is more than sufficient.  The result is
        upsampled back to ``sp.N`` via nearest-neighbour before returning.

    Returns
    -------
    extended_mask_binary : ndarray
        Boolean-valued float array (``0.0`` or ``1.0``) of length ``sp.N``.
        ``1.0`` indicates samples that are clean and not contaminated.
    gap_contamination : ndarray
        Float array of length ``sp.N`` containing the raw filter-leakage
        signal ``filtered(1 − smoothed_mask)`` after downsampling and trimming.
        Inspect this to tune *contamination_threshold*.

    Examples
    --------
    >>> extended_mask, gap_contamination = compute_extended_mask(
    ...     smoothed_mask, sp_0,
    ...     filter_kwargs, downsample_kwargs, trim_kwargs,
    ...     fs_raw=data["fs"],
    ...     contamination_threshold=1e-4,
    ...     min_clean_hours=12.0,
    ...     mask_fs=0.2,
    ... )
    """
    target_fs: float = downsample_kwargs["target_fs"]
    trim_fraction: float = trim_kwargs.get("fraction", 0.0)

    _mask_fs: float = mask_fs if mask_fs is not None else target_fs
    _ds_factor: int = round(fs_raw / _mask_fs)
    _upsample_factor: int = round(target_fs / _mask_fs)

    # ── Determine trim sample count at the mask rate ──────────────────────────
    # len(arr[::_ds_factor]) == ceil(len(arr) / _ds_factor) for any arr.
    # Any 1-sample rounding discrepancy vs sp.N is handled by _match_length below.
    n_ds_total = math.ceil(len(smoothed_mask) / _ds_factor)
    trim_n = int(round(n_ds_total * trim_fraction / 2))

    def _trim(arr: np.ndarray) -> np.ndarray:
        return arr[trim_n:-trim_n] if trim_n > 0 else arr

    # ── Design Butterworth filter at the mask rate ────────────────────────────
    highpass: float = filter_kwargs["highpass_cutoff"]
    lowpass = filter_kwargs.get("lowpass_cutoff", None)
    order: int = int(filter_kwargs.get("order", 2))

    # Drop the lowpass if it meets or exceeds the Nyquist at _mask_fs.
    # Gap leakage arises solely from the highpass impulse response; the
    # lowpass only attenuates high-frequency content and does not introduce
    # contamination around gap edges.
    if lowpass is not None and lowpass >= _mask_fs / 2.0:
        lowpass = None

    if lowpass is not None:
        sos = scipy_signal.butter(
            order, [highpass, lowpass], btype="bandpass", fs=_mask_fs, output="sos"
        )
    else:
        sos = scipy_signal.butter(
            order, highpass, btype="highpass", fs=_mask_fs, output="sos"
        )

    # ── Stride-decimate the gap indicator ────────────────────────────────────
    # Stride first (view, no copy of the raw array), then compute 1 − mask on
    # the small result.  The smoothed mask's transitions span hours so it has
    # no energy above _mask_fs/2; stride decimation is lossless here.
    gap_contamination_input = _trim(1.0 - smoothed_mask[::_ds_factor])

    # ── Filter the downsampled gap indicator ─────────────────────────────────
    gap_contamination = scipy_signal.sosfiltfilt(sos, gap_contamination_input)
    del gap_contamination_input

    # ── Stride-decimate and trim the mask ────────────────────────────────────
    smoothed_mask_ds = _trim(smoothed_mask[::_ds_factor].astype(float))

    # ── Length-match to the coarse target length ──────────────────────────────
    # resample_poly in process_pipeline can produce sp.N that differs by 1–2
    # samples from what pure striding gives; clip or pad before upsampling.
    coarse_N = math.ceil(sp.N / _upsample_factor) if _upsample_factor > 1 else sp.N

    def _match_length(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)), constant_values=arr[-1])

    gap_contamination = _match_length(gap_contamination, coarse_N)
    smoothed_mask_ds = _match_length(smoothed_mask_ds, coarse_N)

    # ── Build extended mask ───────────────────────────────────────────────────
    original_gap = smoothed_mask_ds < 0.5
    contaminated = np.abs(gap_contamination) > contamination_threshold
    excluded = original_gap | contaminated

    # Merge short clean stretches between excluded regions
    min_clean_samples = max(1, int(min_clean_hours * 3600.0 * _mask_fs))
    excluded = binary_closing(excluded, structure=np.ones(min_clean_samples))

    extended_mask_binary = (~excluded).astype(float)

    # ── Upsample back to sp.N if mask was computed at a lower rate ───────────
    if _upsample_factor > 1:
        extended_mask_binary = np.repeat(extended_mask_binary, _upsample_factor)[: sp.N]
        gap_contamination = np.repeat(gap_contamination, _upsample_factor)[: sp.N]

    return extended_mask_binary, gap_contamination
