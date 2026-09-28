"""
MojitoProcessor gaps — gap-handling utilities for LISA TDI data.

This subpackage provides a non-mutating, modular interface for applying and
investigating the effect of data gaps on the processing pipeline.

The typical workflow for the v1 pipeline is:

1. :func:`apply_raw_mask` — zero-out gapped samples in the raw data dict
   before passing it to :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
2. :func:`compute_extended_mask` — quantify Butterworth filter leakage and
   produce a conservative binary mask at the processed sampling rate.

For the v2 pipeline (recommended):

1. :func:`extract_clean_segments` — extract contiguous clean segments from raw
   data using a binary gap mask.
2. :func:`compute_settling_trim_samples` — compute the number of samples to
   trim from each segment end based on filter settling time.
"""

from .extend import compute_extended_mask
from .mask import apply_raw_mask
from .segment import extract_clean_segments
from .settling import compute_settling_trim_samples

__all__ = [
    "apply_raw_mask",
    "compute_extended_mask",
    "compute_settling_trim_samples",
    "extract_clean_segments",
]
