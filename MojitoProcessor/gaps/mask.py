"""
Gap masking utilities.

:func:`apply_raw_mask` multiplies a scalar mask onto every TDI channel in
a raw data dictionary, returning a deep copy so the original is untouched.
"""

from copy import deepcopy

import numpy as np

__all__ = ["apply_raw_mask"]

# Channels on which the mask is applied when working with a raw data dict.
_TDI_CHANNELS = ("X", "Y", "Z", "A", "E", "T")


def apply_raw_mask(data: dict, mask: np.ndarray) -> dict:
    """Apply a gap mask to all TDI channels in a raw data dictionary.

    Creates a deep copy of *data* before multiplying, so the original dict is
    never modified.  Only channels present in the dict's ``"tdis"`` sub-dict
    are touched; all other entries (ltts, orbits, metadata, …) are preserved
    unchanged.

    Parameters
    ----------
    data : dict
        Raw data dict as returned by
        :func:`~MojitoProcessor.io.read.load_file`.  Must contain a ``"tdis"``
        key whose value is a dict of 1-D numpy arrays.
    mask : ndarray
        1-D float array of length ``data["tdis"]["X"].size`` (or whichever
        channel is present).  Values should be in ``[0, 1]``.

    Returns
    -------
    data_masked : dict
        Deep copy of *data* with every TDI channel multiplied by *mask*.

    Examples
    --------
    >>> data_masked = apply_raw_mask(data, smoothed_mask)
    """
    mask = np.asarray(mask, dtype=float)
    data_masked = deepcopy(data)
    for ch in _TDI_CHANNELS:
        if ch in data_masked["tdis"]:
            data_masked["tdis"][ch] = data_masked["tdis"][ch] * mask
    return data_masked
