"""
Filter settling-time utilities for per-segment gap-aware processing.

When each clean data segment is filtered independently (rather than the full
dataset), the Butterworth filter must settle from its initial conditions at the
start and end of every segment.  :func:`compute_settling_trim_samples` computes
how many samples to remove from each end so that the residual filter transient
falls below a configurable level.
"""

import math

__all__ = ["compute_settling_trim_samples"]


def compute_settling_trim_samples(
    filter_kwargs: dict,
    target_fs: float,
    *,
    k: float = 3.0,
) -> int:
    """Samples to trim from each end of a segment after zero-phase filtering.

    For an order-*n* Butterworth highpass filter applied with ``sosfiltfilt``
    (forward + backward pass), the effective time constant of the slowest pole
    is halved compared to a single-pass filter:

    .. math::

        \\tau_{\\text{eff}} = \\frac{1}{2 \\times 2\\pi f_c \\sin\\!\\left(\\pi / (2n)\\right)}

    The required trim from each end to reduce the initial transient to
    :math:`e^{-k}` of its starting amplitude is:

    .. math::

        \\text{trim\\_n} = \\left\\lceil k \\cdot \\tau_{\\text{eff}} \\cdot f_s \\right\\rceil

    For ``order=2`` this simplifies to
    :math:`\\lceil k \\cdot f_s / (\\sqrt{2} \\cdot 2\\pi f_c) \\rceil`.

    Parameters
    ----------
    filter_kwargs : dict
        Must contain ``"highpass_cutoff"`` (Hz).  ``"order"`` (int) is optional
        and defaults to 2.  ``"lowpass_cutoff"`` is ignored — the highpass corner
        dominates settling.
    target_fs : float
        Output sampling rate in Hz (after any downsampling, or the raw rate if
        no downsampling is applied).  Trim is expressed in *output* samples.
    k : float, optional
        Number of effective time constants to trim.  ``k=3`` reduces the
        transient amplitude to :math:`e^{-3} \\approx 5\\%`; ``k=5`` to
        :math:`e^{-5} \\approx 0.7\\%`.  Default ``3.0``.

    Returns
    -------
    trim_n : int
        Number of samples to remove from **each end** of the filtered segment.

    Raises
    ------
    ValueError
        If ``filter_kwargs`` does not contain ``"highpass_cutoff"``, or if any
        of the numerical inputs are non-positive.

    Examples
    --------
    >>> from MojitoProcessor.gaps.settling import compute_settling_trim_samples
    >>> compute_settling_trim_samples(
    ...     {"highpass_cutoff": 5e-5, "order": 2}, target_fs=4.0, k=3.0
    ... )
    27014
    """
    fc = filter_kwargs.get("highpass_cutoff")
    if fc is None:
        raise ValueError(
            "'filter_kwargs' must contain 'highpass_cutoff'. "
            "Settling trim is undefined without a highpass cutoff frequency."
        )
    if fc <= 0:
        raise ValueError(f"'highpass_cutoff' must be positive, got {fc}")
    if target_fs <= 0:
        raise ValueError(f"'target_fs' must be positive, got {target_fs}")
    if k <= 0:
        raise ValueError(f"'k' must be positive, got {k}")

    order = int(filter_kwargs.get("order", 2))
    if order <= 0:
        raise ValueError(f"filter 'order' must be a positive integer, got {order}")

    # Slowest Butterworth pole, halved for sosfiltfilt (forward+backward)
    tau_eff = 1.0 / (2.0 * 2.0 * math.pi * fc * math.sin(math.pi / (2.0 * order)))
    return math.ceil(k * tau_eff * target_fs)
