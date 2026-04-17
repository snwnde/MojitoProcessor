"""
Load data for the Mojito Processor.

Two loaders are provided:

* :func:`load_file` — reads a raw Mojito L1 HDF5 file via the ``mojito``
  package and returns a data dict suitable for
  :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.

* :func:`load_processed` — reads an HDF5 file written by
  :func:`~MojitoProcessor.io.write.write` and returns a
  ``dict[str, SignalProcessor]`` identical in structure to the output of
  :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.
"""

import json
import pathlib
from typing import Dict

import h5py
import mojito
import numpy as np

from ..process.sigprocess import SignalProcessor

__all__ = ["load_file", "load_processed"]


def load_file(
    paths: str | pathlib.Path | list[str | pathlib.Path],
    *,
    load_days: float | None = None,
) -> dict:
    """Load a raw Mojito L1 HDF5 file.

    Uses the ``mojito`` package to open the file and extracts TDI observables,
    light travel times, spacecraft orbits, noise estimates, and metadata into a
    flat dictionary suitable for
    :func:`~MojitoProcessor.process.sigprocess.process_pipeline`.

    Parameters
    ----------
    paths : str, Path, or list thereof
        Path(s) to the Mojito L1 ``.h5`` file(s).
    load_days : float, optional
        Number of days to load from the start of the file (lazy slicing).
        ``None`` loads the full dataset.

    Returns
    -------
    data : dict
        Dictionary containing:

        - ``tdis`` — dict of TDI channel arrays (X, Y, Z, A, E, T)
        - ``fs``, ``dt``, ``t_tdi`` — TDI sampling parameters and timestamps
        - ``ltts``, ``ltt_derivatives``, ``ltt_times`` — light travel times
        - ``orbits``, ``velocities``, ``orbit_times`` — spacecraft kinematics
        - ``noise_estimates`` — frequency-domain noise covariance cubes
        - ``metadata`` — laser frequency and pipeline names
    """
    with mojito.MojitoL1File(paths) as f:
        tdi_sampling = f.tdis.time_sampling
        ltt_sampling = f.ltts.time_sampling
        orbit_sampling = f.orbits.time_sampling

        # Consistent sample counts across all data streams
        n_tdi = (
            int(load_days * 86400 * tdi_sampling.fs) if load_days else tdi_sampling.size
        )
        n_ltt = (
            int(load_days * 86400 * ltt_sampling.fs) if load_days else ltt_sampling.size
        )
        n_orbit = (
            int(load_days * 86400 * orbit_sampling.fs)
            if load_days
            else orbit_sampling.size
        )

        data = {
            # ── TDI observables ──────────────────────────────────────────────────
            "tdis": {
                "X": f.tdis.x2[:n_tdi],
                "Y": f.tdis.y2[:n_tdi],
                "Z": f.tdis.z2[:n_tdi],
                "A": f.tdis.a2[:n_tdi],
                "E": f.tdis.e2[:n_tdi],
                "T": f.tdis.t2[:n_tdi],
            },
            "fs": tdi_sampling.fs,
            "dt": tdi_sampling.dt,
            "t_tdi": tdi_sampling.t(slice(None, n_tdi)),
            # ── Light travel times ───────────────────────────────────────────────
            "ltts": {
                "12": f.ltts.ltt_12[:n_ltt],
                "13": f.ltts.ltt_13[:n_ltt],
                "21": f.ltts.ltt_21[:n_ltt],
                "23": f.ltts.ltt_23[:n_ltt],
                "31": f.ltts.ltt_31[:n_ltt],
                "32": f.ltts.ltt_32[:n_ltt],
            },
            "ltt_derivatives": {
                "12": f.ltts.ltt_derivative_12[:n_ltt],
                "13": f.ltts.ltt_derivative_13[:n_ltt],
                "21": f.ltts.ltt_derivative_21[:n_ltt],
                "23": f.ltts.ltt_derivative_23[:n_ltt],
                "31": f.ltts.ltt_derivative_31[:n_ltt],
                "32": f.ltts.ltt_derivative_32[:n_ltt],
            },
            "ltt_times": ltt_sampling.t(slice(None, n_ltt)),
            # ── Spacecraft orbits ────────────────────────────────────────────────
            "orbits": f.orbits.positions[:n_orbit],  # (n_orbit, 3, 3)
            "velocities": f.orbits.velocities[:n_orbit],  # (n_orbit, 3, 3)
            "orbit_times": orbit_sampling.t(slice(None, n_orbit)),
            # ── Noise estimates (frequency-domain, not truncated) ────────────────
            "noise_estimates": {
                "xyz": f.noise_estimates.xyz[:],
                "aet": f.noise_estimates.aet[:],
            },
            # ── Metadata ─────────────────────────────────────────────────────────
            "metadata": {
                "laser_frequency": f.laser_frequency,
                "pipeline_names": f.pipeline_names,
            },
        }
    return data


def load_processed(
    path: str | pathlib.Path,
) -> tuple[Dict[str, SignalProcessor], dict]:
    """Load processed segments from an HDF5 file written by :func:`~MojitoProcessor.io.write.write`.

    Reconstructs each :class:`~MojitoProcessor.process.sigprocess.SignalProcessor`
    from the stored channel arrays and metadata attributes.  Per-segment orbit
    and LTT data (written under ``/raw/<segment_name>/``) are returned alongside
    the segments in a raw data dict.

    Parameters
    ----------
    path : str or Path
        Path to a ``.h5`` file previously written by
        :func:`~MojitoProcessor.io.write.write`.

    Returns
    -------
    segments : dict of SignalProcessor
        Dictionary mapping segment names to reconstructed
        :class:`~MojitoProcessor.process.sigprocess.SignalProcessor` objects.
    raw_data : dict
        Auxiliary data dict.  Top-level keys:

        - ``'noise_estimates'`` — dict of noise covariance arrays (if present)
        - ``'metadata'`` — dict with ``laser_frequency`` / ``pipeline_names`` (if present)

        Per-segment keys (one entry per written segment, absent if no raw data
        was written for that segment):

        - ``raw_data[seg_name]['orbits']`` — positions array
        - ``raw_data[seg_name]['velocities']`` — velocities array (if present)
        - ``raw_data[seg_name]['orbit_times']`` — orbit timestamps
        - ``raw_data[seg_name]['ltts']`` — dict of LTT arrays
        - ``raw_data[seg_name]['ltt_derivatives']`` — dict of derivative arrays (if present)
        - ``raw_data[seg_name]['ltt_times']`` — LTT timestamps

    Raises
    ------
    ValueError
        If the file does not contain a ``/processed`` group, indicating it was
        not written by :func:`~MojitoProcessor.io.write.write`.

    Examples
    --------
    >>> from MojitoProcessor import write, load_processed
    >>> write("processed.h5", segments, raw_data=data)
    >>> segments, raw = load_processed("processed.h5")
    >>> sp = segments["segment0"]
    >>> orbit_positions = raw["segment0"]["orbits"]
    """
    path = pathlib.Path(path)
    segments: Dict[str, SignalProcessor] = {}
    raw_data: dict = {}

    with h5py.File(path, "r") as f:
        if "processed" not in f:
            raise ValueError(
                f"No '/processed' group found in '{path}'. "
                "Is this a file written by MojitoProcessor.io.write.write()?"
            )
        processed: h5py.Group = f["processed"]  # type: ignore[assignment]
        for seg_name, grp in processed.items():
            channels = json.loads(grp.attrs["channels"])
            data = {ch: grp[ch][:] for ch in channels}
            fs = float(grp.attrs["fs"])
            t0_raw = float(grp.attrs["t0"])
            t0 = None if np.isnan(t0_raw) else t0_raw
            segments[seg_name] = SignalProcessor(data, fs=fs, t0=t0)

        if "raw" not in f:
            return segments, raw_data

        raw_grp: h5py.Group = f["raw"]  # type: ignore[assignment]

        # Top-level: noise estimates
        if "noise_estimates" in raw_grp:
            raw_data["noise_estimates"] = {
                k: raw_grp["noise_estimates"][k][:] for k in raw_grp["noise_estimates"]
            }

        # Top-level: metadata
        if "metadata" in raw_grp:
            meta = raw_grp["metadata"]
            md: dict = {}
            if "laser_frequency" in meta.attrs:
                md["laser_frequency"] = float(meta.attrs["laser_frequency"])
            if "pipeline_names" in meta.attrs:
                md["pipeline_names"] = json.loads(meta.attrs["pipeline_names"])
            raw_data["metadata"] = md

        # Per-segment: orbits and LTTs
        for seg_name in segments:
            if seg_name not in raw_grp:
                continue
            seg_grp: h5py.Group = raw_grp[seg_name]  # type: ignore[assignment]
            seg_raw: dict = {}

            if "orbits" in seg_grp:
                orb = seg_grp["orbits"]
                seg_raw["orbits"] = orb["positions"][:]
                if "velocities" in orb:
                    seg_raw["velocities"] = orb["velocities"][:]
                if "times" in orb:
                    seg_raw["orbit_times"] = orb["times"][:]

            if "ltts" in seg_grp:
                ltt = seg_grp["ltts"]
                seg_raw["ltts"] = {
                    k: ltt[k][:] for k in ltt if k not in ("derivatives", "times")
                }
                if "derivatives" in ltt:
                    seg_raw["ltt_derivatives"] = {
                        k: ltt["derivatives"][k][:] for k in ltt["derivatives"]
                    }
                if "times" in ltt:
                    seg_raw["ltt_times"] = ltt["times"][:]

            if seg_raw:
                raw_data[seg_name] = seg_raw

    return segments, raw_data
