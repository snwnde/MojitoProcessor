"""Unit tests for MojitoProcessor.io.write and MojitoProcessor.io.read.load_processed."""

import json

import h5py
import numpy as np
import pytest

from MojitoProcessor.io.read import load_processed
from MojitoProcessor.io.write import write
from MojitoProcessor.process.sigprocess import SignalProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_T0 = 9.77298893e7  # shared t0 used by both fixtures


@pytest.fixture(name="two_segments")
def _two_segments() -> dict:
    """Two SignalProcessor segments with known data."""
    rng = np.random.default_rng(0)
    n, fs = 500, 4.0
    segments = {}
    for i in range(2):
        data = {ch: rng.standard_normal(n) for ch in ["X", "Y", "Z"]}
        segments[f"segment{i}"] = SignalProcessor(data, fs=fs, t0=_T0 + i * n / fs)
    return segments


@pytest.fixture(name="raw_data")
def _raw_data() -> dict:
    """Minimal raw auxiliary data dict mirroring load_file output.

    Orbit and LTT times are aligned with the two_segments fixture so that the
    per-segment slicing in write() produces non-empty arrays.

    - segment0 spans [_T0, _T0 + 124.75]
    - segment1 spans [_T0 + 125, _T0 + 249.75]
    Both orbit_times and ltt_times cover [_T0, _T0 + 250].
    """
    rng = np.random.default_rng(1)
    n_orbit, n_ltt, n_freq = 50, 200, 100
    # Times span both segments
    orbit_times = _T0 + np.linspace(0, 250, n_orbit)
    ltt_times = _T0 + np.linspace(0, 250, n_ltt)
    return {
        "noise_estimates": {
            "xyz": rng.standard_normal((1, n_freq, 3, 3)),
            "aet": rng.standard_normal((1, n_freq, 3, 3)),
        },
        "ltts": {
            "12": rng.standard_normal(n_ltt),
            "13": rng.standard_normal(n_ltt),
        },
        "ltt_derivatives": {
            "12": rng.standard_normal(n_ltt),
            "13": rng.standard_normal(n_ltt),
        },
        "ltt_times": ltt_times,
        "orbits": rng.standard_normal((n_orbit, 3, 3)),
        "velocities": rng.standard_normal((n_orbit, 3, 3)),
        "orbit_times": orbit_times,
        "metadata": {
            "laser_frequency": 2.816e14,
            "pipeline_names": ["emri", "noise"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests: file creation and basic structure
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteFileCreation:
    def test_creates_file(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        assert out.exists()

    def test_overwrites_existing_file(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        write(out, two_segments)  # second call must not raise
        assert out.exists()

    def test_processed_group_exists(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "processed" in f

    def test_pipeline_params_group_exists(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "pipeline_params" in f

    def test_no_raw_group_without_raw_data(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "raw" not in f

    def test_raw_group_written_when_provided(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "raw" in f


# ─────────────────────────────────────────────────────────────────────────────
# Tests: processed segments
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteProcessedSegments:
    def test_all_segment_groups_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            for key in two_segments:
                assert key in f["processed"]

    def test_channel_datasets_present(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            for ch in ["X", "Y", "Z"]:
                assert ch in f["processed/segment0"]

    def test_time_dataset_present(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            assert "t" in f["processed/segment0"]

    def test_channel_data_correct(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment0"]
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["processed/segment0/X"][:], sp._data["X"])

    def test_time_data_correct(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment0"]
        with h5py.File(out, "r") as f:
            np.testing.assert_allclose(f["processed/segment0/t"][:], sp.t)

    def test_segment_scalar_attrs(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment0"]
        with h5py.File(out, "r") as f:
            grp = f["processed/segment0"]
            assert grp.attrs["fs"] == pytest.approx(sp.fs)
            assert grp.attrs["N"] == sp.N
            assert grp.attrs["dt"] == pytest.approx(sp.dt)
            assert grp.attrs["T"] == pytest.approx(sp.T)
            assert grp.attrs["t0"] == pytest.approx(sp.t0)

    def test_channels_attr_is_json_list(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            channels = json.loads(f["processed/segment0"].attrs["channels"])
        assert set(channels) == {"X", "Y", "Z"}

    def test_t0_none_stored_as_nan(self, tmp_path):
        sp = SignalProcessor({"X": np.zeros(100)}, fs=1.0, t0=None)
        out = tmp_path / "out.h5"
        write(out, {"segment0": sp})
        with h5py.File(out, "r") as f:
            assert np.isnan(f["processed/segment0"].attrs["t0"])

    def test_second_segment_data_correct(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        sp = two_segments["segment1"]
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(f["processed/segment1/Y"][:], sp._data["Y"])


# ─────────────────────────────────────────────────────────────────────────────
# Tests: pipeline parameters
# ─────────────────────────────────────────────────────────────────────────────


class TestWritePipelineParams:
    def test_all_param_subgroups_present(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            for name in ["filter", "downsample", "trim", "truncate", "window"]:
                assert name in f["pipeline_params"]

    def test_filter_attrs_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(
            out,
            two_segments,
            filter_kwargs={"highpass_cutoff": 5e-6, "order": 2},
        )
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/filter"].attrs[
                "highpass_cutoff"
            ] == pytest.approx(5e-6)
            assert f["pipeline_params/filter"].attrs["order"] == 2

    def test_downsample_attrs_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments, downsample_kwargs={"target_fs": 0.2})
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/downsample"].attrs["target_fs"] == pytest.approx(
                0.2
            )

    def test_none_value_stored_as_string(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments, filter_kwargs={"lowpass_cutoff": None})
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/filter"].attrs["lowpass_cutoff"] == "None"

    def test_window_attrs_written(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments, window_kwargs={"window": "tukey", "alpha": 0.05})
        with h5py.File(out, "r") as f:
            assert f["pipeline_params/window"].attrs["window"] == "tukey"
            assert f["pipeline_params/window"].attrs["alpha"] == pytest.approx(0.05)

    def test_empty_kwargs_writes_empty_group(self, tmp_path, two_segments):
        """Passing no kwargs at all must still create the param groups."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        with h5py.File(out, "r") as f:
            grp = f["pipeline_params/filter"]
            assert len(grp.attrs) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: raw auxiliary data
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteRawData:
    # Top-level data (noise estimates, metadata) — unchanged layout
    def test_noise_estimates_groups_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "noise_estimates" in f["raw"]
            assert "xyz" in f["raw/noise_estimates"]
            assert "aet" in f["raw/noise_estimates"]

    def test_noise_xyz_data_correct(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            np.testing.assert_array_equal(
                f["raw/noise_estimates/xyz"][:],
                raw_data["noise_estimates"]["xyz"],
            )

    def test_metadata_laser_frequency(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert f["raw/metadata"].attrs["laser_frequency"] == pytest.approx(2.816e14)

    def test_metadata_pipeline_names(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            names = json.loads(f["raw/metadata"].attrs["pipeline_names"])
        assert "emri" in names
        assert "noise" in names

    # Per-segment orbit data
    def test_orbit_groups_written_per_segment(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            for seg in ["segment0", "segment1"]:
                assert seg in f["raw"], f"/raw/{seg} missing"
                assert "orbits" in f[f"raw/{seg}"]
                assert "positions" in f[f"raw/{seg}/orbits"]
                assert "velocities" in f[f"raw/{seg}/orbits"]
                assert "times" in f[f"raw/{seg}/orbits"]

    def test_orbit_data_is_correct_slice(self, tmp_path, two_segments, raw_data):
        """Orbit positions written for each segment must match the expected time slice."""
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        orbit_times = raw_data["orbit_times"]
        with h5py.File(out, "r") as f:
            for seg_name, sp in two_segments.items():
                t_start, t_end = float(sp.t[0]), float(sp.t[-1])
                i0 = int(np.searchsorted(orbit_times, t_start, side="left"))
                i1 = int(np.searchsorted(orbit_times, t_end, side="right"))
                expected = raw_data["orbits"][i0:i1]
                np.testing.assert_array_equal(
                    f[f"raw/{seg_name}/orbits/positions"][:], expected
                )

    def test_orbit_times_are_within_segment(self, tmp_path, two_segments, raw_data):
        """Orbit times stored for each segment must lie within that segment's window."""
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            for seg_name, sp in two_segments.items():
                t_start, t_end = float(sp.t[0]), float(sp.t[-1])
                orb_t = f[f"raw/{seg_name}/orbits/times"][:]
                assert np.all(orb_t >= t_start)
                assert np.all(orb_t <= t_end)

    def test_orbit_segments_are_disjoint(self, tmp_path, two_segments, raw_data):
        """The two per-segment orbit arrays must together cover the full range
        without duplication (total rows == all orbit samples in the joint window)."""
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            n0 = f["raw/segment0/orbits/positions"].shape[0]
            n1 = f["raw/segment1/orbits/positions"].shape[0]
        # Neither segment should be empty
        assert n0 > 0
        assert n1 > 0
        # Combined should equal the total number of orbit samples in the full window
        orbit_times = raw_data["orbit_times"]
        t_min = float(two_segments["segment0"].t[0])
        t_max = float(two_segments["segment1"].t[-1])
        i0 = int(np.searchsorted(orbit_times, t_min, side="left"))
        i1 = int(np.searchsorted(orbit_times, t_max, side="right"))
        assert n0 + n1 == i1 - i0

    # Per-segment LTT data
    def test_ltt_groups_written_per_segment(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            for seg in ["segment0", "segment1"]:
                assert "ltts" in f[f"raw/{seg}"]
                assert "12" in f[f"raw/{seg}/ltts"]
                assert "13" in f[f"raw/{seg}/ltts"]
                assert "times" in f[f"raw/{seg}/ltts"]
                assert "derivatives" in f[f"raw/{seg}/ltts"]
                assert "12" in f[f"raw/{seg}/ltts/derivatives"]

    def test_ltt_times_are_within_segment(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            for seg_name, sp in two_segments.items():
                t_start, t_end = float(sp.t[0]), float(sp.t[-1])
                ltt_t = f[f"raw/{seg_name}/ltts/times"][:]
                assert np.all(ltt_t >= t_start)
                assert np.all(ltt_t <= t_end)

    # Segments without t0 must not produce raw sub-groups
    def test_no_raw_segment_group_when_t0_is_none(self, tmp_path, raw_data):
        sp = SignalProcessor({"X": np.zeros(100)}, fs=1.0, t0=None)
        out = tmp_path / "out.h5"
        write(out, {"segment0": sp}, raw_data=raw_data)
        with h5py.File(out, "r") as f:
            assert "segment0" not in f["raw"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_processed (round-trip with write)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadProcessed:
    """Round-trip tests: write then load_processed must recover the original data."""

    def test_returns_tuple(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        result = load_processed(out)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_dict(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        assert isinstance(segments, dict)

    def test_second_element_is_dict(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        _, raw = load_processed(out)
        assert isinstance(raw, dict)

    def test_segment_keys_match(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        assert set(segments.keys()) == set(two_segments.keys())

    def test_values_are_signal_processors(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        for sp in segments.values():
            assert isinstance(sp, SignalProcessor)

    def test_fs_round_trips(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        for key, sp in two_segments.items():
            assert segments[key].fs == pytest.approx(sp.fs)

    def test_t0_round_trips(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        for key, sp in two_segments.items():
            assert segments[key].t0 == pytest.approx(sp.t0)

    def test_t0_none_round_trips(self, tmp_path):
        sp = SignalProcessor({"X": np.zeros(100)}, fs=1.0, t0=None)
        out = tmp_path / "out.h5"
        write(out, {"segment0": sp})
        segments, _ = load_processed(out)
        assert segments["segment0"].t0 is None

    def test_channel_data_round_trips(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        for key, sp in two_segments.items():
            for ch in sp.channels:
                np.testing.assert_array_equal(segments[key]._data[ch], sp._data[ch])

    def test_channels_list_round_trips(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        for key, sp in two_segments.items():
            assert set(segments[key].channels) == set(sp.channels)

    def test_N_round_trips(self, tmp_path, two_segments):
        out = tmp_path / "out.h5"
        write(out, two_segments)
        segments, _ = load_processed(out)
        for key, sp in two_segments.items():
            assert segments[key].N == sp.N

    def test_raises_for_non_mojitoprocessor_file(self, tmp_path):
        out = tmp_path / "other.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("data", data=np.zeros(10))
        with pytest.raises(ValueError, match="processed"):
            load_processed(out)

    def test_raw_empty_without_raw_data(self, tmp_path, two_segments):
        """When write() is called without raw_data, the raw dict must be empty."""
        out = tmp_path / "out.h5"
        write(out, two_segments)
        _, raw = load_processed(out)
        assert raw == {}

    def test_raw_noise_estimates_round_trip(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        _, raw = load_processed(out)
        np.testing.assert_array_equal(
            raw["noise_estimates"]["xyz"], raw_data["noise_estimates"]["xyz"]
        )

    def test_raw_metadata_round_trip(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        _, raw = load_processed(out)
        assert raw["metadata"]["laser_frequency"] == pytest.approx(2.816e14)
        assert "emri" in raw["metadata"]["pipeline_names"]

    def test_per_segment_orbit_keys_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        _, raw = load_processed(out)
        for seg in ["segment0", "segment1"]:
            assert seg in raw
            assert "orbits" in raw[seg]
            assert "velocities" in raw[seg]
            assert "orbit_times" in raw[seg]

    def test_per_segment_ltt_keys_present(self, tmp_path, two_segments, raw_data):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        _, raw = load_processed(out)
        for seg in ["segment0", "segment1"]:
            assert "ltts" in raw[seg]
            assert "ltt_derivatives" in raw[seg]
            assert "ltt_times" in raw[seg]

    def test_per_segment_orbit_data_matches_write(
        self, tmp_path, two_segments, raw_data
    ):
        """Orbit positions returned by load_processed must match what write() stored."""
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        _, raw = load_processed(out)
        orbit_times = raw_data["orbit_times"]
        for seg_name, sp in two_segments.items():
            t_start, t_end = float(sp.t[0]), float(sp.t[-1])
            i0 = int(np.searchsorted(orbit_times, t_start, side="left"))
            i1 = int(np.searchsorted(orbit_times, t_end, side="right"))
            expected = raw_data["orbits"][i0:i1]
            np.testing.assert_array_equal(raw[seg_name]["orbits"], expected)

    def test_per_segment_orbit_times_within_segment(
        self, tmp_path, two_segments, raw_data
    ):
        out = tmp_path / "out.h5"
        write(out, two_segments, raw_data=raw_data)
        _, raw = load_processed(out)
        for seg_name, sp in two_segments.items():
            t_start, t_end = float(sp.t[0]), float(sp.t[-1])
            orb_t = raw[seg_name]["orbit_times"]
            assert np.all(orb_t >= t_start)
            assert np.all(orb_t <= t_end)
