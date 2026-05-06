"""Tests for the v2 segment-first gap-aware pipeline."""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from MojitoProcessor.gaps.settling import compute_settling_trim_samples
from MojitoProcessor.process.sigprocess import SignalProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

FS = 4.0  # Hz — sampling rate used in most tests
FC = 5e-5  # Hz — highpass cutoff
ORDER = 2
K = 3.0

# Expected trim for the standard configuration (used as a regression anchor)
# tau_eff = 1 / (2 * 2*pi * 5e-5 * sin(pi/4)) = 1 / (sqrt(2) * 2*pi * 5e-5)
# trim_n  = ceil(3.0 * tau_eff * 4.0)
_TAU_EFF = 1.0 / (2.0 * 2.0 * math.pi * FC * math.sin(math.pi / (2.0 * ORDER)))
_EXPECTED_TRIM = math.ceil(K * _TAU_EFF * FS)  # = 27014


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeSettlingTrimSamples
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeSettlingTrimSamples:
    def _call(self, fc=FC, order=ORDER, fs=FS, k=K):
        return compute_settling_trim_samples(
            {"highpass_cutoff": fc, "order": order}, fs, k=k
        )

    def test_returns_int(self):
        assert isinstance(self._call(), int)

    def test_order2_regression(self):
        """Standard configuration must hit the pre-computed anchor."""
        assert self._call() == _EXPECTED_TRIM

    def test_order2_matches_simplified_formula(self):
        """For order=2 the result equals ceil(k*fs / (sqrt(2)*2*pi*fc))."""
        expected = math.ceil(K * FS / (math.sqrt(2) * 2 * math.pi * FC))
        assert self._call() == expected

    def test_higher_order_gives_larger_trim(self):
        """sin(pi/(2n)) decreases with n → tau_eff increases → trim_n increases."""
        trim2 = self._call(order=2)
        trim4 = self._call(order=4)
        assert trim4 > trim2

    def test_larger_k_gives_larger_trim(self):
        trim3 = self._call(k=3.0)
        trim5 = self._call(k=5.0)
        assert trim5 > trim3

    def test_higher_fs_gives_more_output_samples(self):
        """Same wall-clock settling time → more samples at higher fs."""
        trim_4 = self._call(fs=4.0)
        trim_8 = self._call(fs=8.0)
        assert trim_8 > trim_4

    def test_lower_cutoff_gives_larger_trim(self):
        """Lower fc → longer time constant → more samples."""
        trim_hi = self._call(fc=5e-4)
        trim_lo = self._call(fc=5e-5)
        assert trim_lo > trim_hi

    def test_lowpass_only_key_ignored(self):
        """lowpass_cutoff in filter_kwargs is silently ignored."""
        result = compute_settling_trim_samples(
            {"highpass_cutoff": FC, "order": ORDER, "lowpass_cutoff": 0.1},
            FS,
            k=K,
        )
        assert result == _EXPECTED_TRIM

    def test_default_order_is_2(self):
        """Omitting 'order' should give the same result as order=2."""
        without_order = compute_settling_trim_samples({"highpass_cutoff": FC}, FS, k=K)
        with_order = compute_settling_trim_samples(
            {"highpass_cutoff": FC, "order": 2}, FS, k=K
        )
        assert without_order == with_order

    # ── Error cases ──────────────────────────────────────────────────────────

    def test_raises_missing_highpass(self):
        with pytest.raises(ValueError, match="highpass_cutoff"):
            compute_settling_trim_samples({"lowpass_cutoff": 0.1}, FS)

    def test_raises_empty_dict(self):
        with pytest.raises(ValueError, match="highpass_cutoff"):
            compute_settling_trim_samples({}, FS)

    def test_raises_nonpositive_highpass(self):
        with pytest.raises(ValueError, match="positive"):
            compute_settling_trim_samples({"highpass_cutoff": 0.0}, FS)

    def test_raises_negative_highpass(self):
        with pytest.raises(ValueError, match="positive"):
            compute_settling_trim_samples({"highpass_cutoff": -1e-4}, FS)

    def test_raises_nonpositive_fs(self):
        with pytest.raises(ValueError, match="positive"):
            compute_settling_trim_samples({"highpass_cutoff": FC}, 0.0)

    def test_raises_nonpositive_k(self):
        with pytest.raises(ValueError, match="positive"):
            compute_settling_trim_samples({"highpass_cutoff": FC}, FS, k=0.0)

    def test_raises_zero_order(self):
        with pytest.raises(ValueError):
            compute_settling_trim_samples({"highpass_cutoff": FC, "order": 0}, FS)


# ─────────────────────────────────────────────────────────────────────────────
# TestTrimNSamples  (new keyword path on SignalProcessor.trim)
# ─────────────────────────────────────────────────────────────────────────────


def _make_sp(n: int = 1000, fs: float = 1.0, t0: float = 0.0) -> SignalProcessor:
    rng = np.random.default_rng(42)
    return SignalProcessor(
        {ch: rng.standard_normal(n) for ch in ["X", "Y", "Z"]},
        fs=fs,
        t0=t0,
    )


class TestTrimNSamples:
    def test_removes_correct_endpoints(self):
        sp = _make_sp(n=100)
        original_x = sp.data["X"].copy()
        n = 10
        sp.trim(n_samples=n)
        np.testing.assert_array_equal(sp.data["X"], original_x[n:-n])

    def test_updates_n(self):
        sp = _make_sp(n=100)
        sp.trim(n_samples=10)
        assert sp.N == 80

    def test_updates_t0(self):
        sp = _make_sp(n=100, fs=4.0, t0=1000.0)
        sp.trim(n_samples=10)
        assert np.isclose(sp.t0, 1000.0 + 10 * sp.dt)

    def test_zero_n_samples_is_noop(self):
        sp = _make_sp(n=100)
        original_x = sp.data["X"].copy()
        sp.trim(n_samples=0)
        np.testing.assert_array_equal(sp.data["X"], original_x)
        assert sp.N == 100

    def test_n_samples_too_large_raises(self):
        sp = _make_sp(n=100)
        with pytest.raises(ValueError, match="remove all"):
            sp.trim(n_samples=50)  # 2*50 == 100 == N

    def test_raises_if_both_provided(self):
        sp = _make_sp()
        with pytest.raises(ValueError, match="not both"):
            sp.trim(fraction=0.1, n_samples=10)

    def test_raises_if_neither_provided(self):
        sp = _make_sp()
        with pytest.raises(ValueError, match="Exactly one"):
            sp.trim()

    def test_fraction_keyword_still_works(self):
        sp = _make_sp(n=100)
        original_x = sp.data["X"].copy()
        sp.trim(fraction=0.1)
        # 10% of 100 = 10 total → 5 from each end
        np.testing.assert_array_equal(sp.data["X"], original_x[5:-5])

    def test_fraction_positional_still_works(self):
        sp = _make_sp(n=100)
        original_x = sp.data["X"].copy()
        sp.trim(0.1)  # positional
        np.testing.assert_array_equal(sp.data["X"], original_x[5:-5])

    def test_n_samples_negative_raises(self):
        sp = _make_sp()
        with pytest.raises(ValueError):
            sp.trim(n_samples=-1)

    def test_all_channels_trimmed(self):
        sp = _make_sp(n=100)
        sp.trim(n_samples=10)
        for ch in sp.channels:
            assert len(sp.data[ch]) == 80


# ─────────────────────────────────────────────────────────────────────────────
# TestGapspipelineV2  (integration tests using mocked load_file)
# ─────────────────────────────────────────────────────────────────────────────

_N_RAW = 4000
_FS_RAW = 4.0
_LASER_FREQ = 2.816e14


def _make_raw_data(n: int = _N_RAW, fs: float = _FS_RAW) -> dict:
    """Minimal raw data dict matching the shape of load_file output."""
    rng = np.random.default_rng(0)
    return {
        "tdis": {ch: rng.standard_normal(n) for ch in ["X", "Y", "Z"]},
        "fs": fs,
        "dt": 1.0 / fs,
        "t_tdi": np.arange(n) / fs,
        "ltts": {},
        "ltt_derivatives": {},
        "ltt_times": np.array([]),
        "orbits": np.zeros((10, 3, 3)),
        "velocities": np.zeros((10, 3, 3)),
        "orbit_times": np.zeros(10),
        "noise_estimates": {
            "xyz": np.ones((1, 100, 3, 3)) * 1e-40,
            "aet": np.ones((1, 100, 3, 3)) * 1e-40,
        },
        "metadata": {
            "laser_frequency": _LASER_FREQ,
            "pipeline_names": ["test"],
        },
    }


def _make_binary_mask(n: int = _N_RAW, gap_start: int = 1800, gap_end: int = 2200):
    mask = np.ones(n, dtype=int)
    mask[gap_start:gap_end] = 0
    return mask


_FILTER_KW = {"highpass_cutoff": 0.05, "order": 2}
_DS_KW = {"target_fs": 1.0}
_WIN_KW = {"window": "tukey", "alpha": 0.05}


class TestGapspipelineV2:
    """Integration tests — load_file is mocked to avoid real HDF5 I/O."""

    def _run(self, mask=None, extra_kw=None):
        from MojitoProcessor.pipelines.gapspipeline_v2 import gapspipeline_v2

        if mask is None:
            mask = _make_binary_mask()
        kw = dict(
            filter_kwargs=_FILTER_KW,
            downsample_kwargs=_DS_KW,
            trim_settling_factor=1.0,  # small k so short test segments survive
            window_kwargs=_WIN_KW,
            min_clean_hours=0.0,  # keep even very short segments
        )
        if extra_kw:
            kw.update(extra_kw)

        with patch(
            "MojitoProcessor.pipelines.gapspipeline_v2.load_file",
            return_value=_make_raw_data(),
        ):
            return gapspipeline_v2("dummy.h5", mask, **kw)

    def test_returns_dict(self):
        result = self._run()
        assert isinstance(result, dict)

    def test_returns_signal_processors(self):
        result = self._run()
        assert all(isinstance(v, SignalProcessor) for v in result.values())

    def test_not_a_tuple(self):
        """v2 returns Dict only, not a (Dict, ndarray) tuple like v1."""
        result = self._run()
        assert not isinstance(result, tuple)

    def test_segment_naming(self):
        result = self._run()
        for k in result:
            assert k.startswith("segment")

    def test_all_ones_mask_single_segment(self):
        mask = np.ones(_N_RAW, dtype=int)
        result = self._run(mask=mask)
        assert len(result) == 1

    def test_all_zeros_mask_empty(self):
        mask = np.zeros(_N_RAW, dtype=int)
        result = self._run(mask=mask)
        assert len(result) == 0

    def test_gap_produces_two_segments(self):
        result = self._run()
        assert len(result) == 2

    def test_segments_have_target_fs(self):
        result = self._run()
        for sp in result.values():
            assert np.isclose(sp.fs, _DS_KW["target_fs"])

    def test_no_downsampling_keeps_raw_fs(self):
        result = self._run(extra_kw={"downsample_kwargs": None})
        for sp in result.values():
            assert np.isclose(sp.fs, _FS_RAW)

    def test_t0_set_on_segments(self):
        result = self._run()
        for sp in result.values():
            assert sp.t0 is not None

    def test_t0_values_ordered(self):
        result = self._run()
        t0s = [sp.t0 for sp in result.values()]
        assert t0s == sorted(t0s)

    def test_binary_mask_validation_rejects_float(self):
        from MojitoProcessor.pipelines.gapspipeline_v2 import gapspipeline_v2

        bad_mask = np.linspace(0, 1, _N_RAW)  # smoothed — should be rejected
        with patch(
            "MojitoProcessor.pipelines.gapspipeline_v2.load_file",
            return_value=_make_raw_data(),
        ):
            with pytest.raises(ValueError, match="smoothed"):
                gapspipeline_v2("dummy.h5", bad_mask, filter_kwargs=_FILTER_KW)

    def test_mask_length_mismatch_raises(self):
        from MojitoProcessor.pipelines.gapspipeline_v2 import gapspipeline_v2

        short_mask = np.ones(_N_RAW // 2, dtype=int)
        with patch(
            "MojitoProcessor.pipelines.gapspipeline_v2.load_file",
            return_value=_make_raw_data(),
        ):
            with pytest.raises(ValueError, match="length"):
                gapspipeline_v2("dummy.h5", short_mask, filter_kwargs=_FILTER_KW)

    def test_short_segment_discarded_when_trim_too_large(self):
        """With a very large k, short segments should be skipped."""
        result = self._run(extra_kw={"trim_settling_factor": 1000.0})
        # All segments should have been dropped
        assert len(result) == 0

    def test_no_apply_raw_mask_called(self):
        """v2 must not call apply_raw_mask (v1 function)."""
        from MojitoProcessor.pipelines import gapspipeline_v2

        with (
            patch(
                "MojitoProcessor.pipelines.gapspipeline_v2.load_file",
                return_value=_make_raw_data(),
            ) as _,
            patch(
                "MojitoProcessor.pipelines.gapspipeline_v2.extract_clean_segments",
                wraps=__import__(
                    "MojitoProcessor.gaps.segment", fromlist=["extract_clean_segments"]
                ).extract_clean_segments,
            ),
        ):
            # Import apply_raw_mask and confirm it is never touched
            with patch("MojitoProcessor.gaps.mask.apply_raw_mask") as mock_arm:
                gapspipeline_v2(
                    "dummy.h5",
                    _make_binary_mask(),
                    filter_kwargs=_FILTER_KW,
                    trim_settling_factor=1.0,
                    min_clean_hours=0.0,
                )
                mock_arm.assert_not_called()

    def test_window_not_applied_when_kwargs_absent(self):
        """Omitting window_kwargs must produce un-windowed segments."""
        from MojitoProcessor.pipelines.gapspipeline_v2 import gapspipeline_v2

        with patch(
            "MojitoProcessor.pipelines.gapspipeline_v2.load_file",
            return_value=_make_raw_data(),
        ):
            result = gapspipeline_v2(
                "dummy.h5",
                _make_binary_mask(),
                filter_kwargs=_FILTER_KW,
                downsample_kwargs=_DS_KW,
                trim_settling_factor=1.0,
                window_kwargs=None,
                min_clean_hours=0.0,
            )
        assert len(result) > 0

    def test_write_called_when_output_path_given(self, tmp_path):
        from MojitoProcessor.pipelines.gapspipeline_v2 import gapspipeline_v2

        out = tmp_path / "out.h5"
        with (
            patch(
                "MojitoProcessor.pipelines.gapspipeline_v2.load_file",
                return_value=_make_raw_data(),
            ),
            patch("MojitoProcessor.pipelines.gapspipeline_v2.write") as mock_write,
        ):
            gapspipeline_v2(
                "dummy.h5",
                _make_binary_mask(),
                filter_kwargs=_FILTER_KW,
                trim_settling_factor=1.0,
                min_clean_hours=0.0,
                output_path=str(out),
            )
            mock_write.assert_called_once()
