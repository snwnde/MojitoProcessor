"""
Process gapped LISA TDI data and write the first N clean segments to HDF5.

Processing steps
----------------
1. Load raw data
2. Generate smoothed gap mask (lisaglitch + lisagap)
3. Apply mask to raw data
4. Run processing pipeline (highpass → downsample → trim → window)
5. Compute extended mask (filter leakage at low rate, upsample)
6. Extract clean segments
7. Tukey-taper the first N segments
8. Write to HDF5

Run from the project root:
    uv run python run_gaps_pipeline.py
"""

import logging

import numpy as np
from lisagap import GapWindowGenerator
from lisaglitch import GapMaskGenerator
from scipy.signal import windows

import MojitoProcessor as mp
import MojitoProcessor.gaps as gaps

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger("run_gaps_pipeline")

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_FILE = "Mojito_Data/NOISE_731d_0.25s_L1_source0_0_20251206T220508924302Z.h5"
OUTPUT_FILE = "Processed_Data/gaps_clean_segments.h5"
N_SEGMENTS = 10

# ── Pipeline parameters ────────────────────────────────────────────────────────
downsample_kwargs = {
    "target_fs": 4.0,
    "kaiser_window": 31.0,
}
filter_kwargs = {
    "highpass_cutoff": 5e-5,
    "order": 1,
}
trim_kwargs = {
    "fraction": 0.02,
}
truncate_kwargs = {
    "days": None,
}
window_kwargs = {
    "window": "tukey",
    "alpha": 0.0,
}

# ── Gap schedule ───────────────────────────────────────────────────────────────
gap_definitions = {
    "planned": {
        "antenna_repointing": {
            "rate_per_year": 26,
            "duration_hr": 7,
        }
    },
    "unplanned": {},
}
taper_definitions = {
    "planned": {
        "antenna_repointing": {"lobe_lengths_hr": 3.0},
    },
    "unplanned": {},
}

# ── 1. Load data ───────────────────────────────────────────────────────────────
log.info("Loading %s ...", INPUT_FILE)
data = mp.io.load_file(INPUT_FILE)
n_samples = len(data["tdis"]["X"])
duration = n_samples * data["dt"]
log.info(
    "Loaded %d samples @ %.1f Hz (%.2f days)", n_samples, data["fs"], duration / 86400
)

# ── 2. Generate gap mask ───────────────────────────────────────────────────────
log.info("Generating gap mask ...")
gap_gen = GapMaskGenerator(
    sim_t=data["t_tdi"],
    gap_definitions=gap_definitions,
    treat_as_nan=False,
)
window_func = GapWindowGenerator(gap_gen)
smoothed_mask = window_func.generate_window(
    include_planned=True,
    include_unplanned=False,
    apply_tapering=True,
    taper_definitions=taper_definitions,
)[0]
smoothed_mask = windows.tukey(len(smoothed_mask), alpha=0.01) * smoothed_mask
log.info("Raw gap fraction: %.4f%%", (1 - smoothed_mask.mean()) * 100)

# ── 3. Apply mask to raw data ─────────────────────────────────────────────────
log.info("Applying mask to raw data ...")
data_gapped = gaps.apply_raw_mask(data, smoothed_mask)

# ── 4. Run processing pipeline ────────────────────────────────────────────────
log.info("Running processing pipeline ...")
processed_segments = mp.process_pipeline(
    data_gapped,
    downsample_kwargs=downsample_kwargs,
    filter_kwargs=filter_kwargs,
    trim_kwargs=trim_kwargs,
    truncate_kwargs=truncate_kwargs,
    window_kwargs=window_kwargs,
)
del data_gapped

sp_0 = processed_segments["segment0"]
log.info("segment0: N=%d, fs=%.1f Hz, T=%.2f days", sp_0.N, sp_0.fs, sp_0.T / 86400)

# ── 5. Compute extended mask ───────────────────────────────────────────────────
log.info("Computing extended mask (mask_fs=0.2 Hz) ...")
extended_mask, _ = gaps.compute_extended_mask(
    smoothed_mask,
    sp_0,
    filter_kwargs,
    downsample_kwargs,
    trim_kwargs,
    fs_raw=data["fs"],
    contamination_threshold=1e-4,
    min_clean_hours=8.0,
    mask_fs=0.2,
)
del smoothed_mask
log.info("Extended gap fraction: %.4f%%", (1 - extended_mask.mean()) * 100)

# ── 6. Extract clean segments ─────────────────────────────────────────────────
log.info("Extracting clean segments ...")
sp_0_segments = gaps.extract_clean_segments(
    sp_0,
    extended_mask,
    min_clean_hours=8.0,
    max_segment_days=7.0,
)
del extended_mask
log.info("Found %d clean segment(s), writing first %d", len(sp_0_segments), N_SEGMENTS)

# ── 7. Tukey-taper the first N segments ───────────────────────────────────────
tukey_alpha = 0.05
segment_ids = list(range(min(N_SEGMENTS, len(sp_0_segments))))
for i in segment_ids:
    seg = sp_0_segments[f"segment{i}"]
    seg.apply_window("tukey", alpha=tukey_alpha)
    log.info("segment%d: Tukey α=%.2f applied, N=%d", i, tukey_alpha, seg.N)

# ── 8. Write to file ──────────────────────────────────────────────────────────
log.info("Writing to %s ...", OUTPUT_FILE)
mp.io.write(
    OUTPUT_FILE,
    sp_0_segments,
    raw_data=data,
    segment_ids=segment_ids,
    filter_kwargs=filter_kwargs,
    downsample_kwargs=downsample_kwargs,
    trim_kwargs=trim_kwargs,
    truncate_kwargs=truncate_kwargs,
    window_kwargs=window_kwargs,
)
log.info("Done.")
