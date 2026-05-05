"""
gapspipeline — Process and Write Clean Segments to HDF5

Runs the full gap-aware pipeline using MojitoProcessor.pipelines.gapspipeline:

    1. Load raw L1 data
    2. Apply smoothed gap mask to TDI channels
    3. Filter → downsample → trim
    4. Compute extended mask (Butterworth leakage around gaps)
    5. Extract contiguous clean segments
    6. Window each segment independently
    7. Write clean segments + raw auxiliary data to HDF5

Then loads the written file back and spot-checks the output.
"""

import logging

import numpy as np
from lisagap import GapWindowGenerator
from lisaglitch import GapMaskGenerator
from scipy.signal import windows

import MojitoProcessor as mp
from MojitoProcessor.pipelines.gapspipeline import gapspipeline

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# ── 1. Configuration ──────────────────────────────────────────────────────────

mojito_data_file = (
    "../Mojito_Data/NOISE_731d_0.25s_L1_source0_0_20251206T220508924302Z.h5"
)
output_file = "../Mojito_Data/clean_segments_gaps_pipeline.h5"

# Set to a float (days) to load only a subset of the file
load_days = None

downsample_kwargs = {
    "target_fs": 4.0,
    "kaiser_window": 31.0,
}
filter_kwargs = {
    "highpass_cutoff": 5e-5,
    "order": 2,
}
trim_kwargs = {
    "fraction": 0.015,
}
window_kwargs = {
    "window": "tukey",
    "alpha": 0.05,
}

contamination_threshold = 1e-4
min_clean_hours = 8.0
max_segment_days = 7.0

# ── 2. Generate the Smoothed Gap Mask ─────────────────────────────────────────

data_meta = mp.io.load_file(mojito_data_file, load_days=load_days)

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

gap_gen = GapMaskGenerator(
    sim_t=data_meta["t_tdi"],
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

# Free the mask-generation objects: data_meta holds all TDI channels (~12 GB)
# purely to supply t_tdi to GapMaskGenerator, and is not needed after this point.
del data_meta, gap_gen, window_func

print(
    f"Raw mask: {len(smoothed_mask):,} samples, gap fraction {1 - smoothed_mask.mean():.4%}"
)

# ── 3. Run the Full Pipeline and Write to HDF5 ────────────────────────────────

clean_segments, extended_mask = gapspipeline(
    mojito_data_file,
    smoothed_mask,
    load_days=load_days,
    filter_kwargs=filter_kwargs,
    downsample_kwargs=downsample_kwargs,
    trim_kwargs=trim_kwargs,
    window_kwargs=window_kwargs,
    contamination_threshold=contamination_threshold,
    min_clean_hours=min_clean_hours,
    max_segment_days=max_segment_days,
    output_path=output_file,
)

print(f"\nExtracted {len(clean_segments)} clean segment(s):")
for name, sp in clean_segments.items():
    print(
        f"  {name}: N={sp.N:>7,}, fs={sp.fs} Hz, "
        f"t0={sp.t0 / 86400:.4f} d, duration={sp.T / 3600:.1f} h ({sp.T / 86400:.2f} d)"
    )
print(f"\nExtended mask: {extended_mask.mean():.4%} clean")
print(f"Written to: {output_file}")

# ── 4. Verify — Load the Written File Back ────────────────────────────────────

loaded_segments, loaded_raw = mp.io.load_processed(output_file)

print(f"\nLoaded {len(loaded_segments)} segment(s) from {output_file}")
print()
for name, sp in loaded_segments.items():
    orig = clean_segments[name]
    t0_match = np.isclose(sp.t0, orig.t0)
    data_match = all(np.allclose(sp.data[ch], orig.data[ch]) for ch in sp.channels)
    print(
        f"  {name}: N={sp.N:>7,}, fs={sp.fs} Hz, "
        f"t0 match={t0_match}, data match={data_match}"
    )

if loaded_raw is not None:
    print(f"\nRaw auxiliary data keys: {list(loaded_raw.keys())}")
