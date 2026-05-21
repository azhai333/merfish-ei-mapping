import argparse
import time
import sys
import numpy as np
import nrrd
import pandas as pd
from pathlib import Path

from config import (
    DATA_DIR, RESULTS_DIR, DENSITY_DIR, FIGURES_DIR,
    DEV_SAMPLE_FRAC,
    BLOB_MIN_SIGMA, BLOB_MAX_SIGMA, BLOB_NUM_SIGMA, BLOB_THRESHOLD,
)
from download_data import (
    load_ccf_annotation, load_taxonomy, load_parcellation_terms,
    load_merfish_with_celltypes, inspect_merged_df,
)
from make_density_maps import make_density_maps, compute_ei_ratio
from preprocess import preprocess_pair
from detect_peaks import run_peak_detection
from atlas_analysis import (
    build_parcellation_map, build_region_stats,
    compute_subregion_stats, test_ei_cortex_vs_striatum,
)
from qc import run_all_qc
from visualize import make_all_figures


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true",
                   help=f"Use {DEV_SAMPLE_FRAC*100:.0f}%% sample (fast)")
    p.add_argument("--no-peaks", action="store_true", help="Skip blob detection")
    p.add_argument("--skip-maps", action="store_true",
                   help="Use cached NRRD density maps — skip download and map creation")
    p.add_argument("--force", action="store_true", help="Re-download all files")
    return p.parse_args()


def load_cached_maps():
    excit, _ = nrrd.read(str(DENSITY_DIR / "excitatory_density.nrrd"))
    inhib, _ = nrrd.read(str(DENSITY_DIR / "inhibitory_density.nrrd"))
    print(f"  Loaded cached density maps: {excit.shape}")
    return excit, inhib


def main():
    args = parse_args()
    t0 = time.time()

    for d in [DATA_DIR, RESULTS_DIR, DENSITY_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.dev:
        print("DEV MODE: 5% sample\n")
    sample_frac = DEV_SAMPLE_FRAC if args.dev else None

    print("\n--- atlas ---")
    ann, ds_step = load_ccf_annotation(force=args.force)

    print("\nLoading parcellation terms...")
    parc_df = load_parcellation_terms(force=args.force)
    pmap = build_parcellation_map(parc_df)
    print(f"  {len(pmap)} structures in parcellation map")

    print("\n--- merfish data (Zhang et al. 2023, Science) ---")
    if args.skip_maps:
        print("Skipping download (--skip-maps)")
        df = None
    else:
        df = load_merfish_with_celltypes(sample_frac=sample_frac, force=args.force)
        inspect_merged_df(df)

    print("\n--- density maps ---")
    if args.skip_maps:
        excit_raw, inhib_raw = load_cached_maps()
    else:
        excit_raw, inhib_raw, _ = make_density_maps(df, ann.shape)

    print("\n--- preprocessing ---")
    excit_pp, inhib_pp = preprocess_pair(excit_raw, inhib_raw)
    ei_ratio = compute_ei_ratio(excit_pp, inhib_pp)

    del excit_raw, inhib_raw

    print("\n--- peak detection ---")
    if args.no_peaks:
        print("Skipping (--no-peaks)")
        excit_peaks, inhib_peaks = [], []
    else:
        excit_peaks, inhib_peaks = run_peak_detection(
            excit_pp, inhib_pp,
            use_log=True,
            min_sigma=BLOB_MIN_SIGMA, max_sigma=BLOB_MAX_SIGMA,
            num_sigma=BLOB_NUM_SIGMA, threshold=BLOB_THRESHOLD,
        )

    print("\n--- regional analysis ---")
    region_df = build_region_stats(excit_pp, inhib_pp, ei_ratio, ann, pmap)
    print("\nMajor forebrain regions:")
    cols = ["region", "ei_ratio_mean", "ei_ratio_median", "coverage_frac"]
    print(region_df[cols].to_string(index=False))

    print("\nComputing sub-region stats...")
    subregion_df = compute_subregion_stats(excit_pp, inhib_pp, ei_ratio, ann, pmap)
    print(f"  {len(subregion_df)} sub-regions")

    region_df.to_csv(RESULTS_DIR / "region_stats.csv", index=False)
    subregion_df.to_csv(RESULTS_DIR / "subregion_stats.csv", index=False)
    print(f"saved to {RESULTS_DIR}/")

    print("\n--- validation ---")
    val = test_ei_cortex_vs_striatum(excit_pp, inhib_pp, ei_ratio, ann, pmap)

    print("\n--- qc ---")
    qc, all_passed = run_all_qc(
        excit_pp, inhib_pp, ei_ratio, ann,
        subregion_df, excit_peaks, inhib_peaks
    )

    print("\n--- figures ---")
    make_all_figures(
        excit_pp, inhib_pp, ei_ratio,
        ann, pmap, subregion_df,
        excit_peaks, inhib_peaks
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  |  QC: {'PASSED' if all_passed else 'FAILED'}")
    print(f"figures → {FIGURES_DIR}/  results → {RESULTS_DIR}/")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
