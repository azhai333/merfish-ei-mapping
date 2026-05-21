import numpy as np
import pandas as pd
import nrrd
from scipy.ndimage import gaussian_filter

from config import DENSITY_DIR, VOXEL_SIZE_UM, GAUSSIAN_SIGMA


DENSITY_DIR.mkdir(parents=True, exist_ok=True)

# coords are in mm, multiply by this to get voxel index at VOXEL_SIZE_UM resolution
MM_TO_VOXEL = 1000.0 / VOXEL_SIZE_UM


def classify_cells(df: pd.DataFrame):
    nt = df["neurotransmitter"].fillna("")

    # GABA-Glyc counts as inhibitory, Glut-GABA is ambiguous so skip
    glut = df[nt.str.startswith("Glut") & ~nt.str.contains("-")].copy()
    gaba = df[nt.isin(["GABA", "GABA-Glyc"])].copy()
    other = df[~df.index.isin(glut.index) & ~df.index.isin(gaba.index)]

    print(f"  Glutamatergic (excitatory): {len(glut):>10,}")
    print(f"  GABAergic    (inhibitory):  {len(gaba):>10,}")
    print(f"  Other/mixed/non-neuronal:   {len(other):>10,}  (not used)")

    return glut, gaba


def cells_to_density(cells_df: pd.DataFrame, atlas_shape: tuple, sigma: float) -> np.ndarray:
    # CCF axes: x=AP, y=DV, z=ML — matches atlas axis order (0, 1, 2)
    ap = (cells_df["x"].values * MM_TO_VOXEL).astype(int)
    dv = (cells_df["y"].values * MM_TO_VOXEL).astype(int)
    ml = (cells_df["z"].values * MM_TO_VOXEL).astype(int)

    valid = (
        (ap >= 0) & (ap < atlas_shape[0]) &
        (dv >= 0) & (dv < atlas_shape[1]) &
        (ml >= 0) & (ml < atlas_shape[2])
    )
    if (~valid).sum() > 0:
        print(f"  {(~valid).sum():,} cells ({(~valid).mean()*100:.1f}%) outside atlas bounds — skipped")

    density = np.zeros(atlas_shape, dtype=np.float32)
    np.add.at(density, (ap[valid], dv[valid], ml[valid]), 1)
    print(f"  Raw: {(density > 0).sum():,} occupied voxels, max {density.max():.0f} cells/voxel")

    return gaussian_filter(density, sigma=sigma)


def compute_ei_ratio(excit: np.ndarray, inhib: np.ndarray, min_total=0.005) -> np.ndarray:
    total = excit + inhib
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total >= min_total, excit / total, np.nan)
    return ratio.astype(np.float32)


def make_density_maps(df: pd.DataFrame, atlas_shape: tuple):
    glut_df, gaba_df = classify_cells(df)

    print("\n  Building excitatory density map...")
    excit = cells_to_density(glut_df, atlas_shape, GAUSSIAN_SIGMA)

    print("  Building inhibitory density map...")
    inhib = cells_to_density(gaba_df, atlas_shape, GAUSSIAN_SIGMA)

    print("  Computing E/I ratio map...")
    ei_ratio = compute_ei_ratio(excit, inhib)
    print(f"  E/I coverage: {(~np.isnan(ei_ratio)).mean()*100:.1f}% of volume has data")

    for name, vol in [("excitatory", excit), ("inhibitory", inhib), ("ei_ratio", ei_ratio)]:
        path = DENSITY_DIR / f"{name}_density.nrrd"
        nrrd.write(str(path), vol)
        print(f"  Saved {path}")

    return excit, inhib, ei_ratio


if __name__ == "__main__":
    import argparse
    from download_data import load_merfish_with_celltypes, load_ccf_annotation

    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=float, default=None)
    args = p.parse_args()

    ann, _ = load_ccf_annotation()
    df = load_merfish_with_celltypes(sample_frac=args.sample)
    excit, inhib, ei = make_density_maps(df, ann.shape)
    print(f"\nDone. Shape: {excit.shape}")
