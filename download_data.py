import requests
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_DIR,
    CCF_COORDS_URL, CELL_METADATA_URL, TAXONOMY_URL,
    CCF_ANNOTATION_URL, PARCELLATION_TERM_URL,
    VOXEL_SIZE_UM,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)


def _stream_download(url, dest: Path, chunk_size=1 << 17):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def download_file(url, name, force=False) -> Path:
    dest = DATA_DIR / name
    if dest.exists() and not force:
        print(f"  Using cached: {dest} ({dest.stat().st_size / 1e6:.0f}MB)")
        return dest
    print(f"  Downloading → {dest}")
    _stream_download(url, dest)
    return dest


def load_ccf_annotation(force=False):
    dest = download_file(CCF_ANNOTATION_URL, "annotation_10.nii.gz", force=force)

    print("  Loading annotation volume...")
    ann = np.asarray(nib.load(str(dest)).dataobj)
    print(f"  Raw shape (10um): {ann.shape}")

    # downsample to ~20um, use slicing to keep integer labels intact
    step = round(VOXEL_SIZE_UM / 10)
    ann_ds = ann[::step, ::step, ::step]
    print(f"  Downsampled: {ann_ds.shape}")
    return ann_ds, step


def load_taxonomy(force=False) -> pd.DataFrame:
    dest = download_file(TAXONOMY_URL, "cluster_taxonomy_pivoted.csv", force=force)
    df = pd.read_csv(dest)
    print(f"  Taxonomy: {len(df)} clusters")
    return df


def load_parcellation_terms(force=False) -> pd.DataFrame:
    dest = download_file(PARCELLATION_TERM_URL, "parcellation_terms_acronym.csv", force=force)
    df = pd.read_csv(dest)
    print(f"  Parcellation terms: {len(df)} rows")
    return df


def load_merfish_with_celltypes(sample_frac=None, force=False) -> pd.DataFrame:
    print("\n[1/3] Taxonomy")
    taxonomy = load_taxonomy(force=force)

    print("\n[2/3] CCF coordinates")
    coords_path = download_file(CCF_COORDS_URL, "ccf_coordinates.csv", force=force)
    print("  Reading...")
    coords_df = pd.read_csv(coords_path, usecols=["cell_label", "x", "y", "z"],
                            dtype={"cell_label": str})
    print(f"  {len(coords_df):,} cells")

    if sample_frac is not None:
        coords_df = coords_df.sample(frac=sample_frac, random_state=42)
        print(f"  Sampled to {len(coords_df):,} ({sample_frac*100:.1f}%)")

    print("\n[3/3] Cell metadata")
    meta_path = download_file(CELL_METADATA_URL, "cell_metadata.csv", force=force)
    meta_df = pd.read_csv(meta_path, usecols=["cell_label", "cluster_alias"],
                          dtype={"cell_label": str})
    print(f"  {len(meta_df):,} cells in metadata")

    print("\n  Joining...")
    df = coords_df.merge(meta_df, on="cell_label", how="inner")
    print(f"  After join: {len(df):,} cells")

    df = df.merge(taxonomy[["cluster_alias", "neurotransmitter", "class", "subclass"]],
                  on="cluster_alias", how="left")
    nt_coverage = df["neurotransmitter"].notna().mean()
    print(f"  Neurotransmitter coverage: {nt_coverage*100:.1f}%")

    return df


def inspect_merged_df(df):
    print("\n--- Data summary ---")
    print(f"Total cells: {len(df):,}")
    print("Neurotransmitter distribution:")
    nt = df["neurotransmitter"].fillna("(none)").value_counts()
    for val, cnt in nt.items():
        print(f"  {val:30s} {cnt:>9,}  ({cnt/len(df)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=float, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    ann, step = load_ccf_annotation(force=args.force)
    print(f"\nAnnotation: {ann.shape}, {len(np.unique(ann)):,} unique IDs")

    df = load_merfish_with_celltypes(sample_frac=args.sample, force=args.force)
    inspect_merged_df(df)
