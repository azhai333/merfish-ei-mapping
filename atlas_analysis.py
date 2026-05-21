import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from config import FOREBRAIN_ACRONYMS


def build_parcellation_map(parc_df: pd.DataFrame) -> dict:
    pmap = {}
    for _, row in parc_df.iterrows():
        pid = int(row["parcellation_index"])
        pmap[pid] = {
            "division": str(row.get("division", "")),
            "structure": str(row.get("structure", "")),
            "substructure": str(row.get("substructure", "")),
        }
    return pmap


def get_division_mask(annotation: np.ndarray, division: str, pmap: dict) -> np.ndarray:
    matching_ids = {pid for pid, info in pmap.items() if info["division"] == division}
    if not matching_ids:
        return None
    return np.isin(annotation, list(matching_ids))


def build_region_stats(excit, inhib, ei_ratio, annotation, pmap, acronyms=None):
    if acronyms is None:
        acronyms = FOREBRAIN_ACRONYMS

    records = []
    for acr in acronyms:
        mask = get_division_mask(annotation, acr, pmap)
        if mask is None or mask.sum() == 0:
            print(f"  Warning: division '{acr}' not found in parcellation map")
            continue

        ei_vals = ei_ratio[mask]
        valid = ~np.isnan(ei_vals)
        coverage = valid.mean()

        if valid.sum() < 100:
            continue

        e_v = excit[mask][valid]
        i_v = inhib[mask][valid]
        ei_v = ei_vals[valid]

        records.append({
            "region": acr,
            "n_voxels_total": int(mask.sum()),
            "n_voxels_covered": int(valid.sum()),
            "coverage_frac": float(coverage),
            "excit_mean": float(e_v.mean()),
            "inhib_mean": float(i_v.mean()),
            "ei_ratio_mean": float(ei_v.mean()),
            "ei_ratio_median": float(np.median(ei_v)),
            "ei_ratio_std": float(ei_v.std()),
        })

    df = pd.DataFrame(records).sort_values("ei_ratio_mean", ascending=False)
    return df


def compute_subregion_stats(excit, inhib, ei_ratio, annotation, pmap,
                             min_voxels=200):
    print("  Flattening arrays for vectorized stats...")

    # Work only on brain voxels (annotation > 0) to save memory
    brain_mask = annotation > 0
    ann_flat = annotation[brain_mask].astype(np.int32)
    excit_flat = excit[brain_mask]
    inhib_flat = inhib[brain_mask]
    ei_flat = ei_ratio[brain_mask]

    df = pd.DataFrame({
        "pid": ann_flat,
        "excit": excit_flat,
        "inhib": inhib_flat,
        "ei": ei_flat,
    })
    # Only include voxels with E/I data
    df_valid = df[~np.isnan(df["ei"])].copy()

    print(f"  {len(df_valid):,} valid voxels across {df_valid['pid'].nunique()} structures")

    # Groupby structure
    grp = df_valid.groupby("pid")
    stats_df = grp.agg(
        n_covered=("ei", "count"),
        ei_mean=("ei", "mean"),
        ei_median=("ei", "median"),
        ei_std=("ei", "std"),
        excit_density=("excit", "mean"),
        inhib_density=("inhib", "mean"),
    ).reset_index()

    # Total voxel count (including NaN) for coverage fraction
    total_counts = df.groupby("pid").size().reset_index(name="n_voxels")
    stats_df = stats_df.merge(total_counts, on="pid")
    stats_df["coverage"] = stats_df["n_covered"] / stats_df["n_voxels"]

    # Filter by min voxel count
    stats_df = stats_df[stats_df["n_voxels"] >= min_voxels]

    # Add region name info from pmap
    def get_info(pid, field):
        return pmap.get(int(pid), {}).get(field, str(pid))

    stats_df["division"] = stats_df["pid"].apply(lambda x: get_info(x, "division"))
    stats_df["structure"] = stats_df["pid"].apply(lambda x: get_info(x, "structure"))
    stats_df["acronym"] = stats_df["pid"].apply(lambda x: get_info(x, "substructure") or get_info(x, "structure"))

    stats_df = stats_df.sort_values("ei_mean", ascending=False).reset_index(drop=True)
    print(f"  {len(stats_df)} sub-regions with ≥{min_voxels} voxels")
    return stats_df


def test_ei_cortex_vs_striatum(excit, inhib, ei_ratio, annotation, pmap):
    ctx_mask = get_division_mask(annotation, "Isocortex", pmap)
    str_mask = get_division_mask(annotation, "STR", pmap)

    if ctx_mask is None or str_mask is None:
        print("  Could not get Isocortex or STR masks for validation")
        return None

    ctx_ei = ei_ratio[ctx_mask & ~np.isnan(ei_ratio)]
    str_ei = ei_ratio[str_mask & ~np.isnan(ei_ratio)]

    if len(ctx_ei) < 50 or len(str_ei) < 50:
        print(f"  Not enough data for comparison (ctx={len(ctx_ei)}, str={len(str_ei)})")
        return None

    u_stat, p_val = stats.mannwhitneyu(ctx_ei, str_ei, alternative="greater")

    print(f"\n--- Cortex vs Striatum E/I validation ---")
    print(f"  Isocortex:  E/I = {ctx_ei.mean():.3f} ± {ctx_ei.std():.3f}  (n={len(ctx_ei):,} voxels)")
    print(f"  Striatum:   E/I = {str_ei.mean():.3f} ± {str_ei.std():.3f}  (n={len(str_ei):,} voxels)")
    print(f"  Mann-Whitney U={u_stat:.0f}, p={p_val:.2e}")
    if p_val < 0.001:
        print("  PASS: cortex significantly more excitatory than striatum")
    else:
        print("  FAIL: expected cortex > striatum — check data alignment")

    return {"cortex_ei": ctx_ei, "striatum_ei": str_ei, "p": p_val, "u": u_stat}
