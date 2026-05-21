import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path

from config import FIGURES_DIR, FOREBRAIN_ACRONYMS
from atlas_analysis import get_division_mask


FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EI_CMAP = "RdBu_r"  # blue=inhibitory, red=excitatory


def _save(fig, name):
    p = FIGURES_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def _annotation_rgb(annotation_slice: np.ndarray) -> np.ndarray:
    # rough colormap — just assign a hue per structure ID, good enough for orientation
    out = np.zeros((*annotation_slice.shape, 3), dtype=np.float32)
    for uid in np.unique(annotation_slice):
        if uid == 0:
            continue
        mask = annotation_slice == uid
        # stable per-structure color via hash
        hue = (int(uid) * 1664525 + 1013904223) % 256 / 256.0
        r, g, b = mcolors.hsv_to_rgb([hue, 0.5, 0.7])
        out[mask, 0] = r
        out[mask, 1] = g
        out[mask, 2] = b
    return out


def plot_ei_ratio_sections(ei_ratio, annotation, n_sections=6):
    ap_len = ei_ratio.shape[0]
    idxs = np.linspace(ap_len * 0.1, ap_len * 0.9, n_sections).astype(int)

    fig, axes = plt.subplots(2, n_sections, figsize=(3 * n_sections, 6),
                              gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle("Excitatory fraction (E/[E+I]) — Coronal Sections\nBlue=inhibitory, Red=excitatory",
                 fontsize=12)

    for i, idx in enumerate(idxs):
        ann_slice = annotation[idx].T
        ei_slice = ei_ratio[idx].T

        brain_mask = ann_slice > 0
        ei_masked = np.ma.masked_where(~brain_mask | np.isnan(ei_slice), ei_slice)

        # Top row: atlas annotation (for orientation)
        axes[0, i].imshow(_annotation_rgb(ann_slice), origin="lower", aspect="auto")
        axes[0, i].set_title(f"AP={idx}", fontsize=8)
        axes[0, i].axis("off")

        # Bottom row: E/I ratio
        ax = axes[1, i]
        ax.imshow(np.zeros_like(ann_slice), cmap="gray", origin="lower", aspect="auto")
        im = ax.imshow(ei_masked, cmap=EI_CMAP, vmin=0, vmax=1,
                       origin="lower", aspect="auto", alpha=0.9)
        ax.axis("off")

    plt.colorbar(im, ax=axes[1, -1], shrink=0.8, label="E fraction")
    _save(fig, "ei_ratio_sections.png")


def plot_region_violin(ei_ratio, annotation, pmap, acronyms=None):
    if acronyms is None:
        acronyms = FOREBRAIN_ACRONYMS

    region_data = {}
    for acr in acronyms:
        mask = get_division_mask(annotation, acr, pmap)
        if mask is None:
            continue
        vals = ei_ratio[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 100:
            region_data[acr] = vals

    if not region_data:
        print("  No region data for violin plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(region_data.keys())
    data = [region_data[k] for k in labels]

    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmedians=True, showextrema=False)

    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=0, vmax=1)
    for body, d in zip(parts["bodies"], data):
        body.set_facecolor(cmap(norm(np.median(d))))
        body.set_alpha(0.85)

    medlines = parts["cmedians"]
    medlines.set_color("black")
    medlines.set_linewidth(1.5)

    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Excitatory fraction E/(E+I)")
    ax.set_title("E/I Ratio Distribution by Major Forebrain Region\n(MERFISH WMB, Zhang et al. 2023)")
    ax.set_ylim(-0.05, 1.05)

    # Annotate medians
    for i, d in enumerate(data):
        ax.text(i, np.median(d) + 0.04, f"{np.median(d):.2f}", ha="center", fontsize=7)

    _save(fig, "ei_ratio_by_region.png")


def plot_subregion_ranking(subregion_df, top_n=15):
    df = subregion_df[subregion_df["n_covered"] > 200].copy()

    top_exc = df.nlargest(top_n, "ei_mean")
    top_inh = df.nsmallest(top_n, "ei_mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(f"Sub-region E/I Ranking (top {top_n} each end)", fontsize=12)

    for ax, sub, title, color in [
        (ax1, top_exc, "Most Excitatory", "#c0392b"),
        (ax2, top_inh, "Most Inhibitory",  "#2471a3"),
    ]:
        # Truncate long names
        names = [f"{r['acronym'][:22]}" for _, r in sub.iterrows()]
        vals = sub["ei_mean"].values
        divs = sub["division"].values

        bars = ax.barh(range(len(names)), vals, color=color, alpha=0.75)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)

        # Label with division name
        for j, (div, val) in enumerate(zip(divs, vals)):
            ax.text(val + 0.01, j, div[:20], va="center", fontsize=6, color="gray")

        ax.set_xlabel("Excitatory fraction")
        ax.set_title(title)
        ax.set_xlim(0, 1.1)
        ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.4)

    plt.tight_layout()
    _save(fig, "subregion_ei_ranking.png")


def plot_ei_scatter(subregion_df):
    df = subregion_df.dropna(subset=["ei_mean", "excit_density", "inhib_density"])
    df = df[df["n_covered"] > 200]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["excit_density"], df["inhib_density"],
        c=df["ei_mean"], cmap=EI_CMAP, vmin=0, vmax=1,
        s=30, alpha=0.75, edgecolors="none"
    )
    plt.colorbar(sc, ax=ax, label="E fraction (E/[E+I])")

    # Label interesting points: top/bottom 4 E/I
    extremes = pd.concat([df.nlargest(4, "ei_mean"), df.nsmallest(4, "ei_mean")])
    for _, row in extremes.iterrows():
        ax.annotate(str(row["acronym"])[:12],
                    (row["excit_density"], row["inhib_density"]),
                    fontsize=7, alpha=0.85,
                    xytext=(5, 2), textcoords="offset points")

    ax.set_xlabel("Mean excitatory density (a.u.)")
    ax.set_ylabel("Mean inhibitory density (a.u.)")
    ax.set_title("Cell Density per Sub-region\n(MERFISH WMB, colored by E/I ratio)")
    _save(fig, "excit_vs_inhib_scatter.png")


def plot_ei_gradient_along_axis(ei_ratio, annotation, pmap=None, axis=0):
    brain_mask = annotation > 0
    n_slices = ei_ratio.shape[axis]
    positions = []
    means = []
    stds = []

    for i in range(n_slices):
        if axis == 0:
            ei_slice = ei_ratio[i]
            bm_slice = brain_mask[i]
        elif axis == 1:
            ei_slice = ei_ratio[:, i, :]
            bm_slice = brain_mask[:, i, :]
        else:
            ei_slice = ei_ratio[:, :, i]
            bm_slice = brain_mask[:, :, i]

        vals = ei_slice[bm_slice & ~np.isnan(ei_slice)]
        if len(vals) < 50:
            continue
        positions.append(i)
        means.append(vals.mean())
        stds.append(vals.std())

    positions = np.array(positions)
    means = np.array(means)
    stds = np.array(stds)

    axis_names = ["AP (anterior→posterior)", "DV (dorsal→ventral)", "ML (medial→lateral)"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, means, color="#e74c3c", linewidth=1.5, label="Mean E/I")
    ax.fill_between(positions, means - stds, means + stds, alpha=0.2, color="#e74c3c")
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(f"Voxel index along {axis_names[axis]} (1 voxel ≈ 20µm)")
    ax.set_ylabel("Excitatory fraction")
    ax.set_title(f"E/I Gradient Along {axis_names[axis]}")
    ax.set_ylim(0, 1)
    ax.legend()
    _save(fig, f"ei_gradient_axis{axis}.png")


def make_all_figures(excit_pp, inhib_pp, ei_ratio, annotation, pmap,
                     subregion_df, excit_peaks, inhib_peaks):
    print("\nGenerating figures...")

    plot_ei_ratio_sections(ei_ratio, annotation)
    plot_region_violin(ei_ratio, annotation, pmap)
    plot_ei_gradient_along_axis(ei_ratio, annotation, pmap, axis=0)  # AP gradient
    plot_ei_gradient_along_axis(ei_ratio, annotation, pmap, axis=2)  # ML gradient

    if subregion_df is not None and len(subregion_df) > 0:
        plot_subregion_ranking(subregion_df)
        plot_ei_scatter(subregion_df)

    print(f"  All figures in {FIGURES_DIR}/")
