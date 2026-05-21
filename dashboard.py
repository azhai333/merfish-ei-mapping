import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

RESULTS = Path("results")
FIGURES = RESULTS / "figures"

st.set_page_config(page_title="Forebrain E/I Mapping", layout="wide")

st.title("Forebrain E/I Ratio Mapping using MERFISH data in whole mouse brain")
st.caption("Zhang et al. 2023, *Science* · Allen Brain Cell Atlas · ~4.4M cells, mouse forebrain")

st.markdown("""
**Question:** Can volumetric image analysis recover the known excitatory/inhibitory organization of the
mouse forebrain from MERFISH cell coordinates alone. No manual annotation, just neurotransmitter labels
and 3D positions?

Each cell (glutamatergic = excitatory, GABAergic = inhibitory) is binned into a 20µm voxel grid
and blurred into a continuous density field. The E/I ratio at each voxel is `excit / (excit + inhib)`.
""")

st.divider()

# ---- load data ----
@st.cache_data
def load_data():
    reg = pd.read_csv(RESULTS / "region_stats.csv")
    sub = pd.read_csv(RESULTS / "subregion_stats.csv")
    return reg, sub

region_df, subregion_df = load_data()

# ---- top-level metrics ----
col1, col2, col3, col4 = st.columns(4)
ctx = region_df[region_df["region"] == "Isocortex"].iloc[0]
str_ = region_df[region_df["region"] == "STR"].iloc[0]
col1.metric("Isocortex E/I (mean)", f"{ctx['ei_ratio_mean']:.2f}")
col2.metric("Striatum E/I (mean)", f"{str_['ei_ratio_mean']:.2f}")
col3.metric("Subregions analyzed", len(subregion_df))
col4.metric("Cortex vs. Striatum", "p < 10⁻³⁰⁰", delta="Mann-Whitney U", delta_color="off")

st.divider()

# ---- tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["Brain sections", "By region", "Subregion explorer", "Gradient"])

with tab1:
    st.subheader("E/I ratio in coronal sections")
    st.caption("Top row: CCF atlas annotation (region boundaries). Bottom row: E/I ratio map (blue = inhibitory, red = excitatory). NaN voxels (no MERFISH data) are black.")
    img = Image.open(FIGURES / "ei_ratio_sections.png")
    st.image(img, use_container_width=True)

with tab2:
    st.subheader("E/I distribution by major forebrain division")
    st.caption("Each violin is the distribution of per-voxel E/I values within that region. Median annotated. Dashed line at 0.5.")

    left, right = st.columns([2, 1])
    with left:
        img2 = Image.open(FIGURES / "ei_ratio_by_region.png")
        st.image(img2, use_container_width=True)

    with right:
        st.markdown("**Major region summary**")
        cols = ["region", "ei_ratio_mean", "ei_ratio_median", "coverage_frac"]
        display = region_df[cols].copy()
        display.columns = ["Region", "Mean E/I", "Median E/I", "Coverage"]
        display["Mean E/I"] = display["Mean E/I"].round(3)
        display["Median E/I"] = display["Median E/I"].round(3)
        display["Coverage"] = (display["Coverage"] * 100).round(1).astype(str) + "%"
        st.dataframe(display, hide_index=True, use_container_width=True)

        st.markdown("""
        **What to expect:**
        - Isocortex ~0.75 (mostly Glut)
        - Striatum ~0.05 (mostly GABA)
        - Hippocampus (HPF) ~0.70
        - PAL/OLF intermediate

        Coverage is ~23% of CCF volume, which is expected for MERFISH sections,
        which don't tile continuously like whole-brain LSFM.
        """)

with tab3:
    st.subheader("Subregion E/I explorer")

    left, right = st.columns([1, 2])

    with left:
        divisions = ["All"] + sorted(subregion_df["division"].dropna().unique().tolist())
        selected_div = st.selectbox("Filter by division", divisions)

        min_vox = st.slider("Min voxels covered", 100, 2000, 300, step=100)

        sort_by = st.radio("Sort by", ["ei_mean (high→low)", "ei_mean (low→high)", "n_covered"])

    df_filt = subregion_df[subregion_df["n_covered"] >= min_vox].copy()
    if selected_div != "All":
        df_filt = df_filt[df_filt["division"] == selected_div]

    if sort_by == "ei_mean (high→low)":
        df_filt = df_filt.sort_values("ei_mean", ascending=False)
    elif sort_by == "ei_mean (low→high)":
        df_filt = df_filt.sort_values("ei_mean", ascending=True)
    else:
        df_filt = df_filt.sort_values("n_covered", ascending=False)

    with right:
        fig = px.bar(
            df_filt.head(40),
            x="ei_mean", y="acronym",
            orientation="h",
            color="ei_mean",
            color_continuous_scale="RdBu",
            range_color=[0, 1],
            labels={"ei_mean": "Mean E/I", "acronym": ""},
            hover_data={"division": True, "n_covered": True, "ei_std": ":.3f"},
        )
        fig.update_layout(
            height=600,
            coloraxis_showscale=False,
            yaxis={"autorange": "reversed"},
            margin=dict(l=0, r=10, t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full table**")
    show_cols = ["acronym", "division", "structure", "ei_mean", "ei_std", "n_covered", "coverage"]
    tbl = df_filt[show_cols].copy()
    tbl["ei_mean"] = tbl["ei_mean"].round(3)
    tbl["ei_std"] = tbl["ei_std"].round(3)
    tbl["coverage"] = (tbl["coverage"] * 100).round(1)
    tbl.columns = ["Acronym", "Division", "Structure", "Mean E/I", "Std", "Voxels", "Coverage %"]
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=300)

    st.markdown("---")
    st.subheader("Most vs. least excitatory subregions")
    img3 = Image.open(FIGURES / "subregion_ei_ranking.png")
    st.image(img3, use_container_width=True)

    st.subheader("Excitatory vs. inhibitory density per subregion")
    st.caption("Each dot is a CCF substructure. Color = E/I ratio. Regions with high excitatory density and low inhibitory density (upper-left) are the most excitatory.")
    img4 = Image.open(FIGURES / "excit_vs_inhib_scatter.png")
    st.image(img4, use_container_width=True)

with tab4:
    st.subheader("E/I spatial gradients")
    st.markdown("Mean ± 1 SD of the per-voxel E/I ratio along each brain axis. Shows how excitatory vs. inhibitory balance shifts spatially.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Anterior → Posterior**")
        img5 = Image.open(FIGURES / "ei_gradient_axis0.png")
        st.image(img5, use_container_width=True)
    with c2:
        st.markdown("**Medial → Lateral**")
        img6 = Image.open(FIGURES / "ei_gradient_axis2.png")
        st.image(img6, use_container_width=True)

    st.markdown("""
    The AP gradient drops toward the posterior, posterior structures like the brainstem have higher
    inhibitory fractions. The ML gradient is more symmetric but shows slight medial elevation
    (thalamus, hypothalamus are mixed), dropping laterally into the cortical mantle.
    """)
