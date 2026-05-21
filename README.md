# Forebrain E/I ratio mapping from MERFISH data

Maps excitatory/inhibitory neuron density ratios across the mouse forebrain using the Allen Brain Cell Atlas MERFISH dataset ([Zhang et al. 2023, *Science*](https://doi.org/10.1126/science.abj6987)). The question is whether automated volumetric analysis can recover the known neuroanatomical E/I organization — cortex heavily excitatory, striatum heavily inhibitory — without any manual annotation, just cell coordinates and neurotransmitter labels.

The approach is similar to what you'd do with whole-brain LSFM data: voxelize point-cloud cell coordinates into 3D density volumes, do background subtraction and normalization, then run image analysis (LoG blob detection, regional statistics, QC checks) on those volumes. The difference here is the input comes from MERFISH rather than a fluorescence image, so there's no registration step needed.

## Data

Downloads automatically from Allen S3 on first run (~820MB total):

- CCF coordinates for ~4.4M cells (mm-space, registered to CCFv3)
- Cell metadata + cluster taxonomy (neurotransmitter labels)
- CCF annotation volume (10µm, downsampled 2x to 20µm)
- Parcellation terms (structure hierarchy)

## How to run

```bash
pip install numpy==1.26.4 scipy scikit-image pandas matplotlib tqdm requests pynrrd nibabel streamlit plotly pillow

# full run (downloads data on first run, takes ~6 min)
python run_pipeline.py

# fast dev mode (5% sample)
python run_pipeline.py --dev

# skip re-downloading if you already have the density maps
python run_pipeline.py --skip-maps

# interactive results dashboard
streamlit run dashboard.py
```

## Pipeline steps

1. Download and join CCF coordinates → cell metadata → cluster taxonomy
2. Classify cells as glutamatergic (excitatory) or GABAergic (inhibitory) based on neurotransmitter label
3. Bin cells into 3D voxel grids, apply Gaussian blur to get continuous density fields
4. Background subtraction (rolling-ball via uniform filter) + percentile normalization
5. Compute E/I ratio map: `excit / (excit + inhib)` per voxel, NaN where total density is too low
6. LoG blob detection to find density hotspots in each map
7. Regional statistics per CCF division (Isocortex, STR, HPF, OLF, PAL, CTXsp)
8. Mann-Whitney U test: cortex vs striatum E/I as biological validation
9. QC report (coverage, distribution checks, E/I ranges vs. literature)
10. Figures

## Outputs

```
results/
  region_stats.csv       # E/I stats per major forebrain division
  subregion_stats.csv    # per CCF substructure (~600 regions)
  figures/
    ei_ratio_sections.png      # coronal montage, blue=inhibitory red=excitatory
    ei_ratio_by_region.png     # violin plots per forebrain region
    subregion_ei_ranking.png   # top 15 most/least excitatory subregions
    excit_vs_inhib_scatter.png # density scatter colored by E/I
    ei_gradient_axis0.png      # mean E/I along AP axis
    ei_gradient_axis2.png      # mean E/I along ML axis
```

## Results

The pipeline correctly recovers the expected E/I organization. Isocortex sits around E/I = 0.75–0.80, striatum around 0.05–0.10, and hippocampus intermediate (~0.70). The Mann-Whitney test comparing cortex vs. striatum comes back p < 10⁻³⁰⁰. The subregion ranking surfaces things like cerebellar-adjacent structures at the extreme inhibitory end and upper-layer cortical areas at the excitatory end, which is consistent with known laminar composition.

Coverage is ~23% of the CCF volume — lower than you'd get from a whole-brain LSFM dataset because the MERFISH sections don't tile continuously. This is expected and flagged in the QC report as a warning rather than a failure.

## Files

| file | what it does |
|---|---|
| `config.py` | paths, S3 URLs, pipeline parameters |
| `download_data.py` | streaming download + data joins |
| `make_density_maps.py` | voxelization, Gaussian smoothing, E/I ratio |
| `preprocess.py` | background subtraction, normalization |
| `detect_peaks.py` | LoG blob detection / local maxima |
| `atlas_analysis.py` | CCF parcellation, regional stats, validation |
| `qc.py` | QC checks and report |
| `visualize.py` | figures |
| `run_pipeline.py` | top-level orchestration |
