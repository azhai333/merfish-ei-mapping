from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
DENSITY_DIR = DATA_DIR / "density_maps"

# Zhang et al. 2023, Science - https://doi.org/10.1126/science.abj6987
ABC_S3_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"

CCF_COORDS_URL = f"{ABC_S3_BASE}/metadata/MERFISH-C57BL6J-638850-CCF/20231215/ccf_coordinates.csv"
CELL_METADATA_URL = f"{ABC_S3_BASE}/metadata/MERFISH-C57BL6J-638850/20231215/cell_metadata.csv"
TAXONOMY_URL = (
    f"{ABC_S3_BASE}/metadata/WMB-taxonomy/20231215/views/"
    "cluster_to_cluster_annotation_membership_pivoted.csv"
)
CCF_ANNOTATION_URL = f"{ABC_S3_BASE}/image_volumes/Allen-CCF-2020/20230630/annotation_10.nii.gz"
PARCELLATION_TERM_URL = (
    f"{ABC_S3_BASE}/metadata/Allen-CCF-2020/20230630/views/"
    "parcellation_to_parcellation_term_membership_acronym.csv"
)

# annotation is 10um native, we downsample 2x -> 20um effective
# cell coords are in mm, so mm * 50 = voxel index at 20um
VOXEL_SIZE_UM = 20
GAUSSIAN_SIGMA = 1.5

BLOB_MIN_SIGMA = 1.0
BLOB_MAX_SIGMA = 4.0
BLOB_NUM_SIGMA = 5
BLOB_THRESHOLD = 0.05

FOREBRAIN_ACRONYMS = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL"]

DEV_SAMPLE_FRAC = 0.05
