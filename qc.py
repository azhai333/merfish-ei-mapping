import numpy as np
import pandas as pd


# Rough expected E/I ranges from literature (fraction excitatory)
EXPECTED_EI_RANGES = {
    "Isocortex":  (0.55, 0.90),
    "STR":        (0.02, 0.20),
    "HPF":        (0.55, 0.90),
    "PAL":        (0.05, 0.40),
    "OLF":        (0.25, 0.70),
}


class QCReport:
    def __init__(self):
        self.checks = []

    def add(self, name, passed, details=""):
        status = "PASS" if passed else "FAIL"
        self.checks.append({"name": name, "status": status, "details": details})

    def warn(self, name, details=""):
        self.checks.append({"name": name, "status": "WARN", "details": details})

    def summary(self):
        n_pass = sum(1 for c in self.checks if c["status"] == "PASS")
        n_warn = sum(1 for c in self.checks if c["status"] == "WARN")
        n_fail = sum(1 for c in self.checks if c["status"] == "FAIL")
        print(f"\n=== QC Report: {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL ===")
        for c in self.checks:
            sym = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[c["status"]]
            print(f"  [{sym}] {c['name']}: {c['details']}")
        failures = [c["name"] for c in self.checks if c["status"] == "FAIL"]
        if failures:
            print(f"\n  FAILED: {failures}")
        return n_fail == 0


def check_coverage(excit, inhib, annotation, qc: QCReport, min_frac=0.25):
    brain = annotation > 0
    for name, vol in [("excitatory", excit), ("inhibitory", inhib)]:
        covered = (vol > 1e-6) & brain
        frac = covered.sum() / brain.sum()
        if frac >= min_frac:
            qc.add(f"coverage_{name}", True, f"{frac*100:.1f}% of brain voxels have data")
        elif frac >= min_frac / 2:
            qc.warn(f"coverage_{name}", f"Low: {frac*100:.1f}% (expected ≥{min_frac*100:.0f}%)")
        else:
            qc.add(f"coverage_{name}", False,
                   f"Very low: {frac*100:.1f}% — likely a coordinate alignment issue")


def check_density_distribution(vol, name, qc: QCReport):
    valid = vol[~np.isnan(vol) & (vol > 1e-6)]
    if len(valid) < 1000:
        qc.add(f"dist_{name}", False, f"Only {len(valid)} positive voxels")
        return
    if valid.std() < 1e-8:
        qc.add(f"dist_{name}", False, "Zero variance")
        return
    p99 = np.percentile(valid, 99)
    p50 = np.percentile(valid, 50)
    ratio = p99 / max(p50, 1e-10)
    if ratio > 100:
        qc.warn(f"dist_{name}", f"High p99/p50 ratio ({ratio:.0f}) — possible outlier voxels")
    else:
        qc.add(f"dist_{name}", True, f"mean={valid.mean():.4f} std={valid.std():.4f}")


def check_ei_biology(subregion_df: pd.DataFrame, qc: QCReport):
    for acr, (lo, hi) in EXPECTED_EI_RANGES.items():
        # The division column in subregion_df should match CCF acronyms directly
        row = subregion_df[subregion_df["division"] == acr]
        if len(row) == 0:
            qc.warn(f"ei_range_{acr}", f"No sub-regions found for division '{acr}'")
            continue
        ei = row["ei_mean"].median()
        passed = lo <= ei <= hi
        qc.add(f"ei_range_{acr}", passed,
               f"E/I={ei:.3f}, expected [{lo:.2f}, {hi:.2f}]")


def check_peaks_in_brain(peaks, annotation, qc: QCReport, cell_type=""):
    if not peaks:
        qc.warn(f"peaks_in_brain_{cell_type}", "No peaks detected")
        return
    n_inside = sum(
        1 for p in peaks
        if 0 <= p.z < annotation.shape[0]
        and 0 <= p.y < annotation.shape[1]
        and 0 <= p.x < annotation.shape[2]
        and annotation[p.z, p.y, p.x] > 0
    )
    frac = n_inside / len(peaks)
    qc.add(f"peaks_in_brain_{cell_type}", frac > 0.8,
           f"{n_inside}/{len(peaks)} ({frac*100:.0f}%) peaks inside brain mask")


def run_all_qc(excit_pp, inhib_pp, ei_ratio, annotation,
               subregion_df, excit_peaks, inhib_peaks):
    qc = QCReport()
    print("\nRunning QC checks...")

    check_coverage(excit_pp, inhib_pp, annotation, qc)
    check_density_distribution(excit_pp, "excitatory", qc)
    check_density_distribution(inhib_pp, "inhibitory", qc)

    if subregion_df is not None and len(subregion_df) > 0:
        check_ei_biology(subregion_df, qc)

    if excit_peaks:
        check_peaks_in_brain(excit_peaks, annotation, qc, "excitatory")
    if inhib_peaks:
        check_peaks_in_brain(inhib_peaks, annotation, qc, "inhibitory")

    all_passed = qc.summary()
    return qc, all_passed
