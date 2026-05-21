import numpy as np
from skimage.feature import blob_log
from scipy.ndimage import label, maximum_filter, gaussian_filter
from dataclasses import dataclass, field
from typing import List


@dataclass
class DensityPeak:
    z: int   # AP (anterior-posterior)
    y: int   # DV (dorsal-ventral)
    x: int   # ML (medial-lateral)
    sigma: float
    intensity: float
    cell_type: str  # "excitatory" or "inhibitory"

    @property
    def coords(self):
        return (self.z, self.y, self.x)


def detect_blobs_log(vol, min_sigma, max_sigma, num_sigma, threshold, cell_type=""):
    nan_mask = np.isnan(vol)
    v = np.where(nan_mask, 0.0, vol)

    print(f"  Running LoG blob detection ({cell_type})...")
    blobs = blob_log(
        v,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=0.5,
    )

    if len(blobs) == 0:
        print(f"  No blobs found — try lowering threshold (current: {threshold})")
        return []

    print(f"  Found {len(blobs)} density hotspots")

    peaks = []
    for blob in blobs:
        z, y, x, sigma = blob
        zi, yi, xi = int(z), int(y), int(x)
        if nan_mask[zi, yi, xi]:
            continue
        intensity = float(v[zi, yi, xi])
        peaks.append(DensityPeak(z=zi, y=yi, x=xi, sigma=sigma, intensity=intensity, cell_type=cell_type))

    return peaks


def detect_local_maxima_simple(vol, min_distance=5, threshold_abs=None, cell_type=""):
    nan_mask = np.isnan(vol)
    v = np.where(nan_mask, 0.0, vol)

    # Suppress anything below this
    if threshold_abs is None:
        threshold_abs = np.nanpercentile(vol[~nan_mask], 90)

    local_max = maximum_filter(v, size=min_distance) == v
    local_max &= (v >= threshold_abs)
    local_max &= ~nan_mask

    coords = np.argwhere(local_max)
    print(f"  Found {len(coords)} local maxima ({cell_type}), threshold={threshold_abs:.4f}")

    peaks = [
        DensityPeak(z=int(c[0]), y=int(c[1]), x=int(c[2]),
                    sigma=min_distance / 2, intensity=float(v[c[0], c[1], c[2]]),
                    cell_type=cell_type)
        for c in coords
    ]
    return peaks


def run_peak_detection(excit_pp, inhib_pp, use_log=True,
                       min_sigma=1.0, max_sigma=4.0, num_sigma=5, threshold=0.05,
                       min_distance=5):
    if use_log:
        excit_peaks = detect_blobs_log(excit_pp, min_sigma, max_sigma, num_sigma, threshold, "excitatory")
        inhib_peaks = detect_blobs_log(inhib_pp, min_sigma, max_sigma, num_sigma, threshold, "inhibitory")
    else:
        excit_peaks = detect_local_maxima_simple(excit_pp, min_distance, cell_type="excitatory")
        inhib_peaks = detect_local_maxima_simple(inhib_pp, min_distance, cell_type="inhibitory")

    return excit_peaks, inhib_peaks
