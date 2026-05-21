import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def normalize_percentile(vol, pmin=1, pmax=99.5):
    lo = np.nanpercentile(vol, pmin)
    hi = np.nanpercentile(vol, pmax)
    if hi == lo:
        return np.zeros_like(vol)
    normed = np.clip((vol - lo) / (hi - lo), 0, 1)
    normed[np.isnan(vol)] = np.nan
    return normed.astype(np.float32)


def subtract_background(vol, radius=15):
    # approximate rolling-ball via local mean subtraction
    # need to fill NaN before filtering otherwise it spreads
    nan_mask = np.isnan(vol)
    v = np.where(nan_mask, 0.0, vol)

    background = uniform_filter(v, size=radius * 2 + 1)
    corrected = np.maximum(v - background, 0)
    corrected[nan_mask] = np.nan

    return corrected.astype(np.float32)


def smooth(vol, sigma):
    nan_mask = np.isnan(vol)
    v = np.where(nan_mask, 0.0, vol)
    v = gaussian_filter(v, sigma=sigma)
    v[nan_mask] = np.nan
    return v.astype(np.float32)


def preprocess_pair(excit, inhib, background_radius=15):
    print("Preprocessing excitatory density map...")
    excit_pp = subtract_background(excit, radius=background_radius)
    excit_pp = normalize_percentile(excit_pp)
    print(f"  range: [{np.nanmin(excit_pp):.3f}, {np.nanmax(excit_pp):.3f}]")

    print("Preprocessing inhibitory density map...")
    inhib_pp = subtract_background(inhib, radius=background_radius)
    inhib_pp = normalize_percentile(inhib_pp)
    print(f"  range: [{np.nanmin(inhib_pp):.3f}, {np.nanmax(inhib_pp):.3f}]")

    return excit_pp, inhib_pp
