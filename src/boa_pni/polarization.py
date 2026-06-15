from __future__ import annotations

import numpy as np


def safe_clip_intensity(array: np.ndarray, min_value: float) -> np.ndarray:
    """
    Clip only intensity-like data.

    This should not be used to hide invalid polarization ratios.
    """
    return np.clip(array, min_value, None)


def compute_intensity_and_polarization(
    up: np.ndarray,
    down: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute total intensity and neutron polarization from spin-up/down images.

    I = I_up + I_down

    P = (I_up - I_down) / (I_up + I_down)
    """
    intensity = up + down
    polarization = (up - down) / (intensity + eps)
    return intensity, polarization


def compute_pni_projections(
    sample_up: np.ndarray,
    sample_down: np.ndarray,
    open_up: np.ndarray,
    open_down: np.ndarray,
    dark: np.ndarray | float = 0.0,
    dark_up: np.ndarray | float | None = None,
    dark_down: np.ndarray | float | None = None,
    eps: float = 1e-8,
    intensity_clip_min: float = 1e-8,
    p0_min: float = 1e-3,
    pratio_min: float = 1e-4,
    invalid_policy: str = "nan",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert BOA-style spin-up/down data into scalar CT projection stacks.

    Parameters
    ----------
    sample_up, sample_down
        Spin-up and spin-down sample projection stacks.
        Expected shape: [n_angles, n_rows, n_cols]

    open_up, open_down
        Spin-up and spin-down open-beam images.
        Expected shape: [n_rows, n_cols] or broadcast-compatible.

    dark
        Single detector dark image, used if dark_up/dark_down are not provided.

    dark_up, dark_down
        Optional spin-state-specific dark images.

    eps
        Small number to avoid division by zero.

    intensity_clip_min
        Minimum positive value for intensity-like quantities.

    p0_min
        Minimum acceptable absolute open-beam polarization.

    pratio_min
        Minimum acceptable P/P0 ratio.

    invalid_policy
        "nan" or "zero" for invalid depolarization pixels.

    Returns
    -------
    attenuation_proj
        Scalar attenuation projections:
        -log(I / I0)

    depolarization_proj
        Scalar depolarization projections:
        -log(P / P0), with invalid pixels masked.

    qc
        Dictionary with quality-control statistics and masks.
    """

    if invalid_policy not in {"nan", "zero"}:
        raise ValueError("invalid_policy must be 'nan' or 'zero'.")

    if dark_up is None:
        dark_up = dark

    if dark_down is None:
        dark_down = dark

    # Dark correction
    su = sample_up - dark_up
    sd = sample_down - dark_down
    ou = open_up - dark_up
    od = open_down - dark_down

    # Clip intensity-like data only
    su = safe_clip_intensity(su, intensity_clip_min)
    sd = safe_clip_intensity(sd, intensity_clip_min)
    ou = safe_clip_intensity(ou, intensity_clip_min)
    od = safe_clip_intensity(od, intensity_clip_min)

    # Compute sample and open-beam intensity/polarization
    sample_i, sample_p = compute_intensity_and_polarization(su, sd, eps=eps)
    open_i, open_p = compute_intensity_and_polarization(ou, od, eps=eps)

    # Attenuation projection
    transmission = sample_i / (open_i + eps)
    transmission = safe_clip_intensity(transmission, intensity_clip_min)
    attenuation_proj = -np.log(transmission).astype(np.float32)

    # Depolarization projection
    polarization_ratio = sample_p / (open_p + eps)

    valid_pol = np.isfinite(sample_p)
    valid_pol &= np.isfinite(open_p)
    valid_pol &= np.isfinite(polarization_ratio)
    valid_pol &= np.abs(open_p) > p0_min
    valid_pol &= polarization_ratio > pratio_min

    if invalid_policy == "nan":
        depolarization_proj = np.full_like(
            polarization_ratio,
            np.nan,
            dtype=np.float32,
        )
    else:
        depolarization_proj = np.zeros_like(
            polarization_ratio,
            dtype=np.float32,
        )

    depolarization_proj[valid_pol] = -np.log(
        polarization_ratio[valid_pol]
    ).astype(np.float32)

    qc = {
        "mean_open_polarization": float(np.nanmean(open_p)),
        "min_open_polarization": float(np.nanmin(open_p)),
        "max_open_polarization": float(np.nanmax(open_p)),
        "mean_sample_polarization": float(np.nanmean(sample_p)),
        "min_sample_polarization": float(np.nanmin(sample_p)),
        "max_sample_polarization": float(np.nanmax(sample_p)),
        "invalid_polarization_fraction": float(1.0 - np.mean(valid_pol)),
        "p0_min": float(p0_min),
        "pratio_min": float(pratio_min),
        "invalid_policy": invalid_policy,
    }

    return attenuation_proj, depolarization_proj, qc