from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle


def tv_denoise_volume(
    volume: np.ndarray,
    weight: float = 0.03,
) -> np.ndarray:
    """Apply total variation denoising to a 3D volume."""
    return denoise_tv_chambolle(
        volume,
        weight=weight,
        channel_axis=None,
    ).astype(np.float32)


def make_material_mask(
    attenuation_volume: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Create material mask from attenuation volume."""
    return attenuation_volume > threshold