from __future__ import annotations

import numpy as np
import astra
from tqdm import tqdm


def reconstruct_sirt_parallel_2d_stack(
    projections: np.ndarray,
    angles_rad: np.ndarray,
    center_of_rotation_px: float | None = None,
    iterations: int = 100,
    use_cuda: bool = False,
    recon_size_px: int | None = None,
) -> np.ndarray:
    """
    Slice-by-slice parallel-beam SIRT reconstruction using ASTRA.

    Parameters
    ----------
    projections
        Projection stack with shape [n_angles, n_rows, n_cols].

    angles_rad
        Projection angles in radians, shape [n_angles].

    center_of_rotation_px
        Not used yet. This will be handled later.
        Current version assumes centered rotation axis.

    iterations
        Number of SIRT iterations.

    use_cuda
        If True, use ASTRA CUDA SIRT.
        If False, use CPU SIRT with an explicit ASTRA projector.

    recon_size_px
        Output reconstruction size.
        If None, uses detector width.

    Returns
    -------
    volume
        Reconstructed volume with shape [n_rows, recon_size_px, recon_size_px].
    """

    if projections.ndim != 3:
        raise ValueError("projections must have shape [n_angles, n_rows, n_cols].")

    n_angles, n_rows, n_cols = projections.shape

    if len(angles_rad) != n_angles:
        raise ValueError("Number of angles does not match projection stack.")

    if iterations <= 0:
        raise ValueError("iterations must be a positive integer.")

    if recon_size_px is None:
        recon_size_px = n_cols

    if center_of_rotation_px is not None:
        print(
            "Warning: center_of_rotation_px is provided but not yet applied. "
            "Current reconstruction assumes centered rotation axis."
        )

    volume = np.zeros(
        (n_rows, recon_size_px, recon_size_px),
        dtype=np.float32,
    )

    vol_geom = astra.create_vol_geom(recon_size_px, recon_size_px)

    proj_geom = astra.create_proj_geom(
        "parallel",
        1.0,
        n_cols,
        angles_rad.astype(np.float32),
    )

    projector_id = None

    if use_cuda:
        algorithm_name = "SIRT_CUDA"
    else:
        algorithm_name = "SIRT"
        projector_id = astra.create_projector(
            "linear",
            proj_geom,
            vol_geom,
        )

    try:
        for row in tqdm(range(n_rows), desc="Reconstructing slices"):
            sinogram = projections[:, row, :].astype(np.float32)

            if not np.all(np.isfinite(sinogram)):
                # ASTRA cannot handle NaN/Inf.
                # For now, replace invalid values with zero.
                # Later we can improve this with interpolation/masking.
                sinogram = np.nan_to_num(
                    sinogram,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)

            sino_id = astra.data2d.create(
                "-sino",
                proj_geom,
                sinogram,
            )

            rec_id = astra.data2d.create(
                "-vol",
                vol_geom,
            )

            cfg = astra.astra_dict(algorithm_name)
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = rec_id

            if projector_id is not None:
                cfg["ProjectorId"] = projector_id

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, iterations)

            reconstruction = astra.data2d.get(rec_id)
            volume[row] = reconstruction.astype(np.float32)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sino_id)
            astra.data2d.delete(rec_id)

    finally:
        if projector_id is not None:
            astra.projector.delete(projector_id)

    return volume