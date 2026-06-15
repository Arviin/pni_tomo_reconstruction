import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

import astra

from boa_pni.reconstruct_astra import reconstruct_sirt_parallel_2d_stack


def make_phantom(n=128):
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (n, n), anti_aliasing=True)
    phantom = phantom.astype(np.float32)

    yy, xx = np.mgrid[:n, :n]
    r = np.sqrt((xx - 80) ** 2 + (yy - 65) ** 2)

    magnetic = np.zeros_like(phantom, dtype=np.float32)
    magnetic[r < 12] = 1.0

    return phantom, magnetic


def astra_forward_project(image, angles_rad):
    """
    Create synthetic parallel-beam projections using ASTRA itself.

    This avoids geometry mismatch between skimage.radon and ASTRA.
    """
    n = image.shape[0]

    vol_geom = astra.create_vol_geom(n, n)

    proj_geom = astra.create_proj_geom(
        "parallel",
        1.0,
        n,
        angles_rad.astype(np.float32),
    )

    projector_id = astra.create_projector(
        "linear",
        proj_geom,
        vol_geom,
    )

    volume_id = astra.data2d.create(
        "-vol",
        vol_geom,
        image.astype(np.float32),
    )

    sinogram_id, sinogram = astra.create_sino(
        volume_id,
        projector_id,
    )

    astra.data2d.delete(volume_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(projector_id)

    return sinogram.astype(np.float32)


def main():
    n = 128
    n_angles = 180

    angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    mu_true, eta_true = make_phantom(n)

    sino_mu = astra_forward_project(mu_true, angles_rad)
    sino_eta = astra_forward_project(eta_true, angles_rad)

    # Convert 2D sinogram to projection stack:
    # [n_angles, n_rows, n_cols]
    # Here n_rows = 1 for a simple synthetic slice test.
    proj_mu = sino_mu[:, None, :]
    proj_eta = sino_eta[:, None, :]

    rec_mu = reconstruct_sirt_parallel_2d_stack(
        proj_mu,
        angles_rad,
        iterations=100,
        use_cuda=False,
        recon_size_px=n,
    )[0]

    rec_eta = reconstruct_sirt_parallel_2d_stack(
        proj_eta,
        angles_rad,
        iterations=100,
        use_cuda=False,
        recon_size_px=n,
    )[0]

    print("Synthetic reconstruction finished.")
    print("mu sinogram shape:", sino_mu.shape)
    print("eta sinogram shape:", sino_eta.shape)
    print("mu reconstruction shape:", rec_mu.shape)
    print("eta reconstruction shape:", rec_eta.shape)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(mu_true, cmap="gray")
    axes[0, 0].set_title("True attenuation phantom")

    axes[0, 1].imshow(rec_mu, cmap="gray")
    axes[0, 1].set_title("Reconstructed attenuation")

    axes[1, 0].imshow(eta_true, cmap="gray")
    axes[1, 0].set_title("True depolarization phantom")

    axes[1, 1].imshow(rec_eta, cmap="gray")
    axes[1, 1].set_title("Reconstructed depolarization")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()