import numpy as np

from boa_pni.polarization import (
    compute_intensity_and_polarization,
    compute_pni_projections,
)


def test_intensity_and_polarization_simple_case():
    up = np.array([[6.0, 8.0]])
    down = np.array([[2.0, 4.0]])

    intensity, polarization = compute_intensity_and_polarization(up, down)

    expected_intensity = np.array([[8.0, 12.0]])
    expected_polarization = np.array([[0.5, 1.0 / 3.0]])

    np.testing.assert_allclose(intensity, expected_intensity)
    np.testing.assert_allclose(polarization, expected_polarization)


def test_zero_polarization_case():
    up = np.array([[5.0]])
    down = np.array([[5.0]])

    intensity, polarization = compute_intensity_and_polarization(up, down)

    np.testing.assert_allclose(intensity, np.array([[10.0]]))
    np.testing.assert_allclose(polarization, np.array([[0.0]]))


def test_pni_projection_no_depolarization():
    """
    If sample polarization equals open-beam polarization,
    depolarization projection should be zero.
    """

    sample_up = np.array([[[60.0]]])
    sample_down = np.array([[[20.0]]])

    open_up = np.array([[120.0]])
    open_down = np.array([[40.0]])

    attenuation, depol, qc = compute_pni_projections(
        sample_up=sample_up,
        sample_down=sample_down,
        open_up=open_up,
        open_down=open_down,
        dark=0.0,
    )

    # Transmission = 80 / 160 = 0.5
    expected_attenuation = -np.log(0.5)

    np.testing.assert_allclose(attenuation, expected_attenuation)

    # P_sample = (60 - 20) / 80 = 0.5
    # P_open   = (120 - 40) / 160 = 0.5
    # P/P0 = 1, so -log(1) = 0
    np.testing.assert_allclose(depol, 0.0, atol=1e-6)

    assert qc["invalid_polarization_fraction"] == 0.0


def test_invalid_low_open_beam_polarization_is_masked():
    """
    If open-beam polarization is too small, depolarization is unreliable
    and should be marked invalid.
    """

    sample_up = np.array([[[51.0]]])
    sample_down = np.array([[[49.0]]])

    open_up = np.array([[100.01]])
    open_down = np.array([[100.00]])

    attenuation, depol, qc = compute_pni_projections(
        sample_up=sample_up,
        sample_down=sample_down,
        open_up=open_up,
        open_down=open_down,
        dark=0.0,
        p0_min=0.001,
        invalid_policy="nan",
    )

    assert np.isnan(depol[0, 0, 0])
    assert qc["invalid_polarization_fraction"] == 1.0