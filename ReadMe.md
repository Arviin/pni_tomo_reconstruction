# Polarized Neutron Imaging Tomography Reconstruction

A Python-based reconstruction pipeline for **polarization contrast neutron tomography**.

This repository is designed to reconstruct scalar 3D volumes from spin-resolved neutron tomography data, especially for experiments where magnetic or ferromagnetic phases cause neutron depolarization.

The current reconstruction backend is **ASTRA Toolbox** using parallel-beam **SIRT** reconstruction.

## Author

**Arvin (Fazel) Mirzaei**  
Postdoctoral Researcher, Paul Scherrer Institute (PSI), Switzerland  
Email: `fazel.mirzaei@psi.ch`

---

---

## Scientific Scope

This project reconstructs two scalar tomographic volumes:

1. **Attenuation contrast volume**

   [
   \mu(x,y,z)
   ]

   reconstructed from:

   [
   -\log(I/I_0)
   ]

2. **Depolarization contrast volume**

   [
   \eta(x,y,z)
   ]

   reconstructed from:

   [
   -\log(P/P_0)
   ]

where the total intensity is calculated as:

[
I = I_\uparrow + I_\downarrow
]

and the neutron polarization is calculated as:

[
P = \frac{I_\uparrow - I_\downarrow}{I_\uparrow + I_\downarrow}
]

Here, (I_\uparrow) and (I_\downarrow) are the spin-up and spin-down neutron images.

---

## Important Limitation

This repository is intended for **scalar polarization-contrast / depolarization tomography**.

It is **not** a full magnetic vector-field polarimetric neutron tomography solver.

In other words, this code reconstructs quantities such as:

```text
attenuation contrast
depolarization contrast
relative magnetic phase distribution
```

It does not directly reconstruct:

```text
Bx(x,y,z), By(x,y,z), Bz(x,y,z)
```

For quantitative magnetic phase-fraction mapping, calibration with reference samples is required.

---

## Reconstruction Workflow

The intended processing chain is:

```text
spin-up / spin-down raw images
        ↓
dark correction
        ↓
open-beam correction
        ↓
total intensity and polarization calculation
        ↓
attenuation projection:    -log(I / I0)
depolarization projection: -log(P / P0)
        ↓
artifact correction
        ↓
ASTRA SIRT reconstruction
        ↓
optional TV denoising
        ↓
visualization / masking / calibration
```

---

## Installation

This project uses **Pixi** for environment management.

From the project root directory:

```bash
pixi install
```

To check that ASTRA is available:

```bash
pixi run python scripts/check_astra.py
```

---

## Run Tests

Before using the reconstruction pipeline, run:

```bash
pixi run pytest
```

A clean installation should report:

```text
4 passed
```

---

## Synthetic Reconstruction Test

Before applying the code to real data, run the synthetic ASTRA test:

```bash
pixi run python scripts/synthetic_reconstruction_test.py
```

This verifies that:

```text
ASTRA forward projection works
ASTRA SIRT reconstruction works
the projection-stack convention is correct
attenuation and depolarization scalar volumes can be reconstructed
```

---

## Current Status

under development.

---

## Scientific Warning

Depolarization contrast should not automatically be interpreted as absolute magnetic phase fraction.

To report quantitative phase fractions, the depolarization signal must be calibrated using reference samples with known magnetic phase content.

Without calibration, the safe output is:

```text
relative depolarization contrast
```

not:

```text
absolute martensite volume percent
```
