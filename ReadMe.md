\# BOA PNI Reconstruction



Professional reconstruction pipeline for BOA-style polarization contrast neutron tomography.



This project reconstructs two scalar volumes from spin-resolved neutron tomography data:



\- attenuation coefficient volume from `-log(I / I0)`

\- depolarization contrast volume from `-log(P / P0)`



where:



`I = I\_up + I\_down`



and:



`P = (I\_up - I\_down) / (I\_up + I\_down)`



The reconstruction backend is ASTRA Toolbox using parallel-beam SIRT.



\## Scope



This repository is designed for scalar polarization-contrast neutron tomography, such as mapping ferromagnetic phase distributions.



It is not a full magnetic vector-field polarimetric neutron tomography solver.



\## Basic usage



```bash

pixi run python scripts/reconstruct\_pni.py --config configs/boa\_default.yaml

