# Forward-backward Plug-and-Play with dictionary-based denoisers

This repo contains the scripts to reproduce the experiments of the "[Analysis and Synthesis Denoisers for Forward-Backward Plug-and-Play Algorithms](https://hal.science/hal-04786802)",  Matthieu Kowalski, Benoît Malézieux, Thomas Moreau, Audrey Repetti, preprint 2024.

This paper aims to study the properties of a PnP scheme in which the denoiser is dictionary-based, either using a synthesis or an analysis formulation.

### Install

This repo's dependencies can be installed using `pip install -r requirements`. 
The experiments are performed using the `BSD500` dataset, which can be retrieved using the `create_dataset.py` script.

### Experiments

The paper's experiments can be reproduced using two scripts:

- `convergence_algorithm.py`: this script runs the computations necessary to produce Figures 1 and 2.

- `pnp_comparisons.py`: this script runs the comparison between DRUNet, and various versions of SD/AD unrolling in PnP. It compares the runtime, the convergence profile, and the PSNR for varying images and regularization parameters. It also displays some reconstruction examples.

Example of reconstruction provides the following figures:
![image](https://github.com/user-attachments/assets/ba090eab-7304-4cb1-8923-846252ee1b3b)
![image](https://github.com/user-attachments/assets/07a01385-4c2d-4671-ab30-48913946f733)


