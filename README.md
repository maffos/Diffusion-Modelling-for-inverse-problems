# Diffusion Modelling for Inverse Problems

This repository contains the code for my master thesis, which applies score-based generative models to inverse problems. The project includes two examples: one linear problem and one real-world application from scatterometry. In particular it leverages a newly developed partial differential equation (PDE) that is based on the Fokker-Planck Equation and acts on the score. The PDE was independently developed in [3] and termed 'Score-Fokker-Planck-Equation'. 

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
- [References](#references)

## Description
This project explores the application of score-based generative models to inverse problems. In particular the 'Conditional Denoising Estimator (CDE)' and 'Conditional Diffusive Estimator' (CDiffE) in [1], as well as the 'Diffusion Posterior Sampler' in [4] are implemented. Furthermore, these models are equipped with a novel loss function (PINNLoss) that is based on the PDE called 'ScoreFPE' [3]. In cases where the score of the posterior distribution is known (which is often the case in inverse problems), one can use the ScoreFPE to train score-based generative models as a Physics-Informed-Neural-Network (PINN) [5]. The resulting loss function is called PINNLoss and can be found in the losses.py module. It includes benchmarks of these models against other methods such as stochastic normalizing flows [2]. The project provides two example applications:
1. A toy linear problem.
2. A real-world application from scatterometry.

For the scatterometry problem, the pretrained surrogate of [2] was used.

## Features
- **Benchmarking**: Compare score-based generative models with other methods like stochastic normalizing flows.
- **Novel Loss Function**: Implementation of a novel loss function (PINNLoss) for score-based generative models, similar to [3], in the `losses.py` module.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
Four main files are provided to train and evaluate the baseline models and the diffusion models on the linear and scatterometry datasets, respectively:
- `main_baselines_linear.py`
- `main_diffusion_linear.py`
- `main_baselines_scatterometry.py`
- `main_diffusion_scatterometry.py`

To try out different model architectures and loss functions, adjust the corresponding fields in the respective configuration files.

## Contact
For any questions or further information, please contact:
Matthias Wamhoff  
Email: matthias_wamhoff@web.de

## References
[1] Batzolis, Georgios, et al. "Conditional image generation with score-based diffusion models." arXiv preprint arXiv:2111.13606 (2021).  
[2] Hagemann, Paul, Johannes Hertrich, and Gabriele Steidl. "Stochastic normalizing flows for inverse problems: a Markov Chains viewpoint." SIAM/ASA Journal on Uncertainty Quantification 10.3 (2022): 1162-1190.  
[3] Lai, Chieh-Hsin, et al. "Regularizing score-based models with score fokker-planck equations." NeurIPS 2022 Workshop on Score-Based Methods. 2022.  
[4] Chung, Hyungjin, et al. "Diffusion posterior sampling for general noisy inverse problems." arXiv preprint arXiv:2209.14687 (2022).  
[5] Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational physics 378 (2019): 686-707.


