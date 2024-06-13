# Diffusion Modelling for Inverse Problems

This repository contains the code for my master thesis, which applies score-based generative models to inverse problems. The project includes two examples: one linear problem and one real-world application from scatterometry.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
- [References](#references)

## Description
This project explores the application of score-based generative models to inverse problems. It includes benchmarks of these models against other methods such as stochastic normalizing flows. The project provides two example applications:
1. A linear problem
2. A real-world application from scatterometry.

## Features
- **Benchmarking**: Compare score-based generative models with other methods like stochastic normalizing flows [2].
- **Novel Loss Function**: Implementation of a novel loss function for score-based generative models, similar to [3], in the `loss.py` module.

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
1. [Score-Based Generative Models](link to reference)
2. [Scatterometry Application](link to reference)
3. [Similar Loss Function] Here's a draft for your README file based on the provided details:

---

# Diffusion Modelling for Inverse Problems

This repository contains the code for my master thesis, which applies score-based generative models to inverse problems. The project includes two examples: one linear problem and one real-world application from scatterometry.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contact](#contact)
- [References](#references)

## Description
This project explores the application of score-based generative models to inverse problems. It includes benchmarks of these models against other methods such as stochastic normalizing flows. The project provides two example applications:
1. A linear problem
2. A real-world application from scatterometry.

## Features
- **Benchmarking**: Compare score-based generative models with other methods like stochastic normalizing flows.
- **Novel Loss Function**: Implementation of a novel loss function for score-based generative models, similar to [3], in the `loss.py` module.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
Four main files are provided to train and evaluate the baseline models and the diffusion models on the linear and scatterometry datasets, respectively:
- `train_linear.py`
- `evaluate_linear.py`
- `train_scatterometry.py`
- `evaluate_scatterometry.py`

To try out different model architectures and loss functions, adjust the corresponding fields in the respective configuration files.

## Requirements
Please refer to the `requirements.txt` file for a complete list of dependencies.

## Contact
For any questions or further information, please contact:
Matthias Wamhoff  
Email: matthias_wamhoff@web.de

## References
1. [Score-Based Generative Models](link to reference)
2. [Scatterometry Application](link to reference)
3. [Similar Loss Function] Here's a draft for your README file based on the provided details:

---

# Diffusion Modelling for Inverse Problems

This repository contains the code for my master thesis, which applies score-based generative models to inverse problems. The project includes two examples: one linear problem and one real-world application from scatterometry.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contact](#contact)
- [References](#references)

## Description
This project explores the application of score-based generative models to inverse problems. In particular the 'Conditional Denoising Estimator (CDE)' and 'Conditional Diffusive Estimator' (CDiffE) in [1], as well as the 'Diffusion Posterior Sampler' in [4] are implemented. Furthermore, these models are equipped with a novel loss function that incorporates knowledge of the underlying Fokker-Planck Equation, similar to the one in [3]. It includes benchmarks of these models against other methods such as stochastic normalizing flows [2]. The project provides two example applications:
1. A toy linear problem.
2. A real-world application from scatterometry.

## Features
- **Benchmarking**: Compare score-based generative models with other methods like stochastic normalizing flows.
- **Novel Loss Function**: Implementation of a novel loss function for score-based generative models, similar to [3], in the `loss.py` module.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
Four main files are provided to train and evaluate the baseline models and the diffusion models on the linear and scatterometry datasets, respectively:
- `train_linear.py`
- `evaluate_linear.py`
- `train_scatterometry.py`
- `evaluate_scatterometry.py`

To try out different model architectures and loss functions, adjust the corresponding fields in the respective configuration files.

## Requirements
Please refer to the `requirements.txt` file for a complete list of dependencies.

## Contact
For any questions or further information, please contact:
Matthias Wamhoff  
Email: matthias_wamhoff@web.de

## References
1. [Score-Based Generative Models] Batzolis, Georgios, et al. "Conditional image generation with score-based diffusion models." arXiv preprint arXiv:2111.13606 (2021).
2. [Scatterometry Application] Hagemann, Paul, Johannes Hertrich, and Gabriele Steidl. "Stochastic normalizing flows for inverse problems: a Markov Chains viewpoint." SIAM/ASA Journal on Uncertainty Quantification 10.3 (2022): 1162-1190.
3. [Similar Loss Function] Lai, Chieh-Hsin, et al. "Regularizing score-based models with score fokker-planck equations." NeurIPS 2022 Workshop on Score-Based Methods. 2022.
4. [Diffusion POsterior Sampling] Chung, Hyungjin, et al. "Diffusion posterior sampling for general noisy inverse problems." arXiv preprint arXiv:2209.14687 (2022).

For the scatterometry problem, the pretrained surrogate of [2] was used.

---

Feel free to adjust any sections or add more details as needed. If you have links to the references, you can replace the placeholders with the actual URLs.

For the scatterometry problem, the pretrained surrogate of [2] was used.

---

Feel free to adjust any sections or add more details as needed. If you have links to the references, you can replace the placeholders with the actual URLs.

For the scatterometry problem, the pretrained surrogate of [2] was used.

---

Feel free to adjust any sections or add more details as needed. If you have links to the references, you can replace the placeholders with the actual URLs.
