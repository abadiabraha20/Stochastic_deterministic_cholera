# Stochastic and deterministic modeling of cholera dynamics with multiple transmission pathways, asymptomatic infection, and vaccination

**CPython code for bifurcation analysis and optimal control in a cholera transmission model. Generates all figures for the manuscript.**

## Overview

This repository contains the complete source code for the manuscript:

> *"Stochastic and deterministic modeling of cholera dynamics with multiple transmission pathways, asymptomatic infection, and vaccination"*

The model integrates seven compartments (susceptible $S$, vaccinated $V$, asymptomatic $A$, symptomatic $I$, treated $T$, recovered $R$, and environmental pathogen concentration $B$) with direct and environmental transmission, seasonality, vaccination, treatment, and stochastic noise.

## Key Features

- **Deterministic simulation** – 4th order Runge-Kutta method
- **Stochastic simulation** – Euler-Maruyama with 5,000 realizations
- **Bifurcation analysis** – Transcritical and Hopf bifurcations via numerical continuation
- **Optimal control** – Forward-backward sweep with Pontryagin's minimum principle
- **Sensitivity analysis** – PRCC with Latin Hypercube Sampling (10,000 samples)
- **Extinction probability** – Estimated from 10,000 simulations per parameter set

## Repository Structure

```
├── src/
│   ├── deterministic.py      # Deterministic model (RK4)
│   ├── stochastic.py         # Stochastic model (Euler-Maruyama)
│   ├── bifurcation.py        # Numerical continuation (MATCONT interface)
│   ├── optimal_control.py    # Forward-backward sweep
│   ├── sensitivity.py        # PRCC with LHS
│   └── utils.py              # Helper functions
├── figures/                  # All manuscript figures generated here
├── data/                     # Parameter tables and simulation outputs
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Required packages (install via `pip install -r requirements.txt`)

```bash
pip install -r requirements.txt
```

### Running the Code

To reproduce all figures from the manuscript:

```bash
python src/main.py
```

To run individual components:

```bash
python src/deterministic.py   # Generate deterministic trajectories
python src/stochastic.py      # Generate stochastic ensemble simulations
python src/bifurcation.py     # Generate bifurcation diagrams
python src/optimal_control.py # Generate optimal control profiles
python src/sensitivity.py     # Generate PRCC sensitivity analysis
```

## Parameters

All parameter values used in the analysis are provided in **Table 1** of the manuscript, with the following baseline values:

| Parameter | Description | Baseline value |
|-----------|-------------|----------------|
| $\beta_0$ | Baseline direct transmission rate | 0.25 day$^{-1}$ |
| $\eta$ | Environmental transmission rate | 0.001 day$^{-1}$ |
| $\nu_V$ | Vaccination rate | 0.003 day$^{-1}$ |
| $\epsilon$ | Vaccine efficacy | 0.75 |
| $\gamma_T$ | Treatment rate | 0.3 day$^{-1}$ |
| $\nu_i$ | Noise intensities | 0.05 |

Complete parameter descriptions and sources are available in the manuscript.

## Figure Generation

This code generates all figures presented in the manuscript:

| Figure | Description |
|--------|-------------|
| Fig. 1 | Compartmental flow diagram |
| Fig. 2 | Deterministic vs. stochastic comparison |
| Fig. 3 | Extinction probability vs. $\mathcal{R}_0^S$ |
| Fig. 4 | Vaccination intervention strategies |
| Fig. 5 | Environmental interventions |
| Fig. 6 | Optimal control strategies |
| Fig. 7 | Seasonal forcing dynamics |
| Fig. 8 | Sensitivity analysis (PRCC, tornado)|
| Fig. 9 | Bifurcation diagrams (transcritical and Hopf) |
| Fig. 10 | Phase portraits ($I$-$B$, $S$-$I$, $A$-$I$) |
| Fig. 11 | Validation against Haiti and Yemen outbreaks |


## Run in Google Colab

Click any link below to open and run the code in your browser:

| Figure | Colab Link |
|--------|------------|
| Fig2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig2.py) |
| Fig3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig3.py) |
| Fig4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig4.py) |
| Fig5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig5.py) |
| Fig6 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig6.py) |
| Fig7 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig7.py) |
| Fig8 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig8.py) |
| Fig9 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig9.py) |
| Fig10 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig10.py) |
| Fig11 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abadiabraha20/Stochastic_deterministic_cholera/blob/main/Fig11.py) |

### How to use:

1. Click the badge above
2. In Colab, go to **Runtime → Run all**
3. The figure will be displayed at the bottom

## Citation

If you use this code in your research, please cite the manuscript:

```bibtex
@article{Asgedom2025Cholera,
  title = {Stochastic and deterministic modeling of cholera dynamics with multiple transmission pathways, asymptomatic infection, and vaccination},
  author = {Asgedom, Abadi Abraha and Kefela, Yohannes Yirga and Gebrehiwot, Berihu Teklu and Hatzikirou, Haralampos},
  journal = {Submitted for publication},
  year = {2025}
}
```
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/abadiabraha20/Stochastic_deterministic_cholera/main)

## Contact

Corresponding author: Abadi Abraha Asgedom  
Email: [abadi.abraha@mu.edu.et](mailto:abadi.abraha@mu.edu.et)

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments

The authors acknowledge institutional support from Mekelle University and Khalifa University.
