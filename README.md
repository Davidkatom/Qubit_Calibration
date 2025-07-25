# Qubit Calibration

This repository contains Python code that accompanies the paper **"Optimal Calibration of Qubit Detuning and Crosstalk"**. The scripts implement simulations of multi-qubit Ramsey experiments and routines to estimate detuning and crosstalk errors.

## Repository Contents

- `Ramsey_ExperimentV3.py` – tools for simulating Ramsey sequences with optional decoherence.
- `estimator.py` – functions to fit Ramsey data and compute estimation errors.
- `symbolic_evolution.py` – symbolic utilities used to derive expectation values and Fisher information.
- `fisher_information.ipynb` – Jupyter notebook demonstrating calculation of the Fisher information for different experiment settings.
- `Comparisons/` – notebooks exploring how estimation accuracy changes with experiment parameters such as the number of qubits or measurement shots.
- `requirements.txt` – list of Python dependencies.
- The PDFs `Optimal_calibration_of_qubit_detuning_and_crosstalk.pdf` and `Supplementary material.pdf` contain the accompanying paper and supplementary information.

## Installation

Create a Python environment (e.g. using `venv`) and install the dependencies:

```bash
pip install -r requirements.txt
```

The code relies on QuTiP and standard scientific Python packages.

## Usage

The modules are mainly intended to be imported from the notebooks. As a basic example you can compute simulated Ramsey data with

```python
import Ramsey_ExperimentV3 as ramsey

# number of qubits
n = 2
# delays at which the system is probed
delay = [0.0, 1.0, 2.0]
# detuning terms
W = [0.1] * n
# crosstalk couplings
J = {(0, 1): 0.02}
# dephasing rates
Gamma_phi = [0.0] * n

batch = ramsey.ramsey_local(n, total_shots=1000, delay=delay,
                            Gamma_phi=Gamma_phi, W=W, J=J)
```

See the notebooks in the `Comparisons` directory and `fisher_information.ipynb` for demonstrations of how these routines were used in the paper.

## License

This repository is distributed for academic use only. See the PDFs for citation details.
