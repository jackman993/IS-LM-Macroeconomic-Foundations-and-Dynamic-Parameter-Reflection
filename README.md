# ISLM_v02: Analytic IS–LM Model, Neural Approximation, and QQSS Dynamic Shock Prototype

This repository contains **Prototype 0.1** of an IS–LM analytic equilibrium solver, a neural surrogate model, and an experimental QQSS four-state dynamic shock module. The system is intended as a conceptual testbed for dynamic parameter reflection and macroeconomic simulation.

Prototype Status
Completed
[x] Analytic IS–LM model with closed-form equilibrium.

[x] Data generator for supervised learning.

[x] Neural surrogate model (trained checkpoint included).

[x] QQSS four-state dynamic module.

[x] End-to-end simulation pipeline executes without errors.

Current Limitations
Dynamics: QQSS dynamics are currently over-damped and do not produce oscillatory cycles.

Coupling: Coupling from QQSS states to policy variables is intentionally weak.

Calibration: Parameters have not been calibrated for realistic macroeconomic behavior.

Scope: Not intended for empirical or policy analysis in its current form.

Future Directions
Complex Dynamics: Introduce complex eigenvalue dynamics to generate stable oscillations.

Mapping: Strengthen the mapping from QQSS states to fiscal and monetary parameters.

Interpretation: Assign interpretable economic meaning to each QQSS channel.

Target Sensitivity: Aim for a target where one moderate shock generates approximately 3–5 percent movement in output.

How to Run
1. Analytic IS–LM Simulation
Run the main simulation pipeline:

Bash

python src/simulate_islm_equilibrium.py
2. Generate Training Data
If you need to regenerate the dataset for the neural model:

Bash

python src/ISLMdata_generator.py
3. Train the Neural Surrogate
Train the neural network using the generated data:

Bash

python src/ISLMtrain.py
Example Usage
Here is a snippet showing how to integrate the QQSS module with the IS–LM model:

Python

from ISLMMode import ISLMModel
from ISLMqqss_module import QQSSModule

# Initialize models
islm = ISLMModel(...)
qqss = QQSSModule(...)

# Execute a step
z_t = qqss.step(shock=1.0)

# Apply shock to Government Spending (G_eff)
G_eff = base_G + qqss.to_G_eff(z_t)

# Solve for equilibrium
Y, r = islm.solve_equilibrium(G_eff=G_eff, ...)
License
MIT License (recommended for this prototype). You may replace this with a different open license if needed.

Citation
ISLM_v02: Analytic IS–LM Model with Neural Approximation and QQSS Dynamic Shock Prototype. GitHub repository, 2025.
