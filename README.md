# ISLM_v02: Analytic IS–LM Model, Neural Approximation, and QQSS Dynamic Shock Prototype

This repository contains Prototype 0.1 of an IS–LM analytic equilibrium solver, a neural surrogate model, and an experimental four-state QQSS dynamic shock module.  
The system serves as a conceptual testbed for dynamic parameter reflection and macroeconomic simulation.

---

## Overview

This prototype system includes:

### 1. Analytic IS–LM Equilibrium Solver
A closed-form solver that computes simultaneous equilibrium (Y, r) based on fiscal and monetary parameters.  
Used as the ground-truth generator for surrogate training.

### 2. Neural Surrogate Model (ISLM NeuroCore)
A trained neural network mapping:
(G, T, M, P) → (Y, r)

Useful for embedding the IS–LM mechanism into larger dynamic systems without repeatedly solving the analytic form.

### 3. QQSS Dynamic Shock Module (Prototype 0.1)
A four-state dynamical system that reacts to external shocks and produces reflected fiscal parameters.  
Currently generates small, over-damped perturbations and weak coupling to macro variables.

### 4. End-to-End Pipeline
shock → QQSS state → effective fiscal variables → IS–LM equilibrium

---

## Prototype Status (Completed)

- Analytic IS–LM model with closed-form equilibrium  
- Training data generator for supervised learning  
- Trained neural surrogate model included  
- Four-state QQSS dynamic module  
- End-to-end simulation pipeline executes without errors  

---

## Current Limitations

- QQSS dynamics are over-damped and do not generate oscillatory cycles  
- Weak mapping from QQSS states to fiscal/monetary parameters  
- Parameters not yet calibrated for realistic macroeconomic behavior  
- Not intended for policy or empirical analysis in current form  

---

## Future Directions

- Introduce complex eigenvalues to produce stable oscillatory macro-dynamics  
- Strengthen QQSS → (G, T, M, P) coupling  
- Interpret each QQSS channel as an economic factor  
  (expectation shock, external demand, financial stress, policy noise)  
- Target behavior: 3–5% Y deviation from a moderate single shock  

---

## How to Run

### 1. Run analytic IS–LM simulation
```bash
python simulate_islm_equilibrium.py
2. Generate training data
bash
複製程式碼
python ISLMdata_generator.py
3. Train the neural surrogate
bash
複製程式碼
python ISLMtrain.py
4. Example usage: QQSS + IS–LM
python
複製程式碼
from ISLMMode import ISLMModel
from ISLMqqss_module import QQSSModule

islm = ISLMModel(...)
qqss = QQSSModule(...)

z = qqss.step(shock=1.0)
G_eff = base_G + qqss.to_G_eff(z)

Y, r = islm.solve_equilibrium(G_eff=G_eff, ...)
License
MIT License (recommended for open-source and academic use)

Citation
A CITATION.cff file may be added for Zenodo DOI and ORCID integration.

Suggested citation:

ISLM_v02: Analytic IS–LM Model with Neural Approximation and QQSS Dynamic Shock Prototype. GitHub, 2025.
