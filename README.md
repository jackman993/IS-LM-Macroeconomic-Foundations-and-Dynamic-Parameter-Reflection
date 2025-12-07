# ISLM_v02 — Analytic IS–LM Model × Neural Approximation × QQSS Dynamic Shock Prototype
**Macroeconomic Foundations and Dynamic Parameter Reflection**

This repository contains the **Prototype 0.1** implementation of the  
**IS–LM analytic equilibrium model**, a **Neural Network surrogate (ISLM NeuroCore)**,  
and the experimental **QQSS four-channel dynamic shock module**.

The goal is **not** to present a fully validated macroeconomic model, but to provide a  
**research testbed** for dynamic interaction between:

- Structural IS–LM equilibrium  
- Learned neural mappings  
- Shock-driven dynamic parameter reflection (QQSS)

This prototype is suitable for exploration, simulation, and future expansion.

---

## 🔍 Project Overview

### **1. Analytic IS–LM Equilibrium Model**
**File:** `simulate_islm_equilibrium.py`

- Computes closed-form simultaneous equilibrium for **(Y, r)**  
  given fiscal and monetary parameters.
- Serves as ground-truth generator for training the neural surrogate.

---

### **2. Neural Surrogate: ISLM NeuroCore**

**Files:**

- `ISLMdata_generator.py`
- `ISLMtrain.py`
- `models/islm_neurocore_model.ckpt`

The neural network approximates:

\[
(G, T, M, P) \longrightarrow (Y, r)
\]

This allows embedding IS–LM inside large dynamic systems  
without recalculating analytic solutions repeatedly.

---

### **3. QQSS Dynamic Shock Module (Prototype 0.1)**

**File:** `ISLMqqss_module.py`

A **4-state dynamic system** reacting to shocks and modifying policy parameters:

\[
G_{\text{eff}} = G + f(z_t)
\]


This version is conceptual and intended for further calibration.

---

## 📂 Folder Structure

```text
ISLM_v02/
│
├── src/
│   ├── ISLMMode.py
│   ├── ISLMqqss_module.py
│   ├── ISLMdata_generator.py
│   ├── ISLMtrain.py
│   ├── simulate_islm_equilibrium.py
│   └── models/
│       └── islm_neurocore_model.ckpt
│
├── output/                # Simulation logs, plots, intermediate results
├── lightning_logs/        # PyTorch Lightning training logs
│
├── docs/
│   ├── ISLM_QQSS_NOTES.md📘 Prototype Status
✔ Completed

Analytic IS–LM core with closed-form solution

Training data generator

Neural surrogate (checkpoint included)

QQSS 4-state dynamic module

End-to-end simulation pipeline

⚠ Current Limitations

QQSS is over-damped → no oscillatory macro cycles

Weak coupling from 
𝑧
𝑡
z
t
	​

 → fiscal/monetary variables

Parameters not yet calibrated for realistic macro behavior

Not yet suitable for empirical macroeconomic claims

🎯 Future Directions

Introduce complex eigenvalues for stable oscillations

Strengthen QQSS → (G, T, M, P) mapping

Assign economic interpretation to each QQSS state

Target: 3–5% movement in output 
𝑌
Y per moderate shock

More details: docs/ISLM_QQSS_NOTES.md

🚀 How to Run
1. Analytic IS–LM simulation
python src/simulate_islm_equilibrium.py

2. Generate NN training data
python src/ISLMdata_generator.py

3. Train the neural surrogate
python src/ISLMtrain.py

4. Use QQSS in your own experiment
from ISLMMode import ISLMModel
from ISLMqqss_module import QQSSModule

islm = ISLMModel(...)
qqss = QQSSModule(...)

z_t = qqss.step(shock=1.0)
G_eff = base_G + qqss.to_G_eff(z_t)
Y, r = islm.solve_equilibrium(G_eff=G_eff, ...)

📄 License

MIT License (recommended for academic + open-source use)

📎 Citation

You may cite this repository as:

ISLM_v02: Analytic IS–LM Model with Neural Approximation and QQSS Dynamic Shock Prototype.
GitHub repository, 2025.
│   └── ssrn/
│       └── paper.pdf
│
├── README.md
└── .gitignore


Current behavior:

- Produces **small, fast-decaying ripples** in \(z_t\)  
- Output \(Y_t\) changes slightly; \(r_t\) remains nearly unchanged  
- Full pipeline works:

