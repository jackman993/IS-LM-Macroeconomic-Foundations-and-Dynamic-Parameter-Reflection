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
│   ├── ISLM_QQSS_NOTES.md
│   └── ssrn/
│       └── paper.pdf
│
├── README.md
└── .gitignore





## 📘 Prototype Status

### ✔ Completed
- Analytic IS–LM core with closed-form equilibrium  
- Data generator for supervised learning  
- Neural surrogate model (trained checkpoint)  
- QQSS 4-state dynamic shock module  
- End-to-end pipeline runs without error  

---

### ⚠ Current Limitations
- QQSS is **over-damped** → no clear oscillatory macro cycles  
- Weak coupling from \(z_t\) to policy variables  
- Parameters not yet calibrated for realistic macro dynamics  
- Not ready for publication-grade empirical evaluation  

---

### 🎯 Future Directions
- Introduce **complex eigenvalues** for structured oscillations  
- Strengthen QQSS → (G, T, M, P) mappings  
- Assign interpretations to each QQSS dynamic channel  
- Target: moderate shock induces **3–5% movement** in output \(Y\)  

More details: [`docs/ISLM_QQSS_NOTES.md`](docs/ISLM_QQSS_NOTES.md)


