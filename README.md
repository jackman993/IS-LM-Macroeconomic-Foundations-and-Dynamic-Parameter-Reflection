ISLM_v02 — Analytic IS–LM Model × Neural Approximation × QQSS Dynamic Shock Prototype

Macroeconomic Foundations and Dynamic Parameter Reflection

This repository contains the Prototype 0.1 implementation of the
IS–LM analytic equilibrium model,
a Neural Network surrogate (ISLM NeuroCore),
and the experimental QQSS four-channel dynamic shock module.

The goal is not to present a fully validated macroeconomic model,
but to provide a research testbed for dynamic interaction between:

structural IS–LM equilibrium

learned neural mappings

shock-driven dynamic parameter reflection (QQSS)

This prototype is suitable for exploration, simulation, and future expansion.

🔍 Project Overview
1. Analytic IS–LM Equilibrium Model

File: simulate_islm_equilibrium.py

Computes closed-form simultaneous equilibrium for 
(
𝑌
,
𝑟
)
(Y,r)
given fiscal and monetary parameters.

Serves as ground-truth generator for training the neural surrogate.

2. Neural Surrogate: ISLM NeuroCore

Files:

ISLMdata_generator.py

ISLMtrain.py

models/islm_neurocore_model.ckpt

The neural network approximates:

(
𝐺
,
𝑇
,
𝑀
,
𝑃
)
⟶
(
𝑌
,
𝑟
)
(G,T,M,P)⟶(Y,r)

This is useful when embedding IS–LM inside larger dynamic systems,
avoiding repeated analytic solving during simulation.

3. QQSS Dynamic Shock Module (Prototype 0.1)

File: ISLMqqss_module.py

A 4-state dynamic system designed to react to external shocks
and produce reflected fiscal parameters:

𝐺
eff
=
𝐺
+
𝑓
(
𝑧
𝑡
)
G
eff
	​

=G+f(z
t
	​

)

Current behavior (Prototype 0.1):

A single shock produces small, fast-decaying ripples in 
𝑧
𝑡
z
t
	​


Output 
𝑌
𝑡
Y
t
	​

 moves slightly; 
𝑟
𝑡
r
t
	​

 is almost unchanged

The full pipeline is functional:

shock → z_t → G_eff → IS–LM → (Y_t, r_t)

This version is conceptual and intended for further calibration and theory work,
not yet for empirical macroeconomic claims.

📂 Folder Structure
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


If your actual repo structure is slightly different,
feel free to adjust file names / folders accordingly.

📘 Prototype Status
✔ Completed

Analytic IS–LM core with closed-form equilibrium

Data generator for supervised learning

Neural surrogate model (trained checkpoint)

QQSS 4-state dynamic module

End-to-end pipeline runs without error

⚠ Current Limitations

QQSS is over-damped → no clear oscillatory macro cycles

Coupling from 
𝑧
𝑡
z
t
	​

 to policy variables is intentionally weak

Parameters not yet calibrated for realistic macro dynamics

Not ready for publication-grade empirical evaluation (yet)

🎯 Future Directions

Introduce complex eigenvalues for stable macro-oscillations

Strengthen QQSS → 
(
𝐺
,
𝑇
,
𝑀
,
𝑃
)
(G,T,M,P) mappings

Map each QQSS channel to interpretable economic factors
(expectations, external demand, financial stress, policy noise…)

Target: a single moderate shock induces ~3–5% movement in 
𝑌
Y
with visible but stable variation in 
𝑟
r

For more technical notes, see: docs/ISLM_QQSS_NOTES.md.

🚀 How to Run
1. Run analytic IS–LM simulation
python src/simulate_islm_equilibrium.py

2. Generate training data
python src/ISLMdata_generator.py

3. Train the neural surrogate
python src/ISLMtrain.py

4. Use QQSS in your own experiment
from ISLMMode import ISLMModel
from ISLMqqss_module import QQSSModule

# Example (pseudo-code):
islm = ISLMModel(...)
qqss = QQSSModule(...)

z_t = qqss.step(shock=1.0)
G_eff = base_G + qqss.to_G_eff(z_t)
Y, r = islm.solve_equilibrium(G_eff=G_eff, ...)


(Adjust the actual API to match your current code.)

📄 License

Suggested: MIT License (good for academic + open-source use).
If you prefer CC-BY 4.0 for data/paper-aligned release, you can replace it.

📎 Citation

A CITATION.cff file can be added later for Zenodo DOI and ORCID integration.
For now, you may cite this repo as:

ISLM_v02: Analytic IS–LM Model with Neural Approximation and QQSS Dynamic Shock Prototype.
GitHub repository, 2025.
