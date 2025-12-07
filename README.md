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

given fiscal/monetary parameters.

Serves as the ground-truth generator for training the neural surrogate.

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

Useful for embedding IS–LM into larger dynamic simulations
without recomputing closed-form solutions repeatedly.

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

Current behavior:

Produces small ripples in 
𝑧
𝑡
z
t
	​

 after a shock

Output decays quickly (over-damped)

Visible but mild influence on 
𝑌
𝑡
Y
t
	​

 and almost no change in 
𝑟
𝑡
r
t
	​


Fully functional pipeline:
shock → zₜ → G_eff → IS–LM → (Yₜ, rₜ)

This version is conceptual and intended for research iteration.

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
├── docs/
│   ├── ISLM_QQSS_NOTES.md
│   └── ssrn/
│       └── paper.pdf
│
├── README.md
└── .gitignore

📘 Prototype Status
✔ Completed

Analytic IS–LM solution

Data generator for supervised training

Neural surrogate model (trained)

QQSS 4-state dynamic module

End-to-end pipeline

⚠ Current Limitations

QQSS is over-damped → no oscillatory cycles

Coupling into fiscal variables is intentionally weak

Not yet calibrated for macro-scale dynamics

Not suitable for publication-grade econometric validation (yet)

🎯 Future Directions

Introduce complex eigenvalues for stable macro-oscillations

Strengthen QQSS → (G,T,M,P) mapping

Explore multi-channel interpretation: expectations, liquidity tension, policy noise

Tune for 3–5% output sensitivity per moderate shock

For detailed notes, see:
docs/ISLM_QQSS_NOTES.md

🚀 Running the Project
Run the analytic IS–LM simulation
python src/simulate_islm_equilibrium.py

Generate training data
python src/ISLMdata_generator.py

Train the neural surrogate
python src/ISLMtrain.py

Integrate QQSS dynamic shocks

Import and combine inside your simulation code:

from ISLMqqss_module import QQSSModule
from ISLMMode import ISLMModel

📄 License

MIT License (recommended for academic dissemination).
You may replace it with CC-BY 4.0 if preferred.

📎 Citation

A CITATION.cff file can be added upon request
for Zenodo DOI + ORCID-linked publication.
