# ISLM_v02: Analytic IS–LM Model, Neural Approximation, and QQSS Dynamic Shock Prototype

This repository contains Prototype 0.1 of the IS–LM analytic equilibrium solver, a neural surrogate model, and the experimental QQSS four-state dynamic shock module. The system is intended as a conceptual testbed for dynamic parameter reflection and macroeconomic simulation.

---

## Folder Structure

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
├── output/                
├── lightning_logs/        
│
├── docs/
│   ├── ISLM_QQSS_NOTES.md
│   └── ssrn/
│       └── paper.pdf
│
└── .gitignore
Prototype Status
Completed
Analytic IS–LM model with closed-form equilibrium.

Data generator for supervised learning.

Neural surrogate model (trained checkpoint included).

QQSS four-state dynamic module.

End-to-end simulation pipeline executes without errors.

Current Limitations
QQSS dynamics are over-damped and do not produce oscillatory cycles.

Coupling from QQSS states to policy variables is intentionally weak.

Parameters have not been calibrated for realistic macroeconomic behavior.

Not intended for empirical or policy analysis in its current form.

Future Directions
Introduce complex eigenvalue dynamics to generate stable oscillations.

Strengthen the mapping from QQSS states to fiscal/monetary parameters.

Assign interpretable economic meaning to QQSS channels.

Target: one moderate shock should generate approximately 3–5% movement in output.

How to Run
Analytic IS–LM simulation
bash
複製程式碼
python src/simulate_islm_equilibrium.py
Generate training data
css
複製程式碼
python src/ISLMdata_generator.py
Train the neural surrogate
css
複製程式碼
python src/ISLMtrain.py
Example (using QQSS + ISLM)
python
複製程式碼
from ISLMMode import ISLMModel
from ISLMqqss_module import QQSSModule

islm = ISLMModel(...)
qqss = QQSSModule(...)

z_t = qqss.step(shock=1.0)
G_eff = base_G + qqss.to_G_eff(z_t)
Y, r = islm.solve_equilibrium(G_eff=G_eff, ...)
License
MIT License (recommended). You may replace this with a different open license if needed.

Citation
ISLM_v02: Analytic IS–LM Model with Neural Approximation and QQSS Dynamic Shock Prototype. GitHub repository, 2025.
