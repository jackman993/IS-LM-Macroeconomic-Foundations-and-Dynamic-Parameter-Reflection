# ISLM_v02: Analytic IS–LM Model, Neural Approximation, and QQSS Dynamic Shock Prototype

This repository contains **Prototype 0.1** of an IS–LM analytic equilibrium solver, a neural surrogate model, and an experimental QQSS four-state dynamic shock module. The system is intended as a conceptual testbed for dynamic parameter reflection and macroeconomic simulation.

## Folder Structure

```text
ISLM_v02/
│
├── src/
│   ├── ISLMMode.py                 # Core IS-LM logic
│   ├── ISLMqqss_module.py          # QQSS Dynamic Shock Module
│   ├── ISLMdata_generator.py       # Data generation for training
│   ├── ISLMtrain.py                # Training script for neural surrogate
│   ├── simulate_islm_equilibrium.py # Main simulation script
│   └── models/
│       └── islm_neurocore_model.ckpt # Pre-trained model checkpoint
│
├── output/                         # Simulation outputs
├── lightning_logs/                 # Training logs
│
├── docs/
│   ├── ISLM_QQSS_NOTES.md          # Technical notes
│   └── ssrn/
│       └── paper.pdf               # Related paper/documentation
│
└── .gitignore
