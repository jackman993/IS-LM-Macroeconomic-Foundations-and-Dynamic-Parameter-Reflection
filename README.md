# ISLM_v02: Analytic IS–LM Model, Neural Approximation, and QQSS Dynamic Shock Prototype

This repository contains Prototype 0.1 of an IS–LM analytic equilibrium solver, a neural surrogate model, and an experimental QQSS four-state dynamic shock module. The system is intended as a conceptual testbed for dynamic parameter reflection and macroeconomic simulation.

## Files Included

ISLMMode.py  
ISLMqqss_module.py  
ISLMdata_generator.py  
ISLMtrain.py  
simulate_islm_equilibrium.py  
islm_neurocore_model.ckpt (in models/)  
Simulation logs and plots (in output/)  
PyTorch Lightning logs (in lightning_logs/)  
ISLM_QQSS_NOTES.md (in docs/)  
paper.pdf (in docs/ssrn/)  
README.md  

This listing avoids tree-format folders so that GitHub renders consistently.

## Prototype Status

### Completed
- Analytic IS–LM model with closed-form equilibrium.  
- Data generator for supervised learning.  
- Neural surrogate model (trained checkpoint included).  
- QQSS four-state dynamic module.  
- End-to-end simulation pipeline executes without errors.

### Current Limitations
- QQSS is over-damped and does not generate oscillatory macro cycles.  
- Coupling from QQSS states to policy variables is intentionally weak.  
- Parameters have not been calibrated for realistic macroeconomic behavior.  
- Not intended for empirical or policy evaluation in the current form.

### Future Directions
- Introduce complex eigenvalues to produce stable oscillations.  
- Strengthen QQSS-to-policy mappings.  
- Map QQSS channels to interpretable macroeconomic forces.  
- Target behavior: moderate shocks should cause 3–5% movements in output with stable interest-rate variation.

---

## How to Run

### 1. Run analytic IS–LM simulation
```bash
python simulate_islm_equilibrium.py
```

### 2. Generate training data
```bash
python ISLMdata_generator.py
```

### 3. Train the neural surrogate
```bash
python ISLMtrain.py
```

### 4. Example usage: QQSS + IS–LM
```python
from ISLMMode import ISLMModel
from ISLMqqss_module import QQSSModule

islm = ISLMModel(...)
qqss = QQSSModule(...)

z = qqss.step(shock=1.0)
G_eff = base_G + qqss.to_G_eff(z)

Y, r = islm.solve_equilibrium(G_eff=G_eff, ...)
```

---
<img width="2662" height="1764" alt="islm_equilibrium" src="https://github.com/user-attachments/assets/dde46ce9-c352-4564-b5be-0f6307371900" />



## License
MIT License (recommended for open-source academic use).

## Citation (Draft)

If you use this dataset or pipeline, please cite:

**Wu C.-H. (2025). _BRCA-Multi: A Mini Multi-Omics Benchmark Dataset for Methodological Prototyping._**  
https://orcid.org/0009-0001-3396-6835  

 DOI: https://doi.org/10.5281/zenodo.17845433

---

## License

Open Data Commons (ODC-By) license for dataset.  
MIT license for software code.

---

## Maintainer

Chi-Hsing Wu  
TaiScience Research Group  
https://www.taiscience.org

