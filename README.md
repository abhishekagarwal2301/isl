# Variational Quantum Algorithms

## Features
- Approximate circuit recompilation
- VQE

## Approximate circuit recompilation
Ref. arXiv:1807.00800v5

Recompilation features:
- Recompile a particular state (circuit with a fixed initial state)
- Recompile a general circuit (circuit with an arbitrary initial state)
- Recompile onto a particular coupling map
- Recompile onto a custom gate set
- Local, and non-local cost function (C_{LHST},C_{HST} in arXiv:1807.00800v5)

### Using ISL
```python
from variationalalgorithms.recompilers import ISLRecompiler
from qiskit import QuantumCircuit

# Setup the circuit
qc = QuantumCircuit(3)
qc.rx(1.23,0)
qc.cx(0,1)
qc.ry(2.5,1)
qc.rx(-1.6,2)
qc.ccx(2,1,0)

# Recompile
recompiler = ISLRecompiler(qc)
result = recompiler.recompile()
recompiled_circuit = result['circuit']

# See the recompiled output
print(recompiled_circuit)
```

### Citing usage

We respectfully ask any publication, project or whitepaper using ISL to cite the original literature:

B Jaderberg, A Agarwal, K Leonhardt, M Kiffner, D Jaksch, 2020 Quantum Sci. Technol. 5 034015.
https://doi.org/10.1088/2058-9565/ab972b

