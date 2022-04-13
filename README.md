
# Incremental Structure Learning (ISL)

ISL is a circuit recompilation algorithm [1,2] that finds an approximate representation of a circuit when acting on the |0>|0>...|0> state [3].

[1] Jones, Tyson, and Simon C. Benjamin. "Quantum compilation and circuit optimisation via energy dissipation." arXiv preprint arXiv:1811.03147 (2018).

[2] Khatri, Sumeet, et al. "Quantum-assisted quantum compiling." Quantum 3 (2019): 140.

[3] B Jaderberg, A Agarwal, K Leonhardt, M Kiffner, D Jaksch, 2020 Quantum Sci. Technol. 5 034015

### Using ISL
```python
from isl.recompilers import ISLRecompiler
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

