
# Incremental Structure Learning (ISL)

An open-source implementation of ISL [1], a circuit recompilation algorithm that finds an approximate representation of
any circuit acting on the |0>|0>...|0> state. Created for the IBM Quantum Awards: Open Sciece Prize 2021. More details of ISL,
alongside it's use in the Quantum Awards can be found [here](https://ibmquantumawards.bemyapp.com/#/projects).

[1] B Jaderberg, A Agarwal, K Leonhardt, M Kiffner, D Jaksch, 2020 Quantum Sci. Technol. 5 034015

## Installing ISL

The best way of installing ISL is through `pip`:

```
pip install quantum-isl
```

## Using ISL

### Minimal example
A circuit can be recompiled and the result accessed with only 3 lines if using the 
default settings.
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

### Specifying additional configuration

The default settings can be changed by specifying arguments when
building `ISLRecompiler()`. Many of the configuration options are bundled into the 
`ISLConfig` class.

```python
from isl.recompilers import ISLRecompiler, ISLConfig
from qiskit.circuit.random import random_circuit

qc = random_circuit(5, 5, seed=2)

# Recompile
config = ISLConfig(sufficient_cost=1e-3, max_2q_gates=25)
recompiler = ISLRecompiler(qc, entanglement_measure='EM_TOMOGRAPHY_CONCURRENCE', isl_config=config)
result = recompiler.recompile()
recompiled_circuit = result['circuit']

# See the original circuit
print(qc)

# See the recompiled solution
print(recompiled_circuit)
```

Here we have specified a number of things
* `sufficient_cost=1e-3`: The state produced by the recompiled solution will have an overlap of at least 99.9% with respect to the state produced by the original circuit.
* `max_2q_gates=25`: If our solution contains more than 25 CNOT gates, return early. Setting this to the number of 2-qubit gates in the original circuit provides a useful upper limit.
* `entanglement_measure`: This argument on the recompiler itself specifies the type of entanglement measure used when deciding which qubits to add the next layer to.

More configuration options can be explored in the documentation of `ISLConfig` and `ISLRecompiler`.

### Comparing quantum resources
Taking the above example, lets compare the number of gates and circuit depth before and after recompilation.
```python
from qiskit import transpile

# Transpile the original circuits to the common basis set
qc_in_basis_gates = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
print(qc_in_basis_gates.count_ops())
print(qc_in_basis_gates.depth())

# Compare with recompiled circuit
print(recompiled_circuit.count_ops())
print(recompiled_circuit.depth())
```
In the above example, the original circuit contains 25 CNOT gates and 
32 single-qubit gates with a depth of 33. By comparison, the recompiled solution
prepares the same state to 99.9% overlap with on average 6 CNOT gates and
8 two-qubit gates with a depth of 9 (average tested over 10 runs).

### Citing usage

We respectfully ask any publication, project or whitepaper using ISL to cite the original literature:

B Jaderberg, A Agarwal, K Leonhardt, M Kiffner, D Jaksch, 2020 Quantum Sci. Technol. 5 034015.
https://doi.org/10.1088/2058-9565/ab972b

