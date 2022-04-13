import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer.noise import NoiseModel

QULACS_BASIS_GATES = ["cx", "rx", "ry", "rz", "u1", "u2", "u3", "h", "x"]


def run_on_qulacs_noiseless(circuit: QuantumCircuit, transpile_circuit=False):
    from qulacs import QuantumCircuit as QQC
    from qulacs import QuantumState

    circuit_to_use = circuit
    if transpile_circuit:
        circuit_to_use = transpile(
            circuit, basis_gates=["rx", "ry", "rz", "cx"], optimization_level=0
        )

    num_qubits = circuit.num_qubits
    state = QuantumState(num_qubits)

    circuit_qulacs = QQC(num_qubits)

    for gate, qargs, cargs in circuit_to_use.data:
        if gate.name == "rx":
            circuit_qulacs.add_U3_gate(
                qargs[0].index, gate.params[0], -0.5 * np.pi, 0.5 * np.pi
            )
        elif gate.name == "ry":
            circuit_qulacs.add_U3_gate(qargs[0].index, gate.params[0], 0, 0)
        elif gate.name == "rz":
            circuit_qulacs.add_RZ_gate(qargs[0].index, -1 * gate.params[0])
        elif gate.name == "u1":
            circuit_qulacs.add_U1_gate(qargs[0].index, gate.params[0])
        elif gate.name == "u2":
            circuit_qulacs.add_U2_gate(qargs[0].index, gate.params[0], gate.params[1])
        elif gate.name == "u3":
            circuit_qulacs.add_U3_gate(
                qargs[0].index, gate.params[0], gate.params[1], gate.params[2]
            )
        elif gate.name == "h":
            circuit_qulacs.add_H_gate(qargs[0].index)
        elif gate.name == "x":
            circuit_qulacs.add_X_gate(qargs[0].index)
        elif gate.name == "y":
            circuit_qulacs.add_Y_gate(qargs[0].index)
        elif gate.name == "z":
            circuit_qulacs.add_Z_gate(qargs[0].index)
        elif gate.name == "cx":
            circuit_qulacs.add_CNOT_gate(qargs[0].index, qargs[1].index)
        elif gate.name == "cz":
            circuit_qulacs.add_CZ_gate(qargs[0].index, qargs[1].index)
        else:
            raise ValueError(f"Qulacs converter does not support {gate.name} gate")

    circuit_qulacs.update_quantum_state(state)
    return state.get_vector()
