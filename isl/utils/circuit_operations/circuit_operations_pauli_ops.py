import numpy as np
from openfermion import QubitOperator
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.extensions import U1Gate, U3Gate
from qiskit.quantum_info import Pauli

from isl.utils.circuit_operations.circuit_operations_full_circuit import (
    add_classical_operations,
    add_to_circuit,
    remove_classical_operations,
    remove_inner_circuit,
)
from isl.utils.circuit_operations.circuit_operations_running import (
    run_circuit_without_transpilation,
)
from isl.utils.utilityfunctions import (
    expectation_value_of_pauli_observable,
    is_statevector_backend,
)


def add_pauli_operators_to_circuit(
    circuit: QuantumCircuit, pauli: Pauli, location=None
):
    if location is None:
        location = len(circuit.data)
    original_circuit_length = len(circuit.data)
    # Add rotation gates
    pauli_circuit = QuantumCircuit(circuit.num_qubits)
    for i, pauli_axis in enumerate(reversed(pauli.to_label())):
        if pauli_axis == "I":
            continue
        elif pauli_axis == "X":
            U3Gate(np.pi, -0.5 * np.pi, 0.5 * np.pi)
        elif pauli_axis == "Y":
            U3Gate(np.pi, 0, 0)
        elif pauli_axis == "Z":
            U1Gate(np.pi)
        else:
            raise ValueError(f"Unexpected pauli axis {pauli_axis}")

    # Add post rotation gates (copied from pauli_measurement in
    # qiskit.aqua.operators.common)
    for qubit_idx in range(circuit.num_qubits):
        if pauli.x[qubit_idx]:
            if pauli.z[qubit_idx]:
                # Measure Y
                pauli_circuit.u1(-np.pi / 2, qubit_idx)  # sdg
                pauli_circuit.u3(np.pi / 2, 0.0, np.pi, qubit_idx)  # h
            else:
                # Measure X
                pauli_circuit.u3(np.pi / 2, 0.0, np.pi, qubit_idx)  # h
    add_to_circuit(
        circuit, pauli_circuit, location=location, transpile_before_adding=False
    )
    pauli_circuit_len = len(circuit.data) - original_circuit_length
    pauli_operators_gate_range = (location, location + pauli_circuit_len)
    return pauli_operators_gate_range


def expectation_value_of_pauli_operator(
    circuit: QuantumCircuit,
    operator: dict,
    backend,
    backend_options=None,
    execute_kwargs=None,
):
    expectation_value = 0
    cl_ops_data = remove_classical_operations(circuit)
    creg = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(creg)
    for pauli_lbl in operator.keys():
        if pauli_lbl == "I" * len(pauli_lbl):
            expectation_value += operator[pauli_lbl] * 1
            continue
        pauli_obj = Pauli.from_label(pauli_lbl)
        pauli_circuit_gate_range = add_pauli_operators_to_circuit(circuit, pauli_obj)
        if not is_statevector_backend(backend):
            [
                circuit.measure(circuit.qregs[0][x], creg[x])
                for x in range(circuit.num_qubits)
            ]
        counts = run_circuit_without_transpilation(
            circuit, backend, backend_options, execute_kwargs
        )
        remove_classical_operations(circuit)
        eval_po = expectation_value_of_pauli_observable(counts, pauli_obj)
        expectation_value += operator[pauli_lbl] * eval_po

        remove_inner_circuit(circuit, pauli_circuit_gate_range)
    circuit.cregs.remove(creg)
    for clbit in creg:
        if clbit in circuit.clbits:
            circuit.clbits.remove(clbit)
    add_classical_operations(circuit, cl_ops_data)
    return expectation_value


def convert_qubit_op_to_pauli_dict(qubit_op: QubitOperator):
    paulis = []
    base_pauli = ["I"]
    for action_pairs, coeff in qubit_op.terms.items():
        if not np.isreal(coeff):
            raise ValueError("Complex coefficients unsupported")
        else:
            coeff = np.real(coeff)
        this_pauli = list(base_pauli)
        for qubit_index, pauli_op in action_pairs:
            if qubit_index >= len(base_pauli):
                # Add extra ops to all pauli strings
                diff = (qubit_index + 1) - len(base_pauli)
                base_pauli += ["I"] * diff
                this_pauli += ["I"] * diff
                for key in [x[0] for x in paulis]:
                    key += ["I"] * diff
            this_pauli[qubit_index] = pauli_op
        paulis.append((this_pauli, coeff))

    pauli_dict = {"".join(pauli_list[::-1]): coeff for (pauli_list, coeff) in paulis}
    return pauli_dict
