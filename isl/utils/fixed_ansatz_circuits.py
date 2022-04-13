"""Contains variational circuit ansatz such as hardware efficient ansatz"""
from qiskit import QuantumCircuit, QuantumRegister

import isl.utils.circuit_operations as co
import isl.utils.constants as vconstants


def hardware_efficient_circuit(
    num_qubits,
    ansatz_kind,
    ansatz_depth,
    entangling_gate="cx",
    coupling_map=None,
    gates_to_fix=None,
    gates_to_remove=None,
):
    """
    Create a hardware efficient ansatz circuit. Each depth has a layers of
    rotation gates followed by entangling gates.
    The indexes are relative to the order in which the rotation gates are
    added.
    In example circuit, index of gate is shown after the gate name.
    Example circuit (num_qubits:3, ansatz_kind: 'rxry', ansatz_depth:2,
    linear entangling):
        ┌───────┐┌───────┐     ┌───────┐┌───────┐
        ┤ Rx(0) ├┤ Ry(1) ├──■──┤ Rx(6) ├┤ Ry(7) ├───────────■───────────
        ├───────┤├───────┤┌─┴─┐└───────┘├───────┤┌───────┐┌─┴─┐
        ┤ Rx(2) ├┤ Ry(3) ├┤ X ├────■────┤ Rx(8) ├┤ Ry(9) ├┤ X ├────■────
        ├───────┤├───────┤└───┘  ┌─┴─┐  ├───────┤├───────┤└───┘  ┌─┴─┐
        ┤ Rx(4) ├┤ Ry(5) ├───────┤ X ├──┤ Rx(10)├┤ Ry(11)├───────┤ X ├──
        └───────┘└───────┘       └───┘  └───────┘└───────┘       └───┘
    :param num_qubits: The number of qubits in circuit
    :param ansatz_kind: String name of rotation gates (e.g. 'ry', 'rxry',
    'rzryrz')
    :param ansatz_depth: Number of layers of (rotation gates+entangling gates)
    :param entangling_gate: Entangling gate to use ('cx' or 'cz')
    :param coupling_map:  Map of entangling gates of the form [(control,
    target)]. Gates are added sequentially.
    If None, linear coupling layout is used
    :param gates_to_fix: The indexes and angles of the gates which are to be
    fixed (FIXED_GATE_LABEL is added to gate)
    Must be of form {index:angle}
    :param gates_to_remove: Indexes of gates which are to be removed
    :return: QuantumCircuit
    """
    qr = QuantumRegister(num_qubits)
    qc = QuantumCircuit(qr)

    if coupling_map is None:
        coupling_map = vconstants.coupling_map_linear(num_qubits)
    if gates_to_remove is None:
        gates_to_remove = []
    if gates_to_fix is None:
        gates_to_fix = {}

    index = 0
    for _ in range(ansatz_depth):

        # Add rotation gates
        for qubit in range(num_qubits):
            for gate_name in [
                ansatz_kind[i : i + 2] for i in range(0, len(ansatz_kind), 2)
            ]:
                gate = co.create_1q_gate(gate_name, 0)
                if index in gates_to_fix:
                    gate.label = vconstants.FIXED_GATE_LABEL
                    gate.params[0] = gates_to_fix[index]
                if index not in gates_to_remove:
                    qc.append(gate, [qr[qubit]])
                index += 1

        for control, target in coupling_map:
            qc.append(co.create_2q_gate(entangling_gate), [qr[control], qr[target]])

    return qc


def number_preserving_ansatz(num_qubits, ansatz_depth):
    coupling_map = vconstants.coupling_map_ladder(num_qubits)

    qc = QuantumCircuit(num_qubits)
    index = 0
    for layer in range(ansatz_depth):
        for control, target in coupling_map:
            rz_gate = co.create_independent_parameterised_gate(
                "rz", f"theta_" f"{index}"
            )
            minus_rz_gate = co.create_dependent_parameterised_gate(
                "rz", f"-theta_" f"{index}"
            )
            ry_gate = co.create_independent_parameterised_gate("ry", f"phi_{index}")
            minus_ry_gate = co.create_dependent_parameterised_gate(
                "ry", f"-phi_" f"{index}"
            )

            qc.cx(control, target)
            co.add_gate(qc, minus_rz_gate.copy(), qubit_indexes=[control])
            co.add_gate(qc, minus_ry_gate.copy(), qubit_indexes=[control])
            qc.cx(target, control)
            co.add_gate(qc, ry_gate.copy(), qubit_indexes=[control])
            co.add_gate(qc, rz_gate.copy(), qubit_indexes=[control])
            qc.cx(control, target)
            index += 1
    return qc


def custom_ansatz(num_qubits, two_qubit_circuit, ansatz_depth, coupling_map=None):
    if coupling_map is None:
        coupling_map = vconstants.coupling_map_ladder(num_qubits)

    qc = QuantumCircuit(num_qubits)
    for layer in range(ansatz_depth):
        for control, target in coupling_map:
            co.add_to_circuit(
                qc, two_qubit_circuit.copy(), qubit_subset=[control, target]
            )
    return qc
