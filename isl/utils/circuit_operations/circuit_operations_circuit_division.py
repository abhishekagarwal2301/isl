from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, Instruction, Qubit

from isl.utils.circuit_operations.circuit_operations_full_circuit import (
    unroll_to_basis_gates,
)


def find_previous_gate_on_qubit(circuit, gate_index):
    """
    Find the gate just before specified gate that acts on at least 1 same
    qubit as the specified gate
    :param circuit: QuantumCircuit
    :param gate_index: The index of the specified gate
    :return: (previous_gate_object, index) (or (None,None) if no such gate)
    """
    # circuit.data has form [(gate_object, [(register, qubit)], cargs)]
    required_qubits = set(circuit.data[gate_index][1])
    index = gate_index - 1
    while index >= 0:
        gate, qargs, cargs = circuit.data[index]
        # If at least one of the qargs (register, qubit) of the gate is the
        # same as the qargs of the specified circuit
        if len(required_qubits & set(qargs)) > 0:
            return gate, index
        index -= 1
    return None, None


def index_of_bit_in_circuit(bit, circuit):
    """
    Calculate the index of the qubit/clbit in the circuit. Location of clbit
    is number of qubits + relative location of
    clbit
    :param bit: Qubit or Clbit
    :param circuit: QuantumCircuit
    :return:
    """
    if isinstance(bit, Qubit):
        return circuit.qubits.index(bit)
    elif isinstance(bit, Clbit):
        return circuit.clbits.index(bit)
    else:
        raise TypeError(f"{bit} not a Qubit or Clbit")


def calculate_next_gate_indexes(
    current_next_gate_indexes, circuit, gate_qargs, gate_cargs
):
    """
    Updates the position (in bit wire) at which a gate on that bit will
    occur. For multi-bit gates, all the bits
    involved in that gate will have the same index which is 1 + the largest
    position in the list of relevant bits
    :param circuit: QuantumCircuit
    :param current_next_gate_indexes: Current next_gate_indexes
    :param gate_qargs: Qubits the gate acts on
    :param gate_cargs: Clbits the gate acts on
    :return: New next_gate_indexes
    """
    qubit_indexes = [index_of_bit_in_circuit(qubit, circuit) for qubit in gate_qargs]
    clbit_indexes = [
        len(circuit.qubits) + index_of_bit_in_circuit(clbit, circuit)
        for clbit in gate_cargs
    ]

    largest_index = max(
        current_next_gate_indexes[i] for i in qubit_indexes + clbit_indexes
    )

    resulting_next_gate_indexes = list(current_next_gate_indexes)
    for i in qubit_indexes + clbit_indexes:
        resulting_next_gate_indexes[i] = largest_index + 1

    return resulting_next_gate_indexes


def vertically_divide_circuit(circuit, max_depth_per_division=10):
    """
    ----------|----|----|---|---------
    ----- ____|____|____|____|__ -----
    -----|    |    |    |    |  |-----
    -----|____|____|____|____|__|-----
    ----------|----|----|---|---------
    :param circuit: Circuit to divide (QuantumCircuit/Instruction)
    :param max_depth_per_division: Upper limit of depth of each of the
    subcircuits resulting from the division
    :return List of subcircuits [QuantumCircuit]
    """
    if isinstance(circuit, Instruction):
        if circuit.num_clbits > 0:
            remaining_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        else:
            remaining_circuit = QuantumCircuit(circuit.num_qubits)
        remaining_circuit.append(
            circuit, remaining_circuit.qubits, remaining_circuit.clbits
        )
    else:
        remaining_circuit = circuit.copy()

    remaining_circuit = unroll_to_basis_gates(remaining_circuit)
    all_subcircuits = []
    while len(remaining_circuit) > 0:

        subcircuit = QuantumCircuit(*remaining_circuit.qregs, *remaining_circuit.cregs)
        gate_indexes_to_remove = []
        next_gate_indexes = [0] * (
            len(remaining_circuit.qubits) + len(remaining_circuit.clbits)
        )
        for i in range(len(remaining_circuit.data)):
            instr, qargs, cargs = remaining_circuit.data[i]

            next_gate_indexes = calculate_next_gate_indexes(
                next_gate_indexes, remaining_circuit, qargs, cargs
            )

            if max(next_gate_indexes) <= max_depth_per_division:
                subcircuit.append(instr, qargs, cargs)
                gate_indexes_to_remove.append(i)
            elif min(next_gate_indexes) >= max_depth_per_division:
                break

        for j in reversed(gate_indexes_to_remove):
            del remaining_circuit.data[j]

        all_subcircuits.append(subcircuit)

    return all_subcircuits
