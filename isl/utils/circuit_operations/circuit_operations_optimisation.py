import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import OneQubitEulerDecomposer

from isl.utils.circuit_operations.circuit_operations_basic import (
    is_supported_1q_gate,
    replace_1q_gate,
)
from isl.utils.circuit_operations.circuit_operations_circuit_division import (
    find_previous_gate_on_qubit,
)

MINIMUM_ROTATION_ANGLE = 1e-3


def remove_unnecessary_gates_from_circuit(
    circuit: QuantumCircuit,
    remove_zero_gates=True,
    remove_small_gates=False,
    gate_range=None,
):
    """
    Remove unnecessary gates from circuit by merging adjacent gates of same
    kind, converting 3+ consecutive single
    qubit gates on a single qubit to an rzryrz decomposition, removing
    similar consecutive cx, cz gates.
    :param circuit: Circuit from which gates are to be removed
    :param remove_zero_gates: If true, single qubit gates with 0 angle will
    be removed
    :param remove_small_gates: If true, single qubit gates with angle less
    than MINIMUM_ROTATION_ANGLE will be removed
    :param gate_range: If provided, only gates in that range relative to
    circuit.data will be modified
    """
    if gate_range is None:
        gate_range = [0, len(circuit.data)]
    else:
        gate_range = list(gate_range)

    last_circuit_length = len(circuit.data)
    i = 0
    while True:
        if i == 0:
            remove_unnecessary_1q_gates_from_circuit(
                circuit, remove_zero_gates, remove_small_gates, gate_range
            )
            i = 1
        else:
            remove_unnecessary_2q_gates_from_circuit(circuit, gate_range)
            i = 0
        new_circuit_length = len(circuit.data)
        if new_circuit_length != last_circuit_length:
            # Update the gate range maximum to account for the shortened
            # circuit
            gate_range[1] -= last_circuit_length - new_circuit_length
            last_circuit_length = new_circuit_length
        elif i == 0:
            return


def remove_unnecessary_1q_gates_from_circuit(
    circuit,
    remove_zero_gates=True,
    remove_small_gates=False,
    gate_range=None,
    min_rotation_angle=MINIMUM_ROTATION_ANGLE,
):
    """
    Remove unnecessary 1-qubit gates from circuit by converting 3+
    consecutive single qubit gates on a single qubit
    to an rzryrz decomposition
    :param circuit: Circuit from which gates are to be removed
    :param remove_zero_gates: If true, single qubit gates with 0 angle will
        be removed
    :param remove_small_gates: If true, single qubit gates with angle less
        than MINIMUM_ROTATION_ANGLE will be removed
    :param gate_range: If provided, only gates in that range relative to
        circuit.data will be modified
        (lower index is inclusive and upper index is exclusive)
    :param min_rotation_angle: If remove_small_gates, rotation gates
        with angles smaller than min_rotation_angle will be removed
    """
    if gate_range is None:
        gate_range = (0, len(circuit.data))

    indexes_to_remove = []
    indexes_dealt_with = []

    # Reverse iterate over all gates
    for gate_index in range(gate_range[1] - 1, gate_range[0] - 1, -1):
        gate, _, _ = circuit.data[gate_index]
        if (
            gate_index in indexes_to_remove
            or gate_index in indexes_dealt_with
            or not is_supported_1q_gate(gate)
        ):
            continue
        remove_because_zero = remove_zero_gates and gate.params[0] == 0
        remove_because_small = (
            remove_small_gates and np.absolute(gate.params[0]) < min_rotation_angle
        )
        if remove_because_zero or remove_because_small:
            indexes_to_remove += [gate_index]
            continue

        # Any single qubit operation can be reduced to phase * Rz(phi) * Ry(
        # theta) * Rz(lambda)
        # RXGate, RYGate, RZGate do not implement to_matrix() but their
        # definitions (U3Gate or U1Gate) do
        matrix = circuit.data[gate_index][0].definition[0][0].to_matrix()
        prev_gate_indexes = [gate_index]
        prev_gate, prev_gate_index = find_previous_gate_on_qubit(circuit, gate_index)

        # Get all previous gates on qubit (until end or non rx/rz/rz gate is
        # met)
        while (
            prev_gate is not None
            and is_supported_1q_gate(prev_gate)
            and prev_gate_index >= gate_range[0]
        ):
            # If that gate is small, add it to indexes_to_remove and do not
            # add it in decomposition
            remove_because_zero = remove_zero_gates and prev_gate.params[0] == 0
            remove_because_small = (
                remove_small_gates
                and np.absolute(prev_gate.params[0]) < min_rotation_angle
            )
            if remove_because_zero or remove_because_small:
                indexes_to_remove += [prev_gate_index]
            else:
                prev_gate_indexes += [prev_gate_index]
                prev_gate_matrix = (
                    circuit.data[prev_gate_index][0].definition[0][0].to_matrix()
                )
                matrix = np.matmul(matrix, prev_gate_matrix)
            prev_gate, prev_gate_index = find_previous_gate_on_qubit(
                circuit, prev_gate_index
            )

        if len(prev_gate_indexes) > 3:
            theta, phi, lam = OneQubitEulerDecomposer().angles(matrix)
            replace_1q_gate(circuit, prev_gate_indexes[0], "rz", phi)
            replace_1q_gate(circuit, prev_gate_indexes[1], "ry", theta)
            replace_1q_gate(circuit, prev_gate_indexes[2], "rz", lam)
            # replace_1q_gate(circuit, prev_gate_indexes[3], 'ph', phase)
            indexes_dealt_with += [prev_gate_indexes[1], prev_gate_indexes[2]]
            indexes_to_remove += prev_gate_indexes[3:]
        else:
            indexes_dealt_with += prev_gate_indexes
    for index in sorted(indexes_to_remove, reverse=True):
        del circuit.data[index]


def remove_unnecessary_2q_gates_from_circuit(circuit, gate_range=None):
    """
    Remove unnecessary 2-qubit gates from circuit by removing pairs of
    consecutive CX/CZ gates
    :param circuit: Circuit from which gates are to be removed
    :param gate_range: If provided, only gates in that range relative to
        circuit.data will be modified
        (lower index is inclusive and upper index is exclusive)
    """
    if gate_range is None:
        gate_range = (0, len(circuit.data))

    indexes_to_remove = []
    indexes_dealt_with = []

    # Reverse iterate over all gates
    for gate_index in range(gate_range[1] - 1, gate_range[0] - 1, -1):
        gate, qargs, _ = circuit.data[gate_index]
        if gate.name not in ["cx", "cy", "cz"]:
            continue
        if gate_index in indexes_to_remove or gate_index in indexes_dealt_with:
            continue
        prev_gate, prev_gate_index = find_previous_gate_on_qubit(circuit, gate_index)
        if prev_gate is None or prev_gate.name != gate.name:
            continue
        if prev_gate_index < gate_range[0]:
            continue
        if (
            prev_gate_index in indexes_to_remove
            or prev_gate_index in indexes_dealt_with
        ):
            continue
        if circuit.data[prev_gate_index][1] == qargs:
            indexes_to_remove += [gate_index, prev_gate_index]
    for index in sorted(indexes_to_remove, reverse=True):
        del circuit.data[index]
