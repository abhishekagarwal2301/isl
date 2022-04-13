import multiprocessing
import random
from random import seed
from typing import Union

import numpy as np
from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit import transpile as qiskit_transpile
from qiskit.circuit import Clbit, Gate, Instruction, Qubit, Reset
from qiskit.quantum_info import random_statevector

from isl.utils.circuit_operations import (
    BASIS_GATES,
    DEFAULT_GATES,
    SUPPORTED_1Q_GATES,
    SUPPORTED_2Q_GATES,
)
from isl.utils.circuit_operations.circuit_operations_alternate_emulators import (
    QULACS_BASIS_GATES,
)
from isl.utils.circuit_operations.circuit_operations_basic import (
    add_gate,
    create_1q_gate,
    create_2q_gate,
)


def find_register(circuit, bit):
    for reg in circuit.qregs + circuit.cregs:
        if bit in reg:
            return reg
    return None


def find_bit_index(reg, bit):
    for i, reg_bit in enumerate(reg):
        if bit == reg_bit:
            return i
    return None


def create_random_circuit(
    num_qubits, depth=5, one_qubit_gates=None, two_qubit_gates=None
):
    qc = QuantumCircuit(num_qubits)
    if one_qubit_gates is None:
        one_qubit_gates = SUPPORTED_1Q_GATES
    if two_qubit_gates is None:
        two_qubit_gates = SUPPORTED_2Q_GATES
    rs = np.random.RandomState(multiprocessing.current_process().pid)
    while qc.depth() < depth:
        random_gate = rs.choice(one_qubit_gates + two_qubit_gates)
        if random_gate in one_qubit_gates:
            qubits = rs.choice(list(range(num_qubits)), [1])
            add_gate(
                qc,
                create_1q_gate(random_gate, random.uniform(-np.pi, np.pi)),
                qubit_indexes=qubits,
            )
        elif random_gate in two_qubit_gates:
            qubits = rs.choice(list(range(num_qubits)), [2], replace=False)
            add_gate(qc, create_2q_gate(random_gate), qubit_indexes=qubits)
    return qc


def are_circuits_identical(
    qc1: QuantumCircuit, qc2: QuantumCircuit, match_labels=False, match_registers=False
):
    if len(qc1.data) != len(qc2.data):
        return False
    for (gate1, qargs1, cargs1), (gate2, qargs2, cargs2) in zip(qc1.data, qc2.data):
        # Checks that gates match
        gate1_name = gate1.label if gate1.label is not None else gate1.name
        gate2_name = gate2.label if gate2.label is not None else gate2.name

        if gate1_name != gate2_name:
            return False

        if len(gate1.params) != len(gate2.params) and gate1_name in ["rx", "ry", "rz"]:
            gate1_params = [gate1.params[0]]
            gate2_params = [gate2.params[0]]
        else:
            gate1_params = gate1.params
            gate2_params = gate2.params

        if gate1_params != gate2_params:
            return False

        if match_labels and gate1.label != gate2.label:
            return False

        # Check that qargs match
        for qubit1, qubit2 in zip(qargs1, qargs2):
            if match_registers and qubit1 != qubit2:
                return False
            if qubit1.index != qubit2.index:
                return False

        # Check that cargs match
        for clbit1, clbit2 in zip(cargs1, cargs2):
            if match_registers and clbit1 != clbit2:
                return False
            if clbit1.index != clbit2.index:
                return False
    return True


def change_circuit_register(
    circuit: QuantumCircuit,
    new_circuit_reg: Union[QuantumRegister, ClassicalRegister],
    bit_mapping=None,
):
    """
    Only supports 1 quantum/classical register circuits
    :param circuit:
    :param new_circuit_reg:
    :param bit_mapping:
    """
    change_quantum = isinstance(new_circuit_reg, QuantumRegister)
    if change_quantum:
        old_reg = circuit.qregs[0]
        # If qregs used in multiple circuits (e.g. if circuit is copied)
        # then don't affect other circuits
        circuit.qregs = circuit.qregs.copy()
        num_bits = circuit.num_qubits
    else:
        old_reg = circuit.cregs[0]
        # If cregs used in multiple circuits (e.g. if circuit is copied)
        # then don't affect other circuits
        circuit.cregs = circuit.cregs.copy()
        num_bits = circuit.num_clbits
    bit_mapping = {} if bit_mapping is None else bit_mapping
    for bit in range(num_bits):
        if bit not in bit_mapping:
            bit_mapping[bit] = bit

    # Add new register to circuit if necessary
    if new_circuit_reg not in circuit.qregs + circuit.cregs:
        if change_quantum:
            circuit.qregs = []
            circuit.add_register(new_circuit_reg)
        else:
            circuit.cregs = []
            circuit.add_register(new_circuit_reg)

    for index, (gate, qargs, cargs) in enumerate(circuit.data):
        if change_quantum:
            new_qargs = [
                Qubit(new_circuit_reg, bit_mapping[find_bit_index(old_reg, qubit)])
                for qubit in qargs
            ]
            circuit.data[index] = (gate, new_qargs, cargs)
        else:
            new_cargs = [
                Clbit(new_circuit_reg, bit_mapping[find_bit_index(old_reg, clbit)])
                for clbit in cargs
            ]
            circuit.data[index] = (gate, qargs, new_cargs)


def add_to_circuit(
    original_circuit: QuantumCircuit,
    circuit_to_be_added: QuantumCircuit,
    location=None,
    transpile_before_adding=False,
    transpile_kwargs=None,
    qubit_subset=None,
    clbit_subset=None,
):
    """
    Only supports 1 quantum register circuits
    :param original_circuit:
    :param circuit_to_be_added:
    :param location:
    :param transpile_before_adding:
    :param transpile_kwargs:
    :param qubit_subset:
    :param clbit_subset:
    """
    circuit_to_be_added_copy = circuit_to_be_added.copy()
    if location is None:
        location = len(original_circuit.data)
    if transpile_before_adding:
        circuit_to_be_added_copy = unroll_to_basis_gates(
            circuit_to_be_added_copy, DEFAULT_GATES
        )
        if transpile_kwargs is not None:
            circuit_to_be_added_copy = transpile(
                circuit_to_be_added_copy, **transpile_kwargs
            )
    qubit_mapping = None
    if qubit_subset is not None:
        qubit_mapping = (
            {index: value for index, value in enumerate(qubit_subset)}
            if isinstance(qubit_subset, list)
            else qubit_subset
        )

    clbit_mapping = None
    if clbit_subset is not None:
        clbit_mapping = {index: value for index, value in enumerate(clbit_subset)}

    # Change quantum register
    change_circuit_register(
        circuit_to_be_added_copy,
        find_register(original_circuit, original_circuit.qubits[0]),
        qubit_mapping,
    )

    # Change classical register if present
    if len(circuit_to_be_added_copy.clbits) > 0 and len(original_circuit.clbits) > 0:
        change_circuit_register(
            circuit_to_be_added_copy,
            find_register(original_circuit, original_circuit.clbits[0]),
            clbit_mapping,
        )

    for gate in circuit_to_be_added_copy:
        original_circuit.data.insert(location, gate)
        location += 1


def remove_inner_circuit(circuit: QuantumCircuit, gate_range_to_remove):
    for index in list(range(*gate_range_to_remove))[::-1]:
        del circuit.data[index]


def extract_inner_circuit(circuit: QuantumCircuit, gate_range):
    inner_circuit = QuantumCircuit()
    [inner_circuit.add_register(qreg) for qreg in circuit.qregs]
    [inner_circuit.add_register(creg) for creg in circuit.cregs]
    for gate_index in range(*gate_range):
        gate, qargs, cargs = circuit.data[gate_index]
        inner_circuit.data.append((gate, qargs, cargs))
    return inner_circuit


def replace_inner_circuit(
    circuit: QuantumCircuit,
    inner_circuit_replacement,
    gate_range,
    transpile_before_adding=False,
    transpile_kwargs=None,
):
    remove_inner_circuit(circuit, gate_range)
    if (
        inner_circuit_replacement is not None
        and len(inner_circuit_replacement.data) > 0
    ):
        add_to_circuit(
            circuit,
            inner_circuit_replacement,
            gate_range[0],
            transpile_before_adding=transpile_before_adding,
            transpile_kwargs=transpile_kwargs,
        )


def find_num_gates(
    circuit, transpile_before_counting=False, transpile_kwargs=None, gate_range=None
):
    """
    Find the number of 2 qubit and 1 qubit (non classical) gates in circuit
    :param circuit: QuantumCircuit
    :param transpile_before_counting: Whether circuit should be transpiled
    before counting
    :param transpile_kwargs: transpile kwargs (e.g {'backend':backend})
    :param gate_range: The range of gates to include in search space (full
    circuit if None)
    :return: (num_2q_gates, num_1q_gates)
    """
    if circuit is None:
        return 0, 0
    if transpile_before_counting:
        if transpile_kwargs is None:
            circuit = unroll_to_basis_gates(circuit)
        else:
            circuit = transpile(circuit, **transpile_kwargs)
    if gate_range is None:
        gate_range = (0, len(circuit.data))
    num_2q_gates = 0
    num_1q_gates = 0
    for gate_index in range(*gate_range):
        if (
            len(circuit.data[gate_index][1]) == 1
            and len(circuit.data[gate_index][2]) == 0
        ):
            num_1q_gates += 1
        elif (
            len(circuit.data[gate_index][1]) == 2
            and len(circuit.data[gate_index][2]) == 0
        ):
            num_2q_gates += 1
    return num_2q_gates, num_1q_gates


def transpile(circuit, **transpile_kwargs):
    if transpile_kwargs is None:
        transpile_kwargs = {}

    if "backend" in transpile_kwargs and transpile_kwargs["backend"] == "qulacs":
        backend_removed_kwargs = dict(transpile_kwargs)
        del backend_removed_kwargs["backend"]
        # if 'basis_gates' not in backend_removed_kwargs:
        backend_removed_kwargs["basis_gates"] = QULACS_BASIS_GATES
        return qiskit_transpile(circuit, **backend_removed_kwargs)
    return qiskit_transpile(circuit, **transpile_kwargs)


def unroll_to_basis_gates(circuit, basis_gates=None):
    """
    Create circuit by unrolling given circuit to basis_gates
    :param circuit: Circuit to unroll
    :param basis_gates: Basis gate set to unroll to (BASIS_GATES by default)
    :return: Transpiled circuit
    """
    basis_gates = basis_gates if basis_gates is not None else BASIS_GATES
    return transpile(circuit, basis_gates=basis_gates, optimization_level=0)


def append_to_instruction(main_ins, ins_to_append):
    qc = QuantumCircuit(main_ins.num_qubits)
    if main_ins.definition is not None and len(main_ins.definition) > 0:
        qc.append(main_ins, qc.qubits)
    if ins_to_append.definition is not None and len(ins_to_append.definition) > 0:
        qc.append(ins_to_append, qc.qubits)
    return qc.to_instruction()


def remove_classical_operations(circuit: QuantumCircuit):
    gates_and_locations = []
    for index, (gate, qargs, cargs) in list(enumerate(circuit.data))[::-1]:
        if len(cargs) > 0:
            gates_and_locations.append((index, (gate, qargs, cargs)))
            del circuit.data[index]
    return gates_and_locations[::-1]


def add_classical_operations(circuit: QuantumCircuit, gates_and_locations):
    for index, (gate, qargs, cargs) in gates_and_locations:
        circuit.data.insert(index, (gate, qargs, cargs))


def make_quantum_only_circuit(circuit: QuantumCircuit):
    new_qc = QuantumCircuit(*circuit.qregs)
    no_classical_circuit = circuit.copy()
    remove_classical_operations(no_classical_circuit)
    for i in no_classical_circuit.data:
        new_qc.data.append(i)
    # remove_classical_operations(new_qc)
    # new_qc.cregs = []
    # new_qc.clbits = []
    return new_qc


def circuit_by_inverting_circuit(circuit: QuantumCircuit):
    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)

    for (gate, qargs, cargs) in circuit.data[::-1]:
        if not isinstance(gate, Gate):
            new_circuit.data.append((gate, qargs, cargs))
            continue
        if gate.label not in ["rx", "ry", "rz"]:
            inverted_gate = gate.inverse()
        else:
            inverted_gate = gate.copy()
            inverted_gate.params[0] *= -1
        inverted_gate.label = gate.label
        new_circuit.data.append((inverted_gate, qargs, cargs))
    return new_circuit


def initial_state_to_circuit(initial_state):
    """
    Convert to QuantumCircuit
    :param initial_state: Can either be a circuit (
    QuantumCircuit/Instruction) or vector (list/np.ndarray) or None
    :return: QuantumCircuit or None
    """
    if initial_state is None:
        return None
    elif isinstance(initial_state, (list, np.ndarray)):
        num_qubits = int(np.log2(len(initial_state)))
        qc = QuantumCircuit(num_qubits)
        qc.initialize(initial_state, qc.qubits)
        # Unrolling will remove 'reset' gates from circuit
        qc = unroll_to_basis_gates(qc)
        remove_reset_gates(qc)
        return qc
    elif isinstance(initial_state, Instruction):
        num_qubits = initial_state.num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.append(initial_state, qc.qubits)
        return qc
    elif isinstance(initial_state, QuantumCircuit):
        return initial_state.copy()
    else:
        raise TypeError("Invalid type of initial_state provided")


def calculate_overlap_between_circuits(
    circuit1, circuit2, initial_state=None, qubit_subset=None
):
    initial_state_circuit = initial_state_to_circuit(initial_state)
    if initial_state_circuit is None:
        total_num_qubits = circuit1.num_qubits
    else:
        total_num_qubits = initial_state_circuit.num_qubits

    qubit_subset_to_recompile = (
        qubit_subset if qubit_subset else list(range(total_num_qubits))
    )
    qr1 = QuantumRegister(total_num_qubits)
    qr2 = QuantumRegister(total_num_qubits)
    qc1 = QuantumCircuit(qr1)
    qc2 = QuantumCircuit(qr2)

    if initial_state_circuit is not None:
        add_to_circuit(qc1, initial_state_circuit)
        add_to_circuit(qc2, initial_state_circuit)
    qc1.append(circuit1, [qr1[i] for i in qubit_subset_to_recompile])
    qc2.append(circuit2, [qr2[i] for i in qubit_subset_to_recompile])

    sv1 = (
        execute(qc1, Aer.get_backend("statevector_simulator"))
        .result()
        .get_statevector()
    )
    sv2 = (
        execute(qc2, Aer.get_backend("statevector_simulator"))
        .result()
        .get_statevector()
    )
    return np.absolute(np.vdot(sv1, sv2))


def create_random_initial_state_circuit(num_qubits, return_statevector=False):
    rand_state = random_statevector(2**num_qubits, seed()).data
    qc = QuantumCircuit(num_qubits)
    qc.initialize(rand_state, qc.qubits)
    qc = unroll_to_basis_gates(qc)

    # Delete reset gates
    for i in range(len(qc.data) - 1, -1, -1):
        gate, qargs, cargs = qc.data[i]
        if isinstance(gate, Reset):
            del qc.data[i]

    if return_statevector:
        return qc, rand_state
    else:
        return qc


def remove_reset_gates(circuit: QuantumCircuit):
    for i, (gate, qargs, cargs) in list(enumerate(circuit.data))[::-1]:
        if isinstance(gate, Reset):
            del circuit.data[i]
