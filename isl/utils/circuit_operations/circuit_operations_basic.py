import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.extensions import CXGate, CZGate, U1Gate, U3Gate
from sympy import parse_expr


def create_1q_gate(gate_name, angle):
    """
    Create 1 qubit rotation gate with given name and angle
    :param gate_name: Name of rotation gate ('rx','ry','rz')
    :param angle: Angle of rotation
    :return: New gate
    """
    if gate_name == "rx":
        return U3Gate(angle, -0.5 * np.pi, 0.5 * np.pi, label="rx")
    elif gate_name == "ry":
        return U3Gate(angle, 0, 0, label="ry")
    elif gate_name == "rz":
        return U1Gate(angle, label="rz")
    else:
        raise ValueError(f"Unsupported gate {gate_name}")


def create_2q_gate(gate_name):
    """
    Create 2 qubit gate with given name
    :param gate_name: Name of rotation gate ('cx','cz')
    :return: New gate
    """
    if gate_name == "cx":
        return CXGate()
    elif gate_name == "cz":
        return CZGate()
    else:
        raise ValueError("Unsupported gate")


def add_gate(
    circuit: QuantumCircuit,
    gate,
    gate_index=None,
    qubit_indexes=None,
    clbit_indexes=None,
):
    if gate_index is None:
        gate_index = len(circuit.data)
    qubits = (
        [circuit.qubits[i] for i in qubit_indexes] if qubit_indexes is not None else []
    )
    clbits = (
        [circuit.clbits[i] for i in clbit_indexes] if clbit_indexes is not None else []
    )
    circuit.data.insert(gate_index, (gate, qubits, clbits))


def replace_1q_gate(circuit, gate_index, gate_name, angle):
    """
    Replace the gate at the specified index of circuit
    :param circuit: QuantumCircuit
    :param gate_index: The index of the gate that is to be replaced
    :param gate_name: New gate name
    :param angle: New gate angle
    """
    if gate_name is None:
        return
    old_gate, qargs, cargs = circuit.data[gate_index]
    if "#" in gate_name:
        circuit.data[gate_index] = (
            create_independent_parameterised_gate(*gate_name.split("#"), angle),
            qargs,
            cargs,
        )
        reevaluate_dependent_parameterised_gates(
            circuit, calculate_independent_variable_values(circuit)
        )
    elif "@" in gate_name:
        raise ValueError("Cant replace dependent parameterised gate")
    else:
        circuit.data[gate_index] = (create_1q_gate(gate_name, angle), qargs, cargs)


def replace_2q_gate(circuit, gate_index, control, target, gate_name="cx"):
    """
    Replace the gate at the specified index of circuit
    :param circuit: QuantumCircuit
    :param gate_index: The index of the gate that is to be replaced
    :param control: New gate control qubit
    :param target: New gate target qubit
    :param gate_name: New gate name
    """
    old_gate, old_qargs, cargs = circuit.data[gate_index]
    qr = old_qargs[0].register
    new_qargs = [qr[control], qr[target]]
    new_gate = create_2q_gate(gate_name)
    circuit.data[gate_index] = (new_gate, new_qargs, cargs)


def is_supported_1q_gate(gate):
    if not isinstance(gate, Gate):
        return False
    gate_name = gate.label if gate.label is not None else gate.name

    if "@" in gate_name:
        return False
    if "#" in gate_name:
        gate_name = gate_name.split("#")[0]
    return gate_name in SUPPORTED_1Q_GATES


def add_dressed_cnot(
    circuit: QuantumCircuit,
    control,
    target,
    thinly_dressed=False,
    gate_index=None,
    v1=True,
    v2=True,
    v3=True,
    v4=True,
):
    """
    Add a dressed cnot gate (cx surrounded by 4 general-rotation(rzryrz
    decomposition) gates)
    :param circuit: QuantumCircuit
    :param control: Control qubit
    :param target: Target qubit
    :param thinly_dressed: Whether only a single rz  gate should be added
    instead of the 3 gate rzryrz decomposition
    :param gate_index: The location of the dressed CNOT gate in circuit.data
    (gates are added to the end if None)
    :param v1: Whether there should be rotation gates before control qubit
    :param v2: Whether there should be rotation gates before target qubit
    :param v3: Whether there should be rotation gates after control qubit
    :param v4: Whether there should be rotation gates after target qubit
    """
    if gate_index is None:
        gate_index = len(circuit.data)

    rz_gate = create_1q_gate("rz", 0)
    ry_gate = create_1q_gate("ry", 0)
    cx_gate = create_2q_gate("cx")

    def add_appropriate_gates(qubit, loc):
        add_gate(circuit, rz_gate.copy(), loc, [qubit])
        loc += 1
        if not thinly_dressed:
            add_gate(circuit, ry_gate.copy(), loc, [qubit])
            loc += 1
            add_gate(circuit, rz_gate.copy(), loc, [qubit])
            loc += 1
        return loc

    if v1:
        gate_index = add_appropriate_gates(control, gate_index)
    if v2:
        gate_index = add_appropriate_gates(target, gate_index)

    add_gate(circuit, cx_gate.copy(), gate_index, [control, target])
    gate_index += 1
    if v3:
        gate_index = add_appropriate_gates(control, gate_index)
    if v4:
        add_appropriate_gates(target, gate_index)


def random_1q_gate():
    """
    Create rotation gate with random angle and axis randomly chosen from x,y,z
    :return: New gate
    """
    return create_1q_gate(
        random.choice(SUPPORTED_1Q_GATES), random.uniform(-np.pi, np.pi)
    )


SUPPORTED_1Q_GATES = ["rx", "ry", "rz"]
SUPPORTED_2Q_GATES = ["cx", "cz"]
BASIS_GATES = ["u3", "cx", "cz", "rx", "ry", "rz", "x", "y", "z", "XY", "ZZ", "h"]
DEFAULT_GATES = ["u1", "u2", "u3", "cx", "id", "measure", "reset"]


def create_independent_parameterised_gate(gate_type, variable_name, angle=0):
    gate = create_1q_gate(gate_type, angle)
    gate.label = f"{gate.label}#{variable_name}"
    return gate


def create_dependent_parameterised_gate(gate_type, equation_string, angle=0):
    gate = create_1q_gate(gate_type, angle)
    gate.label = f"{gate.label}@{equation_string}"
    return gate


def calculate_independent_variable_values(circuit: QuantumCircuit):
    variable_dict = {}
    for (gate, _, _) in circuit.data:
        if gate.label is not None and "#" in gate.label:
            variable_name = gate.label.split("#")[1]
            variable_value = gate.params[0]
            variable_dict[variable_name] = variable_value
    return variable_dict


def reevaluate_dependent_parameterised_gates(
    circuit: QuantumCircuit, independent_variable_values
):
    for (gate, _, _) in circuit.data:
        if gate.label is not None and "@" in gate.label:
            equation = gate.label.split("@")[1]
            result = parse_expr(equation, independent_variable_values)
            angle = np.float(result)
            gate.params[0] = angle


def add_subscript_to_all_variables(circuit: QuantumCircuit, subscript_value):
    substitution_dict = {}
    for (gate, _, _) in circuit.data:
        if gate.label is not None and "#" in gate.label:
            gate_type, variable_name = gate.label.split("#")
            gate.label = f"{gate_type}#{variable_name}_{subscript_value}"

            substitution_dict[variable_name] = f"{variable_name}_{subscript_value}"

    for (gate, _, _) in circuit.data:
        if gate.label is not None and "@" in gate.label:
            gate_type, equation = gate.label.split("@")
            for old_name, new_name in substitution_dict.items():
                equation = equation.replace(old_name, new_name)
            gate.label = f"{gate_type}@{equation}"
