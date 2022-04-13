from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


class TestBasicCircuitOperations(TestCase):
    def test_create_1q_gate(self):
        from isl.utils.circuit_operations.circuit_operations_basic import create_1q_gate

        rx_gate = create_1q_gate("rx", 0.5)
        ry_gate = create_1q_gate("ry", -0.5)
        rz_gate = create_1q_gate("rz", 0.23)
        assert (
            rx_gate.name == "u3"
            and rx_gate.params[0] == 0.5
            and rx_gate.params[1] == -np.pi / 2
            and rx_gate.params[2] == np.pi / 2
            and rx_gate.label == "rx"
        )
        assert (
            ry_gate.name == "u3"
            and ry_gate.params[0] == -0.5
            and ry_gate.params[1] == 0
            and ry_gate.params[2] == 0
            and ry_gate.label == "ry"
        )
        assert (
            rz_gate.name == "u1" and rz_gate.params[0] == 0.23 and rz_gate.label == "rz"
        )

    def test_create_2q_gate(self):
        from isl.utils.circuit_operations.circuit_operations_basic import create_2q_gate

        cx_gate = create_2q_gate("cx")
        cz_gate = create_2q_gate("cz")
        assert cx_gate.name == "cx"
        assert cz_gate.name == "cz"

    def test_replace_1q_gate(self):
        from isl.utils.circuit_operations.circuit_operations_basic import (
            replace_1q_gate,
        )

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.3, 1)
        qc.z(2)
        replace_1q_gate(qc, 2, "rz", 1.2)
        assert qc.data[2][0].label == "rz" and qc.data[2][0].params[0] == 1.2

    def test_replace_2q_gate(self):
        from isl.utils.circuit_operations.circuit_operations_basic import (
            replace_2q_gate,
        )

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.5, 1)
        qc.z(2)
        replace_2q_gate(qc, 1, 1, 2, "cz")
        # qc.data has form [(gate,qargs,cargs)] where qargs,cargs have form
        # [Qubit]
        assert (
            qc.data[1][0].name == "cz"
            and qc.data[1][1][0].index == 1
            and qc.data[1][1][1].index == 2
        )

    def test_is_supported_1q_gate(self):
        from qiskit.circuit import Gate

        from isl.utils.circuit_operations.circuit_operations_basic import (
            is_supported_1q_gate,
        )
        from isl.utils.constants import FIXED_GATE_LABEL

        assert is_supported_1q_gate(Gate("rx", 1, [0.5], "rx")) is True
        assert is_supported_1q_gate(Gate("rx", 1, [0.5], FIXED_GATE_LABEL)) is False
        assert is_supported_1q_gate(Gate("cx", 2, [])) is False
        assert is_supported_1q_gate(Gate("ZZ", 1, [0.5])) is False

    def test_add_dressed_cnot(self):
        from isl.utils.circuit_operations.circuit_operations_basic import (
            add_dressed_cnot,
        )
        from isl.utils.circuit_operations.circuit_operations_full_circuit import (
            are_circuits_identical,
        )

        qr = QuantumRegister(3)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.5, 1)
        # Add dressed CNOT here
        qc.h(1)
        qc.z(2)

        expected_qc = QuantumCircuit(qr)
        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.rx(0.5, 1)
        # Before control rzryrz decomposition
        expected_qc.rz(0, 1)
        expected_qc.ry(0, 1)
        expected_qc.rz(0, 1)
        # CNOT
        expected_qc.cx(1, 2)
        # After target rzryrz decomposition
        expected_qc.rz(0, 2)
        expected_qc.ry(0, 2)
        expected_qc.rz(0, 2)
        expected_qc.h(1)
        expected_qc.z(2)

        add_dressed_cnot(qc, 1, 2, gate_index=3, v2=False, v3=False)
        assert are_circuits_identical(qc, expected_qc)

        # Test thinly dressed CNOT
        expected_qc.rz(0, 2)
        expected_qc.rz(0, 0)
        expected_qc.cx(2, 0)
        expected_qc.rz(0, 2)
        expected_qc.rz(0, 0)

        add_dressed_cnot(qc, 2, 0, thinly_dressed=True)
        assert are_circuits_identical(qc, expected_qc)


class TestFullCircuitOperations(TestCase):
    def test_change_circuit_register(self):
        from isl.utils.circuit_operations.circuit_operations_full_circuit import (
            change_circuit_register,
        )

        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(3)
        qc = QuantumCircuit(qr2)
        qc.cx(0, 2)
        qc.h(0)
        qc.cx(2, 1)
        qc.rx(2.3, 1)
        qc.x(0)

        expected_qc = QuantumCircuit(qr1)
        expected_qc.cx(3, 1)
        expected_qc.h(3)
        expected_qc.cx(1, 2)
        expected_qc.rx(2.3, 2)
        expected_qc.x(3)
        qubit_mapping = {0: 3, 1: 2, 2: 1}
        change_circuit_register(qc, qr1, qubit_mapping)

        assert all(
            gate[1] == expected_gate[1]
            for gate, expected_gate in zip(qc.data, expected_qc.data)
        )
        assert qr2 not in qc.qregs  # Make sure old register was removed
        # from circuit

    def test_add_to_circuit(self):
        from isl.utils.circuit_operations.circuit_operations_full_circuit import (
            add_to_circuit,
            calculate_overlap_between_circuits,
        )

        qr = QuantumRegister(3)
        left_circuit = QuantumCircuit(qr)
        left_circuit.h(0)
        left_circuit.cx(0, 1)
        # Add right circuit will be added at this location
        left_circuit.rx(0.23, 0)
        left_circuit.cx(1, 2)
        left_circuit.h(1)

        right_circuit = QuantumCircuit(2)
        right_circuit.rx(-1.5, 1)
        right_circuit.cx(1, 0)
        right_circuit.rz(2.1, 0)
        # The following two gates should cancel each other out when transpiling
        right_circuit.cx(0, 1)
        right_circuit.cx(0, 1)
        right_circuit.cx(0, 1)
        right_circuit.ry(0.2, 0)

        expected_full_circuit = QuantumCircuit(qr)
        expected_full_circuit.h(0)
        expected_full_circuit.cx(0, 1)
        expected_full_circuit.rx(-1.5, 2)
        expected_full_circuit.cx(2, 1)
        expected_full_circuit.rz(2.1, 1)
        expected_full_circuit.cx(1, 2)
        expected_full_circuit.ry(0.2, 1)
        expected_full_circuit.rx(0.23, 0)
        expected_full_circuit.cx(1, 2)
        expected_full_circuit.h(1)

        add_to_circuit(
            left_circuit,
            right_circuit,
            location=2,
            transpile_before_adding=True,
            transpile_kwargs={"optimization_level": 1},
            qubit_subset=[1, 2],
        )
        assert len(expected_full_circuit.data) == len(left_circuit.data)
        assert np.isclose(
            calculate_overlap_between_circuits(left_circuit, expected_full_circuit), 1
        )

    def test_find_num_gates(self):
        from isl.utils.circuit_operations.circuit_operations_full_circuit import (
            find_num_gates,
        )

        qc = QuantumCircuit(3)
        qc.rx(0.6, 0)
        qc.cx(0, 1)
        # Counting start
        qc.rx(0.3, 1)
        # Next 2 cx gates should cancel each other if transpiling
        qc.ry(1.3, 0)
        qc.cx(1, 0)
        qc.cx(1, 0)
        qc.cx(1, 0)
        qc.rz(-2.3, 2)
        qc.cz(2, 0)
        # Counting end
        qc.cx(0, 1)
        qc.rx(0.3, 1)

        assert find_num_gates(qc, False, gate_range=(2, 9)) == (4, 3)
        assert find_num_gates(qc, True, {"optimization_level": 1}) == (4, 5)


class TestVariationalCircuitOperations(TestCase):
    def test_find_angles_in_circuit(self):
        from qiskit import QuantumCircuit

        from isl.utils.circuit_operations.circuit_operations_variational import (
            find_angles_in_circuit,
        )
        from isl.utils.constants import FIXED_GATE_LABEL

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.23, 0)
        qc.cx(1, 0)
        qc.rz(3.1, 2)
        qc.rx(2.9, 2)
        qc.data[-1][0].label = FIXED_GATE_LABEL
        qc.cx(1, 2)
        qc.ry(-1.4, 1)
        qc.measure_all()

        assert find_angles_in_circuit(qc) == [0.23, 3.1, -1.4]
        assert find_angles_in_circuit(qc, (4, 5)) == [3.1]

    def test_update_angles_in_circuit(self):
        import isl.utils.constants as vconstants
        from isl.utils.circuit_operations.circuit_operations_basic import create_1q_gate
        from isl.utils.circuit_operations.circuit_operations_variational import (
            update_angles_in_circuit,
        )

        fixed_gate = create_1q_gate("rx", 2.3)
        fixed_gate.label = vconstants.FIXED_GATE_LABEL
        parameterized_gate1 = create_1q_gate("rz", 1.0)
        parameterized_gate2 = create_1q_gate("ry", 1.0)
        qr = QuantumRegister(3)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.append(parameterized_gate1.copy(), [qr[0]])
        qc.cx(0, 1)
        qc.append(parameterized_gate1.copy(), [qr[0]])
        qc.append(fixed_gate, [qr[2]])
        qc.append(parameterized_gate1.copy(), [qr[1]])
        qc.z(2)
        qc.append(parameterized_gate2.copy(), [qr[2]])

        new_angles = [-1, 0, 0.5, 0.23]
        update_angles_in_circuit(qc, new_angles)

        for index, angle in zip([1, 3, 5, 7], new_angles):
            assert qc.data[index][0].params[0] == angle

        new_angles = [0.5, 0.23]
        update_angles_in_circuit(qc, new_angles, (4, 8))

        for index, angle in zip([5, 7], new_angles):
            assert qc.data[index][0].params[0] == angle

    # -----------------------------------------Circuit optimisation
    # operations-----------------------------------------

    # -------------------------------------Circuit division(and helper)
    # operations------------------------------------


class TestMiscCircuitOperations(TestCase):
    def test_initial_state_to_circuit(self):
        from qiskit import Aer, execute

        from isl.utils.circuit_operations.circuit_operations_full_circuit import (
            initial_state_to_circuit,
        )

        # Test None
        assert initial_state_to_circuit(None) is None

        # Test Vector
        qubits = 3
        x = np.random.RandomState().standard_normal(2**qubits)
        rand_state = x / np.linalg.norm(x)

        qc = QuantumCircuit(qubits)
        qc.append(initial_state_to_circuit(rand_state), qc.qubits)
        sv = (
            execute(qc, Aer.get_backend("statevector_simulator"), shots=1)
            .result()
            .get_statevector()
        )
        overlap_minus_1 = np.abs(np.abs(np.vdot(rand_state, sv)) - 1)
        assert overlap_minus_1 < 1e-3
