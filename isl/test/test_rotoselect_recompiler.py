from unittest import TestCase

import isl.utils.circuit_operations as co
from isl.recompilers import RotoselectRecompiler


class TestRotoselectRecompiler(TestCase):
    def test_no_initial_state(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        roto_recompiler = RotoselectRecompiler(qc)

        result = roto_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999

    def test_with_initial_state_and_qubit_subset(self):
        # 3 qubit initial state
        rand_initial_state = co.create_random_initial_state_circuit(3)

        # Random 2 qubit circuit
        qc = co.create_random_initial_state_circuit(2)
        qc = co.unroll_to_basis_gates(qc)

        # Random 2 qubit circuit acts on qubit 0,2 of initial state circuit
        qubit_subset = [0, 2]
        roto_recompiler = RotoselectRecompiler(
            qc, rand_initial_state, qubit_subset=qubit_subset
        )

        result = roto_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(
            approx_circuit, qc, rand_initial_state, qubit_subset
        )
        assert overlap > 0.999

    def test_general_initial_state(self):
        num_qubits = 2
        qc = co.create_random_initial_state_circuit(num_qubits)
        qc = co.unroll_to_basis_gates(qc)

        roto_recompiler = RotoselectRecompiler(
            qc, num_layers=10, general_initial_state=True, backend="qulacs"
        )

        result = roto_recompiler.recompile()
        approx_circuit = result["circuit"]
        rand_initial_state = co.create_random_initial_state_circuit(num_qubits)
        print(f'Recompilation final overlap: {result["overlap"]}')
        overlap = co.calculate_overlap_between_circuits(
            approx_circuit, qc, rand_initial_state
        )
        print(overlap)
        assert overlap > 0.999

    def test_no_initial_state_qulacs(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        roto_recompiler = RotoselectRecompiler(qc, backend="qulacs")

        result = roto_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999
