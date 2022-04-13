from unittest import TestCase

from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

import isl.utils.circuit_operations as co
from isl.recompilers import ISLConfig, ISLRecompiler
from isl.utils.circuit_operations import QASM_SIM, SV_SIM


class TestISL(TestCase):
    def test_basic_sv(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc, backend=SV_SIM)

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999

    def test_basic_qasm(self):
        qc = co.create_random_initial_state_circuit(2)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler_qasm = ISLRecompiler(qc, backend=QASM_SIM)

        result_qasm = isl_recompiler_qasm.recompile()
        approx_circuit_qasm = result_qasm["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit_qasm, qc)
        assert overlap > 0.999

    def test_exact_overlap_close_to_approx_overlap(self):
        qc = co.create_random_initial_state_circuit(2)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc)

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        approx_overlap = result["overlap"]
        exact_overlap = result["exact_overlap"]
        self.assertAlmostEquals(approx_overlap, exact_overlap, delta=1e-2)

    def test_exact_overlap_calculated_correctly(self):
        qc = co.create_random_initial_state_circuit(2)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc)

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        exact_overlap1 = result["exact_overlap"]
        exact_overlap2 = co.calculate_overlap_between_circuits(approx_circuit, qc)
        self.assertAlmostEquals(exact_overlap1, exact_overlap2, delta=1e-2)

    def test_local_measurements(self):
        qc = co.create_random_initial_state_circuit(2)
        qc = co.unroll_to_basis_gates(qc)
        isl_config = ISLConfig(cost_improvement_num_layers=10)

        for backend in [SV_SIM, QASM_SIM]:
            isl_recompiler = ISLRecompiler(
                qc, local_measurements_only=True, backend=backend, isl_config=isl_config
            )
            result = isl_recompiler.recompile()
            approx_circuit = result["circuit"]
            overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
            assert overlap > 0.999

    def test_custom_layer_gate(self):
        from qiskit import QuantumCircuit

        from isl.utils.fixed_ansatz_circuits import number_preserving_ansatz

        # Initialize to a supervision of states with bit sum 2
        statevector = [
            0,
            0,
            0,
            -((1 / 3) ** 0.5),
            0,
            1j * (1 / 3) ** 0.5,
            -1 * (1 / 3) ** 0.5,
            0,
        ]
        qc = co.initial_state_to_circuit(statevector)

        initial_circuit = QuantumCircuit(3)
        initial_circuit.x(0)
        initial_circuit.x(1)

        isl_recompiler = ISLRecompiler(
            qc,
            custom_layer_2q_gate=number_preserving_ansatz(2, 1),
            starting_circuit=initial_circuit,
        )

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999

    def test_with_initial_ansatz(self):
        from isl.utils.fixed_ansatz_circuits import hardware_efficient_circuit

        qc = hardware_efficient_circuit(3, "rxrz", 3)

        qc_mod = qc.copy()
        qc_mod.cx(0, 1)
        qc_mod.h(1)
        qc_mod.cx(1, 2)

        isl_recompiler = ISLRecompiler(qc_mod)

        result = isl_recompiler.recompile_using_initial_ansatz(qc)
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc_mod)
        assert overlap > 0.99

    def test_heuristic_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="heuristic")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999

    def test_basic_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="basic")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999

    def test_random_methods(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        config = ISLConfig(method="random")

        isl_recompiler = ISLRecompiler(qc, isl_config=config)
        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]
        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999

    def test_given_circuit_with_non_basis_gates_when_recompiling_then_no_error(self):
        qc1 = QuantumCircuit(2)
        qc1.h([0, 1])
        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.append(qc1.to_instruction(), qc2.qregs[0])
        recompiler = ISLRecompiler(qc2)
        recompiler.recompile()

    def test_given_two_registers_when_recompiling_then_no_error(self):
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)
        qc = QuantumCircuit(qr1, qr2)
        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        print(result.get("circuit"))

    def test_given_two_registers_when_recompiling_then_register_names_preserved(self):
        qr1 = QuantumRegister(2, "reg1")
        qr2 = QuantumRegister(2, "reg2")
        qc = QuantumCircuit(qr1, qr2)
        qc.h(1)
        qc.cx(1, 2)
        qc.x(3)
        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        final_circuit = result.get("circuit")
        assert final_circuit.qregs == qc.qregs

    def test_given_circuit_with_cregs_when_recompiling_then_no_error(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)

        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        print(result)

    def test_given_circuit_with_cregs_when_recompiling_then_register_names_preserved(
        self,
    ):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)

        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        final_circuit = result.get("circuit")
        assert final_circuit.cregs == qc.cregs

    def test_given_circuit_with_measurements_when_recompiling_then_no_error(self):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        qc.measure(0, 0)
        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        print(result["circuit"])
        # assert False

    def test_given_circuit_with_one_measurement_when_recompiling_then_preserve_measurement(
        self,
    ):
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        qc.measure(0, 0)
        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        assert result["circuit"].data[-1] == qc.data[-1]
        print(result["circuit"])

    def test_given_circuit_with_multi_measurement_when_recompiling_then_preserve_measurement(
        self,
    ):
        num_measurements = 3
        qreg = QuantumRegister(num_measurements + 2)
        creg = ClassicalRegister(num_measurements + 2)
        qc = QuantumCircuit(qreg, creg)
        qc.cx(0, 1)
        for i in range(num_measurements):
            qc.measure(i, i)
        recompiler = ISLRecompiler(qc)
        result = recompiler.recompile()
        assert result["circuit"].data[-num_measurements:] == qc.data[-num_measurements:]
        print(result["circuit"])

    def test_qulacs_recompiler(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)

        isl_recompiler = ISLRecompiler(qc, backend="qulacs")

        result = isl_recompiler.recompile()
        approx_circuit = result["circuit"]

        overlap = co.calculate_overlap_between_circuits(approx_circuit, qc)
        assert overlap > 0.999

    def test_qulacs_recompiler_noise_give_error(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        nm = co.create_noisemodel(0.1, 0.1, False)
        isl_recompiler = ISLRecompiler(
            qc, backend="qulacs", execute_kwargs={"noise_model": nm}
        )
        with self.assertRaises(ValueError):
            result = isl_recompiler.recompile()

    def test_with_initial_ansatz(self):
        qc = co.create_random_initial_state_circuit(3)
        qc = co.unroll_to_basis_gates(qc)
        isl_recompiler = ISLRecompiler(qc, backend="qulacs")
        result = isl_recompiler.recompile(qc.copy())
        num_2q_before = co.find_num_gates(qc)[0]
        num_2q_after = co.find_num_gates(result["circuit"])[0]
        self.assertLessEqual(num_2q_after, num_2q_before)
