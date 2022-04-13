from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit

import isl.utils.circuit_operations as co


class TestQulacs(TestCase):
    def test_simple_circuit_u1_u2_u3_cx_gives_same_result_as_qiskit(self):
        qc = QuantumCircuit(2)
        qc.u2(1.9, 2.1, 0)
        qc.u1(0.7, 0)
        qc.u2(-0.5, -1.1, 1)
        qc.u3(-0.5, -1.1, 2.3, 1)
        qc.cx(0, 1)
        qc.u1(2.7, 0)
        qc.u3(-0.5, -1.1, -1.94, 1)
        qc.u2(-0.15, -2.1, 1)
        qc.u1(-1.2, 1)
        sv_qulacs = co.run_circuit_without_transpilation(
            qc, "qulacs", return_statevector=True
        )
        sv_qiskit = co.run_circuit_without_transpilation(
            qc, co.SV_SIM, return_statevector=True
        )
        assert np.allclose(sv_qiskit, sv_qulacs, 1e-10)

    def test_simple_circuit_rx_ry_cx_h_gives_same_result_as_qiskit(self):
        qc = QuantumCircuit(2)
        qc.rx(0.7, 0)
        qc.ry(-1.7, 0)
        qc.rz(2.2, 0)
        qc.rx(1.2, 1)
        qc.ry(-2.9, 1)
        qc.rz(-1.2, 1)
        qc.h(1)
        qc.x(0)
        qc.cx(0, 1)
        qc.rx(1.7, 0)
        qc.ry(-2.7, 0)
        qc.rz(1.2, 0)
        qc.rx(2.2, 1)
        qc.ry(-1.9, 1)
        qc.rz(-0.2, 1)
        sv_qulacs = co.run_circuit_without_transpilation(
            qc, "qulacs", return_statevector=True
        )
        sv_qiskit = co.run_circuit_without_transpilation(
            qc, co.SV_SIM, return_statevector=True
        )
        assert np.allclose(sv_qiskit, sv_qulacs, 1e-10)

    def test_simple_circuit_h_y_z_cz_gives_same_result_as_qiskit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.y(0)
        qc.z(1)
        qc.cz(0, 1)
        qc.y(0)
        qc.z(0)
        qc.cz(0, 1)
        qc.y(1)
        qc.z(0)
        qc.cz(1, 0)
        sv_qulacs = co.run_circuit_without_transpilation(
            qc, "qulacs", return_statevector=True
        )
        sv_qiskit = co.run_circuit_without_transpilation(
            qc, co.SV_SIM, return_statevector=True
        )
        assert np.allclose(sv_qiskit, sv_qulacs, 1e-10)
