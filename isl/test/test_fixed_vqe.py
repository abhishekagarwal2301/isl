from unittest import TestCase

import numpy as np
from openfermion import QubitOperator, get_sparse_operator, hermitian_conjugated
from scipy.linalg import eigh

import isl.utils.circuit_operations as co
from isl.utils.fixed_ansatz_circuits import hardware_efficient_circuit
from isl.vqe import FixedAnsatzVQE

TEST_QO = (
    QubitOperator("", 0.2)
    - QubitOperator("Y0 X1", 1.1)
    + QubitOperator("Z1 X1", 0.9)
    - QubitOperator("X1", -0.2)
)
TEST_QO = TEST_QO + hermitian_conjugated(TEST_QO)


class TestMatrixMode(TestCase):
    def test_real_hamiltonian(self):
        matrix = get_sparse_operator(TEST_QO, 2).toarray()

        eigvals, eigvecs = eigh(matrix)
        min_eigval = eigvals[0]
        min_eigvec = eigvecs[:, 0]
        vqe = FixedAnsatzVQE(
            hardware_efficient_circuit(2, "rxryrz", 2), evaluation_matrix=matrix
        )

        result = vqe.run()
        assert np.isclose(min_eigval, result["energy"], atol=1e-5)

        sv = co.run_circuit_without_transpilation(
            result["circuit"], co.SV_SIM, return_statevector=True
        )
        overlap = np.absolute(np.vdot(sv, min_eigvec))
        assert np.isclose(overlap, 1, atol=1e-3)


class TestPauliMode(TestCase):
    def test_real_hamiltonian_qasm_sim(self):
        matrix = get_sparse_operator(TEST_QO, 2).toarray()

        eigvals, eigvecs = eigh(matrix)
        min_eigval = eigvals[0]

        vqe = FixedAnsatzVQE(
            hardware_efficient_circuit(2, "rxryrz", 2),
            backend=co.QASM_SIM,
            evaluation_paulis=TEST_QO,
        )

        result = vqe.run()
        assert np.isclose(min_eigval, result["energy"], atol=1e-2)
        # min_eigvec = eigvecs[:, 0]
        # sv = co.run_circuit_without_transpilation(result['circuit'],
        #                                           co.SV_SIM,
        #                                           return_statevector=True)
        # overlap = np.absolute(np.vdot(sv, min_eigvec))
        # print(np.vdot(sv, min_eigvec), overlap)
        # print(sv.T.conj() @ matrix @ sv)
        # print(sv, min_eigvec)
        # assert np.isclose(overlap, 1, atol=1e-2)

    def test_real_hamiltonian_sv_sim(self):
        matrix = get_sparse_operator(TEST_QO, 2).toarray()

        eigvals, eigvecs = eigh(matrix)
        min_eigval = eigvals[0]

        vqe = FixedAnsatzVQE(
            hardware_efficient_circuit(2, "rxryrz", 2), evaluation_paulis=TEST_QO
        )

        result = vqe.run()
        assert np.isclose(min_eigval, result["energy"], atol=1e-5)
        # min_eigvec = eigvecs[:, 0]
        # sv = co.run_circuit_without_transpilation(result['circuit'],
        #                                           co.SV_SIM,
        #                                           return_statevector=True)
        # overlap = np.absolute(np.vdot(sv, min_eigvec))
        # print(np.vdot(sv, min_eigvec), overlap)
        # print(sv.T.conj() @ matrix @ sv)
        # print(sv,min_eigvec)
        # assert np.isclose(overlap, 1, atol=1e-3)
