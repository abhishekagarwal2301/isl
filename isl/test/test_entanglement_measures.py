from unittest import TestCase

import numpy as np
from qiskit import Aer

import isl.utils.circuit_operations as co


class TestEntanglementMeasures(TestCase):
    def test_quantum_tomography(self):
        from isl.utils.entanglement_measures import perform_quantum_tomography

        qc = co.create_random_initial_state_circuit(3)
        dm = perform_quantum_tomography(qc, 0, 1, Aer.get_backend("qasm_simulator"))
        assert isinstance(dm, np.ndarray)

    def test_tomography_entanglement_measures(self):
        import isl.utils.entanglement_measures as em

        qc = co.create_random_initial_state_circuit(3)
        for backend in [
            Aer.get_backend("qasm_simulator"),
            Aer.get_backend("statevector_simulator"),
        ]:
            for method in [
                em.EM_TOMOGRAPHY_CONCURRENCE,
                em.EM_TOMOGRAPHY_EOF,
                em.EM_TOMOGRAPHY_NEGATIVITY,
            ]:
                em.calculate_entanglement_measure(method, qc, 0, 1, backend)

    def test_observable_min_concurrence(self):
        import isl.utils.entanglement_measures as em

        qc = co.create_random_initial_state_circuit(3)
        em.measure_concurrence_lower_bound(qc, 0, 1, Aer.get_backend("qasm_simulator"))
