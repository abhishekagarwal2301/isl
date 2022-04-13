"""Contains VariationalQuantumEigensolver"""
from abc import ABC, abstractmethod

import numpy as np
from openfermion import QubitOperator
from qiskit import QuantumCircuit

import isl.utils.circuit_operations as co
from isl.utils.cost_minimiser import CostMinimiser
from isl.utils.utilityfunctions import is_statevector_backend


class VariationalQuantumEigensolver(ABC):
    """
    Variational Algorithm that minimizes
    """

    def __init__(
        self,
        evaluation_matrix=None,
        evaluation_paulis=None,
        backend=co.SV_SIM,
        execute_kwargs=None,
    ):
        """
        :param evaluation_matrix: The hamiltonian of which the ground state(
            circuit) is to be found
        :param evaluation_paulis: The pauli labels and coefficients which
            define the hamiltonian e.g. {'IXII':0.5,'ZZZY':-0.23} or
            QubitOperator.
        :param execute_kwargs: keyword arguments passed into circuit runs (
            excluding backend)
            e.g. {'noise_model:NoiseModel, shots=10000}
        """
        if (evaluation_matrix is None and evaluation_paulis is None) or (
            evaluation_matrix is not None and evaluation_paulis is not None
        ):
            raise ValueError(
                "Exact one of evaluation_matrix and evaluation_paulis must be "
                "provided"
            )
        elif evaluation_paulis is None:
            if not is_statevector_backend(backend):
                raise ValueError(
                    "Only statevector simulator supported in matrix " "evaluation mode"
                )
            self.hamiltonian = evaluation_matrix
            self.matrix_mode = True
            self.num_qubits = int(np.log2(evaluation_matrix.shape[0]))
        elif evaluation_matrix is None:
            if isinstance(evaluation_paulis, QubitOperator):
                self.hamiltonian = co.convert_qubit_op_to_pauli_dict(evaluation_paulis)
            else:
                self.hamiltonian = evaluation_paulis
            self.matrix_mode = False
            self.num_qubits = len(list(self.hamiltonian)[0])
        else:
            raise ValueError(
                "Only one of evaluation_matrix and evaluation_hamiltonian "
                "must be provided"
            )

        self.backend = backend
        self.execute_kwargs = self.parse_default_execute_kwargs(execute_kwargs)
        self.full_circuit, self.lhs_gates, self.rhs_gates = self._prepare_full_circuit()
        self.minimizer = CostMinimiser(
            self.evaluate_cost, self.variational_circuit_range, self.full_circuit
        )

    def variational_circuit_range(self):
        return 0, len(self.full_circuit.data)

    def parse_default_execute_kwargs(self, execute_kwargs):
        kwargs = {} if execute_kwargs is None else dict(execute_kwargs)
        if "shots" not in kwargs:
            if self.backend == "qulacs":
                kwargs["shots"] = 2**30
            elif not is_statevector_backend(self.backend):
                kwargs["shots"] = 8192
            else:
                kwargs["shots"] = 1
        if "optimization_level" not in kwargs:
            kwargs["optimization_level"] = 0
        return kwargs

    @abstractmethod
    def run(self):
        """
        Run algorithm
        :return: Dictionary object containing the resulting circuit,
        minimum_energy, and other optional
        entries(such as circuit parameters)
        """
        return

    def _prepare_full_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        if self.matrix_mode or is_statevector_backend(self.backend):
            return qc, 0, 0
        else:
            # qc.measure_all()
            return qc, 0, len(qc.data)

    def evaluate_cost(self):
        """
        Run circuit(s) and evaluate the cost. If matrix_mode,
        the statevector_simulator is used to find the wavefunction
        and cost is evaluated by calculating <Ψ|H|Ψ>. If not matrix_mode,
        expectation value is calculated by evaluating
        the expectation value of each given pauli_operator (w.r.t.
        wavefunction) and then summing over the expectation
        values with the provided coefficients.
        :return:
        """
        if self.matrix_mode:
            sv = co.run_circuit_without_transpilation(
                self.full_circuit, co.SV_SIM, return_statevector=True
            )
            # Calculate expectation value of hamiltonian <Ψ|H|Ψ>
            energy = np.real(np.vdot(sv, np.dot(self.hamiltonian, sv)))
        else:
            # <Ψ|H|Ψ> = sum(coefficient*<Ψ|pauli_operator|Ψ>)
            energy = co.expectation_value_of_pauli_operator(
                self.full_circuit,
                self.hamiltonian,
                self.backend,
                None,
                self.execute_kwargs,
            )
        return energy
