"""Contains FixedAnsatzVQE"""
import logging

import isl.utils.circuit_operations as co
import isl.utils.constants as vconstants
from isl.vqe.variational_quantum_eigensolver import VariationalQuantumEigensolver

logger = logging.getLogger(__name__)


class FixedAnsatzVQE(VariationalQuantumEigensolver):
    """
    VQE algorithm that uses a fixed ansatz
    """

    def __init__(
        self,
        ansatz_circuit,
        evaluation_matrix=None,
        evaluation_paulis=None,
        initial_params=None,
        optimization_algorithm_kind=vconstants.ALG_ROTOSOLVE,
        optimization_algorithm_identifier=None,
        backend=co.SV_SIM,
    ):
        """
        :param ansatz_circuit: The ansatz circuit that is to be recompiled to.
        Rotation gates with label FIXED_GATE_LABEL will not be varied
        :param initial_params: (list/np.ndarray) Initial ansatz rotation
        angles.
        Ordering must be same as in FixedAnsatzCircuit and list must not
        include angles for gates that are fixed.
        :param optimization_algorithm_kind: One of the ALG_ constants from
        variationalalgorithms.variational_constants
        :param optimization_algorithm_identifier: Relevant optimization
        algorithm method identifier
        """
        super().__init__(
            evaluation_matrix=evaluation_matrix,
            evaluation_paulis=evaluation_paulis,
            backend=backend,
        )

        co.add_to_circuit(
            self.full_circuit,
            ansatz_circuit,
            location=self.variational_circuit_range()[0],
        )

        if initial_params is not None:
            co.update_angles_in_circuit(
                self.full_circuit,
                initial_params,
                gate_range=self.variational_circuit_range(),
            )

        self.optimization_algorithm_kind = optimization_algorithm_kind
        self.optimization_algorithm_identifier = optimization_algorithm_identifier

    def run(self):
        """
        Run algorithm
        :return: {'circuit':resulting circuit,
        'energy':minimum energy,
        'angles':final angles of variational circuit [float]}
        """
        logger.info("Fixed ansatz VQE started")
        cost = self.minimizer.minimize_cost(
            algorithm_kind=self.optimization_algorithm_kind,
            algorithm_identifier=self.optimization_algorithm_identifier,
            tol=1e-10,
        )

        final_angles = co.find_angles_in_circuit(
            self.full_circuit, self.variational_circuit_range()
        )
        final_circuit = co.extract_inner_circuit(
            self.full_circuit, self.variational_circuit_range()
        )
        result_dict = {"circuit": final_circuit, "energy": cost, "angles": final_angles}
        logger.info("Fixed ansatz VQE completed")
        return result_dict
