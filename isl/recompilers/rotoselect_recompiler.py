"""Contains RotoselectRecompiler"""
import logging
import timeit

from qiskit import Aer, QuantumCircuit

import isl.utils.circuit_operations as co
import isl.utils.constants as vconstants
from isl.recompilers.approximate_recompiler import ApproximateRecompiler
from isl.utils.constants import coupling_map_linear

logger = logging.getLogger(__name__)


class RotoselectRecompiler(ApproximateRecompiler):
    """
    Structural learning algorithm that performs recompilation using the
    rotoselect algorithm from arXiv:1905.09692
    """

    def __init__(
        self,
        circuit_to_recompile,
        initial_state=None,
        backend=Aer.get_backend("statevector_simulator"),
        qubit_subset=None,
        max_gpqpl=2,
        num_layers=4,
        max_attempts=10,
        coupling_map=None,
        execute_kwargs=None,
        general_initial_state=False,
        starting_circuit=None,
        local_measurements_only=False,
    ):
        """
        :param max_gpqpl: Maximum number of rotation Gates Per Qubit Per
        Layer (int)
        :param num_layers: Number of layers of (rotation gates + entangling
        gates) (int)
        :param max_attempts: Number of times recompiler will try to escape
        local minimum and retry optimisation
        :param coupling_map: Coupling map of entangling gates [(control),
        target]
        :param general_initial_state: Whether recompilation should be for a
        general initial state
        """
        super().__init__(
            circuit_to_recompile=circuit_to_recompile,
            initial_state=initial_state,
            backend=backend,
            qubit_subset=qubit_subset,
            execute_kwargs=execute_kwargs,
            general_initial_state=general_initial_state,
            starting_circuit=starting_circuit,
            local_measurements_only=local_measurements_only,
        )

        self.max_gpqpl = max_gpqpl
        self.num_layers = num_layers
        self.max_attempts = max_attempts
        self.coupling_map = (
            coupling_map_linear(len(self.qubit_subset_to_recompile))
            if coupling_map is None
            else coupling_map
        )

    def recompile(self, gap_between_minima=None, first_minima_loc=None):
        """
        Perform recompilation algorithm
        :return: {'circuit':resulting circuit(Instruction),
        'overlap':overlap(float),
        'num_1q_gates':number of rotation gates in circuit(int),
        'num_2q_gates':number of entangling gates in circuit(int)}
        'time_taken': total time taken for recompilation
        """
        logger.info("ROTOSELECT recompilation started")
        start_time = timeit.default_timer()
        self.cost_evaluation_counter = 0

        self.prepare_rotoselect_circuit()
        penalty_amp = 0.1
        for i in range(self.max_attempts):
            cost = self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_ROTOSELECT, tol=1e-5, stop_val=1e-3
            )
            if (
                cost > 1e-3
                and gap_between_minima is not None
                and first_minima_loc is not None
            ):
                self.minimizer.try_escaping_periodic_local_minimum(
                    gap_between_minima, first_minima_loc, penalty_amp
                )
                penalty_amp += 0.1
        g_range = self.variational_circuit_range
        co.remove_unnecessary_gates_from_circuit(
            self.full_circuit, True, True, g_range()
        )

        self.minimizer.minimize_cost(
            algorithm_kind=vconstants.ALG_ROTOSOLVE, tol=1e-5, stop_val=1e-7
        )

        co.remove_unnecessary_gates_from_circuit(
            self.full_circuit, True, True, g_range()
        )

        num_2q_gates, num_1q_gates = co.find_num_gates(self.full_circuit, g_range())
        final_cost = self.evaluate_cost()
        end_time = timeit.default_timer()

        recompiled_circuit = self.get_recompiled_circuit()
        exact_overlap = co.calculate_overlap_between_circuits(
            self.circuit_to_recompile.to_instruction(),
            recompiled_circuit.to_instruction(),
            initial_state=self.initial_state_circuit,
            qubit_subset=self.qubit_subset_to_recompile,
        )

        result_dict = {
            "circuit": self.get_recompiled_circuit(),
            "overlap": 1 - final_cost,
            "exact_overlap": exact_overlap,
            "num_1q_gates": num_1q_gates,
            "num_2q_gates": num_2q_gates,
            "time_taken": end_time - start_time,
            "cost_evaluations": self.cost_evaluation_counter,
        }
        logger.info("ROTOSELECT recompilation completed")
        return result_dict

    def prepare_rotoselect_circuit(self):
        variational_circuit = QuantumCircuit(len(self.qubit_subset_to_recompile))
        rz_gate = co.create_1q_gate("rz", 0)
        cx_gate = co.create_2q_gate("cx")
        for _ in range(self.num_layers):
            # Add 1q gates layer (with self.max_gpqpl rotation gates per qubit)
            for i in range(len(self.qubit_subset_to_recompile)):
                for _ in range(self.max_gpqpl):
                    variational_circuit.append(rz_gate, [i])
            # Add 2q gates layer
            for control, target in self.coupling_map:
                variational_circuit.append(cx_gate, [control, target])
        co.add_to_circuit(
            self.full_circuit,
            variational_circuit,
            self.variational_circuit_range()[0],
            transpile_before_adding=True,
            transpile_kwargs={"backend": self.backend, "optimization_level": 0},
            qubit_subset=self.qubit_subset_to_recompile,
        )
