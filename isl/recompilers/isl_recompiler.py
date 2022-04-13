"""Contains ISLRecompiler"""

import logging
import timeit

import numpy as np
from qiskit import QuantumCircuit

import isl.utils.circuit_operations as co
import isl.utils.constants as vconstants
from isl.recompilers.approximate_recompiler import ApproximateRecompiler
from isl.utils.constants import CMAP_FULL, generate_coupling_map
from isl.utils.entanglement_measures import (
    EM_TOMOGRAPHY_CONCURRENCE,
    calculate_entanglement_measure,
)
from isl.utils.utilityfunctions import (
    expectation_value_of_qubits,
    has_stopped_improving,
    is_statevector_backend,
    remove_permutations_from_coupling_map,
)

logger = logging.getLogger(__name__)


class ISLConfig:
    def __init__(
        self,
        max_layers: int = 100000,
        sufficient_cost=1e-4,
        max_2q_gates=1e4,
        cost_improvement_num_layers=10,
        cost_improvement_tol=1e-2,
        max_layers_to_modify=100,
        method="ISL",
        bad_qubit_pair_memory=10,
        rotosolve_frequency=1,
    ):
        """
        Termination criteria:
        :param max_layers: Maximum number of layers where each layer has a
        thinly dressed cnot gate
        :param sufficient_cost: ISL will terminate if the cost (1-overlap)
        reaches below this value
        :param max_2q_gates: ISL will terminate if the number of 2 qubit
        gates reaches this value
        :param cost_improvement_num_layers: The number of layer costs to
        consider when evaluating cost improvement
        :param cost_improvement_tol: The minimum relative cost improvement
        to continue adding layers

        Add layer criteria:
        :param max_layers_to_modify: Only the last max_layers_to_modify
        layers will be modified using Rotoselect
        :param method: Method to choose qubit pair for 2-qubit gates.
            One of 'ISL', 'random', 'heuristic','basic'

        Other parameters:
        :param bad_qubit_pair_memory:
        :param rotosolve_frequency: How often rotosolve is used
        (if n, rotosolve will be used after every n layers)

        """
        self.bad_qubit_pair_memory = bad_qubit_pair_memory
        self.max_layers = max_layers
        self.sufficient_cost = sufficient_cost
        self.max_2q_gates = max_2q_gates
        self.cost_improvement_tol = cost_improvement_tol
        self.cost_improvement_num_layers = cost_improvement_num_layers
        self.max_layers_to_modify = max_layers_to_modify
        self.method = method
        self.rotosolve_frequency = rotosolve_frequency

    def __repr__(self):
        representation_str = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            representation_str += f"{k}={v!r}, "
        representation_str += ")"
        return representation_str


class ISLRecompiler(ApproximateRecompiler):
    """
    Structure learning algorithm that incrementally builds a circuit that
    has the same result when acting on |0> state
    (computational basis) as the given circuit.
    """

    def __init__(
        self,
        circuit_to_recompile,
        entanglement_measure=EM_TOMOGRAPHY_CONCURRENCE,
        backend=co.SV_SIM,
        execute_kwargs=None,
        coupling_map=None,
        isl_config: ISLConfig = None,
        general_initial_state=False,
        custom_layer_2q_gate=None,
        starting_circuit=None,
        use_rotosolve=True,
        perform_final_minimisation=False,
        local_measurements_only=False,
    ):
        """
        :param circuit_to_recompile: Circuit that is to be recompiled
        :param entanglement_measure: The entanglement measurement method to
        use for quantifying local entanglement
        :param backend: Backend to run circuits on
        :param coupling_map: 2-qubit gate coupling map to use
        :param isl_config: ISLConfig object
        :param general_initial_state: Recompile circuit for an arbitrary
        initial state
        :param custom_layer_2q_gate: Entangling gate to use (default is
        thinly dressed CNOT)
        :param starting_circuit: The initial fixed gates on the recompiled
        circuit. WARNING: Using an entangled circuit
            will lead to worse ISL performance because it disrupts the
            measurement of local entanglement between qubits
        :param use_rotosolve: Whether to use rotosolve, rotoselect for cost
        minimisation.
            Disable if custom_layer_2q_gate does not support rotosolve
        :param perform_final_minimisation: Perform a final cost minimisation
        once ISL has ended
        :param local_measurements_only: Use LHST cost function as defined in
        (arXiv:1807.00800v5)
        :param execute_kwargs: keyword arguments passed into circuit runs (
        excluding backend)
            e.g. {'noise_model:NoiseModel, 'shots':10000}
        """
        super().__init__(
            circuit_to_recompile=circuit_to_recompile,
            initial_state=None,
            backend=backend,
            execute_kwargs=execute_kwargs,
            general_initial_state=general_initial_state,
            starting_circuit=starting_circuit,
            local_measurements_only=local_measurements_only,
        )

        self.entanglement_measure_method = entanglement_measure
        self.isl_config = isl_config if isl_config is not None else ISLConfig()

        if coupling_map is None:
            coupling_map = generate_coupling_map(
                self.total_num_qubits, CMAP_FULL, False, False
            )

        # If custom layer gate is provided, do not remove gate during ISL
        # because individual gates
        # might depend on each other.
        self.remove_unnecessary_gates = custom_layer_2q_gate is None
        self.use_rotosolve = use_rotosolve
        self.perform_final_minimisation = perform_final_minimisation
        self.layer_2q_gate = self.construct_layer_2q_gate(custom_layer_2q_gate)

        # Remove permutations so that ISL is not stuck on the same pair of
        # qubits
        self.coupling_map = remove_permutations_from_coupling_map(coupling_map)
        self.coupling_map = [
            (q1, q2)
            for (q1, q2) in self.coupling_map
            if q1 in self.qubit_subset_to_recompile
            and q2 in self.qubit_subset_to_recompile
        ]
        # Used to avoid adding thinly dressed CNOTs to the same qubit pair
        self.qubit_pair_history = []
        # Avoid adding CNOTs to these qubit pairs
        self.bad_qubit_pairs = []
        # Used to keep track of whether ISL/heuristic method was used
        self.pair_selection_method_history = []
        self.entanglement_measures_history = []
        self.e_val_history = []

    def construct_layer_2q_gate(self, custom_layer_2q_gate):
        if custom_layer_2q_gate is None:
            qc = QuantumCircuit(2)
            if self.general_initial_state:
                co.add_dressed_cnot(qc, 0, 1, True)
                co.add_dressed_cnot(qc, 0, 1, True, v1=False, v2=False)
            else:
                co.add_dressed_cnot(qc, 0, 1, True)
            return qc
        else:
            return custom_layer_2q_gate

    def get_layer_2q_gate(self, layer_index):
        qc = self.layer_2q_gate.copy()
        co.add_subscript_to_all_variables(qc, layer_index)
        return qc

    def recompile_using_initial_ansatz(
        self, ansatz: QuantumCircuit, modify_ansatz=True
    ):
        """
        Use the provided ansatz as a starting point for the recompilation
        :param modify_ansatz: If ansatz should be optimised during ISL
        :param ansatz: Quantum Circuit to use
        :return: Recompilation result
        """
        old_vcr = self.variational_circuit_range()
        vcr = self.variational_circuit_range
        ansatz_inv = co.circuit_by_inverting_circuit(ansatz)
        co.add_to_circuit(self.full_circuit, ansatz_inv, vcr()[0])
        if not modify_ansatz:
            self.lhs_gate_count = old_vcr[0] + len(ansatz_inv.data)

        self.minimizer.minimize_cost(
            algorithm_kind=vconstants.ALG_ROTOSOLVE,
            tol=1e-5,
            stop_val=self.isl_config.sufficient_cost,
        )

        res = self.recompile()
        self.lhs_gate_count = old_vcr[0]

        recompiled_circuit = self.get_recompiled_circuit()
        exact_overlap = co.calculate_overlap_between_circuits(
            self.circuit_to_recompile, recompiled_circuit
        )
        num_2q_gates, num_1q_gates = co.find_num_gates(
            self.full_circuit, gate_range=vcr()
        )

        res["circuit"] = recompiled_circuit
        res["exact_overlap"] = exact_overlap
        res["num_1q_gates"] = num_1q_gates
        res["num_2q_gates"] = num_2q_gates
        res["circuit_qasm"] = recompiled_circuit.qasm()
        return res

    def recompile(self, initial_ansatz: QuantumCircuit = None):
        """
        Perform recompilation algorithm.
        :param initial_ansatz: A trial ansatz to start the recompilation
        with instead of starting from scratch
        Termination criteria: SUFFICIENT_COST reached; max_layers reached;
        std(last_5_costs)/avg(last_5_costs) < TOL
        :return: {'circuit':resulting circuit(Instruction),
        'overlap':overlap(float),
        'num_1q_gates':number of rotation gates in circuit(int),
        'num_2q_gates':number of entangling gates in circuit(int)}
        'cost_progression': list of costs after each layer is added
        'time_taken': total time taken for recompilation
        'circuit_qasm': QASM string of the resulting circuit
        """
        logger.info("ISL started")
        start_time = timeit.default_timer()
        self.cost_evaluation_counter = 0
        cost, num_1q_gates, num_2q_gates = None, None, None

        cost_history = []
        g_range = self.variational_circuit_range

        already_successful = False
        # If an initial ansatz has been provided, add that and run minimization
        if initial_ansatz is not None:
            co.add_to_circuit(
                self.full_circuit,
                co.circuit_by_inverting_circuit(initial_ansatz),
                g_range()[1],
                transpile_before_adding=True,
            )
            if self.use_rotosolve:
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    tol=1e-3,
                    stop_val=self.isl_config.sufficient_cost,
                    indexes_to_modify=g_range(),
                )
            else:
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_PYBOBYQA,
                    alg_kwargs={"seek_global_minimum": True},
                )
            if cost < self.isl_config.sufficient_cost:
                already_successful = True
                logger.debug(
                    "ISL successfully found approximate circuit using provided ansatz only"
                )

        for layer_count in range(self.isl_config.max_layers):
            # Make sure recompilation already hasn't been completed using initial ansatz
            if already_successful:
                break

            logger.debug(f"Cost before adding layer: {cost}")
            cost = self._add_layer(layer_count)
            cost_history.append(cost)
            if self.remove_unnecessary_gates:
                co.remove_unnecessary_gates_from_circuit(
                    self.full_circuit, False, False, gate_range=g_range()
                )

            num_2q_gates, num_1q_gates = co.find_num_gates(
                self.full_circuit, gate_range=g_range()
            )
            cinl = self.isl_config.cost_improvement_num_layers
            cit = self.isl_config.cost_improvement_tol
            if len(cost_history) >= cinl and has_stopped_improving(
                cost_history[-1 * cinl :], cit
            ):
                logger.debug("ISL stopped improving")
                break

            if cost < self.isl_config.sufficient_cost:
                logger.debug("ISL successfully found approximate circuit")
                break
            elif num_2q_gates >= self.isl_config.max_2q_gates:
                logger.debug("ISL MAX_2Q_GATES reached. Using ROTOSOLVE one last time")
                self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    max_cycles=10,
                    tol=1e-5,
                    stop_val=self.isl_config.sufficient_cost,
                )
                break

        if self.remove_unnecessary_gates:
            co.remove_unnecessary_gates_from_circuit(
                self.full_circuit, True, True, gate_range=g_range()
            )

        # Perform a final optimisation
        if self.perform_final_minimisation:
            self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_PYBOBYQA,
                alg_kwargs={"seek_global_minimum": False},
            )

        num_2q_gates, num_1q_gates = co.find_num_gates(
            self.full_circuit, gate_range=g_range()
        )

        final_cost = self.evaluate_cost()
        end_time = timeit.default_timer()

        recompiled_circuit = self.get_recompiled_circuit()
        exact_overlap = co.calculate_overlap_between_circuits(
            self.circuit_to_recompile, co.make_quantum_only_circuit(recompiled_circuit)
        )
        result_dict = {
            "circuit": recompiled_circuit,
            "overlap": 1 - final_cost,
            "exact_overlap": exact_overlap,
            "num_1q_gates": num_1q_gates,
            "num_2q_gates": num_2q_gates,
            "cost_progression": cost_history,
            "entanglement_measures_progression": self.entanglement_measures_history,
            "e_val_history": self.e_val_history,
            "qubit_pair_history": self.qubit_pair_history,
            "method_history": self.pair_selection_method_history,
            "time_taken": end_time - start_time,
            "cost_evaluations": self.cost_evaluation_counter,
            "coupling_map": self.coupling_map,
            "circuit_qasm": recompiled_circuit.qasm(),
        }
        logger.info("ISL completed")
        return result_dict

    def _add_layer(self, index):
        """
        Adds a dressed CNOT gate to the qubits with the highest local
        entanglement. If all qubit pairs have no
        local entanglement, adds a dressed CNOT gate to the qubit pair with
        the highest sum of expectation values
        (computational basis).
        :return: New cost
        """
        control, target = self._find_appropriate_qubit_pair()
        co.add_to_circuit(
            self.full_circuit,
            self.get_layer_2q_gate(index),
            self.variational_circuit_range()[1],
            qubit_subset=[control, target],
        )
        self.qubit_pair_history.append((control, target))
        # First modify the gates on dressed cnot (with structural learning)
        # then modify all gates (without
        # structural learning)
        entangling_gate_indexes = (
            self.variational_circuit_range()[1] - len(self.layer_2q_gate.data),
            self.variational_circuit_range()[1],
        )
        rotosolve_gate_start_index = max(
            self.variational_circuit_range()[0],
            self.variational_circuit_range()[1]
            - len(self.layer_2q_gate.data) * self.isl_config.max_layers_to_modify,
        )
        rotosolve_gate_indexes = (
            rotosolve_gate_start_index,
            self.variational_circuit_range()[1],
        )

        if self.use_rotosolve:
            cost = self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_ROTOSELECT,
                tol=1e-5,
                stop_val=self.isl_config.sufficient_cost,
                indexes_to_modify=entangling_gate_indexes,
            )
            if index % self.isl_config.rotosolve_frequency == 0:
                cost = self.minimizer.minimize_cost(
                    algorithm_kind=vconstants.ALG_ROTOSOLVE,
                    tol=1e-3,
                    stop_val=self.isl_config.sufficient_cost,
                    indexes_to_modify=rotosolve_gate_indexes,
                )
        else:
            cost = self.minimizer.minimize_cost(
                algorithm_kind=vconstants.ALG_PYBOBYQA,
                alg_kwargs={"seek_global_minimum": True},
            )
        return cost

    def _find_appropriate_qubit_pair(self):
        e_vals = self._measure_qubit_expectation_values()
        self.e_val_history.append(e_vals)
        ems = self._get_all_qubit_pair_entanglement_measures()
        self.entanglement_measures_history.append(ems)
        priorities = self._get_all_qubit_pair_priorities()
        e_val_sums = self._get_all_qubit_pair_e_val_sums(e_vals)

        if self.isl_config.method == "ISL":
            return self._find_highest_entanglement_qubit_pair(
                ems, e_val_sums, priorities
            )
        elif self.isl_config.method == "heuristic":
            return self._find_best_heuristic_qubit_pair(e_val_sums, priorities)
        elif self.isl_config.method == "basic":
            return self._find_best_priority_qubit_pair(priorities)
        elif self.isl_config.method == "random":
            rand_index = np.random.randint(len(self.coupling_map))
            self.pair_selection_method_history.append(f"random")
            return self.coupling_map[rand_index]
        else:
            raise ValueError(
                f"Invalid ISL method {self.isl_config.method}. "
                f"Method must be one of ISL,heuristic,random"
            )

    def _find_highest_entanglement_qubit_pair(
        self, entanglement_measures, e_val_sums, priorities
    ):

        # First check if the previous qubit pair was 'bad'
        if len(self.entanglement_measures_history) >= 2:
            prev_qp_index = self.coupling_map.index(self.qubit_pair_history[-1])
            pre_em = self.entanglement_measures_history[-2][prev_qp_index]
            post_em = self.entanglement_measures_history[-1][prev_qp_index]
            if post_em >= pre_em:
                # Previous qubit pair was bad. Add to bad_qubit_pairs list
                self.bad_qubit_pairs.append(self.coupling_map[prev_qp_index])
            if len(self.bad_qubit_pairs) > self.isl_config.bad_qubit_pair_memory:
                # Maintain max size of bad_qubit_pairs
                del self.bad_qubit_pairs[0]

        filtered_ems = entanglement_measures.copy()
        for qp in set(self.bad_qubit_pairs):
            # Find the number of times this qubit pair has occurred recently
            reps = len(
                [
                    x
                    for x in self.qubit_pair_history[
                        -1 * self.isl_config.bad_qubit_pair_memory :
                    ]
                    if x == qp
                ]
            )
            if reps >= 1:
                filtered_ems[self.coupling_map.index(qp)] = -1
        if len(self.qubit_pair_history) > 0:
            # Avoid using same qubit pair as the one used immediately before
            filtered_ems[self.coupling_map.index(self.qubit_pair_history[-1])] = -1
        if max(filtered_ems) <= 0:
            # No local entanglement detected in non-bad qubit pairs;
            # defer to using 'basic' method
            return self._find_best_heuristic_qubit_pair(e_val_sums, priorities)
        else:
            self.pair_selection_method_history.append(f"ISL")
            return self.coupling_map[np.argmax(filtered_ems)]

    def _find_best_heuristic_qubit_pair(self, e_val_sums, priorities):
        # Choose the qubit pair to be the
        # one with the highest sum of expectation
        # values multiplied by the 'priority' of that pair. The priority
        # of a pair depends on how long ago a CNOT
        # was added to that qubit pair such that the priority is 0 for
        # the previous qubit pair.
        # This is to avoid the circuit falling into loops which don't
        # lead to improvements
        combined_priorities = [
            e_val_sum * priority
            for (e_val_sum, priority) in zip(e_val_sums, priorities)
        ]
        self.pair_selection_method_history.append(f"heuristic")
        return self.coupling_map[np.argmax(combined_priorities)]

    def _find_best_priority_qubit_pair(self, priorities):
        # Choose the qubit pair with the highest priority
        self.pair_selection_method_history.append(f"basic")
        return self.coupling_map[np.argmax(priorities)]

    def _get_all_qubit_pair_entanglement_measures(self):
        entanglement_measures = []
        for control, target in self.coupling_map:
            this_entanglement_measure = calculate_entanglement_measure(
                self.entanglement_measure_method,
                self.full_circuit,
                control,
                target,
                self.backend,
                self.backend_options,
                self.execute_kwargs,
            )
            entanglement_measures.append(this_entanglement_measure)
        return entanglement_measures

    def _get_all_qubit_pair_e_val_sums(self, e_vals):
        e_val_sums = []
        for control, target in self.coupling_map:
            e_val_sums.append(e_vals[control] + e_vals[target])
        return e_val_sums

    def _get_all_qubit_pair_priorities(self):
        priorities = []
        for qp in self.coupling_map:
            priorities.append(self._get_priority_of_qubit_pair(qp))
        return priorities

    def _get_priority_of_qubit_pair(self, qubit_pair):
        """
        Priority is determined by the distance to the previous occurrence of
        the qubit pair.
        Exponential dependence is such that the last occurrence has priority
        0, 2nd last has priority 0.5, etc.
        If a qubit pair has not been used it will have priority 1
        """
        rev_chronological_cnot_locations = self.qubit_pair_history[::-1]
        try:
            loc = rev_chronological_cnot_locations.index(qubit_pair)
            priority = 1 - np.exp2(-1 * loc)
            return priority
        except ValueError as _:
            return 1

    def _measure_qubit_expectation_values(self):
        if self.local_measurements_only:
            orig_circ_len = len(self.full_circuit.data)
            if not is_statevector_backend(self.backend):
                self.full_circuit.measure_all()
            counts = self._run_full_circuit()
            if not is_statevector_backend(self.backend):
                for i in range(len(self.full_circuit.data) - 1, orig_circ_len - 1, -1):
                    del self.full_circuit.data[i]
                del self.full_circuit.cregs[-1]
            rel_counts = {k[0 : self.total_num_qubits]: v for k, v in counts.items()}
            return expectation_value_of_qubits(rel_counts)
        else:
            counts = self._run_full_circuit()
            return expectation_value_of_qubits(counts)
