"""
Contains ApproximateRecompiler
"""
import logging
import multiprocessing
import os
import timeit
from abc import ABC, abstractmethod

import numpy as np
from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister

import isl.utils.circuit_operations as co
from isl.utils.circuit_operations.circuit_operations_full_circuit import (
    remove_classical_operations,
)
from isl.utils.cost_minimiser import CostMinimiser
from isl.utils.utilityfunctions import is_statevector_backend

logger = logging.getLogger(__name__)


class ApproximateRecompiler(ABC):
    """
    Variational hybrid quantum-classical algorithm that recompiles a given
    circuit into another circuit. The new circuit
    has the same result when acting on the given input state as the given
    circuit.
    """

    def __init__(
        self,
        circuit_to_recompile: QuantumCircuit,
        backend,
        execute_kwargs=None,
        initial_state=None,
        qubit_subset=None,
        general_initial_state=False,
        starting_circuit=None,
        local_measurements_only=False,
    ):
        """
        :param circuit_to_recompile: Circuit that is to be recompiled
        :param backend: Backend that is to be used
        :param execute_kwargs: keyword arguments passed into circuit runs (
        excluding backend)
        e.g. {'noise_model:NoiseModel, shots=10000}
        :param initial_state: Initial state that circuits act on.
        Can be a circuit (QuantumCircuit/Instruction) or vector (
        list/np.ndarray) or None
        :param qubit_subset: The subset of qubits (relative to initial state
        circuit) that circuit_to_recompile acts
        on. If None, it will be assumed that circuit_to_recompile and
        initial_state circuit have the same qubits
        :param general_initial_state: Whether recompilation should be for a
        general initial state
        """
        self.original_circuit = circuit_to_recompile
        self.original_circuit_classical_ops = None
        self.circuit_to_recompile = self.prepare_circuit()
        self.backend = (
            backend if backend is not None else Aer.get_backend("qasm_simulator")
        )
        self.execute_kwargs = self.parse_default_execute_kwargs(execute_kwargs)
        self.backend_options = self.parse_default_backend_options()
        self.initial_state_circuit = co.initial_state_to_circuit(initial_state)
        self.total_num_qubits = self.calculate_total_num_qubits()
        self.qubit_subset_to_recompile = (
            qubit_subset if qubit_subset else list(range(self.total_num_qubits))
        )
        self.general_initial_state = general_initial_state
        self.starting_circuit = starting_circuit
        self.local_measurements_only = local_measurements_only
        if initial_state is not None and general_initial_state:
            raise ValueError(
                "Can't recompile for general initial state when specific "
                "initial state is provided"
            )

        (
            self.full_circuit,
            self.lhs_gate_count,
            self.rhs_gate_count,
        ) = self._prepare_full_circuit()
        self.minimizer = CostMinimiser(
            self.evaluate_cost, self.variational_circuit_range, self.full_circuit
        )

        # Count number of cost evaluations
        self.cost_evaluation_counter = 0

    def prepare_circuit(self):
        """
        TODO Add description
        """
        circuit_to_recompile_copy = self.original_circuit.copy()
        self.original_circuit_classical_ops = remove_classical_operations(
            circuit_to_recompile_copy
        )
        qc2 = QuantumCircuit(len(self.original_circuit.qubits))
        qc2.append(
            co.make_quantum_only_circuit(circuit_to_recompile_copy).to_instruction(),
            qc2.qregs[0],
        )
        prepared_circuit = co.unroll_to_basis_gates(qc2)
        return prepared_circuit

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

    def parse_default_backend_options(self):
        backend_options = {}
        if (
            "noise_model" in self.execute_kwargs
            and self.execute_kwargs["noise_model"] is not None
        ):
            backend_options["method"] = "automatic"
        else:
            backend_options["method"] = "automatic"

        try:
            if os.environ["QISKIT_IN_PARALLEL"] == "TRUE":
                # Already in parallel
                backend_options["max_parallel_experiments"] = 1
            else:
                num_threads = multiprocessing.cpu_count()
                backend_options["max_parallel_experiments"] = num_threads
                logging.debug(
                    "Circuits will be evaluated with {} experiments in "
                    "parallel".format(num_threads)
                )
                os.environ["KMP_WARNINGS"] = "0"

        except KeyError:
            logging.debug(
                "No OMP number of threads defined. Qiskit will autodiscover "
                "the number of parallel shots to run"
            )
        return backend_options

    def calculate_total_num_qubits(self):
        if self.initial_state_circuit is None:
            total_num_qubits = self.circuit_to_recompile.num_qubits
        else:
            total_num_qubits = self.initial_state_circuit.num_qubits
        return total_num_qubits

    def variational_circuit_range(self):
        return self.lhs_gate_count, len(self.full_circuit.data) - self.rhs_gate_count

    @abstractmethod
    def recompile(self) -> dict:
        """
        Run the recompilation algorithm
        :return: Dictionary object containing the resulting circuit,
        the overlap between original and resulting circuit,
        and other optional entries (such as circuit parameters). {
        'circuit':Instruction, 'overlap': float}
        """
        raise NotImplementedError(
            "A recompiler must provide implementation for the recompile() " "method"
        )

    def recompile_in_parts(self, max_depth_per_block=10):
        """
        Recompiles the circuit using the following procedure: First break
        the circuit into n subcircuits.
        Then iteratively find an approximation recompilation for the first
        m+1 subcircuits by finding an approximate
        of (approx_circuit_for_first_m_subcircuits + (m+1)th subcircuit)
        :param max_depth_per_block: The maximum allowed depth of each of the
        n subcircuits
        :return: Dictionary object containing the resulting circuit, overlap
        between original and approximate circuit,
        and other entries (such as circuit parameters). {
        'circuit':Instruction, 'overlap': float}
        """
        logger.info("Started partial recompilation")
        start_time = timeit.default_timer()

        all_subcircuits = co.vertically_divide_circuit(
            self.circuit_to_recompile.copy(), max_depth_per_block
        )
        last_recompiled_subcircuit = None
        individual_results = []
        for subcircuit in all_subcircuits:
            co.replace_inner_circuit(
                self.full_circuit,
                last_recompiled_subcircuit,
                self.variational_circuit_range(),
                True,
                {"backend": self.backend},
            )
            co.add_to_circuit(
                self.full_circuit,
                subcircuit,
                self.variational_circuit_range()[1],
                True,
                {"backend": self.backend},
            )
            partial_recompilation_result = self.recompile()
            last_recompiled_subcircuit = partial_recompilation_result["circuit"]
            del partial_recompilation_result["circuit"]
            individual_results.append(partial_recompilation_result)
            percentage = (
                100 * (1 + all_subcircuits.index(subcircuit)) / len(all_subcircuits)
            )
            logger.debug(f"Completed {percentage}%  of recompilation")

        end_time = timeit.default_timer()
        result_dict = {
            "circuit": last_recompiled_subcircuit,
            "overlap": co.calculate_overlap_between_circuits(
                last_recompiled_subcircuit,
                self.circuit_to_recompile,
                self.initial_state_circuit,
                self.qubit_subset_to_recompile,
            ),
            "individual_results": individual_results,
            "time_taken": end_time - start_time,
        }
        return result_dict

    def get_recompiled_circuit(self):
        recompiled_circuit = co.circuit_by_inverting_circuit(
            co.extract_inner_circuit(
                self.full_circuit, self.variational_circuit_range()
            )
        )
        if self.starting_circuit is not None:
            co.add_to_circuit(
                recompiled_circuit,
                self.starting_circuit,
                0,
                transpile_before_adding=True,
                transpile_kwargs={"backend": self.backend},
            )
        final_circuit = QuantumCircuit(
            *self.circuit_to_recompile.qregs, *self.circuit_to_recompile.cregs
        )
        qubit_map = {
            full_circ_index: subset_index
            for subset_index, full_circ_index in enumerate(
                self.qubit_subset_to_recompile
            )
        }
        co.add_to_circuit(final_circuit, recompiled_circuit, qubit_subset=qubit_map)

        final_circuit_original_regs = QuantumCircuit(
            *self.original_circuit.qregs, *self.original_circuit.cregs
        )
        final_circuit_original_regs.append(
            final_circuit.to_instruction(), final_circuit_original_regs.qubits
        )
        circuit_no_classical_ops = co.unroll_to_basis_gates(final_circuit_original_regs)
        co.add_classical_operations(
            circuit_no_classical_ops, self.original_circuit_classical_ops
        )
        return circuit_no_classical_ops

    def _prepare_full_circuit(self):
        """Circuit is of form:
        -|0>--|initial_state|--|circuit_to_recompile
        |--|variational_circuit|--|initial_state_inverse|--|(measure)|
        With this circuit, the overlap between circuit_to_recompile and
        inverse of full_circuit
        w.r.t initial_state is just the probability of resulting state
        being in all zero |00...00> state
        If self.general_initial_state is true, circuit takes a different
        form described in the papers below.
        (refer to arXiv:1811.03147, arXiv:1908.04416)
        """
        total_qubits = (
            2 * self.total_num_qubits
            if self.general_initial_state
            else self.total_num_qubits
        )
        qr = QuantumRegister(total_qubits)
        qc = QuantumCircuit(qr)

        if self.initial_state_circuit is not None:
            co.add_to_circuit(
                qc,
                self.initial_state_circuit,
                transpile_before_adding=True,
                transpile_kwargs={"backend": self.backend},
            )
        elif self.general_initial_state:
            for qubit in range(self.total_num_qubits):
                qc.h(qubit)
                qc.cx(qubit, qubit + self.total_num_qubits)

        co.add_to_circuit(
            qc,
            self.circuit_to_recompile,
            transpile_before_adding=False,
            qubit_subset=self.qubit_subset_to_recompile,
        )

        lhs_gate_count = len(qc.data)

        if self.initial_state_circuit is not None:
            isc = co.unroll_to_basis_gates(self.initial_state_circuit)
            co.remove_reset_gates(isc)
            co.add_to_circuit(
                qc,
                isc.inverse(),
                transpile_before_adding=True,
                transpile_kwargs={"backend": self.backend},
            )
        if self.starting_circuit is not None:
            co.add_to_circuit(
                qc,
                self.starting_circuit.inverse(),
                transpile_before_adding=True,
                transpile_kwargs={"backend": self.backend},
            )
        elif self.general_initial_state:
            for qubit in range(self.total_num_qubits - 1, -1, -1):
                qc.cx(qubit, qubit + self.total_num_qubits)
                qc.h(qubit)

        if not is_statevector_backend(self.backend):
            if self.local_measurements_only:
                register_size = 2 if self.general_initial_state else 1
                qc.add_register(
                    ClassicalRegister(register_size, name="recompiler_creg")
                )
            else:
                qc.add_register(ClassicalRegister(total_qubits, name="recompiler_creg"))
                [qc.measure(i, i) for i in range(total_qubits)]

        rhs_gate_count = len(qc.data) - lhs_gate_count

        return qc, lhs_gate_count, rhs_gate_count

    def evaluate_cost(self):
        """
        Run the full circuit and evaluate the overlap.
        The cost function is the Loschmidt Echo Test.
        (refer to arXiv:1811.03147, arXiv:1908.04416)
        :return: Cost (float)
        """
        self.cost_evaluation_counter += 1
        if self.local_measurements_only:
            return self._evaluate_cost_measure_local()
        else:
            return self._evaluate_cost_measure_all()

    def _evaluate_cost_measure_all(self):
        counts = self._run_full_circuit()
        total_qubits = (
            2 * self.total_num_qubits
            if self.general_initial_state
            else self.total_num_qubits
        )
        all_zero_string = "".join(str(int(e)) for e in np.zeros(total_qubits))
        total_shots = sum([each_count for _, each_count in counts.items()])
        # '00...00' might not be present in counts if no shot resulted in
        # the ground state
        if all_zero_string in counts:
            overlap = counts[all_zero_string] / total_shots
        else:
            overlap = 0
        cost = 1 - overlap
        return cost

    def _evaluate_cost_measure_local(self):
        qubit_costs = np.zeros(self.total_num_qubits)
        if is_statevector_backend(self.backend):
            counts = self._run_full_circuit()
            total_shots = sum([each_count for _, each_count in counts.items()])
            for i in range(self.total_num_qubits):
                if self.general_initial_state:
                    overlap = (
                        sum(
                            [
                                count
                                for bit_str, count in counts.items()
                                if bit_str[-1 * (1 + i)] == "0"
                                and bit_str[-1 * (1 + i + self.total_num_qubits)] == "0"
                            ]
                        )
                        / total_shots
                    )
                    qubit_costs[i] = 1 - overlap
                else:
                    overlap = (
                        sum(
                            [
                                count
                                for bit_str, count in counts.items()
                                if bit_str[-1 * (1 + i)] == "0"
                            ]
                        )
                        / total_shots
                    )
                    qubit_costs[i] = 1 - overlap
            cost = np.mean(qubit_costs)
            return cost
        for i in range(self.total_num_qubits):
            if self.general_initial_state:
                self.full_circuit.measure(i, 0)
                self.full_circuit.measure(i + self.total_num_qubits, 1)
                counts = self._run_full_circuit()
                del self.full_circuit.data[-1]
                del self.full_circuit.data[-1]
                total_shots = sum([each_count for _, each_count in counts.items()])
                # '00...00' might not be present in counts if no shot
                # resulted in the ground state
                if "00" in counts:
                    overlap = counts["00"] / total_shots
                else:
                    overlap = 0
                qubit_costs[i] = 1 - overlap
            else:
                self.full_circuit.measure(i, 0)
                counts = self._run_full_circuit()
                del self.full_circuit.data[-1]
                total_shots = sum([each_count for _, each_count in counts.items()])
                # '00...00' might not be present in counts if no shot
                # resulted in the ground state
                if "0" in counts:
                    overlap = counts["0"] / total_shots
                else:
                    overlap = 0
                qubit_costs[i] = 1 - overlap
        cost = np.mean(qubit_costs)
        return cost

    def _run_full_circuit(self):
        """
        Run the full circuit
        :rtype: dict
        :return: counts_data or [counts_data] (e.g. counts_data = ['000':10,
        '010':31,'011':20,'110':40])
        """

        # Don't parallelise shots if ISl is already being run in parallel
        already_in_parallel = os.environ["QISKIT_IN_PARALLEL"] == "TRUE"
        backend_options = None if already_in_parallel else self.backend_options

        counts = co.run_circuit_without_transpilation(
            self.full_circuit, self.backend, backend_options, self.execute_kwargs
        )

        return counts
