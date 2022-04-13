"""Contains CostMinimiser"""
import logging
from typing import Tuple

import numpy as np
from scipy.optimize import minimize

import isl.utils.circuit_operations as co
import isl.utils.constants as vconstants
from isl.utils.circuit_operations import SUPPORTED_1Q_GATES
from isl.utils.utilityfunctions import (
    derivative_of_sinusoidal,
    has_stopped_improving,
    minimum_of_sinusoidal,
)

logger = logging.getLogger(__name__)


class CostMinimiser:
    """
    Minimizer that minimizes a cost function
    """

    def __init__(self, cost_finder, variational_circuit_range, full_circuit):
        """
        :param cost_finder: Callable that returns cost(float)
        """
        self.cost_finder = cost_finder
        self.variational_circuit_range = variational_circuit_range
        self.full_circuit = full_circuit

    def minimize_cost(
        self,
        algorithm_kind=vconstants.ALG_ROTOSOLVE,
        algorithm_identifier=None,
        max_cycles=1000,
        stop_val=-np.inf,
        tol=1e-10,
        indexes_to_modify=None,
        alg_kwargs=None,
    ):
        """
        Minimize the cost by varying rotation gate angles (and axes in case
        of ALG_ROTOSELECT). Gates with label
        FIXED_GATE_LABEL will not be varied.
        :param algorithm_kind:
        :param algorithm_identifier:
        :param max_cycles: For ALG_ROTOSOLVE,ALG_ROTOSELECT, this is the max
            number of cycles
        :param stop_val: Minimization will stop when this value is reached
        :param tol: Tolerance (float). Difference algorithms have different
            implementations of this value
        :param indexes_to_modify: If not None, only gates with the given
            indexes (index of gate in variational_circuit.data) will be varied
            (only valid for rotosolve/rotoselect)
        :param alg_kwargs: Keyword arguments supplied to particular optimiser
        :return:
        """
        if alg_kwargs is None:
            alg_kwargs = {}
        if (
            algorithm_kind == vconstants.ALG_ROTOSOLVE
            or algorithm_kind == vconstants.ALG_ROTOSELECT
        ):
            cost_history = []
            cost = self.cost_finder()
            cycles = 0
            while cost > stop_val and cycles < max_cycles:
                cost = self._reduce_cost(
                    algorithm_kind == vconstants.ALG_ROTOSELECT, indexes_to_modify
                )
                cycles += 1
                cost_history.append(cost)
                if len(cost_history) > 3 and has_stopped_improving(
                    cost_history[-3:], tol
                ):
                    break
            if algorithm_kind == vconstants.ALG_ROTOSOLVE:
                alg_name = "ROTOSOLVE"
            else:
                alg_name = "ROTOSELECT"

            logger.debug(f"{alg_name} finished with cost {cost}")
            return cost

        elif algorithm_kind == vconstants.ALG_NLOPT:
            try:
                import nlopt
            except ModuleNotFoundError as e:
                logger.error(
                    "NLOPT not installed. Use 'conda install -c conda-forge "
                    "nlopt' to install nlopt using conda"
                )
                raise e
            initial_angles = co.find_angles_in_circuit(
                self.full_circuit, self.variational_circuit_range()
            )
            if len(initial_angles) == 0:
                return self.cost_finder()
            # Setup optimizer
            opt = nlopt.opt(algorithm_identifier, len(initial_angles))
            opt.set_upper_bounds([np.pi] * len(initial_angles))
            opt.set_lower_bounds([-np.pi] * len(initial_angles))
            opt.set_stopval(stop_val)
            opt.set_ftol_rel(tol)
            opt.set_xtol_abs(1e-10)
            opt.set_min_objective(self._find_cost_with_angles)

            # Start optimization
            try:
                final_angles = opt.optimize(initial_angles)
            except RuntimeError as e:
                logger.error(f"Nlopt optimisation failed")
                raise e

            co.update_angles_in_circuit(
                self.full_circuit, final_angles, self.variational_circuit_range()
            )
            return opt.last_optimum_value()

        elif algorithm_kind == vconstants.ALG_SCIPY:
            initial_angles = co.find_angles_in_circuit(
                self.full_circuit, self.variational_circuit_range()
            )
            optimization_result = minimize(
                fun=self._find_cost_with_angles,
                method=algorithm_identifier,
                x0=initial_angles,
                tol=tol,
                **alg_kwargs,
            )
            co.update_angles_in_circuit(
                self.full_circuit,
                optimization_result["x"],
                self.variational_circuit_range(),
            )
            return optimization_result["fun"]
        elif algorithm_kind == vconstants.ALG_PYBOBYQA:
            try:
                import pybobyqa
            except ModuleNotFoundError as e:
                logger.error(
                    "PyBOBYQA not installed. Use 'pip install Py-BOBYQA' "
                    "to install using pip"
                )
                raise e

            initial_angles = co.find_angles_in_circuit(
                self.full_circuit, self.variational_circuit_range()
            )
            bounds = ([-np.pi] * len(initial_angles), [np.pi] * len(initial_angles))
            try:
                result = pybobyqa.solve(
                    self._find_cost_with_angles,
                    initial_angles,
                    bounds=bounds,
                    objfun_has_noise=True,
                    print_progress=False,
                    do_logging=False,
                    **alg_kwargs,
                )
                co.update_angles_in_circuit(
                    self.full_circuit, result.x, self.variational_circuit_range()
                )
                return result.f
            except Exception as e:
                logger.debug(f"BOBYQA failed with exception: {e}")
                co.update_angles_in_circuit(
                    self.full_circuit, initial_angles, self.variational_circuit_range()
                )
                return self.cost_finder()
        else:
            raise ValueError(f"Invalid algorithm kind {algorithm_kind}")

    def try_escaping_periodic_local_minimum(
        self, gap_between_minima, first_minima_loc, penalty_amp=0.1
    ):
        initial_cost = self.cost_finder()
        initial_angles = co.find_angles_in_circuit(
            self.full_circuit, self.variational_circuit_range()
        )
        num_attempts = 5
        stochastic_param = 1

        def find_cost_with_penalty_for_angles(angles, grad=None):
            cost = self._find_cost_with_angles(angles, grad)
            # Create a sinusoidally varying potential that has maxima at the
            # local minima locations
            penalty = penalty_amp * np.cos(
                np.pi
                + (
                    (cost - first_minima_loc)
                    * 2
                    * np.pi
                    * (1 / gap_between_minima)
                    * stochastic_param
                )
            )
            return cost + penalty

        actual_cost = initial_cost
        for i in range(num_attempts):
            res = minimize(
                find_cost_with_penalty_for_angles, initial_angles, method="Nelder-Mead"
            )
            final_angles = res.x

            co.update_angles_in_circuit(
                self.full_circuit, final_angles, self.variational_circuit_range()
            )
            cost_with_penalty = res.fun

            co.update_angles_in_circuit(
                self.full_circuit, final_angles, self.variational_circuit_range()
            )
            actual_cost = self.cost_finder()
            logging.debug(
                f"{i}th Attempt to escape minima: initial cost = "
                f"{initial_cost}, final cost with penalty "
                f"= {cost_with_penalty}, "
                f"actual final cost = {actual_cost}"
            )
            stochastic_param = np.random.random() * 10
            if actual_cost < initial_cost:
                break
        return actual_cost

    def _find_cost_with_angles(self, angles, grad=None):
        """
        Find the cost with self.full_circuit with the given angles.
        This method changes the angles of self.full_circuit
        :param angles: New angles
        :param grad: Gradient of circuit (used by gradient-based optimizers)
        which is modified in place
        :return: Cost (float)
        """
        co.update_angles_in_circuit(
            self.full_circuit, angles, self.variational_circuit_range()
        )
        if grad is not None and grad.size > 0:
            self._update_gradient_of_circuit(grad)
        cost = self.cost_finder()
        return cost

    def _reduce_cost(
        self, change_1q_gate_kind=False, indexes_to_modify: Tuple[int, int] = None
    ):
        """
        For each gate in the full circuit, find the optimal angle (and gate
        kind) while keeping all other gates fixed.
        Sequentially cycles over gates w.r.t their index in circuit.data
        :param change_1q_gate_kind: If true, the optimal gate kind (
        rx/ry/rz) will be chosen for each gate
        :param indexes_to_modify: If not None, all gates except those at
        specified indexes will be fixed.
        Indexes are relative to full_circuit.data
        :return: New cost
        """
        cost = 1
        variational_circuit_range = self.variational_circuit_range()
        if indexes_to_modify is None:
            indexes_to_modify = variational_circuit_range
        else:
            indexes_to_modify = (
                max(indexes_to_modify[0], variational_circuit_range[0]),
                min(indexes_to_modify[1], variational_circuit_range[1]),
            )
        for index in range(*indexes_to_modify):
            old_gate = self.full_circuit.data[index][0]

            if change_1q_gate_kind and co.is_supported_1q_gate(old_gate):
                cost = self.replace_with_best_1q_gate(index)
            elif co.is_supported_1q_gate(old_gate):
                angle, cost = self.find_best_angle(index, old_gate.label)
                co.replace_1q_gate(self.full_circuit, index, old_gate.label, angle)
            else:
                continue
        return cost

    def replace_with_best_1q_gate(self, gate_index):
        """
        Find the gate which results in the lowest cost and replace the gate
        at gate_index with the best gate
        :param gate_index: The index of the gate that is to be replaced
        :return: New cost
        """
        # Find cost at 0 angle separately because it is the same regardless
        # of gate kind
        co.replace_1q_gate(self.full_circuit, gate_index, "rx", 0)
        cost_identity = self.cost_finder()
        best_gate_name, best_gate_angle, best_gate_cost = None, None, 1
        for gate_name in SUPPORTED_1Q_GATES:
            min_angle, cost = self.find_best_angle(gate_index, gate_name, cost_identity)
            if cost < best_gate_cost:
                best_gate_name, best_gate_angle, best_gate_cost = (
                    gate_name,
                    min_angle,
                    cost,
                )
        co.replace_1q_gate(
            self.full_circuit, gate_index, best_gate_name, best_gate_angle
        )
        return best_gate_cost

    def find_best_angle(self, gate_index, gate_name, cost_for_identity=None):
        """
        Find the angle of the specified gate which results in the lowest cost
        :param gate_index: The index of the gate that is to be checked
        :param gate_name: Name of the gate kind that is to be used
        :param cost_for_identity: The cost when the angle is 0
        :return: best_gate_angle, best_cost
        """
        # Remember original gate
        old_gate, qargs, cargs = self.full_circuit.data[gate_index]

        costs = []
        angles_to_run = [0, np.pi / 2, -np.pi / 2]
        if cost_for_identity is not None:
            costs.append(cost_for_identity)
            angles_to_run.remove(0)

        for theta in angles_to_run:
            co.replace_1q_gate(self.full_circuit, gate_index, gate_name, theta)
            costs.append(self.cost_finder())
        theta_min, cost_min = minimum_of_sinusoidal(costs[0], costs[1], costs[2])

        # Replace with original gate
        self.full_circuit.data[gate_index] = (old_gate, qargs, cargs)
        return theta_min, cost_min

    def _update_gradient_of_circuit(self, grad, method="parameter_shift"):
        """
        Evaluates the gradient of the circuit (list of partial derivatives
        of cost w.r.t each rotation angle)
        :param grad: Old gradient (modified in place)
        """
        angles = co.find_angles_in_circuit(self.full_circuit)
        angle_index = 0
        for gate_index in range(*self.variational_circuit_range()):
            gate, _, _ = self.full_circuit.data[gate_index]
            if co.is_supported_1q_gate(gate):
                # Calculate partial derivative
                if method == "parameter_shift":
                    r = 0.5
                    shift = np.pi * (1 / (4 * r))
                    current_angle = angles[angle_index]
                    co.replace_1q_gate(
                        self.full_circuit, gate_index, gate.label, current_angle + shift
                    )
                    value_plus = self.cost_finder()
                    co.replace_1q_gate(
                        self.full_circuit, gate_index, gate.label, current_angle - shift
                    )
                    value_minus = self.cost_finder()

                    grad[angle_index] = r * (value_plus - value_minus)

                else:
                    co.replace_1q_gate(self.full_circuit, gate_index, gate.label, 0)
                    value_0 = self.cost_finder()
                    co.replace_1q_gate(
                        self.full_circuit, gate_index, gate.label, np.pi / 2
                    )
                    value_pi_by_2 = self.cost_finder()
                    co.replace_1q_gate(
                        self.full_circuit, gate_index, gate.label, -np.pi / 2
                    )
                    value_minus_pi_by_2 = self.cost_finder()

                    grad[angle_index] = derivative_of_sinusoidal(
                        angles[angle_index], value_0, value_pi_by_2, value_minus_pi_by_2
                    )

                # Return circuit back to original
                co.replace_1q_gate(
                    self.full_circuit, gate_index, gate.label, angles[angle_index]
                )

                angle_index += 1
