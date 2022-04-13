"""Contains functions """
import functools
from collections.abc import Iterable

import numpy as np

# ------------------Trigonometric functions------------------ #


def minimum_of_sinusoidal(value_0, value_pi_by_2, value_minus_pi_by_2):
    """
    Find the minimum of a sinusoidal function with period 2*pi and of the
    form f(x) = a*sin(x+b)+c
    :param value_0: f(0)
    :param value_pi_by_2: f(pi/2)
    :param value_minus_pi_by_2: f(-pi/2)
    :return: (x_min, f(x_min))
    """
    theta_min = -(np.pi / 2) - np.arctan2(
        2 * value_0 - value_pi_by_2 - value_minus_pi_by_2,
        value_pi_by_2 - value_minus_pi_by_2,
    )

    theta_min = normalized_angles(theta_min)

    intercept_c = 0.5 * (value_pi_by_2 + value_minus_pi_by_2)
    value_pi = (value_pi_by_2 + value_minus_pi_by_2) - value_0
    amplitude_a = 0.5 * (
        ((value_0 - value_pi) ** 2 + (value_pi_by_2 - value_minus_pi_by_2) ** 2) ** 0.5
    )
    value_theta_min = intercept_c - amplitude_a

    return theta_min, value_theta_min


def amplitude_of_sinusoidal(value_0, value_pi_by_2, value_minus_pi_by_2):
    """
    Find the amplitude of a sinusoidal function with period 2*pi and of the
    form f(x) = a*sin(x+b)+c
    :param value_0: f(0)
    :param value_pi_by_2: f(pi/2)
    :param value_minus_pi_by_2: f(-pi/2)
    :return: Amplitude
    """

    value_pi = (value_pi_by_2 + value_minus_pi_by_2) - value_0
    amplitude_a = 0.5 * (
        ((value_0 - value_pi) ** 2 + (value_pi_by_2 - value_minus_pi_by_2) ** 2) ** 0.5
    )

    return amplitude_a


def derivative_of_sinusoidal(theta, value_0, value_pi_by_2, value_minus_pi_by_2):
    """
    Find the derivative of a sinusoidal function with period 2*pi and of the
    form f(x) = a*sin(x+b)+c at x=theta
    :param theta: Angle at which derivative is to be evaluated
    :param value_0: f(0)
    :param value_pi_by_2: f(pi/2)
    :param value_minus_pi_by_2: f(-pi/2)
    :return: df(x)/dx at x=theta
    """
    value_pi = (value_pi_by_2 + value_minus_pi_by_2) - value_0
    amplitude_a = 0.5 * (
        ((value_0 - value_pi) ** 2 + (value_pi_by_2 - value_minus_pi_by_2) ** 2) ** 0.5
    )
    phase_b = np.arctan2(value_0 - value_pi, value_pi_by_2 - value_minus_pi_by_2)

    derivative = amplitude_a * np.cos(theta + phase_b)
    return derivative


def normalized_angles(angles):
    """
    Normalize angle(s) to between -pi, pi by adding/subtracting multiples of
    2pi
    :param angles: float or Iterable(float)
    :return: float or Iterable(float)
    """
    single = not isinstance(angles, Iterable)
    if single:
        angles = [angles]
    new_angles = []
    for angle in angles:
        while (angle > np.pi) or (angle < -np.pi):
            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle < np.pi:
                angle += 2 * np.pi
        new_angles += [angle]
    return new_angles[0] if single else new_angles


# ------------------Misc. functions------------------ #


def is_statevector_backend(backend):
    """
    Check if backend is a statevector simulator backed
    :param backend: Simulator backend
    :return: Boolean
    """
    if backend == "qulacs":
        return True
    return backend.name().startswith("statevector")


def counts_data_from_statevector(
    statevector,
    num_shots=2**40,
):
    """
    Get counts data from statevector by multiplying amplitude squares with num_shots.
    Note: Doesn't guarantee total number of shots in returned counts data will be num_shots.
    Warning: Doesn't work well if num_shots << number of non-zero elements in statevector
    :param statevector: Statevector (list/array)
    :return: Counts data (e.g. {'00':13, '10':7}) with bitstrings ordered
        with decreasing qubit number
    """
    num_qubits = int(np.log2(len(statevector)))
    counts = {}
    probs = np.absolute(statevector) ** 2
    bit_str_array = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]
    counts = dict(zip(bit_str_array, np.asarray(probs * num_shots, int)))
    # counts = dict(zip(*np.unique(np.random.choice(bit_str_array, num_shots,p=probs),return_counts=True)))
    return counts


# TODO: Add test
def statevector_from_counts_data(counts):
    """
    Get statevector from counts (works only for real, positive states)
    :param: Counts data (e.g. {'00':13, '10':7})
    :return statevector: Statevector (list/array)
    """
    num_qubits = len(list(counts.keys())[0])
    sv = np.zeros(2**num_qubits)
    for i in range(2**num_qubits):
        bitstr = bin(i)[2:].zfill(num_qubits)
        if bitstr in counts:
            sv[i] = counts[bitstr] ** 0.5
    sv /= np.linalg.norm(sv)
    return sv


def expectation_value_of_qubits(counts):
    """
    Expectation value of qubits (in computational basis)
    :param counts: Counts data (e.g. {'00':13, '10':7})
    :return: [expectation_value(float)]
    """
    num_qubits = len(list(counts)[0])
    expectation_values = []
    for i in range(num_qubits):
        expectation_values.append(expectation_value_of_qubit(i, counts))
    return expectation_values


def expectation_value_of_qubit(qubit_index, counts):
    """
    Expectation value of qubit (in computational basis) at given index
    :param qubit_index: Index of qubit (int)
    :param counts: Counts data (e.g. {'00':13, '10':7})
    :return: [expectation_value(float)]
    """
    exp_val = 0
    total_counts = 0
    reverse_index = len(list(counts)[0]) - (qubit_index + 1)
    for state in list(counts):
        exp_val += int(state[reverse_index]) * counts[state]
        total_counts += counts[state]

    return exp_val / total_counts


def expectation_value_of_pauli_observable(counts, pauli):
    """
    Copied from measure_pauli_z in qiskit.aqua.operators.common

    Args:
        counts (dict): a dictionary of the form counts = {'00000': 10} ({
            str: int})
        pauli (Pauli): a Pauli object
    Returns:
        float: Expected value of paulis given data
    """
    observable = 0.0
    num_shots = sum(counts.values())
    p_z_or_x = np.logical_or(pauli.z, pauli.x)
    for key, value in counts.items():
        bitstr = np.asarray(list(key))[::-1].astype(np.bool)
        sign = (
            -1.0
            if functools.reduce(np.logical_xor, np.logical_and(bitstr, p_z_or_x))
            else 1.0
        )
        observable += sign * value
    observable /= num_shots
    return observable


def remove_permutations_from_coupling_map(coupling_map):
    set_of_sets = {frozenset(pair) for pair in coupling_map}
    new_coupling_map = [tuple(pair) for pair in set_of_sets]
    return new_coupling_map


def has_stopped_improving(cost_history, rel_tol=1e-2):
    try:
        poly_fit_res = np.polyfit(list(range(len(cost_history))), cost_history, 1)
        grad = poly_fit_res[0] / np.absolute(np.mean(cost_history))
        return grad > -1 * rel_tol
    except np.linalg.LinAlgError:
        return False
