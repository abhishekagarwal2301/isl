"""Contains functions to measure quantum correlations"""
import copy
import itertools
import logging

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit import quantum_info as qi
from qiskit.ignis.verification.tomography import (
    TomographyFitter,
    state_tomography_circuits,
)
from qiskit.providers.aer.backends.aerbackend import AerBackend
from scipy import linalg
from scipy.linalg import eig

import isl.utils.circuit_operations as co
from isl.utils.utilityfunctions import is_statevector_backend

logger = logging.getLogger(__name__)

EM_OBSERVABLE_CONCURRENCE_LOWER_BOUND = "EM_OBSERVABLE_CONCURRENCE_LOWER_BOUND"
EM_TOMOGRAPHY_EOF = "EM_TOMOGRAPHY_EOF"
EM_TOMOGRAPHY_CONCURRENCE = "EM_TOMOGRAPHY_CONCURRENCE"
EM_TOMOGRAPHY_NEGATIVITY = "EM_TOMOGRAPHY_NEGATIVITY"


def calculate_entanglement_measure(
    method,
    circuit,
    qubit_1,
    qubit_2,
    backend,
    backend_options=None,
    execute_kwargs=None,
):
    """
    Measure quantum correlations between two qubits in a state resulting
    from running a given
    QuantumCircuit
    :param method: Which entanglement measure/method to use
    :param circuit: QuantumCircuit
    :param qubit_1: Index of first qubit
    :param qubit_2: Index of second qubit
    :param backend: Backend on which circuits are to be run. Not relevant
    for tomography based
    entanglement measures.
    Note that observable based entanglement measures can't run on
    statevector_simulators
    :param backend_options:
    :param execute_kwargs:
    :return: Value of quantum correlation
    """
    if method == EM_OBSERVABLE_CONCURRENCE_LOWER_BOUND:
        return measure_concurrence_lower_bound(
            circuit, qubit_1, qubit_2, backend, backend_options, execute_kwargs
        )
    else:
        if is_statevector_backend(backend):
            statevector = co.run_circuit_without_transpilation(
                circuit, backend, return_statevector=True
            )
            rho = partial_trace(statevector, qubit_1, qubit_2)
        else:
            rho = perform_quantum_tomography(
                circuit, qubit_1, qubit_2, backend, backend_options, execute_kwargs
            )
        if method == EM_TOMOGRAPHY_EOF:
            return eof(rho)
        elif method == EM_TOMOGRAPHY_CONCURRENCE:
            return concurrence(rho)
        elif method == EM_TOMOGRAPHY_NEGATIVITY:
            return negativity(rho)
        else:
            raise ValueError("Invalid entanglement measure method")


def perform_quantum_tomography(
    circuit: QuantumCircuit,
    qubit_1,
    qubit_2,
    backend,
    backend_options=None,
    execute_kwargs=None,
):
    """
    Performs quantum state tomography on the reduced state of qubit_1 and
    qubit_2
    :param circuit:
    :param qubit_1:
    :param qubit_2:
    :param backend:
    :param backend_options:
    :param execute_kwargs:
    :return:
    """
    execute_kwargs = {} if execute_kwargs is None else execute_kwargs
    classical_gates = co.remove_classical_operations(circuit)
    old_cregs = circuit.cregs.copy()
    circuit.cregs = []
    tomography_circuits = state_tomography_circuits(circuit, [qubit_1, qubit_2])
    circuit.cregs = old_cregs
    co.add_classical_operations(circuit, classical_gates)

    # Backend options only supported for simulators
    if backend_options is None or not isinstance(backend, AerBackend):
        backend_options = {}

    result = [
        execute(qc, backend, **backend_options, **execute_kwargs).result()
        for qc in tomography_circuits
    ]
    rho = TomographyFitter(result, tomography_circuits).fit()
    assert isinstance(rho, np.ndarray)
    return rho


def measure_concurrence_lower_bound(
    circuit: QuantumCircuit,
    qubit_1,
    qubit_2,
    backend,
    backend_options=None,
    execute_kwargs=None,
):
    """
    Measures the lower limit of the concurrence of the mixed, bipartite
    state resulting from a
    partial trace over all
    qubits except those in qubit_pair
    Lower bound based on 10.1103/PhysRevLett.98.140505 and upper bound based on
    10.1103/PhysRevA.78.042308
    Concurrence C bounds: K_1,K_2>= C^2 >= V_1,V_2 :
    V_1 = 4((P-)-(P+))x(P-) = 4(2(P-)-I)x(P-) = 8(P-)x(P-) - 4(I)x(P-),
    V_2 = 4(P-)x((P-)-(P+)) = 4(P-)x(2(P-)-I) = 8(P-)x(P-) - 4(P-)x(I),
    K_1 = 4(P-)x(I),
    K_2 = 4(I)x(P-),
    where (P+) and (P-) are projectors on the symmetric and antisymmetric
    subspace
    of the two copies of either subsystem. I is the identity operator
    :param circuit: QuantumCircuit
    :param qubit_1: Index of the first qubit forming the bipartite state
    :param qubit_2: Index of the second qubit forming the bipartite state
    :param backend: Backend on which circuit is to be run (can't be
    statevector_simulator)
    :param backend_options:
    :param execute_kwargs:
    :returns Minimum value of concurrence
    """
    # Remove measurements and other classical gates
    classical_gates = co.remove_classical_operations(circuit)
    num_qubits = circuit.num_qubits

    qc = QuantumCircuit(2 * num_qubits, 4)
    co.add_to_circuit(qc, circuit.copy(), qubit_subset=list(range(0, num_qubits)))
    co.add_to_circuit(
        qc, circuit.copy(), qubit_subset=list(range(num_qubits, 2 * num_qubits))
    )

    transpile_kwargs = {"backend": backend}

    p_minus_p_minus_circuit = qc.copy()
    co.add_to_circuit(
        p_minus_p_minus_circuit,
        antisymmetric_subspace_projector_measurement_circuit(),
        qubit_subset=[qubit_1, num_qubits + qubit_1],
        clbit_subset=[0, 1],
        transpile_before_adding=True,
        transpile_kwargs=transpile_kwargs,
    )
    co.add_to_circuit(
        p_minus_p_minus_circuit,
        antisymmetric_subspace_projector_measurement_circuit(),
        qubit_subset=[qubit_2, num_qubits + qubit_2],
        clbit_subset=[2, 3],
        transpile_before_adding=True,
        transpile_kwargs=transpile_kwargs,
    )

    p_minus_i_circuit = qc.copy()
    co.add_to_circuit(
        p_minus_i_circuit,
        antisymmetric_subspace_projector_measurement_circuit(),
        qubit_subset=[qubit_1, num_qubits + qubit_1],
        clbit_subset=[0, 1],
        transpile_before_adding=True,
        transpile_kwargs=transpile_kwargs,
    )

    i_p_minus_circuit = qc.copy()
    co.add_to_circuit(
        i_p_minus_circuit,
        antisymmetric_subspace_projector_measurement_circuit(),
        qubit_subset=[qubit_2, num_qubits + qubit_2],
        clbit_subset=[2, 3],
        transpile_before_adding=True,
        transpile_kwargs=transpile_kwargs,
    )

    p_minus_p_minus_counts = co.run_circuit_without_transpilation(
        p_minus_p_minus_circuit, backend, backend_options, execute_kwargs
    )
    p_minus_i_counts = co.run_circuit_without_transpilation(
        p_minus_i_circuit, backend, backend_options, execute_kwargs
    )
    i_p_minus_counts = co.run_circuit_without_transpilation(
        i_p_minus_circuit, backend, backend_options, execute_kwargs
    )

    if "1111" not in p_minus_p_minus_counts:
        p_minus_p_minus_eval = 0
    else:
        p_minus_p_minus_eval = p_minus_p_minus_counts["1111"] / sum(
            p_minus_p_minus_counts.values()
        )

    if "1100" not in i_p_minus_counts:
        i_p_minus_eval = 0
    else:
        i_p_minus_eval = i_p_minus_counts["1100"] / sum(i_p_minus_counts.values())

    if "0011" not in p_minus_i_counts:
        p_minus_i_eval = 0
    else:
        p_minus_i_eval = p_minus_i_counts["0011"] / sum(p_minus_i_counts.values())

    v1 = 8 * p_minus_p_minus_eval - 4 * i_p_minus_eval
    v2 = 8 * p_minus_p_minus_eval - 4 * p_minus_i_eval
    lower_bound = max(v1, v2)
    # k1 = 4 * p_minus_i_eval
    # k2 = 4 * i_p_minus_eval
    # upper_bound = min(k1, k2)

    # Add back the classical gates
    co.add_classical_operations(circuit, classical_gates)
    return lower_bound


# Tomography based entanglement measures


def eof(rho):
    """
    Mixed state entanglement of formation as defined in PhysRevLett.80.2245
    :param rho: 2-qubit density matrix (pure or mixed)
    :return:
    """

    def h(x):
        return (-x * np.log2(x)) - ((1 - x) * np.log2(1 - x))

    c = concurrence(rho)
    if c == 0:
        return 0
    return h(0.5 * (1 + np.sqrt(1 - c**2)))


def concurrence(rho):
    """
    Mixed state concurrence as defined in PhysRevLett.80.2245
    :param rho: 2-qubit density matrix (pure or mixed)
    :return:
    """
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_y_sigma_y = np.kron(sigma_y, sigma_y)
    rho_tilda = sigma_y_sigma_y @ rho.conjugate() @ sigma_y_sigma_y
    eigenvalues = eig(rho @ rho_tilda, left=False, right=False)
    # Make sure eigenvalues are real
    if np.allclose(np.imag(eigenvalues), 0):
        eigenvalues = np.real(eigenvalues)
    else:
        logger.warning(f"When calculating concurrence,eigenvalues were not real")
        return 0
    eigenvalues[np.isclose(eigenvalues, 0)] = 0
    lambdas = np.sqrt(eigenvalues)
    return np.max([0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]])


def negativity(rho):
    transposed = partial_transpose(rho)
    t_norm = trace_norm(transposed)
    return (t_norm - 1) / 2


# Helper functions


def antisymmetric_subspace_projector_measurement_circuit():
    qr = QuantumRegister(2, "projection_qr")
    cr = ClassicalRegister(2, "projection_cr")
    qc = QuantumCircuit(qr, cr)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc.copy()


def partial_trace(statevector, a, b):
    """
    Partial trace over all subsystems except qubit a and qubit b
    :param statevector: Statevector
    :param a: qubit a
    :param b: qubit b
    :return: Density matrix
    """
    num_qubits = int(np.log2(len(statevector)))
    if num_qubits == 2:
        return np.outer(statevector, statevector.conj())
    qubits_to_trace_over = list(range(num_qubits))
    qubits_to_trace_over.remove(a)
    qubits_to_trace_over.remove(b)

    return qi.partial_trace(statevector, qubits_to_trace_over).data


def partial_transpose(density_matrix, wrt=1):
    """
    Partial transpose of density matrix
    :param density_matrix: Bipartite system density matrix
    :param wrt: Which subsystem transpose is supposed to be carried over
    :return: density matrix
    """
    tp = copy.deepcopy(density_matrix)
    for ja, ka, jb, kb in itertools.product(range(2), range(2), range(2), range(2)):
        if wrt == 1:
            tp[ka * 2 + jb][ja * 2 + kb] = density_matrix[ja * 2 + jb][ka * 2 + kb]
        elif wrt == 2:
            tp[ja * 2 + kb][ka * 2 + jb] = density_matrix[ja * 2 + jb][ka * 2 + kb]
    return tp


def trace_norm(density_matrix):
    """
    Evaluate trace norm of density matrix
    :return: float
    """
    return np.real(
        np.trace(
            linalg.sqrtm(
                np.matmul(density_matrix, np.conjugate(density_matrix).transpose())
            )
        )
    )
