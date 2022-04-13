import logging

import numpy as np
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error
from scipy.optimize import curve_fit

from isl.utils.circuit_operations.circuit_operations_alternate_emulators import (
    run_on_qulacs_noiseless,
)
from isl.utils.circuit_operations.circuit_operations_full_circuit import (
    unroll_to_basis_gates,
)
from isl.utils.utilityfunctions import (
    counts_data_from_statevector,
    expectation_value_of_pauli_observable,
    is_statevector_backend,
)

QASM_SIM = Aer.get_backend("qasm_simulator")
SV_SIM = Aer.get_backend("statevector_simulator")


def run_circuit_with_transpilation(
    circuit: QuantumCircuit,
    backend=QASM_SIM,
    backend_options=None,
    execute_kwargs=None,
    return_statevector=False,
):
    if backend == "qulacs":
        transpiled_circuit = unroll_to_basis_gates(circuit)
    else:
        transpiled_circuit = transpile(circuit, backend)
    return run_circuit_without_transpilation(
        transpiled_circuit, backend, backend_options, execute_kwargs, return_statevector
    )


def run_circuit_without_transpilation(
    circuit: QuantumCircuit,
    backend=QASM_SIM,
    backend_options=None,
    execute_kwargs=None,
    return_statevector=False,
):
    if execute_kwargs is None:
        execute_kwargs = {}

    if backend == "qulacs":
        sv = run_on_qulacs_noiseless(circuit, False)
        if (
            "noise_model" in execute_kwargs
            and execute_kwargs["noise_model"] is not None
        ):
            raise ValueError(f"Noisy emulations on qulacs are not supported yet")
        if return_statevector:
            return sv
        else:
            if execute_kwargs is not None and "shots" in execute_kwargs:
                counts = counts_data_from_statevector(
                    sv, num_shots=execute_kwargs["shots"]
                )
            else:
                counts = counts_data_from_statevector(sv)
            return counts

    # Backend options only supported for simulators
    if backend_options is None or not isinstance(backend, AerBackend):
        backend_options = {}
    # executing the circuits on the backend and returning the job
    job = backend.run(circuit, **backend_options, **execute_kwargs)

    result = job.result()
    if is_statevector_backend(backend):
        if return_statevector:
            counts = result.get_statevector()
        else:
            counts = counts_data_from_statevector(result.get_statevector())
    else:
        counts = result.get_counts()

    return counts


def create_noisemodel(t1, t2, log_fidelities=True):
    # Instruction times (in nanoseconds)
    time_u1 = 0  # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond

    t1 = t1 * 1e6
    t2 = t2 * 1e6

    # QuantumError objects
    error_reset = thermal_relaxation_error(t1, t2, time_reset)
    error_measure = thermal_relaxation_error(t1, t2, time_measure)
    error_u1 = thermal_relaxation_error(t1, t2, time_u1)
    error_u2 = thermal_relaxation_error(t1, t2, time_u2)
    error_u3 = thermal_relaxation_error(t1, t2, time_u3)
    error_cx = thermal_relaxation_error(t1, t2, time_cx).expand(
        thermal_relaxation_error(t1, t2, time_cx)
    )

    # Add errors to noise model
    noise_thermal = NoiseModel()
    noise_thermal.add_all_qubit_quantum_error(error_reset, "reset")
    noise_thermal.add_all_qubit_quantum_error(error_measure, "measure")
    noise_thermal.add_all_qubit_quantum_error(error_u1, "u1")
    noise_thermal.add_all_qubit_quantum_error(error_u2, "u2")
    noise_thermal.add_all_qubit_quantum_error(error_u3, "u3")
    noise_thermal.add_all_qubit_quantum_error(error_cx, "cx")

    if log_fidelities:
        logging.info("Noise model fidelities:")
        for qubit_error in noise_thermal.to_dict()["errors"]:
            logging.info(
                f"{qubit_error['operations']}: " f"{max(qubit_error['probabilities'])}"
            )
    return noise_thermal


def zero_noise_extrapolate(
    circuit: QuantumCircuit, measurement_function, num_points=10
):
    calculated_values = []
    probabilities = np.linspace(0, 1, num_points)
    for prob in probabilities:
        circuit_data_copy = circuit.data.copy()
        for i, (gate, qargs, cargs) in list(enumerate(circuit.data))[::-1]:
            if isinstance(gate, CXGate):
                if np.random.random() < prob:
                    circuit.data.insert(i, (gate, qargs, cargs))
                    circuit.data.insert(i, (gate, qargs, cargs))

        calculated_values.append(measurement_function())
        circuit.data = circuit_data_copy

    def exp_decay(x, intercept, amp, decay_rate):
        return intercept + amp * np.exp(-1 * x / decay_rate)

    try:
        popt, pcov = curve_fit(
            exp_decay, probabilities, calculated_values, [0, calculated_values[0], 1]
        )
        zne_val = exp_decay(-0.5, *popt)
        return zne_val
    except RuntimeError as e:
        logging.warning(f"Failed to zero-noise-extrapolate. Error was {e}")
        return measurement_function()
