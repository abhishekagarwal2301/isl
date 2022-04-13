import logging

import qiskit.test.mock as fb
from qiskit.providers.aer.backends import QasmSimulator

import isl.utils.circuit_operations as co
from isl.recompilers import ISLConfig, ISLRecompiler

logging.basicConfig(level=logging.INFO)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("variationalalgorithms").setLevel(logging.DEBUG)

num_qubits = 4
qc = co.create_random_initial_state_circuit(num_qubits)
qc = co.unroll_to_basis_gates(qc, co.DEFAULT_GATES)
fb = fb.FakeGuadalupe()
# noise_model = NoiseModel.from_backend(fb.FakeAthens())
# backend = Aer.get_backend('qasm_simulator')
backend = QasmSimulator.from_backend(fb)
execute_kwargs = {"shots": 8192}
coupling_map = backend.configuration().coupling_map
isl_config = ISLConfig(
    max_layers=100, cost_improvement_tol=1e-2, cost_improvement_num_layers=3
)
isl_recompiler = ISLRecompiler(
    qc,
    backend=backend,
    coupling_map=None,
    execute_kwargs=execute_kwargs,
    isl_config=isl_config,
    perform_final_minimisation=False,
    general_initial_state=False,
    local_measurements_only=False,
)

result = isl_recompiler.recompile()
approx_circuit = result["circuit"]
print(approx_circuit)
print(f"Overlap between circuits is {result['exact_overlap']}")
print(result["cost_evaluations"])
qc.measure_all()
approx_circuit.measure_all()
exact_counts = co.run_circuit_with_transpilation(
    qc, backend, execute_kwargs={"shots": 8192}
)
approx_counts = co.run_circuit_with_transpilation(
    approx_circuit, backend, execute_kwargs={"shots": 8192}
)
exact_noiseless_counts = co.run_circuit_with_transpilation(
    qc, backend, execute_kwargs={"shots": 8192}
)
approx_noiseless_counts = co.run_circuit_with_transpilation(
    approx_circuit, backend, execute_kwargs={"shots": 8192}
)

print(f"Exact,noiseless : {exact_noiseless_counts}")
print(f"Approx,noiseless: {approx_noiseless_counts}")
print(f"Exact  noisy    : {exact_counts}")
print(f"Approx noisy    : {approx_counts}")
