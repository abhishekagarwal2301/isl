import logging

import isl.utils.circuit_operations as co
from isl.recompilers import FixedAnsatzRecompiler
from isl.utils.fixed_ansatz_circuits import hardware_efficient_circuit

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("qiskit").setLevel(logging.WARNING)

num_qubits = 5
qc = co.create_random_initial_state_circuit(num_qubits)
qc = co.unroll_to_basis_gates(qc)

for anderson_acceleration in [True, False]:
    ansatz = hardware_efficient_circuit(num_qubits, "rxryrz", 7)
    far = FixedAnsatzRecompiler(
        qc,
        ansatz,
        optimization_algorithm_kwargs={"anderson_acceleration": anderson_acceleration},
    )

    result = far.recompile()
    approx_circuit = result["circuit"]
    print(co.unroll_to_basis_gates(approx_circuit))
    print(f"Overlap between circuits is {result['overlap']}")
