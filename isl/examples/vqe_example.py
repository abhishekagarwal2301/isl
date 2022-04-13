import logging

import isl.utils.circuit_operations as co
from isl.utils.constants import ALG_ROTOSOLVE
from isl.utils.fixed_ansatz_circuits import hardware_efficient_circuit
from isl.vqe import FixedAnsatzVQE

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("qiskit").setLevel(logging.WARNING)

num_qubits = 4
qc = co.create_random_initial_state_circuit(num_qubits)
qc = co.unroll_to_basis_gates(qc)

paulis = {"IXII": 0.1, "ZZXX": 0.4, "XXZZ": 0.4, "IIXI": 0.1}

ansatz = hardware_efficient_circuit(num_qubits, "ryrz", 7)
favqe = FixedAnsatzVQE(
    ansatz, evaluation_paulis=paulis, optimization_algorithm_kind=ALG_ROTOSOLVE
)

result = favqe.run()
circuit = result["circuit"]
gs_energy = result["energy"]
approx_circuit = result["circuit"]
print(co.unroll_to_basis_gates(circuit))
print(f"GS Energy = {gs_energy}")
