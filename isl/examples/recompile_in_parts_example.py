import logging

import isl.utils.circuit_operations as co
from isl.recompilers import ISLRecompiler

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("qiskit").setLevel(logging.WARNING)

qc = co.create_random_initial_state_circuit(4)
qc = co.unroll_to_basis_gates(qc)

isl_recompiler = ISLRecompiler(qc)

result = isl_recompiler.recompile_in_parts(20)
approx_circuit = result["circuit"]
print(approx_circuit)
print(f"Overlap between circuits is {result['overlap']}")
