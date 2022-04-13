import numpy as np
from openfermion import (
    FermionOperator,
    QubitOperator,
    get_ground_state,
    get_sparse_operator,
    jordan_wigner,
)
from scipy.linalg import eigh


def heisenberg_hamiltonian(
    n=4, jx=1.0, jy=0.0, jz=0.0, hx=0.0, hy=0.0, hz=0.0, periodic_bc=False
):
    """
    H = -sum_over_nn(j_x*X_i*X_(i+1) + j_y*Y_i*Y_(i+1) + j_z*Z_i*Z_(i+1))
        -sum(h_x*X_i + h_y*Y_i + h_z*Z_i)
    """
    ham = QubitOperator()
    max_index = n if periodic_bc else n - 1
    for i in range(max_index):
        next_neighbour_index = 0 if i == n - 1 and periodic_bc else i + 1
        ham += QubitOperator(f"X{i} X{next_neighbour_index}", -jx)
        ham += QubitOperator(f"Y{i} Y{next_neighbour_index}", -jy)
        ham += QubitOperator(f"Z{i} Z{next_neighbour_index}", -jz)
    for i in range(n):
        ham += QubitOperator(f"X{i}", -hx)
        ham += QubitOperator(f"Y{i}", -hy)
        ham += QubitOperator(f"Z{i}", -hz)
    return ham


def anderson_model_fermionic_hamiltonian(
    v_i=np.array([0, 1]), epsilon_i=np.array([2, 2]), u=4, mu=0
):
    if len(v_i) != len(epsilon_i):
        raise ValueError(
            f"Number of elements in v_i ({len(v_i)}) must equal number of "
            f"elements in epsilon_i({len(epsilon_i)})"
        )
    num_bath_sites = len(v_i) - 1
    ham = FermionOperator()

    # Coulomb repulsion
    ham += FermionOperator(f"0^ 0 {num_bath_sites + 1}^ {num_bath_sites + 1}", float(u))

    # Bath site energies
    for site_index in range(0, 1 + num_bath_sites):
        for spin in range(2):
            i = site_index + (spin * (1 + num_bath_sites))
            ham += FermionOperator(f"{i}^ {i}", float(epsilon_i[site_index] - mu))
    # Hybridization energies
    for site_index in range(1, 1 + num_bath_sites):
        for spin in range(2):
            i = site_index + (spin * (1 + num_bath_sites))
            impurity_index = spin * (1 + num_bath_sites)
            ham += FermionOperator(f"{impurity_index}^ {i}", float(v_i[site_index]))
            ham += FermionOperator(f"{i}^ {impurity_index}", float(v_i[site_index]))

    return ham


def anderson_model_qubit_hamiltonian(
    v_i=np.array([0, 1]), epsilon_i=np.array([2, 2]), u=4, mu=0
):
    f_ham = anderson_model_fermionic_hamiltonian(v_i, epsilon_i, u, mu)
    qubit_ham = jordan_wigner(f_ham)
    return qubit_ham


def calculate_ground_state(hamiltonian):
    gs_energy, gs_wf = get_ground_state(get_sparse_operator(hamiltonian))
    # eigvals, eigvecs = eigh(get_sparse_operator(hamiltonian).toarray())
    # gs_energy = eigvals[0]
    # gs_wf = eigvecs[:,0]
    return gs_energy, gs_wf
