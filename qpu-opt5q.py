"""VQLS on Rigetti: Optimization attempts 5 qubit example system from paper."""

from itertools import product
from math import pi
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import (dual_annealing, fmin, fmin_powell, fmin_cobyla, fmin_l_bfgs_b, minimize)

import pyquil
from pyquil import Program, get_qc
import pyquil.gates as gates

from aqgd import AQGD
import vqls

QVM = True
NOISY = False
np.set_printoptions(precision=3)
nshots = 1


def main():
    """Creates linear system and evaluates cost by sweeping parameters."""
    # ======================================================
    # Terms in effective Hamiltonian for first linear system
    #
    # TODO: Read in the effective Hamiltonian from file.
    # ======================================================
    # Number of qubits
    n = 5

    # Load the Hamiltonian from file
    coeffs = np.loadtxt(f"ham{n}q_paper_coeffs.txt", dtype=float)
    paulis = np.loadtxt(f"ham{n}q_paper_paulis.txt", dtype=str)
    
    # Convert to List[List[float, str]]
    ham = list(zip(coeffs, paulis))

    # ================================
    # Get a quantum computer to run on
    # ================================
    qcomputer = f"Aspen-7-{n}Q-B"
    lattice = get_qc(qcomputer, as_qvm=QVM, noisy=NOISY)  # Change to as_qvm=False to run on QC. Must have reservation.

    # =============
    # Get an ansatz
    # =============
    circ, creg = vqls.yansatz(lattice)

    # =============
    # Cost function
    # =============
    CVALS = []
    def cost(angles, min_weight: float = 0.0, verbose: bool = False):
        start = time.time()
        val = vqls.energy(angles, ham, circ, creg, lattice, min_weight=min_weight, shots=10_000)
        print("Time to eval cost:", round(time.time() - start, 3), "seconds.")
        CVALS.append(val)
        if verbose:
            print("Current angles:", angles)
            print("Current energy:", val)
        return val
    
    # ===================
    # Do the optimization
    # ===================
    
    start = time.time()
#     res = fmin_l_bfgs_b(cost, x0=[pi / 2] * n, approx_grad=True, maxfun=50, args=(0.00, True))
    res = fmin(cost, x0=[pi / 2] * n, maxfun=50, args=(0.00, True))
    print("Runtime:", (time.time() - start) // 60, "minutes.")
    
    print("True energy (evaluating all terms in Ham) at optimal parameters:")
    print("C(xopt) =", cost(res))
    
    key = "_".join(time.asctime().split())
    np.savetxt(f"CVALS_{n}q_{key}.txt", CVALS)

#     # =============
#     # Save the data
#     # =============              
#     CVALS = np.array(CVALS)
#     key = "_".join(time.asctime().split())
#     np.savetxt(f"cost_sweep_data_{key}.txt", CVALS)


if __name__ == "__main__":
    main()
