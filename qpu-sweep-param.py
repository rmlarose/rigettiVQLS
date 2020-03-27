"""VQLS on Rigetti: Parameter search for 3 qubit example system from paper."""

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
    n = 3

    # Define the B matrix of the linear system
    Bcoeffs = [1 / 2**(n / 2)] * 2**n
    paulis = [["X", "Z"]] * n
    prods = list(product(*paulis))
    Bterms = ["".join(p) for p in prods]

    # Define the A matrix of the linear system
    Acoeffs = [1, 0.2, 0.2]
    Aterms = ["III", "XII", "XZI"]

    # Get the effective Hamiltonian
    ham = vqls.effective_hamiltonian(Acoeffs, Aterms, Bcoeffs, Bterms)

    # Display the effective Hamiltonian
    nterms = len(ham)
    print("Number of terms:", nterms)
    print("H = ")
    for ii in range(nterms):
        print(f"{round(ham[ii][0], 3)} \t* \t{ham[ii][1]}", (lambda ii: "+" if ii < nterms - 1 else "")(ii))

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
    def costLS1(angles, min_weight: float = 0.0, verbose: bool = False):
        val = vqls.energy(angles, ham, circ, creg, lattice, min_weight=min_weight, shots=10_000)
        if verbose:
            print("Current angles:", angles)
            print("Current energy:", val)
        return val
    
    # ====================
    # Sweep the parameters
    # ====================
    CVALS = []
    
    for theta in np.linspace(-np.pi, np.pi, 20):
        print("Status: theta =", theta)
        current_cvals = []
        angles = [theta] * n

        for _ in range(nshots):
            start = time.time()
            current_cvals.append(costLS1(angles, min_weight=0.00, verbose=False))
            print(f"Time to evaluate cost on n = {n} qubits:", round(time.time() - start, 3), "seconds.")
        CVALS.append(current_cvals)

    # =============
    # Save the data
    # =============              
    CVALS = np.array(CVALS)
    key = "_".join(time.asctime().split())
    np.savetxt(f"cost_sweep_data_{key}.txt", CVALS)


if __name__ == "__main__":
    main()
