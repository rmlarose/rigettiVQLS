"""Code for running VQLS on Rigetti's quantum computers."""

from itertools import product
from typing import (List, Tuple)

import numpy as np

import pyquil
from pyquil import Program, get_qc
import pyquil.gates as gates

# ====================================
# Pauli matrices and string dictionary
# ====================================

imat = np.array([[1, 0], [0, 1]], dtype=np.complex64)
xmat = np.array([[0, 1], [1, 0]], dtype=np.complex64)
ymat = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
zmat = np.array([[1, 0], [0, -1]], dtype=np.complex64)
pauli_dict = {"I": imat, "X": xmat, "Y": ymat, "Z": zmat}

# ============================================================
# Functions for viewing matrices and vectors of linear systems
# ============================================================

def tensor(*matrices) -> np.ndarray:
    """Returns the tensor product of all matrices."""
    mats = list(matrices)
    mat = mats[0]
    if type(mat) != np.ndarray:
        raise ValueError("All arguments must be numpy.ndarray's.")
    for term in mats[1:]:
        mat = np.kron(mat, term)
    return mat


def matrix(coeffs: List[complex], terms: List[str]):
    """Returns the matrix defined by the coefficients and Paulis.
    
    Args:
        coeffs: List of real/complex numbers specifying coefficients of Paulis.
        terms: List of Pauli operators.
    """
    if len(coeffs) != len(terms):
        raise ValueError("Number of coeffs does not equal number of terms.")
    nqubits = len(terms[0])
    dim = 2**nqubits
    mat = np.zeros((dim, dim), dtype=np.complex64)
    for (c, pauli) in zip(coeffs, terms):
        pauli = [pauli_dict[key] for key in pauli]
        mat += c * tensor(*pauli)
    return mat


def matrix_of_hamiltonian(ham: List[Tuple[complex, str]]) -> np.ndarray:
    """Returns a matrix representation of the Hamiltonian in the standard basis."""
    coeffs = []
    paulis = []
    for (coeff, pauli) in ham:
        coeffs.append(coeff)
        paulis.append(pauli)
    mat = matrix(coeffs, paulis)
    return mat


def vector(coeffs: List[complex], terms: List[str]):
    """Returns the vector of the linear system.
    
    Args:
        coeffs: Weights of B matrix.
        terms: Pauli terms in B matrix.
    """
    return matrix(coeffs, terms)[:, 0]

# ======================================
# Functions for multipling Pauli strings
# ======================================

def mult(a: str, b: str) -> Tuple[complex, str]:
    """Returns product a times b of two single qubit Paulis."""
    if a not in ("I", "X", "Y", "Z") or b not in ("I", "X", "Y", "Z"):
        raise ValueError("One or more invalid Pauli keys.")
        
    prod = {"XY": "Z",
            "YZ": "X",
            "ZX": "Y"}

    if a == "I":
        return (1., b)
    if b == "I":
        return (1., a)
    if a == b:
        return (1., "I")
    
    phase = 1j
    new = a + b
    if new in prod.keys():
        return (phase, prod[new])
    else:
        return (-phase, prod[b + a])
    

def multn(a: str, b: str) -> Tuple[complex, str]:
    """Returns product a times b of two n qubit Paulis."""
    if len(a) != len(b):
        raise ValueError("len(a) != len(b)")
    phase = 1
    paulis = []
    for (aterm, bterm) in zip(a, b):
        p, pauli = mult(aterm, bterm)
        phase *= p
        paulis.append(pauli)
    return phase, "".join(paulis)

# ===========================================================
# Functions for expanding input Pauli string for Pb projector
# ===========================================================

def Pb_expansion(Bcoeffs: List[complex], Bterms: List[str]) -> List[Tuple[complex, str]]:
    """Inputs B as a linear combination of Paulis and 
    outputs B|0><0|B^dagger as a linear combination of Paulis.
    """
    if len(Bcoeffs) != len(Bterms):
        raise ValueError("len(Bcoeffs) != len(Bterms)")
    n = len(Bterms[0])
    
    # Form the |0><0| projector on n qubits
    terms = [["I", "Z"]] * n
    prod = list(product(*terms))
    Pterms = ["".join(p) for p in prod]
    
    coeffs = []
    paulis = []
    
    for ii in range(len(Bterms)):
        for jj in range(len(Pterms)):
            for kk in range(len(Bterms)):
                coeff = Bcoeffs[ii] * np.conj(Bcoeffs[kk]) / 2**n
                phase1, pauli1 = multn(Bterms[ii], Pterms[jj])
                phase2, pauli = multn(pauli1, Bterms[kk])
                
                coeffs.append(coeff * phase1 * phase2)
                paulis.append(pauli)
    return coeffs, paulis

# ====================================================
# Functions to attempt to minimize the number of terms 
# in the Pauli expansion of the Hamiltonian.
# ====================================================

def combine_paulis(
    hamiltonian: List[Tuple[complex, str]]
) -> List[Tuple[complex, str]]:
    """Combines identical Pauli terms in the Hamiltonian, if possible."""
    new = [list(hamiltonian[0])]
    seen_paulis = [new[0][1]]

    for (coeff, pauli) in hamiltonian[1:]:
        if pauli in seen_paulis:
            # Find the index of where the Pauli is in new
            index = seen_paulis.index(pauli)
            new[index][0] += coeff
        else:
            new.append(list((coeff, pauli)))
            seen_paulis.append(pauli)

    return new


def drop_zero(
    hamiltonian: List[Tuple[complex, str]]
) -> List[Tuple[complex, str]]:
    """Drops Pauli terms with zero weight."""
    new = []
    for (coeff, pauli) in hamiltonian:
        if not np.isclose(abs(coeff), 0.0):
            new.append([coeff, pauli])
    return new

# =================================================================
# Function for getting the effective Hamiltonian of a linear system
# =================================================================

def effective_hamiltonian(Acoeffs: List[complex], 
                          Aterms: List[str], 
                          Bcoeffs: List[complex],
                          Bterms: List[str]) -> Tuple[List[complex], List[str]]:
    """Returns the effective Hamiltonian of an input linear system Ax = B|0>.
    
    Args:
        Acoeffs: List of coefficients of A matrix.
        Aterms: Pauli strings for each coefficient of A matrix.
        Bcoeffs: List of coefficients of B matrix.
        Bterms: Pauli strings for each coefficient of B matrix.
    """
    # Input checks
    if len(Acoeffs) != len(Aterms):
        raise ValueError("len(Acoeffs) != len(Aterms)")
    if len(Bcoeffs) != len(Bterms):
        raise ValueError("len(Bceoffs) != len(Bterms)")

    if not np.isclose(np.linalg.norm(vector(Bcoeffs, Bterms)), 1.0):
        raise ValueError("The bvector must have unit norm.")

    # Initialize the Hamiltonian as a List[Tuple[complex, str]]
    ham = []
    n = len(Aterms[0])
    
    # =======================================
    # First term in Hamiltonian (A^\dagger A)
    # =======================================

    for l in range(len(Aterms)):
        for k in range(len(Aterms)):
            coeff = np.conj(Acoeffs[l]) * Acoeffs[k]
            phase, pauli = multn(Aterms[l], Aterms[k])
            ham.append(list((coeff * phase, pauli)))
    
    ham = drop_zero(combine_paulis(ham))
    
    # ============================================
    # Second term in Hamiltonian (A^\dagger P_b A)
    # ============================================
    
    # Get the Pb expansion
    Pbcoeffs, Pbterms = Pb_expansion(Bcoeffs, Bterms)
    
    for k in range(len(Aterms)):
        for l in range(len(Aterms)):
            for m in range(len(Pbterms)):
                # Compute this coefficient and term
                coeff = np.conj(Acoeffs[k]) * Acoeffs[l] * Pbcoeffs[m]
                phase1, pauli1 = multn(Aterms[k], Pbterms[m])
                phase2, pauli = multn(pauli1, Aterms[l])
                
                # Append it to the expansion
                ham.append(list((-1 * phase1 * phase2 * coeff, pauli)))
    
    return drop_zero(combine_paulis(ham))

