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


# ==================================================
# Functions for computing <H> with a quantum circuit
# ==================================================

def yansatz(computer):
    """Returns a circuit with a product state ansatz."""
    n = len(computer.qubits())
    # Get a circuit and classical memory register
    circ = Program()
    creg = circ.declare("ro", memory_type="BIT", memory_size=n)

    # Define parameters for the ansatz
    angles = circ.declare("theta", memory_type="REAL", memory_size=n)

    # Add the ansatz
    circ += [gates.RY(angles[ii], computer.qubits()[ii]) for ii in range(n)]
    
    return circ, creg


def expectation(angles: List[float], 
                coeff: complex, 
                pauli: str,
                ansatz: pyquil.Program,
                creg: pyquil.quilatom.MemoryReference,
                computer: pyquil.api.QuantumComputer,
                shots: int = 10000,
                verbose: bool = False) -> float:
    """Returns coeff * <\theta| paulii |\theta>.
    
    Args:
        angles: List of angles at which to evaluate coeff * <theta| pauli |theta>.
        coeff: Coefficient of Pauli term.
        pauli: Pauli string.
        ansatz: pyQuil program representing the ansatz state.
        creg: Classical register of ansatz to measure into.
        computer: QuantumComputer to execute the circuit on.
        shots: Number of times to execute the circuit (sampling statistics).
        verbose: Option for visualization/debugging.
    """
    if np.isclose(np.imag(coeff), 0.0):
        coeff = np.real(coeff)

    if set(pauli) == {"I"}:
        return coeff
    
    # Set up the circuit
    circuit = ansatz.copy()
    qubits = computer.qubits()
    measured = []
    for (q, p) in enumerate(pauli):
        if p == "X":
            circuit += [gates.H(qubits[q]), gates.MEASURE(qubits[q], creg[q])]
            measured.append(qubits[q])
        elif p == "Y":
            circuit += [gates.S(qubits[q]), gates.H(qubits[q]), gates.MEASURE(qubits[q], creg[q])]
            measured.append(qubits[q])
        elif p == "Z":
            circuit += [gates.MEASURE(qubits[q], creg[q])]
            measured.append(qubits[q])
    
    if verbose:
        print(f"Computing {coeff} x <theta|{pauli}|theta>...")
        print("\nCircuit to be executed:")
        print(circuit)
    
    # Execute the circuit
    circuit.wrap_in_numshots_loop(shots)
    executable = computer.compile(circuit)
    res = computer.run(executable, memory_map={"theta": angles})
    
    if verbose:
        print("\nResults:")
        print(f"{len(res)} total measured bit strings.")
        print(res)
    
    # Do the postprocessing
    tot = 0.0
    for vals in res:
        tot += (-1)**sum(vals)
    return coeff * tot / shots


# TODO: Utilize simulatenous measurements
def energy(angles, hamiltonian, ansatz, creg, computer, shots=10000, min_weight=0, verbose=False):
    """Returns <theta| H |theta>.
    
    Args:
        angles: List of angles at which to evaluate <theta| H |theta>.
        hamiltonian: List[Tuple] of (coeff, pauli) pairs.
        ansatz: pyQuil program representing the ansatz state.
        creg: Classical register of ansatz to measure into.
        computer: QuantumComputer to execute the circuit on.
        shots: Number of times to execute the circuit (sampling statistics).
        min_weight: If a term has abs(coeff) < min_weight, skip the term.
                    Default value = 0, i.e., all terms are present in computation.
        verbose: Option for visualization/debugging.
    """
    value = 0.0
    for (coeff, pauli) in hamiltonian:
        if abs(coeff) >= min_weight:
            value += expectation(angles, coeff, pauli, ansatz, creg, computer, shots, verbose)
    return value

# ==========================================================================
# Function for getting the wavefunction from the ansatz + optimal parameters
# ==========================================================================

def qsolution(ansatz, opt_angles):
    """Returns the wavefunction of the ansatz at the optimal angles."""
    prog = Program()
    memory_map = {"theta": opt_angles}
    for name, arr in memory_map.items():
        for index, value in enumerate(arr):
            prog += gates.MOVE(gates.MemoryReference(name, offset=index), value)

    ansatz = prog + ansatz
    soln = pyquil.quil.percolate_declares(ansatz)

    wfsim = pyquil.api.WavefunctionSimulator()
    return wfsim.wavefunction(soln).amplitudes