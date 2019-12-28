"""Unit tests for VQLS using effective Hamiltonian."""

from itertools import product
from math import pi

import numpy as np
from pyquil import get_qc

import vqls


def test_pauli_matrices():
    """Simple checks for Pauli matrices."""
    assert np.array_equal(vqls.matrix([1], ["Y"]), vqls.ymat)
    assert np.array_equal(vqls.vector([1], ["Y"]), vqls.ymat[:, 0])


def test_single_pauli_mult():
    """Tests single qubit Pauli multiplication."""
    paulis = ("I", "X", "Y", "Z")
    # Check multiplication with identity and squared terms
    for p in paulis:
        assert vqls.mult("I", p) == (1., p)
        assert vqls.mult(p, "I") == (1., p)
        assert vqls.mult(p, p) == (1., "I")

    # Check "cross terms"
    assert vqls.mult("X", "Y") == (1j, "Z")
    assert vqls.mult("Y", "X") == (-1j, "Z")
    assert vqls.mult("X", "Z") == (-1j, "Y")
    assert vqls.mult("Z", "X") == (1j, "Y")
    assert vqls.mult("Y", "Z") == (1j, "X")
    assert vqls.mult("Z", "Y") == (-1j, "X")


def test_pauli_mult():
    """Tests n-qubit Pauli multiplication."""
    assert vqls.multn("IXY", "ZYY") == (1j, "ZZI")
    assert vqls.multn("IIIIII", "XYZXYZ") == (1., "XYZXYZ")
    assert vqls.multn("YY", "ZI") == (1j, "XY")
    assert vqls.multn("XZ", "ZX") == (1 + 0j, "YY")


def Pbexpansion_test_helper(Bcoeffs, Bterms, verbose=False):
    """Tests matrix equality for Pauli expansion of Pb = B|0><0|B^dagger."""
    Pbcoeffs, Pbpaulis = vqls.Pb_expansion(Bcoeffs, Bterms)

    if verbose:
        print("Found Pauli expansion of Pb:")
        print(Pbcoeffs)
        print(Pbpaulis)

    # Determine the bvector
    bvec = vqls.vector(Bcoeffs, Bterms)
    if verbose:
        print("\nbvec computed from Bcoeffs and Bterms:")
        print(bvec)

    Pbmat = vqls.matrix(Pbcoeffs, Pbpaulis)
    if verbose:
        print("\nComputed Pb:")
        print(Pbmat)
        print("\nActual Pb = |b><b|:")
        print(np.outer(bvec, bvec.conj()))

    return np.array_equal(Pbmat, np.outer(bvec, bvec.conj()))


def test_Pbexpansion_single_qubit_single_term():
    """Test Pb expansion with one-term, single qubit B matrices."""
    assert Pbexpansion_test_helper([1], ["I"])
    assert Pbexpansion_test_helper([1], ["X"])
    assert Pbexpansion_test_helper([1], ["Y"])
    assert Pbexpansion_test_helper([1], ["Z"])


def test_Pbexpansion_single_qubit_multiple_terms():
    """Test Pb expansion with multiple terms, single qubit B matrices."""
    assert Pbexpansion_test_helper([1, 2], ["I", "X"])
    assert Pbexpansion_test_helper([1, 1, 3], ["I", "X", "Y"])

def test_Pbexpansion_multi_qubit_multi_term():
    """Tests Pb expansion with multiple qubit, multiple term B matrices."""
    assert Pbexpansion_test_helper([1, 1], ["XZ", "XZ"])
    assert Pbexpansion_test_helper([2, 3, 1], ["IIZ", "ZXY", "YYX"])


def test_simplify_hamiltonain():
    """Tests combining Paulis and dropping zero terms to simplify Hamiltonians."""
    # Test 1
    ham = [(1j, "XY"), (-1j, "XY")]
    assert vqls.drop_zero(vqls.combine_paulis(ham)) == []

    # Test 2
    ham = [(1.0, 'II'), (1j, 'XY'), (-1j, 'XY'), (1.0, 'II')]
    assert vqls.drop_zero(vqls.combine_paulis(ham)) == [[2.0, "II"]]

    # Test 3
    ham = [[2.0, 'II'], [-0.25, 'II'], [-0.25, 'IZ'], [-0.25, 'ZI'], 
           [-0.25, 'ZZ'], [-0.25j, 'XY'], [-0.25, 'XX'], [(0.25+0j), 'YY'], 
           [-0.25j, 'YX'], [0.25j, 'XY'], [(-0.25+0j), 'XX'], [(0.25+0j), 'YY'], 
           [0.25j, 'YX'], [-0.25, 'II'], [(0.25+0j), 'IZ'], [(0.25+0j), 'ZI'],
           [(-0.25+0j), 'ZZ']]
    assert vqls.drop_zero(vqls.combine_paulis(ham)) == [[1.5, 'II'], [(-0.5+0j), 'ZZ'], [(-0.5+0j), 'XX'], [(0.5+0j), 'YY']]


def effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct, 
                               Bcoeffs, Bterms, bvec_correct,
                               verbose = False):
    """Checks matrix equality for the effective Hamiltonian expansion.
    
    Ensures minimum evalue of effective Hamiltonian is 0 and
    corresponding evector is the solution to the linear system."""
    # Get the effective Hamiltonian
    ham = vqls.effective_hamiltonian(Acoeffs, Aterms, Bcoeffs, Bterms)

    # Get the matrix and vector of the linear system, and solve it
    Amat = vqls.matrix(Acoeffs, Aterms)
    assert np.array_equal(Amat, Amat_correct)
    bvec = vqls.vector(Bcoeffs, Bterms)
    if verbose:
        print("Computed bvec:")
        print(bvec)
    assert np.array_equal(bvec, bvec_correct)
    xvec = np.linalg.solve(Amat, bvec)
    xvec /= np.linalg.norm(xvec)
    
    if verbose:
        print("\nEffective Hamiltonian expansion:")
        print(ham)

    # Get the matrix of the Hamiltonian
    Heff = vqls.matrix_of_hamiltonian(ham)

    # Check correctness of the effective Hamiltonian
    Hexact = (Amat_correct.conj().T @ 
              (np.identity(len(Amat)) - np.outer(bvec_correct, bvec_correct.conj())) @ 
              Amat_correct)
    if verbose:
        print("\nExact Heff:")
        print(Hexact)
        print("\nComputed Heff:")
        print(Heff)
    assert np.allclose(Heff, Hexact)
    assert np.allclose(Heff, Heff.conj().T)
    if verbose:
        print("Heff @ xvec:")
        print(Heff @ xvec)
        print("||Heff @ xvec|| = ", np.linalg.norm(Heff @ xvec))
    assert np.isclose(np.linalg.norm(Heff @ xvec), 0.0, atol=1e-5)

    # Do the diagonalization
    evals, evecs = np.linalg.eigh(Heff)
    if verbose:
        print("Min eval:")
        print(evals[0])
        print("Evec of minimum eval:")
        print(evecs[:, 0])
        print("Solution of linear system:")
        print(xvec)
    assert np.isclose(evals[0], 0.0, atol=1e-5)
    assert np.isclose(abs(np.dot(evecs[:, 0], xvec.conj().T))**2, 1.0)

# =======================
# Tests with A = Identity
# =======================

def test_effective_hamiltonian_iden1():
    Bcoeffs = [1 / 2] * 4
    Bterms = ["II", "IX", "XI", "XX"]
    bvec_correct = np.ones(4) / 2
    Acoeffs = [1]
    Aterms = ["II"]
    Amat_correct = np.identity(4)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)


def test_effective_hamiltonian_iden2():
    Bcoeffs = [1 / 4] * 4
    Bterms = ["II", "IZ", "ZI", "ZZ"]
    bvec_correct = np.array([1, 0, 0, 0])
    Acoeffs = [1]
    Aterms = ["II"]
    Amat_correct = np.identity(4)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)


def test_effective_hamiltonian_iden3():
    Bcoeffs = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    Bterms = ["II", "XY"]
    bvec_correct = vqls.vector(Bcoeffs, Bterms)
    Acoeffs = [1]
    Aterms = ["II"]
    Amat_correct = np.identity(4)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)


def test_effective_hamiltonian_iden4():
    Bcoeffs = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    Bterms = ["III", "XYZ"]
    bvec_correct = vqls.vector(Bcoeffs, Bterms)
    Acoeffs = [1]
    Aterms = ["III"]
    Amat_correct = np.identity(8)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)


# ========================
# Tests with A != Identity
# ========================

def test_effective_hamiltonian_noniden1():
    Bcoeffs = [1 / 2] * 4
    Bterms = ["II", "IX", "XI", "XX"]
    bvec_correct = np.ones(4) / 2
    Acoeffs = [1]
    Aterms = ["IZ"]
    Amat_correct = vqls.matrix(Acoeffs, Aterms)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)

def test_effective_hamiltonian_noniden2():
    Bcoeffs = [1]
    Bterms = ["II"]
    bvec_correct = vqls.vector(Bcoeffs, Bterms)
    Acoeffs = [1, 1]
    Aterms = ["IZ", "XX"]
    Amat_correct = vqls.matrix(Acoeffs, Aterms)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)

def test_effective_hamiltonian_noniden3():
    Bcoeffs = [1]
    Bterms = ["ZY"]
    bvec_correct = vqls.vector(Bcoeffs, Bterms)
    Acoeffs = [1, 1]
    Aterms = ["IZ", "XX"]
    Amat_correct = vqls.matrix(Acoeffs, Aterms)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)


# ===========================
# Test example linear systems
# ===========================

def test_effective_hamiltonian_exampleLS1():
    """Three qubit linear system example from VQLS paper."""
    Bcoeffs = [1 / 2**(3/2)] * 8
    Bterms = ["XXX", "XXZ", "XZX", "XZZ", "ZXX", "ZXZ", "ZZX", "ZZZ"]
    bvec_correct = vqls.vector(Bcoeffs, Bterms)
    Acoeffs = [1, 0.2, 0.2]
    Aterms = ["III", "XII", "XZI"]
    Amat_correct = vqls.matrix(Acoeffs, Aterms)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)

def test_effective_hamiltonian_exampleLS2():
    """Five qubit linear system example from VQLS paper."""
    Bcoeffs = [1 / 2**(5/2)] * 32
    paulis = [["X", "Z"]] * 5
    prods = list(product(*paulis))
    Bterms = ["".join(p) for p in prods]
    bvec_correct = vqls.vector(Bcoeffs, Bterms)
    Acoeffs = [1, 0.2, 0.2]
    Aterms = ["IIIII", "XIIII", "XZIII"]
    Amat_correct = vqls.matrix(Acoeffs, Aterms)
    effective_hamiltonian_test_helper(Acoeffs, Aterms, Amat_correct,
                               Bcoeffs, Bterms, bvec_correct)


    
# =================================
# Test computing expectation values
# =================================

def test_expectation_three_qubits():
    """Tests several expectation values for a three qubit circuit."""
    n = 3
    qcomputer = f"Aspen-7-{n}Q-B"
    lattice = get_qc(qcomputer, as_qvm=True)
    circ, creg = vqls.yansatz(lattice)
    SHOTS = 10000
    tol = 1e-1  # Set smaller than necessary to be safe
    assert np.isclose(vqls.expectation([0] * n, 1, "ZZZ", circ, creg,
                                       lattice, shots=SHOTS), 1.0, atol=tol)
    assert np.isclose(vqls.expectation([0] * n, 0.8675309, "ZZZ", circ, creg,
                                       lattice, shots=SHOTS), 0.8675309, atol=tol)
    assert np.isclose(vqls.expectation([0] * n, 1, "XII", circ, creg,
                                       lattice, shots=SHOTS), 0.0, atol=tol)
    assert np.isclose(vqls.expectation([pi / 2, 0, 0], 1, "XII", circ, creg,
                                       lattice, shots=SHOTS), 1.0, atol=tol)
    assert np.isclose(vqls.expectation([pi / 2, pi / 2, 0], 1, "XXI", circ, creg,
                                       lattice, shots=SHOTS), 1.0, atol=tol)
    assert np.isclose(vqls.expectation([pi / 2, pi / 2, pi / 2], 1, "XXX", circ, creg,
                                       lattice, shots=SHOTS), 1.0, atol=tol)
    assert np.isclose(vqls.expectation([pi / 2, pi / 2, 0], 1, "XXZ", circ, creg,
                                       lattice, shots=SHOTS), 1.0, atol=tol)


def test_energy1():
    """Tests energy computation for identity Hamiltonian."""
    test_hamiltonian = [(1.0, "II")]
    test_computer = get_qc("Aspen-7-2Q-B", as_qvm=True)
    test_circuit, test_creg = vqls.yansatz(test_computer)
    assert np.isclose(
        vqls.energy([0, 0], test_hamiltonian, test_circuit, test_creg, test_computer, shots=10000, verbose=False),
        1.0,
        atol=1e-5
    )


def test_energy2():
    """Tests energy for simple Hamiltonian."""
    test_hamiltonian = [(1.0, "II"), (-0.1, "IX")]
    test_computer = get_qc("Aspen-7-2Q-B", as_qvm=True)
    test_circuit, test_creg = vqls.yansatz(test_computer)
    assert np.isclose(
        vqls.energy([0, 0], test_hamiltonian, test_circuit, test_creg, test_computer, shots=10000, verbose=False),
        1.0,
        1e-2
    )
    assert np.isclose(
        vqls.energy([0, pi / 2], test_hamiltonian, test_circuit, test_creg, test_computer, shots=10000, verbose=False),
        0.9,
        1e-2
    )


def test_energy_with_min_weight():
    """Tests computing energy of Hamiltonian with small weight terms."""
    test_hamiltonian = [(1.0, "II"), (-0.1, "II"), (-0.01, "II")]
    test_computer = get_qc("Aspen-7-2Q-B", as_qvm=True)
    test_circuit, test_creg = vqls.yansatz(test_computer)
    assert np.isclose(
        vqls.energy([0, 0], test_hamiltonian, test_circuit, test_creg, test_computer, shots=10000, min_weight=0.1),
        0.9,
        1e-2
    )
    assert np.isclose(
        vqls.energy([0, 0], test_hamiltonian, test_circuit, test_creg, test_computer, shots=10000, min_weight=0.01),
        0.89,
        1e-2
    )
    assert np.isclose(
        vqls.energy([0, 0], test_hamiltonian, test_circuit, test_creg, test_computer, shots=10000, min_weight=2.0),
        0.0,
        1e-2
    )
    