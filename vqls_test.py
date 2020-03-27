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

    
# ========================================
# Unit tests for simultaneous measurements
# ========================================

def test_merge():
    assert vqls.merge("IX", "ZI") == "ZX"
    assert vqls.merge("ZX", "ZI") == "ZX"
    assert vqls.merge("IIZ", "ZZI") == "ZZZ"


def test_squash():
    assert vqls.squash(["III", "IXY", "ZII"]) == "ZXY"
    assert vqls.squash(["I", "Z"]) == "Z"
    assert vqls.squash(
        ["IIII", "IIXI", "ZXII", "IIIY"]
    ) == "ZXXY"
    assert vqls.squash(["IIIII", "IIIXZ"]) == "IIIXZ"
    assert vqls.squash(["IXXI", "ZIII"]) == "ZXXI"

def test_support():
    assert vqls.support("III") == []
    assert vqls.support("IXY") == [1, 2]
    assert vqls.support("ZZI") == [0, 1]


def test_supports():
    paulis = ["III", "IXZ", "ZZI", "XII", "IYI", "IIY", "XYZ"]
    supports = [vqls.support(p) for p in paulis]
    assert supports == [
        [],
        [1, 2],
        [0, 1],
        [0],
        [1],
        [2],
        [0, 1, 2]
    ]
    
def test_islice():
    vals = [0, 1, 2, 3, 4]
    assert vqls.islice(vals, [0, 1, 4]) == [0, 1, 4]
    assert vqls.islice(vals, [0]) == [0]
    assert vqls.islice(vals, []) == []

def test_is_sim_meas():
    """Unit test for determining if two paulis can be measured simultaneously."""
    # Cases which can be measured simultaneously
    assert vqls.is_sim_meas("III", "IXZ")
    assert vqls.is_sim_meas("IZZ", "ZZI")
    assert vqls.is_sim_meas("X" * 20, "X" * 20)
    assert vqls.is_sim_meas("YYIZZ", "IIXZI")
    
    # Cases which cannot be measured simultaneously
    assert not vqls.is_sim_meas("X", "Z")
    assert not vqls.is_sim_meas("Z", "X")
    assert not vqls.is_sim_meas("YY", "YZ")
    assert not vqls.is_sim_meas("XIZ", "IIY")


def test_can_be_grouped_with():
    """Unit test for measurments with a group."""
    # Cases which the grouping works
    assert vqls.can_be_grouped_with("IIZ", ["ZZI", "IZI", "ZZZ"])
    assert vqls.can_be_grouped_with("IIZ", [])
    assert vqls.can_be_grouped_with("XII", ["XXX", "IXI", "IXX"])
    assert vqls.can_be_grouped_with("Z", ["I", "Z"])
    assert vqls.can_be_grouped_with("IXYZ", ["IIII", "XXYZ"])
    
    # Cases which the grouping doesn't work
    assert not vqls.can_be_grouped_with("ZZI", ["ZZZ", "XZI"])
    assert not vqls.can_be_grouped_with("Z", ["I", "X"])
    assert not vqls.can_be_grouped_with("IXYZ", ["IIII", "XXYZ", "IIIX"])
    assert not vqls.can_be_grouped_with("XXX", ["YII", "IIZ"])


def test_is_sim_meas_group():
    """Unit test for seeing if a group is simultaneously measurable."""
    # Groups which are simultaneously measurable
    assert vqls.is_sim_meas_group(["IXZ", "IIZ", "IXI"])
    assert vqls.is_sim_meas_group(["IIZI", "ZZIZ", "ZZZZ", "ZIZI"])
    assert vqls.is_sim_meas_group(["I", "Z"])
    assert vqls.is_sim_meas_group(["IX", "XI"])
    
    # Groups which are not simultaneously measurable
    assert not vqls.is_sim_meas_group(["ZZZ", "XXX"])
    assert not vqls.is_sim_meas_group(["I", "X", "Z"])
    assert not vqls.is_sim_meas_group(["IZY", "IZI", "ZZX"])

def test_split_ham():
    ham = [
        [1, "IIZ"],
        [2, "XIZ"],
        [3, "YYY"]
    ]
    coeffs, paulis = vqls.split_ham_to_coeffs_and_paulis(ham)
    assert coeffs == [1.0, 2.0, 3.0]
    assert paulis == ['IIZ', 'XIZ', 'YYY']


def test_greedy_group():
    ham = [
        [1, "IIZ"],
        [2, "XIZ"],
        [3, "YYY"]
    ]
    grouped = vqls.group_greedy(ham, randomized=False)

    assert len(grouped) == 2
    assert grouped == [[(1.0, 'IIZ'), (2.0, 'XIZ')], [(3.0, 'YYY')]]


def test_greedy_group2():
    ham = [
        [1, "IIZ"],
        [2, "IIX"],
        [3, "ZIZ"]
    ]
    grouped = vqls.group_greedy(ham, randomized=False)
    assert len(grouped) == 2
    assert grouped == [[(1.0, 'IIZ'), (3.0, 'ZIZ')], [(2.0, 'IIX')]]


def test_greedy_group3():
    ham = [
        [1, "IIZ"],
        [2, "IIX"],
        [3, "ZIZ"],
        [4, "XII"]
    ]
    grouped = vqls.group_greedy(ham, randomized=False)
    assert len(grouped) == 2
    assert grouped == [[(1.0, 'IIZ'), (3.0, 'ZIZ')], [(2.0, 'IIX'), (4.0, 'XII')]]


def test_greedy_group4():
    ham = [
        [1, "IIZ"],
        [2, "IIX"],
        [3, "ZIZ"],
        [4, "XII"],
        [5, "IZI"]
    ]
    grouped = vqls.group_greedy(ham, randomized=False)
    assert len(grouped) == 2
    assert grouped == [[(1.0, 'IIZ'), (3.0, 'ZIZ'), (5.0, 'IZI')], [(2.0, 'IIX'), (4.0, 'XII')]]


def test_greedy_group5():
    ham = [
        [1, "IIZ"],
        [2, "IIX"],
        [3, "ZIZ"],
        [4, "XII"],
        [5, "IZI"],
        [6, "ZZZ"],
        [7, "XXX"],
        [8, "YYY"]
    ]
    grouped = vqls.group_greedy(ham, randomized=False)
    assert len(grouped) == 3

def test_measure_group_identity_2q_zgroup():
    # Define number of qubits
    n = 2

    # Get a group to measure
    group = [(1, "IZ"), (-1, "ZI"), (1, "ZZ")]

    # Get a quantum computer to run on
    qcomputer = f"Aspen-7-{n}Q-B"
    lattice = get_qc(qcomputer, as_qvm=True)  # Change to as_qvm=False to run on QC. Must have reservation.

    # Get an ansatz and set the angles
    circ, creg = vqls.yansatz(lattice)
    angles = [0, 0]

    # Compute expectation via individual terms
    itot = 0.
    for coeff, pauli in group:
        itot += vqls.expectation(angles, coeff, pauli, circ, creg, lattice, shots=10_000)
    
    # Compute expectation by grouping
    gtot = vqls.measure_group(angles, group, circ, creg, lattice, shots=10_000)
    
    # Compare to each other
    assert np.isclose(gtot, itot)
    
    # Compare to known answer
    assert np.isclose(gtot, 1.0)


def test_measure_group_identity_2q_xgroup():
    # Define number of qubits
    n = 2

    # Get a group to measure
    group = [(1, "IX"), (-1, "XI")]

    # Get a quantum computer to run on
    qcomputer = f"Aspen-7-{n}Q-B"
    lattice = get_qc(qcomputer, as_qvm=True)  # Change to as_qvm=False to run on QC. Must have reservation.

    # Get an ansatz and set the angles
    circ, creg = vqls.yansatz(lattice)
    angles = [0, 0]

    # Compute expectation via individual terms
    itot = 0.
    for coeff, pauli in group:
        itot += vqls.expectation(angles, coeff, pauli, circ, creg, lattice, shots=10_000)
    
    # Compute expectation by grouping
    gtot = vqls.measure_group(angles, group, circ, creg, lattice, shots=10_000)
    
    # Compare to each other
    assert np.isclose(gtot, itot, atol=0.05)
    
    # Compare to known answer
    assert np.isclose(gtot, 0.0, atol=0.05)


def test_measure_group_2q_xgroup():
    # Define number of qubits
    n = 2

    # Get a group to measure
    group = [(1, "IX"), (-1, "XI")]

    # Get a quantum computer to run on
    qcomputer = f"Aspen-7-{n}Q-B"
    lattice = get_qc(qcomputer, as_qvm=True)  # Change to as_qvm=False to run on QC. Must have reservation.

    # Get an ansatz and set the angles
    circ, creg = vqls.yansatz(lattice)
    angles = [pi / 2, 0]

    # Compute expectation via individual terms
    itot = 0.
    for coeff, pauli in group:
        itot += vqls.expectation(angles, coeff, pauli, circ, creg, lattice, shots=10_000)
    
    # Compute expectation by grouping
    gtot = vqls.measure_group(angles, group, circ, creg, lattice, shots=10_000)
    
    # Compare to each other
    assert np.isclose(gtot, itot, atol=0.05)
    
    # Compare to known anser
    assert np.isclose(gtot, -1.0, atol=0.05)

    
def test_measure_group_2q_xgroup_loop_angles():
    # Define number of qubits
    n = 2

    # Get a group to measure
    group = [(1, "IX"), (-1, "XI")]

    # Get a quantum computer to run on
    qcomputer = f"Aspen-7-{n}Q-B"
    lattice = get_qc(qcomputer, as_qvm=True)  # Change to as_qvm=False to run on QC. Must have reservation.

    # Get an ansatz and set the angles
    circ, creg = vqls.yansatz(lattice)
    
    for angles in ([0, 0], [pi / 4, pi / 4], [pi / 2, pi / 4],
                   [pi / 4, pi / 2], [pi / 2, pi / 2]):
        # Compute expectation via individual terms
        itot = 0.
        for coeff, pauli in group:
            itot += vqls.expectation(angles, coeff, pauli, circ, creg, lattice, shots=10_000)

        # Compute expectation by grouping
        gtot = vqls.measure_group(angles, group, circ, creg, lattice, shots=10_000)

        # Compare to each other
        assert np.isclose(gtot, itot, atol=0.05)

def test_measure_energy_sim():
    """Unit test for measuring <H> using simultaneous measurements."""
    ham = [[(1, "I"), (1, "X")], [(-1, "Z")]]
    n = 1
    qcomputer = f"{n}q-qvm"
    lattice = get_qc(qcomputer, as_qvm=True)
    circ, creg = vqls.yansatz(lattice)
    angles = [0]
    cost = vqls.energy_sim(angles, ham, circ, creg, lattice, shots=10_000)
    assert np.isclose(cost, 0.0, atol=0.05)
