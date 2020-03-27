"""VQE for a scalar quantum field."""

from math import pi
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from pyquil import Program, get_qc
from pyquil.gates import H, S, RX, MEASURE
from pyquil.quilbase import Measurement

def movemeasurements(circuit):
    """Helper function because Rigetti sucks. Moves all measurements
    to the end of the circuit."""
    for instr in circuit.instructions:
        if type(instr) == Measurement:
            circuit.instructions.remove(instr)
            circuit.instructions.append(instr)
    return circuit
        

# Value of \omega_\phi [1] which determines the interaction strength
# omega = 1 <==> no interaction, omega --> 0 <==> large interaction
omega = 0.5

# Computer to run on
simulator = "3q-qvm"
qcomputer = "Aspen-4-3Q-A"
computer = get_qc(qcomputer, as_qvm=False) # Change to as_qvm=False to run on QC. Must have reservation.

def ansatz():
    """Returns a circuit with a product state ansatz."""
    # Get a circuit and classical memory register
    circ = Program()
    creg = circ.declare("ro", memory_type="BIT", memory_size=3)

    # Define parameters for the ansatz
    angles = circ.declare("theta", memory_type="REAL", memory_size=3)

    # Add the ansatz
    circ += [RX(angles[0], 10), RX(angles[1], 11), RX(angles[2], 17)]
    
    return circ, creg

"""Get an example of ansatz."""
circ, creg = ansatz()

print(circ)

# "Global" coefficients on H_basis and \delta H
Cbasis = omega
Cdelta = (1.0 - omega**2) / omega

# Numerical values of coefficeints above
a = (np.sqrt(3.0) - np.sqrt(5.0)) / 8.0
b = (np.sqrt(3.0) + np.sqrt(5.0)) / 8.0
c = (1.0 - np.sqrt(3.0) + np.sqrt(21.0) - np.sqrt(15.0)) / 8 / np.sqrt(2.0)
d = (1.0 + np.sqrt(3.0) - np.sqrt(21.0) - np.sqrt(15.0)) / 8 / np.sqrt(2.0)
e = (1.0 - np.sqrt(3.0) - np.sqrt(21.0) + np.sqrt(15.0)) / 8 / np.sqrt(2.0)
f = (1.0 + np.sqrt(3.0) + np.sqrt(21.0) + np.sqrt(15.0)) / 8 / np.sqrt(2.0)

# Dictionary of coefficients, starting numbering with H_basis and then \delta H
coeffs = {0: -2.0 * Cbasis,
          1: -1.0 * Cbasis,
          2: -0.5 * Cbasis,
          3: a * Cdelta,
          4: b * Cdelta,
          5: a * Cdelta,
          6: b * Cdelta,
          7: c * Cdelta,
          8: d * Cdelta,
          9: e * Cdelta,
         10: f * Cdelta,
         11: -1.0 * Cdelta,
         12: -0.5 * Cdelta,
         13: -0.25 * Cdelta}

# Helper function
def expectation(pauli, qubit):
    """Returns a pyquil.Program with the correct basis measurement 
    for the given Pauli operator and qubit.
    
    Args:
        pauli : str
            Either "X", "Y", or "Z".

        qubit : int
            Index of qubit in the Hamiltonian.
            Note: This should be an actual physical qubit index
                  on the qubit lattice being considered if running
                  on a quantum chip.
    """
    # Get the index of the classical register for the given qubit index
    if qubit == 10:
        ind = 0
    elif qubit == 11:
        ind = 1
    elif qubit == 17:
        ind = 2
    else:
        raise ValueError("Unsupported qubit index for computer. Rigetti will let you know about this...")

    # Do the appropriate basis measurement
    if pauli == "Z":
        return Program(MEASURE(qubit, creg[ind]))
    elif pauli == "X":
        return Program(H(qubit), MEASURE(qubit, creg[ind]))
    elif pauli == "Y":
        return Program(S(qubit), H(qubit), MEASURE(qubit, creg[ind]))
    else:
        raise ValueError("Unsupported operator. Enter X, Y, or Z.")


# Dictionary
measure = {0: expectation("Z", 10),
           1: expectation("Z", 11),
           2: expectation("Z", 17),
           3: expectation("Z", 10) + expectation("X", 11) + expectation("Z", 17),
           4: expectation("X", 10) + expectation("X", 11),
           5: expectation("Y", 10) + expectation("Y", 11) + expectation("Z", 17),
           6: expectation("Y", 10) + expectation("Y", 11),
           7: expectation("Z", 10) + expectation("X", 11) + expectation("Z", 17),
           8: expectation("Z", 10) + expectation("X", 11),
           9: expectation("X", 11) + expectation("Z", 17),
          10: expectation("X", 11),
          11: expectation("Z", 10),
          12: expectation("Z", 11),
          13: expectation("Z", 17)}


def expectation(term, angles, computer, shots=10000, verbose=False):
    """Returns the expectation value <angles|H_{term}|angles>."""   
    circ, creg = ansatz()
    
    circ += measure[term]
    
    # Set the number of shots
    circ.wrap_in_numshots_loop(shots)
    
    # Verbose option
    if verbose:
        print("Circuit:\n=======")
        print(circ)
    
    # Execute the circuit
    circ = movemeasurements(circ)
    executable = computer.compile(circ)
    res = computer.run(executable, memory_map={"theta": angles})
    
    # Do the postprocessing
    tot = 0.0
    for vals in res:
        tot += (-1)**sum(vals)
    return tot / shots



def weighted_expectation(term, angles, computer, shots=10000, verbose=False):
    """Returns coeff * <angles|H_{term}|angles>."""
    return coeffs[term] * expectation(term, angles, computer, shots, verbose)



def energy(angles, computer, shots=10000, verbose=False):
    """Returns the energy <angles|H|angles>."""
    # Variable to store the total energy
    tot = 0.0
    
    # Loop over all terms and add the weighted expectations
    for term in coeffs.keys():
        tot += weighted_expectation(term, angles, computer, shots, verbose)
    
    # Add in the constant values from the Hamiltonian
    tot += 4 * Cbasis
    tot += 2 * Cdelta
    
    return tot


OBJVALS = []

def obj(angles):
    """Returns the energy for a given set of angles."""
    val = energy(angles, computer, shots=10000, verbose=False)
    OBJVALS.append(val)
    print("Current energy =", round(val, 7), end="\r")
    return val



start = time.time()
res = minimize(obj, np.random.rand(3), method="COBYLA")
end = time.time()
print("Total runtime = {} minutes.".format((end - start) / 60))

"""Print out the final result."""
print(res)



# Only include the decreasing costs, since the optimizer tries random values occasionally
energies = [OBJVALS[0]]
for x in OBJVALS:
    if x < energies[-1]:
        energies.append(x)
    else:
        energies.append(energies[-1])
        
# Print the energies just in case they don't save
print(energies)
energies = np.array(energies)
np.savetxt(time.asctime(), energies)

# Do the plotting
plt.figure(figsize=(10, 7))
plt.plot(energies, "--o", linewidth=3, label="VQE Energy")
plt.xlabel("Iteration", fontsize=14, fontweight="bold")
plt.ylabel("Energy", fontsize=14, fontweight="bold")

# Put a line at the actual ground state energy (see below)
GSENERGY = 0.53232723
plt.plot(GSENERGY * np.ones_like(energies), linewidth=3, label="Analytic Energy")

plt.grid()
plt.legend()

plt.savefig(time.asctime() + ".pdf", format="pdf")
