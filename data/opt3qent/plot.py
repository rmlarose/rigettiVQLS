"""Plots cost vs iteration data from QPU and simulator."""

import matplotlib.pyplot as plt
import numpy as np

# ========
# Get data
# ========

# Simulated data
qvm = list(np.loadtxt("CVALS_QVM_3q_Fri_Feb_7_08:11:57_2020.txt"))

# Aspen-7-5Q-B QPU data
qpu1 = list(np.loadtxt("CVALS_3q_Fri_Feb_7_01:55:32_2020.txt"))
qpu2 = list(np.loadtxt("CVALS_3q_Fri_Feb_7_02:06:11_2020.txt"))

def pad(short, long):
    short += [short[-1]] * (len(long) - len(short))

if len(qpu1) < len(qpu2):
    pad(qpu1, qpu2)
else:
    pad(qpu2, qpu1)

assert len(qpu1) == len(qpu2)

allqpu = [qpu1, qpu2]

# ====================
# Do averages and stds
# ====================

# QPU
avgqpu = np.nanmean(allqpu, axis=0)
stdqpu = np.nanstd(allqpu, axis=0)

# ====
# Plot
# ====
xs = list(range(1, len(qpu1) + 1))
plt.rcParams.update({"font.family": "times", "font.weight": "bold", "font.size": 18})
plt.figure(figsize=(9, 9))

# Plot QVM data
plt.plot(qvm, color="black", lw=3, label="QVM")

# Plot Aspen-7-5Q-B QPU data
plt.errorbar(xs, avgqpu, yerr=stdqpu, ls="dashdot", lw=3, capsize=8, label="Aspen-7-3Q-B")

# Final touches
plt.legend()
plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
plt.grid()
plt.show()
