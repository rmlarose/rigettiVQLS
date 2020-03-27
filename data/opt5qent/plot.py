"""Plots cost vs iteration data from QPU and simulator."""

import matplotlib.pyplot as plt
import numpy as np

# ========
# Get data
# ========

# Simulated data
qvm = list(np.loadtxt("CVALS_QVM_5q_Fri_Feb_7_08:29:09_2020.txt"))

# Aspen-7-5Q-B QPU data
qpu1 = list(np.loadtxt("CVALS_5q_Thu_Feb_6_22:07:18_2020.txt"))

allqpu = [qpu1]

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
plt.errorbar(xs, qpu1, yerr=stdqpu, ls="dashdot", lw=3, capsize=8, label="Aspen-7-5Q-B")

# Final touches
plt.legend()
plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
plt.grid()
plt.show()
