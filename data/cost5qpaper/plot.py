"""Plots cost vs iteration data from QPU and simulator."""

import matplotlib.pyplot as plt
import numpy as np

# ========
# Get data
# ========

# Simulated data
qvm = list(np.loadtxt("CVALS_5q_Sun_Feb_2_14:42:11_2020.txt"))

# Aspen-7-5Q-B QPU data
qpu1 = list(np.loadtxt("CVALS_5q_Sun_Feb_2_14:55:47_2020.txt"))
qpu2 = list(np.loadtxt("CVALS_5q_Sun_Feb_2_15:07:05_2020.txt"))
qpu3 = list(np.loadtxt("CVALS_5q_Tue_Feb_4_00:40:29_2020.txt"))

# Aspen-4-5Q-E QPU data
aspen5qpu1 = list(np.loadtxt("CVALS_5q_Tue_Feb_4_20:12:30_2020_ASPEN_4_5Q_E.txt"))

allqpu = np.array([qpu1, qpu2, qpu3])

print(allqpu)

# ====================
# Do averages and stds
# ====================

# QPU
avgqpu = np.nanmean(allqpu, axis=0)
stdqpu = np.nanstd(allqpu, axis=0)

print(avgqpu)
print(len(avgqpu))

# ====
# Plot
# ====
xs = list(range(1, len(avgqpu) + 1))
plt.rcParams.update({"font.family": "times", "font.weight": "bold", "font.size": 18})
plt.figure(figsize=(12, 6))

# Plot Aspen-7-5Q-B QPU data
avqpu_shifted = avgqpu - 0.246
plt.errorbar(xs, avgqpu, yerr=stdqpu, ls="dashdot", lw=3, capsize=8, label="Aspen-7-5Q-B")
plt.errorbar(xs, avqpu_shifted, yerr=stdqpu, ls="dashdot", lw=3, capsize=8, label="Aspen-7-5Q-B Shifted")

# Plot Aspen-4-5Q-E QPU data
#plt.errorbar(xs, aspen5qpu1, yerr=None, ls="dashdot", lw=3, capsize=8, label="Aspen-4-5Q-E")

# Plot QVM data
plt.plot(xs, qvm, "--o", lw=3, label="QVM")

# Final touches
plt.legend()
plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
plt.grid()
plt.show()
