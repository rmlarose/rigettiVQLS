"""Plots landscape data from QPU and simulator."""

import matplotlib.pyplot as plt
import numpy as np

# ========
# Get data
# ========

# Simulated data
qvm = list(np.loadtxt("CVALS_QPU_SWEEP_5_Wed_Feb_5_09:32:58_2020.txt"))

# Aspen-7-5Q-B QPU data
qpu1 = list(np.loadtxt("CVALS_SWEEP_5q_Tue_Feb_4_22:36:19_2020.txt"))
qpu2 = list(np.loadtxt("CVALS_SWEEP_5q_Tue_Feb_4_22:41:36_2020.txt"))
qpu3 = list(np.loadtxt("CVALS_SWEEP_5q_Tue_Feb_4_22:47:21_2020.txt"))
qpu4 = list(np.loadtxt("CVALS_SWEEP_5q_Tue_Feb_4_22:53:01_2020.txt"))
qpu5 = list(np.loadtxt("CVALS_SWEEP_5q_Tue_Feb_4_22:58:19_2020.txt"))

# Aspen-4-5Q-E QPU data
# aspen5qpu1 = list(np.loadtxt("CVALS_5q_Tue_Feb_4_20:12:30_2020_ASPEN_4_5Q_E.txt"))

allqpu = np.array([qpu1, qpu2, qpu3, qpu4, qpu5])

# Get rid of duplicates in the cost. Each value is duplicated adjacently.
allqpu = allqpu[:, ::2]


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

xs = np.linspace(-np.pi, np.pi, len(avgqpu))
print(len(xs))
plt.rcParams.update({"font.family": "times", "font.weight": "bold", "font.size": 18})
plt.figure(figsize=(12, 6))

# Plot Aspen-7-5Q-B QPU data
plt.errorbar(xs, avgqpu, yerr=stdqpu, ls="dashdot", lw=3, capsize=8, label="Aspen-7-5Q-B")

# Plot Aspen-4-5Q-E QPU data
#plt.errorbar(xs, aspen5qpu1, yerr=None, ls="dashdot", lw=3, capsize=8, label="Aspen-4-5Q-E")

# Plot QVM data
plt.plot(xs, qvm, "--o", lw=3, label="QVM")

# Final touches
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r"$-\pi$", r"$-\pi / 2$", r"$0$", r"$\pi / 2$", "$\pi$"])
plt.legend()
plt.xlabel(r"$\theta$")
plt.ylabel(r"$C(\theta) = \langle \theta | H | \theta \rangle$")
plt.grid()
plt.show()


