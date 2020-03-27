"""Plots sweep data from QPU and simulator."""

import matplotlib.pyplot as plt
import numpy as np

# ========
# Get data
# ========

# QPU data
qpu1 = list(np.loadtxt("sweep_data_Sun_Jan_26_02:10:22_2020_QPU_ASPEN_7_3Q_B_COMPLETE_RUN1.txt"))
qpu2 = list(np.loadtxt("sweep_data_Sun_Jan_26_03:35:07_2020_QPU_ASPEN_7_3Q_B_COMPLETE_RUN2.txt"))
qpu3 = list(np.loadtxt("sweep_data_Sun_Jan_26_04:35:16_2020_QPU_ASPEN_7_3Q_B_COMPLETE_RUN3.txt"))
allqpu = np.array([qpu1, qpu2, qpu3])

# QVM data
qvm = np.loadtxt("sweep_data_Sun_Jan_26_21:52:59_2020_QVM.txt")
print(qvm)

# ====================
# Do averages and stds
# ====================

# QPU
avgqpu = np.nanmean(allqpu, axis=0)
stdqpu = np.nanstd(allqpu, axis=0)

# ====
# Plot
# ====
xs = np.linspace(-np.pi, np.pi, len(qpu1))
plt.rcParams.update({"font.family": "times", "font.weight": "bold", "font.size": 18})
plt.figure(figsize=(12, 6))

# Plot QPU data
plt.errorbar(xs, avgqpu, yerr=stdqpu, ls="dashdot", lw=3, capsize=8, label="Aspen-7-3Q-B")

# Plot QVM data
plt.plot(xs, qvm, "--o", lw=3, label="QVM")

# Final touches
plt.legend()
plt.xlabel(r"$\theta$")
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r"$-\pi$", r"$-\pi / 2$", r"$0$", r"$\pi / 2$", "$\pi$"])
plt.ylim(0, 1.2)
plt.ylabel(r"$C(\theta) = \langle \theta | H | \theta \rangle$")
plt.grid()
plt.show()

