"""Plots cost vs iteration data from QPU and simulator."""

import matplotlib.pyplot as plt
import numpy as np

# ========
# Get data
# ========

# Aspen-7-5Q-B QPU data
qpu1 = list(np.loadtxt("CVALS_8q_Thu_Feb_6_07:25:16_2020.txt"))


# Aspen-4-5Q-E QPU data
# aspen5qpu1 = list(np.loadtxt("CVALS_5q_Tue_Feb_4_20:12:30_2020_ASPEN_4_5Q_E.txt"))

# ====
# Plot
# ====
plt.rcParams.update({"font.family": "times", "font.weight": "bold", "font.size": 18})
plt.figure(figsize=(10, 5))

# Plot QVM data
# plt.plot(qvm, color="black", lw=3, label="QVM")

# Plot Aspen-7-5Q-B QPU data
plt.plot(qpu1, ls="dashdot", lw=3, label=f"Aspen-7-8Q-B Finish")

# Plot Aspen-4-5Q-E QPU data
# plt.errorbar(xs, aspen5qpu1, yerr=None, ls="dashdot", lw=3, capsize=8, label="Aspen-4-5Q-E")

# Final touches
plt.legend()
plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
plt.grid()
plt.show()
