import numpy as np
import matplotlib.pyplot as plt
import pandas

baseline = 50.0

profile = np.array([0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.2, 0.9, 0.9, 0.9, 0.9])

L0 = np.array([1.0] * 12)
L1 = np.array([0.95, 0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.95])
L2 = np.array([0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8])

dates = pandas.date_range("2015-01-01", "2015-12-31")

L0_values = np.array([baseline * profile[d.month - 1] * L0[d.month - 1] for d in dates])
L1_values = np.array([baseline * profile[d.month - 1] * L1[d.month - 1] for d in dates])
L2_values = np.array([baseline * profile[d.month - 1] * L2[d.month - 1] for d in dates])

fig = plt.figure(figsize=(8, 5), dpi=300)

plt.plot(np.arange(0, len(dates)), L0_values, linewidth=2)
plt.plot(np.arange(0, len(dates)), L1_values, linewidth=2)
plt.plot(np.arange(0, len(dates)), L2_values, linewidth=2)

plt.xlabel("Day of year")
plt.ylabel("Demand [Ml/d]")

plt.grid(True)
plt.ylim([0.0, 70])
plt.xlim(0, 365)
plt.legend(["Level 0", "Level 1", "Level 2"], loc="lower right")
plt.tight_layout()
plt.show()
