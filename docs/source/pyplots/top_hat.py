import numpy as np
import matplotlib.pyplot as plt
import pandas

profile = [0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.2, 0.9, 0.9, 0.9, 0.9]

dates = pandas.date_range("2015-01-01", "2015-12-31")
values = [profile[d.month - 1] for d in dates]

fig = plt.figure(figsize=(8, 5))

plt.plot(np.arange(0, len(dates)), values, linewidth=2)

plt.xlabel("Day of year")
plt.ylabel("Demand factor")

plt.grid(True)
plt.ylim([0.0, 1.5])
plt.xlim(0, 365)
plt.tight_layout()
plt.show()
