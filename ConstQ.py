import numpy as np
import matplotlib.pylab as plt

x = np.linspace(-np.pi, np.pi, 201)

freq = 1
amp = 1

plt.plot(x, np.sin(x*freq)*amp)
plt.show()