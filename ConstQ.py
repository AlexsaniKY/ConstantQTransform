import numpy as np
import matplotlib.pylab as plt

def create_signal(t, freq, amp, phase):
	return np.sin(t * freq * 2*np.pi + phase)*amp

t_0 = 0
t_f = 10
steps = 200

t = np.linspace(t_0, t_f, steps)

freq = 1
amp = 1
phase = 0

plt.plot(t, create_signal(t, 1, 1, 0))
plt.plot(t, create_signal(t, 1, 1, np.pi/2))
plt.show()