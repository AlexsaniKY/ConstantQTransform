import numpy as np
import matplotlib.pylab as plt
import math

def create_signal(t, freq, amp, phase):
	return np.sin(t * freq * 2*np.pi + phase)*amp
	
notes = { 
	'A' : 0,
	
	'A#': 1,
	'Bb': 1,
	
	'B' : 2,
	
	'C' : 3,
	
	'C#': 4,
	'Db': 4,
	
	'D' : 5,
	
	'D#': 6,
	'Eb': 6,
	
	'E' : 7,
	
	'F' : 8,
	
	'F#' : 9,
	'Gb' : 9,
	
	'G' : 10,
	
	'G#' : 11,
	'Ab' : 11
}
	
def freq_from_note(note, octave):
	return (440. * math.pow(2., (notes[note] - 48 + (octave*12))/12.))
	
def freq_from_midi(midi_note):
	return (440. * math.pow(2., (midi_note-69) /12.))

t_0 = 0
t_f = 1
steps = 200

t = np.linspace(t_0, t_f, steps)

# freq = 1
# amp = 1
# phase = 0

# for note in notes.keys():
	# print note + ": " + str(freq_from_note(note, 0))
print freq_from_note('A', 0)

plt.plot(t, create_signal(t, 440, 1, 0))
plt.plot(t, create_signal(t, 1, 1, np.pi/2))
plt.show()