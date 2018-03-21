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

note_names = [None]*12
for note, num in notes.items():
	if note_names[num] is None:
		note_names[num] = note
	else: note_names[num]+= "/" + note
	
def note_from_midi(midi_note):
	return note_names[(midi_note -9)%12] + str((midi_note+3)//12)
	
# for n in range(100):
	# print note_from_midi(n)
	
def freq_from_note(note, octave):
	return (440. * math.pow(2., (notes[note] - 48 + (octave*12))/12.))
	
def freq_from_midi(midi_note):
	return (440. * math.pow(2., (midi_note-69) /12.))

sample_rate = 48000 #samples/second
t_0 = 0   #seconds
t_f = .01 #seconds
steps = sample_rate * (t_f - t_0)

t = np.linspace(t_0, t_f, steps)

# freq = 1
# amp = 1
# phase = 0

# for note in notes.keys():
	# print note + ": " + str(freq_from_note(note, 0))
window = np.hanning(steps)
#plt.plot(t, window)
	
for n in range(69, 69+12, 1):
	plt.plot(t, window * create_signal(t, freq_from_midi(n), 1,0))

# plt.plot(t, create_signal(t, 440, 1, 0))
# plt.plot(t, create_signal(t, 1, 1, np.pi/2))
plt.show()