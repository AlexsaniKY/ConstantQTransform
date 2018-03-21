import numpy as np
import matplotlib.pylab as plt
import math

def create_signal(t, freq, amp, phase):
	return np.sin(t * freq * 2*np.pi + phase)*amp
	
def create_signal_complex(t, freq, amp, phase):
	return amp * np.exp(1j * (2*np.pi * freq * t + phase))
	
def hann_q(freq, sample_rate, q):
	return np.hanning((1/freq) * sample_rate * q )
	
def hann_q_complex(freq, sample_rate,q):
	return np.hanning((1/freq) * sample_rate * q ).astype(complex)
	

	
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
	return note_names[(midi_note -9)%12] + str((midi_note+3)//12-2)
	
# for n in range(100):
	# print note_from_midi(n)
	
def freq_from_note(note, octave):
	return (440. * math.pow(2., (notes[note] - 48 + (octave*12))/12.))
	
def freq_from_midi(midi_note):
	return (440. * math.pow(2., (midi_note-69) /12.))
	


sample_rate = 48000 #samples/second
t_0 = 0   #seconds
t_f = .1 #seconds
steps = sample_rate * (t_f - t_0)

t = np.linspace(t_0, t_f, steps)

# freq = 1
# amp = 1
# phase = 0
print freq_from_midi(69)
print note_from_midi(69)
print freq_from_note('A', 4)

note_span = 12
step = 1
for n in (x*step + 69 for x in range(0, int(note_span//step), 1)):
	f = freq_from_midi(n)
	print note_from_midi(n)
	h = hann_q_complex(f, sample_rate, 1)
	s = create_signal_complex(t[:h.size],f,1,0)
	output = np.multiply(h,s)
	plt.plot(t[:output.size], output.real)
	plt.plot(t[:output.size], output.imag)
	
	

# for n in range(69, 69+12, 1):
	# f = freq_from_midi(n)
	# s = create_signal(t, f, 1,0)
	# h = hann_q(f, sample_rate, 34)
	# diff = s.size - h.size
	# print diff
	# plt.plot(t, np.multiply(np.pad(h, (np.int_(math.ceil(diff/2.)), np.int_(math.floor(diff/2.))), 'edge'),s))

	
# plt.plot(t, create_signal(t, 440, 1, 0))
# plt.plot(t, create_signal(t, 1, 1, np.pi/2))
plt.show()