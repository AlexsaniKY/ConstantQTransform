import numpy as np
import matplotlib as mt
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplt
import math
from scipy.ndimage import gaussian_filter1d

def create_signal(t, freq, amp, phase):
	return np.sin(t * freq * 2*np.pi + phase)*amp
	
def create_signal_complex(t, freq, amp, phase):
	return amp * np.exp(1j * (2*np.pi * freq * t + phase))
	
def hann_q(freq, sample_rate, q):
	return np.hanning(math.floor((1/freq) * sample_rate * q ))

#from https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib/20146989#20146989
class MidpointNormalize(mt.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mt.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
	
notes = { 
	'A' : 0,
	'A#': 1,	'Bb': 1,
	'B' : 2,
	'C' : 3,
	'C#': 4,	'Db': 4,
	'D' : 5,
	'D#': 6,	'Eb': 6,
	'E' : 7,
	'F' : 8,
	'F#' : 9,	'Gb' : 9,
	'G' : 10,
	'G#' : 11,	'Ab' : 11
}

note_names = [None]*12
for note, num in notes.items():
	if note_names[num] is None:
		note_names[num] = note
	else: note_names[num]+= "/" + note
	
# converts note name from a midi value
def note_from_midi(midi_note):
	return note_names[(midi_note -9)%12] + str((midi_note+3)//12-2)

# returns the frequency of a note with given name and octave
def freq_from_note(note, octave):
	return (440. * math.pow(2., (notes[note] - 48 + (octave*12))/12.))

# returns the frequency of given midi note. Works with fractional notes	
def freq_from_midi(midi_note):
	return (440. * math.pow(2., (midi_note-69) /12.))
	
print (freq_from_midi(69))
print (note_from_midi(69))
print (freq_from_note('A', 4))


def temporal_kernel(frequencies, q, sample_rate, align = 'center'):
	# get lowest frequency (longest wavelength)
	min_freq = min(frequencies)
	#prepare time values 
	time_span = (1./min_freq) * q
	steps = time_span * sample_rate
    #prepare kernel matrix
	kernel = np.zeros((len(frequencies), int(steps)), complex)
    #align the waves to the matrix
	if align is 'left':
		t_0 = 0
		t_f = time_span
	elif align is 'right':
		t_0 = -1 * time_span
		t_f = 0
	else:
		t_0 = -1 * (time_span/2.)
		t_f = time_span/2.
	t = np.linspace(t_0, t_f, steps)
	
    #create the frequency bins
	frequency_row = 0
	for freq in frequencies:
        #create windowing function
		h = hann_q(freq, sample_rate, q).astype(complex)
        #position the samples in a view
		if align is 'left':
			t_start = 0
			t_end   = h.size
		elif align is 'right':
			t_start = t.size - h.size
			t_end   = t.size
		else:
			t_start = t.size//2 - (h.size//2)
			t_end   = t.size//2 + math.ceil(h.size/2.)
		t_view = t[t_start : t_end]
		
        #create complex sinusoid
		s = create_signal_complex(t_view, freq, 1, 0)
        #replace into the kernel array
		kernel[frequency_row][t_start:t_end] = (1./h.size) * np.multiply(h,s)
        #next frequency
		frequency_row +=1
	return kernel
	
def max_slices(array, slice_width, spacing_ratio):
	if spacing_ratio < 1/float(slice_width):
		space_distance = 1
	else: space_distance = slice_width * spacing_ratio
	return math.ceil(array.shape[0]/space_distance)

def generate_slices(array, slice_width, spacing_ratio):
	position = 0
	if spacing_ratio < 1/float(slice_width):
		space_distance = 1
		print("spacing ratio for slicing is too small, offset of 1 sample per slice assumed")
	else: space_distance = math.ceil(slice_width * spacing_ratio)
	
	while position + slice_width <= array.shape[0]:
		yield array[position: position + slice_width]
		position += space_distance
		#print(position)

		
if __name__ == "__main__":
	import scipy.io.wavfile as wavread
	#rate, waveform =  wavread.read("media\\387517__deleted-user-7267864__saxophone-going-up.wav", np.float32)
	#rate, waveform =  wavread.read("media\\Feathers Rise.wav", np.float32)
	rate, waveform =  wavread.read("media\\134010__davidkyoku__c-major-mutted-scale.wav", np.float32)
	#rate, waveform =  wavread.read("media\\387517__deleted-user-7267864__saxophone-going-up.wav", np.float32)
	
	
	note_start = 69-24 -0
	note_span = 48*2
	note_step = .5
	#nyquist limited frequencies
	frequencies = list(y for y in (freq_from_midi(note_start + (x*note_step)) for x in range(0, int(note_span/note_step), 1)) if y < rate/2)
	#print(frequencies)
	k = temporal_kernel(frequencies, 34, rate, align = 'center')
	#pyplt.pcolormesh(k.real)
	
	#print(rate)
	time = np.linspace(0, waveform.size/float(rate) ,waveform.size)
	if len(waveform.shape) > 1:
		norm_wav = waveform[:,0] / 65536.
	else: norm_wav = waveform / 65536.
	
	cqt = np.zeros((max_slices(norm_wav, k.shape[1], .01), k.shape[0]), complex)
	print("output shape: " + str(cqt.shape))
	slice_index = 0
	for slice in generate_slices(norm_wav, k.shape[1], .01):
		cqt[slice_index] = np.matmul(k, slice)#norm_wav[:k.shape[1]] )
		slice_index += 1
	
	plt.subplot(2,1,1)
	
	cqt = np.abs(cqt).astype(float)
	cqt = gaussian_filter1d(cqt, sigma = 2, axis = 0)
	
	spectralchange = np.copy(cqt)
	spectralchange[1:,:] = spectralchange[1:,:] - spectralchange[:-1,:]
	spectralchange *= 10000
	pyplt.pcolormesh(cqt)
	plt.subplot(2, 1, 2)
	pyplt.pcolormesh(gaussian_filter1d(spectralchange, 1, 0), norm = MidpointNormalize(midpoint = 0))#, cmap = mt.cm.get_cmap('Spectral'))
	
	
	#pyplt.pcolormesh(np.abs(output[:, :-1:2] + output[:, 1::2]))
	# note_spectra = np.zeros((output.shape[0], 12), complex)
	# print(note_spectra.shape)
	# for i in range(12):
		# for j in range(int(output.shape[1]/12)):
			# note_spectra[:,i] += output[:,i + 12*j]
	# pyplt.pcolormesh(gaussian_filter1d(np.abs(note_spectra), 3, 0))
	#plt.plot( np.arange(note_start, note_start + len(frequencies)), np.abs(output))

	
	plt.show()
	

# note_span = 12
# step = 1
# for n in (x*step + 69 for x in range(0, int(note_span//step), 1)):
	# f = freq_from_midi(n)
	# print (note_from_midi(n))
	# h = hann_q(f, sample_rate, 1).astype(complex)
	# s = create_signal_complex(t[:h.size],f,1,0)
	# output = np.multiply(h,s)
	# plt.plot(t[:output.size], output.real)
	# plt.plot(t[:output.size], output.imag)
	
	

# for n in range(69, 69+12, 1):
	# f = freq_from_midi(n)
	# s = create_signal(t, f, 1,0)
	# h = hann_q(f, sample_rate, 34)
	# diff = s.size - h.size
	# print diff
	# plt.plot(t, np.multiply(np.pad(h, (np.int_(math.ceil(diff/2.)), np.int_(math.floor(diff/2.))), 'edge'),s))

	
# plt.plot(t, create_signal(t, 440, 1, 0))
# plt.plot(t, create_signal(t, 1, 1, np.pi/2))
