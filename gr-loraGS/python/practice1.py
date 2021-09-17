import numpy
import matplotlib.pyplot as plt

# k = numpy.linspace(0.0, 1024-1.0, 1024)
# dechirp = numpy.exp(1j*numpy.pi*k/1024*k)
# dechirp_8 = numpy.tile(dechirp, 8)

adjusted_signal = numpy.loadtxt('text/in0_signal-2.txt',dtype=numpy.complex128)

plt.plot(adjusted_signal)
plt.show()

symbol_max_mag = numpy.zeros(8, dtype=numpy.complex64)
symbol_max_index = numpy.zeros(8)
for i in range(8):
    symbol_fft = numpy.fft.fftshift(numpy.fft.fft(adjusted_signal[i*1024:(i+1)*1024]))
    # symbol_fft_abs = numpy.abs(symbol_fft)
    symbol_max_mag[i] = numpy.max(symbol_fft)
    symbol_max_index[i] = numpy.argmax(symbol_fft)
print("=====================FFT====================")
print("bin:", symbol_max_index)
print("max:", symbol_max_mag)
print("====================Angle===================")
npa_angle = numpy.angle(symbol_max_mag)
print(npa_angle)
print("============================================")
li_angle_diff = []
for i in range(1, 8):
    angle_diff = npa_angle[i] - npa_angle[i - 1]
    if(angle_diff < 0):
        angle_diff += 2* numpy.pi
    li_angle_diff.append(angle_diff)
print(li_angle_diff)