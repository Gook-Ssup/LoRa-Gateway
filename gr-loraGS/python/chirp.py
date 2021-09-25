import numpy
import matplotlib.pyplot as plt


def draw_specgram(graph, description):
    plt.specgram(graph, Fs=125000)
    plt.savefig(description)
    plt.clf()

M = 1024
k = numpy.linspace(0.0, M-1.0, M)
dechirp = numpy.exp(-1j*numpy.pi*k/M*k)
dechirp_8 = numpy.tile(dechirp, 8)

draw_specgram(dechirp_8, "1")

dechirp = numpy.exp((1j*numpy.pi*(k*k)/M))
dechirp = numpy.conj(dechirp)
dechirp_8 = numpy.tile(dechirp, 8)

draw_specgram(dechirp_8, "2")

upchirp = numpy.exp((1j*2*numpy.pi*((k*k)/(2*M))))
upchirp = dechirp*numpy.exp(1j*2*numpy.pi*(-k/2))
dechirp = numpy.conj(upchirp)
dechirp_8 = numpy.tile(dechirp, 8)

draw_specgram(dechirp_8, "3")