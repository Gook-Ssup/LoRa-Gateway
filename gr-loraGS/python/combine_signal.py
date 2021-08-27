#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 combine_signal.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#


import numpy
from gnuradio import gr

import matplotlib.pyplot as plt
plt.switch_backend('agg')

class combine_signal(gr.sync_block):
    """
    docstring for block combine_signal
    """
    def __init__(self, sf, preamble_length):
        gr.sync_block.__init__(self,
            name="combine_signal",
            in_sig=[numpy.complex64, numpy.complex64],
            out_sig=[numpy.complex64])

        self.M = int(2**sf)
        self.preamble_len = preamble_length

        self.sending_size = self.M * 8
        self.combine_size = self.M * 8
        self.combine_signal = numpy.zeros(self.combine_size, dtype=numpy.complex64)

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.dechirp = numpy.exp(1j*numpy.pi*k/self.M*k)
        self.dechirp_8 = numpy.tile(self.dechirp, self.preamble_len)
        # upchirp
        self.upchirp = numpy.exp(-1j*numpy.pi*k/self.M*k)
        self.upchirp_8 = numpy.tile(self.upchirp, self.preamble_len)

        # draw
        self.image_count = 1
        
        self.set_output_multiple(self.sending_size)

    def draw_graph(self, graph, description, mag, bin):
        plt.plot(graph)
        plt.title("mag: %.2f    bin: %d" %(mag,bin))
        plt.savefig(description)
        plt.clf()

    def work(self, input_items, output_items):
        in0 = input_items[0]
        in1 = input_items[1]
        out = output_items[0]
        self.combine_signal = numpy.zeros(self.combine_size, dtype=numpy.complex64)

        if in0[0] != 0 or in1[0] != 0:
            # signal combine
            if in0[0] != 0:
                channel_est = in0[:self.combine_size] / self.upchirp_8
                conj_h = numpy.conj(channel_est)
                result_h = conj_h * in0[:self.combine_size]
            if in1[0] != 0:
                channel_est = in1[:self.combine_size] / self.upchirp_8
                conj_h = numpy.conj(channel_est)
                result_h = conj_h * in1[:self.combine_size]
            self.combine_signal += result_h
            dechirped_combine_signal = self.combine_signal * self.dechirp_8
            combine_signal_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_combine_signal))
            combine_signal_fft_abs = numpy.abs(combine_signal_fft)

            description4 = "/home/yun/LoRa-Gateway/gr-loraGS/python/image/combine_abs%d.png" %(self.image_count)
            max_combine_mag = numpy.max(combine_signal_fft_abs)
            max_combine_bin = numpy.argmax(combine_signal_fft_abs)
            self.draw_graph(combine_signal_fft_abs, description4, max_combine_mag, max_combine_bin)
            self.image_count += 1
        out[:] = in0[:]
        return len(output_items[0])

