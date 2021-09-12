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
import sys
import time

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
        self.result_h = numpy.zeros(self.combine_size, dtype=numpy.complex64)

        # self.signal0 = numpy.zeros(self.combine_size, dtype=numpy.complex64)
        # self signal1 = numpy.zeros(self.combine_size, dtype=numpy.complex64)

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.dechirp = numpy.exp(1j*numpy.pi*k/self.M*k)
        self.dechirp_8 = numpy.tile(self.dechirp, self.preamble_len)
        # upchirp
        self.upchirp = numpy.exp(-1j*numpy.pi*k/self.M*k)
        self.upchirp_8 = numpy.tile(self.upchirp, self.preamble_len)

        # draw
        self.image_count = 1
        self.image_count2 = 1
        self.image_count3 = 1
        # self.count = 0
        self.index_signal = 1
        self.index_in0 = 0
        self.index_in1 = 0
        self.input_in0 = False
        self.input_in1 = False
        self.set_output_multiple(self.sending_size)
        

    def draw_graph(self, graph, description, mag, bin2):
        plt.plot(graph)
        plt.savefig(description)
        plt.clf()

    def draw_subplot(self, graph, num_mag, num_bin, gatewayNum):
        description = "/home/yun/LoRa-Gateway/gr-loraGS/python/image/"
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        if gatewayNum == 1:
            # ax = fig.add_subplot(2,1,1)
            ax.plot(graph,'r-',lw=1)
            description += 'input0-%d.png' %(self.image_count)
        elif gatewayNum == 2:
            # ax = fig.add_subplot(2,1,2)
            ax.plot(graph,'g-',lw=1)
            description += 'input1-%d.png' %(self.image_count2)
        else:
            ax.plot(graph,'b-',lw=1)
            description += 'combine-%d.png' %(self.image_count3)
        ax.set_title("index: %d   mag: %.2f   bin: %d" %(self.index_signal,num_mag, num_bin))
        fig.tight_layout()
        fig.savefig(description)

    def write_signal_mag(self, signal, combine):
        sys.stdout = open('/home/yun/Desktop/output.txt', 'a')
        if combine != 1:
            print(signal, end='\t\t')
        else:
            print(signal)
        sys.stdout.close()

    def work(self, input_items, output_items):
        in0 = input_items[0]
        in1 = input_items[1]
        out = output_items[0]
    
        if in0[0] != 0 or in1[0] != 0:
            # signal combine
            if in0[0] != 0:
                # channel multiple signal
                channel_est = in0[:self.combine_size] / self.upchirp_8
                conj_h = numpy.conj(channel_est)
                self.result_h = conj_h * in0[:self.combine_size]
                self.combine_signal += self.result_h

                # Find in0 signal FFT & abs
                in0_signal = in0 * self.dechirp_8
                combine_in0_fft = numpy.fft.fftshift(numpy.fft.fft(in0_signal))
                combine_in0_fft_abs = numpy.abs(combine_in0_fft)
                in0_mag = numpy.max(combine_in0_fft_abs)
                in0_bin = numpy.argmax(combine_in0_fft_abs)
                
                self.draw_subplot(combine_in0_fft_abs,in0_mag,in0_bin,1)
                numpy.savetxt('/home/gnuradio-inc/Yun/text/in0_signal-%d.txt'%(self.image_count), in0_signal.view(float).reshape(-1,2))
                self.image_count += 1
                self.index_in0 = self.index_signal
                self.input_in0 = True
                
            if in1[0] != 0:
                # channel multiple signal
                channel_est = in1[:self.combine_size] / self.upchirp_8
                conj_h = numpy.conj(channel_est)
                self.result_h = conj_h * in1[:self.combine_size]
                self.combine_signal += self.result_h
                
                # Find in1 signal FFT & abs
                in1_signal = in1 * self.dechirp_8
                combine_in1_fft = numpy.fft.fftshift(numpy.fft.fft(in1_signal))
                combine_in1_fft_abs = numpy.abs(combine_in1_fft)
                in1_mag = numpy.max(combine_in1_fft_abs)
                in1_bin = numpy.argmax(combine_in1_fft_abs)
                
                self.draw_subplot(combine_in1_fft_abs,in1_mag,in1_bin,2)
                numpy.savetxt('/home/gnuradio-inc/Yun/text/in1_signal-%d.txt'%(self.image_count2), in1_signal.view(float).reshape(-1,2))
                self.image_count2 += 1
                self.index_in1 = self.index_signal
                self.input_in1 = True
                
            # if self.count == 2:
            if self.input_in0 == True and self.input_in1 == True:
                if numpy.abs(self.index_in0 - self.index_in1) < 10:
                    print("index_in0 : ", self.index_in0)
                    print("index_in1 : ", self.index_in1)
                    # self.combine_signal += self.result_h
                    dechirped_combine_signal = self.combine_signal * self.dechirp_8
                    combine_signal_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_combine_signal))
                    combine_signal_fft_abs = numpy.abs(combine_signal_fft)

                    max_combine_mag = numpy.max(combine_signal_fft_abs)
                    max_combine_bin = numpy.argmax(combine_signal_fft_abs)
                    
                    self.draw_subplot(combine_signal_fft_abs,max_combine_mag,max_combine_bin,3)
                    
                    self.image_count3 += 1
                    
                    # self.count = 0
                    self.combine_signal = numpy.zeros(self.combine_size, dtype=numpy.complex64)
                    self.input_in0 = False
                    self.input_in1 = False
                else:
                    self.input_in0 = False
                    self.input_in1 = False
        
        self.index_signal += 1
        out[:] = in0[:]
        return len(output_items[0])

