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
import math

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
        # self.channel_value = numpy.zeros(2, dtype=numpy.complex64)
        self.conj_channel0 = 0
        self.conj_channel1 = 0

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.upchirp = numpy.exp((1j*2*numpy.pi*((k*k)/(2*self.M))))
        self.upchirp = self.upchirp*numpy.exp(1j*2*numpy.pi*(-k/2))
        self.upchirp_8 = numpy.tile(self.upchirp, 8)

        self.dechirp = numpy.conj(self.upchirp)
        self.dechirp_8 = numpy.tile(self.dechirp, 8)


        self.fft_abs_in0 = numpy.zeros(1024, dtype=numpy.float)
        self.fft_abs_in1 = numpy.zeros(1024, dtype=numpy.float)

        # for timming
        self.work_count = 0
        self.work_count_in0 = 0
        self.work_count_in1 = 0
        self.input_in0 = False
        self.input_in1 = False

        self.set_output_multiple(self.sending_size)

    def get_channel_mean(self,signal):
        channel_est = numpy.zeros(self.M * self.preamble_len, dtype=numpy.complex64)
        for i in range(8):
            channel_est[i*self.M : (i+1)*self.M] = signal[i*self.M : (i+1)*self.M] / self.upchirp
        return numpy.mean(channel_est)


    def draw_specgram(self, graph, description):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.specgram(graph, Fs=1)
        fig.savefig(description)
        fig.clf()

    def draw_result(self):
        description = "combine-result-%d" %self.work_count
        # plt.title(description)
        fig = plt.figure()
        ax = fig.add_subplot(1,3,1)
        ax.grid(True)
        ax.plot(self.fft_abs_in0)

        ax = fig.add_subplot(1,3,2)
        ax.grid(True)
        ax.plot(self.fft_abs_in1)

        ax = fig.add_subplot(1,3,3)
        ax.grid(True)
        ax.plot(self.fft_abs_combine_signal)
        fig.savefig(description)
        fig.clf()

    def estimate_result(self):
        snr_ratio_combined = self.fft_abs_combine_signal[512] / (numpy.mean(self.fft_abs_combine_signal[0:512])+numpy.mean(self.fft_abs_combine_signal[513:1024]))
        snr_ratio_in0 = self.fft_abs_in0[512] / (numpy.mean(self.fft_abs_in0[0:512])+numpy.mean(self.fft_abs_in0[513:1024]))
        snr_ratio_in1 = self.fft_abs_in1[512] / (numpy.mean(self.fft_abs_in1[0:512])+numpy.mean(self.fft_abs_in1[513:1024]))

        print("SNR Ratio Combined : ", snr_ratio_combined)
        print("SNR Ratio in0 : ", snr_ratio_in0)
        print("SNR Ratio in1 : ", snr_ratio_in1)
        # self.draw_result()
    
    def work(self, input_items, output_items):
        self.work_count += 1

        in0 = input_items[0]
        in1 = input_items[1]
        out = output_items[0]

        if in0[0] != 0 or in1[0] != 0:
            # check signal in0
            if in0[0] != 0:
                channel_mean = self.get_channel_mean(in0)
                self.conj_channel0 = numpy.conj(channel_mean)
                self.combine_signal += (self.conj_channel0 * in0)
                # for estimation
                in0_dechirped = in0[:1024] * self.dechirp
                dechirped_fft_in0 = numpy.fft.fftshift(numpy.fft.fft(in0_dechirped))
                self.fft_abs_in0 = numpy.abs(dechirped_fft_in0)
                # for sync
                self.work_count_in0 = self.work_count
                self.input_in0 = True
            
            # check signal in1
            if in1[0] != 0:
                channel_mean = self.get_channel_mean(in1)
                self.conj_channel1 = numpy.conj(channel_mean)
                self.combine_signal += (self.conj_channel1 * in1)
                # for estimation
                in1_dechirped = in1[:1024] * self.dechirp
                dechirped_fft_in1 = numpy.fft.fftshift(numpy.fft.fft(in1_dechirped))
                self.fft_abs_in1 = numpy.abs(dechirped_fft_in1)
                # for sync
                self.work_count_in1 = self.work_count
                self.input_in1 = True
            
            if self.input_in0 == True and self.input_in1 == True:
                if numpy.abs(self.work_count_in0 - self.work_count_in1) < 10:
                    print("work_count_in0 : ", self.work_count_in0)
                    print("work_count_in1 : ", self.work_count_in1)
                    # nomalization
                    normalization_factor = math.sqrt(numpy.abs(self.conj_channel0)**2 + numpy.abs(self.conj_channel1)**2)
                    self.combine_signal = self.combine_signal / normalization_factor

                    # for estimation
                    dechirped_combine_signal = (self.combine_signal[:1024]/normalization_factor) * self.dechirp
                    fft_combine_signal = numpy.fft.fftshift(numpy.fft.fft(dechirped_combine_signal))
                    self.fft_abs_combine_signal = numpy.abs(fft_combine_signal)
                    self.estimate_result()

                    self.draw_specgram(self.combine_signal, "combine_signal-%d" %self.work_count)
                    self.combine_signal = numpy.zeros(self.combine_size, dtype=numpy.complex64) # reset

                # reset
                self.fft_abs_in0 = numpy.zeros(1024, dtype=numpy.float)
                self.fft_abs_in1 = numpy.zeros(1024, dtype=numpy.float)
                self.input_in0 = False
                self.input_in1 = False
        
        out[:] = in0[:]
        return len(output_items[0])