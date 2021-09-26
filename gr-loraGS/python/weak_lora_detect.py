#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 goo-gy.
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
from lora2 import css_demod_algo
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time

from pymongo import MongoClient
import datetime

class weak_lora_detect(gr.sync_block):
    """
    docstring for block weak_lora_detect
    """
    def __init__(self, gatewayName, sf, threshold, preamble_len):
        gr.sync_block.__init__(self,
            name="weak_lora_detect",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

        self.gatewayName = gatewayName
        self.M = int(2**sf)
        self.preamble_len = preamble_len
        self.thres = threshold

        # for charm
        self.max_chunk_count = 40
        self.check_index = 30
        self.signal_buffer = numpy.zeros(self.M * self.max_chunk_count, dtype=numpy.complex64)
        self.signal_index = 0
        self.energe_buffer = numpy.zeros(self.max_chunk_count, dtype=numpy.float) - 1
        self.bin_buffer = numpy.zeros(self.max_chunk_count, dtype=numpy.int) - 1
        self.max_mag = 0
        self.increase_count = 0
        self.decrease_count = 0
        self.enough_increase = False
        self.index_signal = 0

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.upchirp = numpy.exp((1j*2*numpy.pi*((k*k)/(2*self.M))))
        self.upchirp = self.upchirp*numpy.exp(1j*2*numpy.pi*(-k/2))
        self.upchirp_8 = numpy.tile(self.upchirp, 8)

        self.dechirp = numpy.conj(self.upchirp)
        self.dechirp_8 = numpy.tile(self.dechirp, 8)

        # for sending
        self.sending_mode = False
        self.sending_size = 8 * self.M

        # for timing & count
        self.detect_count = 0
        self.work_count = 0

        # --------------------------------------------- DB
        self.client = MongoClient('localhost', 27018)
        self.db = self.client['Lora']
        self.signals = self.db['signals']
        # --------------------------------------------- !DB
        self.set_output_multiple(self.sending_size)

    def find_maximum(self):
        max_index = 0
        for k in range(30):
            if(self.energe_buffer[k] > self.energe_buffer[max_index]):
                max_index = k
        return (max_index, self.energe_buffer[max_index])

    def find_maximum_detail(self, k):
        self.energe_buffer_detail = numpy.zeros(3 * self.M, dtype=numpy.float) - 1
        self.bin_buffer_detail = numpy.zeros(3 * self.M, dtype=numpy.int) - 1
        for i in range(3 * self.M):
            try:
                dechirped_signals = self.signal_buffer[self.M * (k - 8) + i: self.M * k + i]*self.dechirp_8
                dechirped_signals_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_signals))
                self.energe_buffer_detail[i] = numpy.max(numpy.abs(dechirped_signals_fft))
                self.bin_buffer_detail[i] = numpy.argmax(numpy.abs(dechirped_signals_fft))
            except:
                print(self.M * (k - 1) + i - self.M *8, self.M * (k - 1) + i)
        max_index_detail = numpy.argmax(numpy.abs(self.energe_buffer_detail))
        return max_index_detail, self.bin_buffer_detail[max_index_detail]

    def set_frequencyOffset(self, bin_number):
        # find frequency offset
        fft_interval = 125000/(self.M*8)
        frequencyOffset_bin = bin_number - self.M * 4 # self.M * 8 / 2
        frequencyOffset = frequencyOffset_bin*fft_interval
        # make signal
        k = numpy.linspace(0.0, self.M * 8 - 1.0, self.M * 8) / 125000
        offset_signal = numpy.exp(1j*2*numpy.pi*(-frequencyOffset)*k)
        # adjust CFO
        self.signal_adjusted_frequency = self.signal_preamble * offset_signal

    def angdiff(self, a, b):
        d = a - b
        d = numpy.mod(d + numpy.pi, 2*numpy.pi) - numpy.pi
        return d
    
    def set_phase_offset(self):
        symbol_mag = numpy.zeros(self.preamble_len, dtype=numpy.complex64)
        symbol_bin = numpy.zeros(self.preamble_len, dtype=numpy.int)
        #-------------------------------- get angle
        phase_angle = numpy.zeros(self.preamble_len, dtype=numpy.float)
        for i in range(8):
            phase_fft = numpy.fft.fft(self.signal_adjusted_frequency[i*self.M : (i+1)*self.M] * self.dechirp)
            phase_angle[i] = numpy.angle(phase_fft[0])
            symbol_mag[i] = numpy.max(phase_fft)
            symbol_bin[i] = numpy.argmax(phase_fft)
        #-------------------------------- get angle diff
        phase_diff = numpy.zeros(self.preamble_len - 1, dtype=numpy.float)
        for i in range(self.preamble_len - 1):
            phase_diff[i] = self.angdiff(phase_angle[i], phase_angle[i + 1])
        phase_mean = numpy.mean(phase_diff)
        #-------------------------------- adjust phase offset
        CFO_fine=(phase_mean*125000)/(1024*2*numpy.pi)
        k = numpy.linspace(0.0, self.M - 1.0, self.M) / 125000
        self.signal_adjusted_phase = numpy.zeros(self.sending_size, dtype=numpy.complex64)
        for i in range(8):
            self.signal_adjusted_phase[i*self.M : (i+1)*self.M] = \
            self.signal_adjusted_frequency[i*self.M : (i+1)*self.M] * \
            numpy.exp(2j*numpy.pi*(-CFO_fine)*k) * numpy.exp(1j*(phase_mean)*i)


    def draw_specgram(self, graph, description):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.specgram(graph, Fs=1)
        fig.savefig(description)
        fig.clf()


    def draw_energe_signal(self, description, index):
        fig = plt.figure()
        ax = fig.add_subplot(3,1,1)
        ax.set_title("%s" %index)
        ax.grid(True)
        ax.plot(self.energe_buffer)

        ax = fig.add_subplot(3,1,2)
        ax.grid(True)
        ax.specgram(self.signal_buffer, Fs=1)

        ax = fig.add_subplot(3,1,3)
        ax.grid(True)
        ax.specgram(self.signal_preamble, Fs=1)
        fig.savefig(description)
        fig.clf()

    def draw_adjusted(self, description):
        fig = plt.figure()

        ax = fig.add_subplot(2,1,1)
        ax.specgram(self.signal_preamble, Fs=1)

        ax = fig.add_subplot(2,1,2)
        ax.specgram(self.signal_adjusted_frequency, Fs=1)

        fig.savefig(description)
        fig.clf()

    def save_signal_to_db(self):
        signal = {
            "gateway": self.gatewayName,
            "sample_rate": 125000,
            "length": self.sending_size,
            "time": datetime.datetime.utcnow(),
            "bin_num": self.max_bin_detail.item(),
            "mag_max": self.max_mag.item(),
            "real": self.signal_preamble.real.tolist(),
            "imag": self.signal_preamble.imag.tolist()
        }
        signal_id = self.signals.insert_one(signal).inserted_id
        print("Save(%s): %s" %(self.gatewayName, signal_id))

    def detect_lora_signal(self):
        if(self.energe_buffer[self.check_index - 1] > self.energe_buffer[self.check_index - 2]):
            self.decrease_count = 0
            self.increase_count += 1
            self.max_mag = self.energe_buffer[self.check_index - 1]
            if(self.increase_count >= 2):
                self.enough_increase = True
        elif(self.energe_buffer[self.check_index - 1] < self.energe_buffer[self.check_index - 2]):
            self.increase_count = 0
            self.decrease_count += 1
            if(self.enough_increase):
                if(self.decrease_count >= 2):
                    self.enough_increase = False
                    if(self.max_mag > self.energe_buffer[0] * 2):
                        return True
        return False

    def work(self, input_items, output_items):
        self.work_count += 1
        ## save signal
        signal_size = len(input_items[0])
        self.signal_buffer = numpy.roll(self.signal_buffer, -signal_size)
        self.signal_buffer[-signal_size:] = input_items[0]
        
        ## signal_size check
        if(self.signal_index < self.M * (self.preamble_len+2)):
            self.signal_index += signal_size
            return len(output_items[0])
        # else
        n_syms = signal_size//self.M
 
        for i in range(0, n_syms):
            ## save energe buffer
            dechirped_signals = self.signal_buffer[(self.max_chunk_count + 1 - n_syms + i - 8)*self.M:(self.max_chunk_count + 1 - n_syms + i)*self.M] * self.dechirp_8
            dechirped_signals_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_signals))

            self.energe_buffer = numpy.roll(self.energe_buffer, -1)
            self.energe_buffer[-1] = numpy.max(numpy.abs(dechirped_signals_fft))
            self.bin_buffer = numpy.roll(self.bin_buffer, -1)
            self.bin_buffer[-1] = numpy.argmax(numpy.abs(dechirped_signals_fft))
        
        # detect
        for i in range(0, n_syms):
            if(self.detect_lora_signal()):
                print("detect lora preamble (%s):%d" %(self.gatewayName, self.work_count))
                self.detect_count += 1
                max_index, energe = self.find_maximum()
                max_index_detail, max_bin_detail = self.find_maximum_detail(max_index)
                self.max_bin_detail = max_bin_detail
                signal_timing_index = self.M * (max_index - 8) + max_index_detail
                # adjust signal
                self.signal_preamble = self.signal_buffer[signal_timing_index: signal_timing_index + self.sending_size].copy()
                self.set_frequencyOffset(max_bin_detail)
                self.set_phase_offset()

                # self.save_signal_to_db()
                self.draw_specgram(self.signal_preamble, "signal_preamble(%s)-%d" %(self.gatewayName, self.work_count))
                # self.draw_energe_signal("energe-signal-%s-%d" %(self.gatewayName, self.work_count), signal_timing_index)
                self.draw_adjusted("signal_adjusted-%s-%d" %(self.gatewayName, self.work_count))
                self.sending_mode = True
            ## check

        # send
        if(self.sending_mode):
            output_items[0][:] = self.signal_adjusted_phase
        else:
            output_items[0][:] = numpy.zeros(self.sending_size, dtype=numpy.complex64)
        self.sending_mode = False
        self.index_signal += 1
        return len(output_items[0])