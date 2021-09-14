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

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.dechirp = numpy.exp(1j*numpy.pi*k/self.M*k)
        self.dechirp_8 = numpy.tile(self.dechirp, 8)

        # for sending
        self.sending_mode = False
        self.sending_size = 8 * self.M

        # for drawing
        self.image_count = 0

        # ------------------------ for checking ----------------------------------
        self.demod = css_demod_algo(self.M)
        self.demod_conj = css_demod_algo(self.M, True)

        if preamble_len > 2:
            self.buffer = numpy.zeros(preamble_len + 2, dtype=numpy.int) - 1
            self.complex_buffer = numpy.zeros(preamble_len + 2, dtype=numpy.complex64)
            self.buffer_meta = [dict() for i in range(0, preamble_len + 2)]
        else:
            self.buffer = numpy.zeros(5, dtype=numpy.int) - 1
            self.complex_buffer = numpy.zeros(5, dtype=numpy.complex64)
            self.buffer_meta = [dict() for i in range(0, 5)]
        # ------------------------ !for checking ----------------------------------

        # --------------------------------------------- DB
        self.register_db = False
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
        # self.draw_graph(self.energe_buffer_detail, "%d-detail" %(self.image_count))
        max_index = numpy.argmax(numpy.abs(self.energe_buffer_detail))
        return max_index, self.bin_buffer_detail[max_index]

    def set_frequencyOffset(self, signal_index, bin_number):
        # find frequency offset
        fft_interval = 125000/(self.M*8)
        frequencyOffset_bin = bin_number - self.M * 4 # self.M * 8 / 2            // because shifted
        frequencyOffset = frequencyOffset_bin*fft_interval
        print("Bin Offset", frequencyOffset_bin)

        # make signal
        k = numpy.linspace(0.0, self.M * 8 - 1.0, self.M * 8) / 125000
        offset_signal = numpy.exp(1j*2*numpy.pi*(-frequencyOffset)*k)

        # adjust CFO
        self.adjusted_signal = self.signal_buffer[signal_index:signal_index + self.M * 8] * offset_signal
        dechirped_adjusted_signal = self.adjusted_signal * self.dechirp_8
        # ----------------- drawing ------------------
        offset_ffted = numpy.fft.fft(offset_signal)
        # self.draw_graph(offset_ffted, '%d_1_offset_ffted' %(-frequencyOffset_bin)) 

        dechirped_origin_ffted = numpy.fft.fft(self.signal_buffer[signal_index:signal_index + self.M * 8]*self.dechirp_8)
        # self.draw_graph(abs(dechirped_origin_ffted), '%d_2_dechirped_origin_ffted' %(-frequencyOffset_bin))

        dechirped_adjusted_ffted = numpy.fft.fft(dechirped_adjusted_signal)
        # self.draw_graph(abs(dechirped_adjusted_ffted), '%d_3_dechirped_adjusted_ffted' %(-frequencyOffset_bin))
        
        # ----------------- !drawing ----------------- 
        adjusted_bin = numpy.argmax(numpy.abs(dechirped_adjusted_ffted))
        print("adjusted:", adjusted_bin)
        # mine
        # self.channel_estimation()
        return adjusted_bin

    def channel_estimation(self):
        k = numpy.linspace(0.0, self.M - 1.0, self.M)
        self.upchirp = numpy.exp(-1j*numpy.pi*k/self.M*k)
        self.upchirp_8 = numpy.tile(self.upchirp, self.preamble_len)

        channel_est = self.adjusted_signal / self.upchirp_8

        for i in range(8):
            plt.plot(channel_est[i*self.M : (i+1)*self.M])
            # mine
            plt.savefig("est-%d-%d.png" %(self.image_count, i))
            plt.clf()

    def detect_preamble(self):
        if self.buffer[0] == -1:
            return False

        mean = numpy.mean(self.buffer[-(self.preamble_len+2):-2])
        mean_err_sq = numpy.sum(numpy.abs(self.buffer[-(self.preamble_len+2):-2] - mean)**2)
        max_err_sq = self.M**2

        if(mean_err_sq/max_err_sq < self.thres):
            self.buffer_meta[self.preamble_len-1]['preamble_value'] = numpy.uint16(numpy.round(mean))
            return True
        else:
            pass

    def draw_graph(self, graph, description):
        plt.plot(graph)
        plt.savefig(description)
        plt.clf()

    def save_signal_to_db(self):
        signal = {
            "gateway": self.gatewayName,
            "sample_rate": 125000,
            "length": self.sending_size,
            "time": datetime.datetime.utcnow(),
            "bin_num": self.max_bin_detail.item(),
            "mag_max": self.max_mag.item(),
            "real": self.signal_buffer[self.signal_timing_index : self.signal_timing_index + self.sending_size].real.tolist(),
            "imag": self.signal_buffer[self.signal_timing_index : self.signal_timing_index + self.sending_size].imag.tolist()
        }
        signal_id = self.signals.insert_one(signal).inserted_id
        print("Save(%s): %s" %(self.gatewayName, signal_id))
        # self.register_db = True
        #

    def work(self, input_items, output_items):
        signal_size = len(input_items[0])

        ## save signal
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
            
            ## check
            if(self.energe_buffer[self.check_index - 1] > self.energe_buffer[self.check_index - 2]):
                self.decrease_count = 0
                self.increase_count += 1
                self.max_mag = self.energe_buffer[self.check_index - 1]
                if(self.increase_count >= 5):
                    self.enough_increase = True
            elif(self.energe_buffer[self.check_index - 1] < self.energe_buffer[self.check_index - 2]):
                self.increase_count = 0
                self.decrease_count += 1
                if(self.enough_increase):
                    if(self.decrease_count >= 5):
                        self.enough_increase = False
                        if(self.max_mag > self.energe_buffer[0] * 5):
                            self.image_count += 1
                            # self.draw_graph(self.energe_buffer, "%d-broad" %(self.image_count))
                            print("detect lora preamble (with charm)")
                            max_index, energe = self.find_maximum()
                            max_index_detail, self.max_bin_detail = self.find_maximum_detail(max_index - (n_syms - i - 1))
                            self.signal_timing_index = self.M * (max_index - 1 - 8) + max_index_detail
                            self.set_frequencyOffset(self.signal_timing_index, self.max_bin_detail)
                            self.sending_mode = True
                            self.save_signal_to_db()
            else:
                pass

            # --------------------------- Checking Start ---------------------------
            self.buffer = numpy.roll(self.buffer, -1)
            self.complex_buffer = numpy.roll(self.complex_buffer, -1)
            self.buffer_meta.pop(0)
            self.buffer_meta.append(dict())

            sig = input_items[0][i*self.M:(i+1)*self.M]
            (hard_sym, complex_sym) = self.demod.complex_demodulate(sig)
            self.buffer[-1] = hard_sym[0]
            self.complex_buffer[-1] = complex_sym[0]

            #Conjugate demod and shift conjugate buffer, if needed
            #AABBC or ABBCC
            #   ^       ^
            if ('sync_value' in self.buffer_meta[-2]) or ('sync_value' in self.buffer_meta[-3]):
                self.conj_buffer = numpy.roll(self.conj_buffer, -1)
                self.conj_complex_buffer = numpy.roll(self.conj_complex_buffer, -1)

                (hard_sym, complex_sym) = self.demod_conj.complex_demodulate(sig)
                self.conj_buffer[-1] = hard_sym[0]
                self.conj_complex_buffer[-1] = complex_sym[0]

            #Check for preamble
            if(self.detect_preamble()):
                print("Detect Preamble(Origin)")
            # --------------------------- Checking End ---------------------------

        # send
        if(self.sending_mode):
            # output_items[0][:] = self.signal_buffer[-self.sending_size :]
            # output_items[0][:] = self.signal_buffer[self.signal_timing_index: self.signal_timing_index + self.sending_size]
            output_items[0][0:self.M * self.preamble_len] = self.adjusted_signal[0:self.sending_size]
        else:
            # output_items[0][:] = numpy.random.normal(size=self.sending_size)
            output_items[0][:] = numpy.zeros(self.sending_size, dtype=numpy.complex64)
        self.sending_mode = False
        return len(output_items[0])