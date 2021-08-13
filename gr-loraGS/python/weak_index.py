#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 weak_index.
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

class weak_index(gr.sync_block):
    """
    docstring for block weak_index
    """
    def __init__(self, sf, threshold, preamble_len):
        gr.sync_block.__init__(self,
            name="weak_index",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

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
        self.dechirp_8 = numpy.tile(self.dechirp, self.preamble_len)

        # for sending
        self.sending_mode = False
        self.sending_size = 10 * self.M

        # for drawing
        self.image_count = 0
        self.image_count2 = 0

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

        self.set_output_multiple(self.sending_size)     

    def find_maximum(self):
        max_index = 0
        for k in range(30):
            if(self.energe_buffer[k] > self.energe_buffer[max_index]):
                max_index = k
        return (max_index, self.energe_buffer[max_index])

    # ---------------------------------------------------------------------------------------------------
    def set_frequency_offset(self, signal_index, bin_number):
        original_signal_dechirped = self.signal_buffer[signal_index : signal_index + (self.M*8)] * self.dechirp_8
        # original_signal_fft = numpy.fft.fft(original_signal_dechirped)
        original_signal_fft = numpy.fft.fftshift(numpy.fft.fft(original_signal_dechirped))
        original_signal_fft_abs = numpy.abs(original_signal_fft)

        description = "/home/yun/LoRa-Gateway/gr-loraGS/python/image/origin_abs%d.png" %(self.image_count)
        max_mag = numpy.max(original_signal_fft_abs)
        max_bin = numpy.argmax(original_signal_fft_abs)
        self.draw_graph2(original_signal_fft_abs, description, max_mag, max_bin)

        # k = numpy.linspace(0.0, self.M*8 -1.0, self.M*8) / 125000
        k = numpy.linspace(signal_index, signal_index + (self.M*8) - 1.0, self.M*8) / 125000
        time_offset = k
        # Method 1
        fft_interval = 125000 / (self.M * 8)
        frequency_offset_bin = max_bin - (self.M * 4)
        print("What I got bin number : ", max_bin)
        frequency_offset = frequency_offset_bin * fft_interval
        frequency_func = numpy.exp(2j*numpy.pi*(-frequency_offset)*(time_offset))

        # # Method 2
        # fft_interval = 1 / (self.M * 8)
        # frequency_offset_bin = max_bin - (self.M * 4)
        # print("What I got bin number : ", max_bin)
        # frequency_offset = frequency_offset_bin * fft_interval
        # frequency_func = numpy.exp(1j*numpy.pi*(-frequency_offset)*(time_offset))

        adjusted_signal_dechirped = self.signal_buffer[signal_index : signal_index + (self.M*8)] * frequency_func * self.dechirp_8
        # adjusted_signal_fft = numpy.fft.fft(adjusted_signal_dechirped)
        adjusted_signal_fft = numpy.fft.fftshift(numpy.fft.fft(adjusted_signal_dechirped))
        adjusted_signal_fft_abs = numpy.abs(adjusted_signal_fft)

        description2 = "/home/yun/LoRa-Gateway/gr-loraGS/python/image/adj_abs%d.png" %(self.image_count2)
        max_adj_mag = numpy.max(adjusted_signal_fft_abs)
        max_adj_bin = numpy.argmax(adjusted_signal_fft_abs)
        print("What I want bin : ", max_adj_bin)
        self.draw_graph2(adjusted_signal_fft_abs, description2, max_adj_mag, max_adj_bin)


    # ---------------------------------------------------------------------------------------------------

    def find_maximum_detail(self, k):
        self.energe_buffer_detail = numpy.zeros(3 * self.M, dtype=numpy.float) - 1
        self.bin_buffer_detail = numpy.zeros(3 * self.M, dtype=numpy.float) - 1
        
        for i in range(3 * self.M):
            try:
                dechirped_signals = self.signal_buffer[self.M * (k - 8) + i: self.M * k + i]*self.dechirp_8
                dechirped_signals_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_signals))
                dechirped_signals_fft2 = numpy.fft.fftshift(numpy.fft.fft(dechirped_signals))
                self.energe_buffer_detail[i] = numpy.max(numpy.abs(dechirped_signals_fft))
                self.bin_buffer_detail[i] = numpy.argmax(numpy.abs(dechirped_signals_fft))
            except:
                print(self.M * (k - 1) + i - self.M *8, self.M * (k - 1) + i)
        # print("Origin bin : ", self.bin_buffer[k])
        origin_mag = numpy.max(numpy.abs(self.energe_buffer_detail))
        finded_index = numpy.argmax(numpy.abs(self.energe_buffer_detail))
        # self.draw_graph(self.energe_buffer_detail, "/home/yun/LoRa-Gateway/gr-loraGS/python/image/image_max%d.png" %(self.image_count),origin_mag, origin_bin, self.bin_buffer[k])
        signal_index = self.M*(k-8) + finded_index
        bin_num = self.bin_buffer[k]
        
        self.set_frequency_offset(signal_index, bin_num)

        return numpy.argmax(numpy.abs(self.energe_buffer_detail)), numpy.max(numpy.abs(self.energe_buffer_detail))

    def detect_preamble(self):
        #Buffer not full yet
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

    def draw_graph(self, graph, description, mag, bin, bin2):
        plt.plot(graph)
        plt.title("mag: %.2f    bin: %d    origin_bin: %d" %(mag,bin,bin2))
        plt.savefig(description)
        plt.clf()

    def draw_graph2(self, graph, description, mag, bin):
        plt.plot(graph)
        plt.title("mag: %.2f    bin: %d" %(mag,bin))
        plt.savefig(description)
        plt.clf()

    def draw_graph3(self, graph, description):
        plt.plot(graph)
        # plt.title("mag: %.2f    bin: %d    origin_bin: %d" %(mag,bin,bin2))
        plt.savefig(description)
        plt.clf()

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
            # if(self.energe_buffer[self.check_index - 1] > self.energe_buffer[self.check_index - 2] + self.thres):
                self.decrease_count = 0
                self.increase_count += 1
                self.max_mag = self.energe_buffer[self.check_index - 1]
                if(self.increase_count >= 5):
                    self.enough_increase = True
            elif(self.energe_buffer[self.check_index - 1] < self.energe_buffer[self.check_index - 2]):
            # elif(self.energe_buffer[self.check_index - 1] < self.energe_buffer[self.check_index - 2] - self.thres):
                self.increase_count = 0
                self.decrease_count += 1
                if(self.enough_increase):
                    if(self.decrease_count >= 5):
                        self.enough_increase = False
                        if(self.max_mag > self.energe_buffer[0] * 5):
                            self.image_count += 1
                            self.image_count2 += 1
                            # self.draw_graph(self.energe_buffer, "/home/yun/LoRa-Gateway/gr-loraGS/python/image/board/energe%d.png" %(self.image_count))
                            print("detect lora preamble (with charm)")
                            max_index, energe = self.find_maximum()
                            # print("max_index:", max_index, n_syms, i)
                            # print("maximum:", energe)
                            self.max_index_detail, max_detail = self.find_maximum_detail(max_index - (n_syms - i - 1))
                            # print("bin : ", self.max_index_detail)
                            self.sending_index = self.M * (max_index - 1 - 8) + i
                            self.sending_mode = True
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
            output_items[0][:] = self.signal_buffer[self.sending_index: self.sending_index + self.sending_size]
        else:
            output_items[0][:] = numpy.random.normal(size=self.sending_size)
        self.sending_mode = False
        return len(output_items[0])