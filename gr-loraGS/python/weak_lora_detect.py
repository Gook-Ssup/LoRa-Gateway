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

class weak_lora_detect(gr.sync_block):
    """
    docstring for block weak_lora_detect
    """
    def __init__(self, sf, threshold, preamble_len):
        gr.sync_block.__init__(self,
            name="weak_lora_detect",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

        self.M = int(2**sf)
        self.preamble_len = preamble_len
        self.thres = threshold

        # for charm
        self.max_chunk_count = 30
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
        self.sending_size = 15 * self.M

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

        self.set_output_multiple(self.sending_size)

    def print_buffers(self):
        for k in range(30):
            print("%04d" %self.bin_buffer[k], end=" ")
        print("")
        for k in range(30):
            print("%04d" %self.energe_buffer[k], end=" ")
        print("")        

    def find_maximum(self):
        self.print_buffers()
        max_index = 0
        for k in range(30):
            if(self.energe_buffer[k] > self.energe_buffer[max_index]):
                max_index = k
        return (self.bin_buffer[max_index], self.energe_buffer[max_index])

    def detect_preamble(self):
        #Buffer not full yet
        if self.buffer[0] == -1:
            return False

        mean = numpy.mean(self.buffer[-(self.preamble_len+2):-2])
        mean_err_sq = numpy.sum(numpy.abs(self.buffer[-(self.preamble_len+2):-2] - mean)**2)
        max_err_sq = self.M**2

        if(mean_err_sq/max_err_sq < self.thres):
            self.buffer_meta[self.preamble_len-1]['preamble_value'] = numpy.uint16(numpy.round(mean))
            # print("-------------------------detect-------------------------------")
            # print("buffer:", self.buffer)
            # print("buffer_meta:", self.buffer_meta)
            # print("input:", in0, len(in0))
            return True
        else:
            pass

    def draw_graph(self):
        # x = numpy.linspace(0, 10, 1)
        #y = buffer[x]
        plt.plot(self.energe_buffer)
        #plt.show()
        plt.savefig("%d.png" %(self.image_count))
        plt.clf()
        self.image_count += 1

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

        ## Step 3
        n_syms = signal_size//self.M
        check_index = 15
 
        for i in range(0, n_syms):
        # for i in range(n_syms, 0, -1):
            # dechirped_signals = numpy.roll(self.signal_buffer, (i-1)*self.M)[-8*self.M:]*self.dechirp_8
            dechirped_signals = numpy.roll(self.signal_buffer, (n_syms - i - 1)*self.M)[-8*self.M:]*self.dechirp_8
            dechirped_signals_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_signals))

            self.energe_buffer = numpy.roll(self.energe_buffer, -1)
            self.energe_buffer[-1] = numpy.max(numpy.abs(dechirped_signals_fft))
            self.bin_buffer = numpy.roll(self.bin_buffer, -1)
            self.bin_buffer[-1] = numpy.argmax(numpy.abs(dechirped_signals_fft))
            
            ## Step 4
            if(self.energe_buffer[-1] > self.energe_buffer[-2]):
                self.decrease_count = 0
                self.increase_count += 1
                self.max_mag = self.energe_buffer[-1]
                if(self.increase_count >= 6):
                    self.enough_increase = True
            else:
                self.increase_count = 0
                self.decrease_count += 1
                if(self.enough_increase):
                    if(self.decrease_count >= 6):
                        self.enough_increase = False
                        if(self.max_mag > self.energe_buffer[-15] * 5):
                            self.draw_graph()
                            print("detect lora preamble (with charm)")
                            max_bin, energe = self.find_maximum()
                            # print("max bin:", max_bin)
                            # print("max mag:", energe)
                            self.sending_mode = True

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
                self.print_buffers()
                print(self.buffer)
            # --------------------------- Checking End ---------------------------

        # send
        if(self.sending_mode):
            output_items[0][:] = self.signal_buffer[-self.M * 15 :]
        else:
            output_items[0][:] = numpy.random.normal(size=self.sending_size)
        self.sending_mode = False
        return len(output_items[0])