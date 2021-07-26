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
        self.max_chunk_count = self.preamble_len + 2
        self.signal_buffer = numpy.zeros(self.M * self.max_chunk_count, dtype=numpy.complex64)
        self.signal_index = 0
        self.energe_buffer = numpy.zeros(self.max_chunk_count, dtype=numpy.int) - 1
        self.increase_count = 0

        #Buffers are initially set to -1
        self.buffer_meta = [dict() for i in range(0, self.max_chunk_count)]

        # ?
        self.set_output_multiple(self.M)

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.dechirp = numpy.exp(1j*numpy.pi*k/self.M*k)
        self.dechirp_8 = numpy.tile(self.dechirp, self.preamble_len)


    def work(self, input_items, output_items):
        signal_size = len(input_items[0])

        ## Step 1
        numpy.roll(self.signal_buffer, -signal_size)
        self.signal_buffer[-signal_size:] = input_items[0]
        
        ## Step 2
        if(self.signal_index < self.M * (self.preamble_len+2)):
            self.signal_index += signal_size
            return len(output_items[0])
        # else
        #dechirped_signals = self.signal_buffer[-self.preamble_len * self.M:] * self.dechirp_8

        step_size=64
        ## Step 3
        n_syms = signal_size//step_size
        for i in range(0, n_syms):
            # print(self.energe_buffer)
            dechirped_signals = self.signal_buffer[(i*step_size):(i*step_size)+(8*self.M)] * self.dechirp_8
            dechirped_signals_fft = numpy.fft.fft(dechirped_signals)
            numpy.roll(self.energe_buffer, -1)
            self.energe_buffer[-1] = numpy.max(numpy.abs(dechirped_signals_fft))
            
            ## Step 4
            if(self.energe_buffer[-1] > self.energe_buffer[-2]):
                self.increase_count += 1
                print(self.energe_buffer)
            elif(self.increase_count >= 5):
                self.increase_count = 0
                print("detect lora preamble (with charm)")
                

                
        return len(output_items[0])