#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-loraGS author.
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

class lora_preamble_detect(gr.sync_block):
    """
    docstring for block lora_preamble_detect
    """
    def __init__(self, sf, threshold, preamble_len):
        gr.sync_block.__init__(self,
            name="lora_preamble_detect",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])
        self.M = int(2**sf)
        self.preamble_len = preamble_len
        self.thres = threshold

        self.demod = css_demod_algo(self.M)
        self.demod_conj = css_demod_algo(self.M, True)

        # for sending
        self.detection_flag=0
        self.maximum_index_size = 5
        self.chunk_index=0
        self.chunk_size = 2048
        self.signal_buffer = numpy.zeros(self.chunk_size * self.maximum_index_size, dtype=numpy.complex64)

        #Buffers are initially set to -1
        self.conj_buffer = numpy.zeros(2, dtype=numpy.int) - 1
        self.conj_complex_buffer = numpy.zeros(2, dtype=numpy.complex64)

        if preamble_len > 2:
            self.buffer = numpy.zeros(preamble_len + 2, dtype=numpy.int) - 1
            self.complex_buffer = numpy.zeros(preamble_len + 2, dtype=numpy.complex64)
            self.buffer_meta = [dict() for i in range(0, preamble_len + 2)]
        else:
            self.buffer = numpy.zeros(5, dtype=numpy.int) - 1
            self.complex_buffer = numpy.zeros(5, dtype=numpy.complex64)
            self.buffer_meta = [dict() for i in range(0, 5)]

        self.set_output_multiple(self.M)

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
            

        return False

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]

        # if(len(in0) == len(out0)):
        #     print("Same")
        # else:
        #     print("Not Same")

        
        n_syms = len(in0)//self.M

        #print("in0 len:", len(in0))
        # print("n_syms:", n_syms)


        for i in range(0, n_syms):
            #Demod and shift buffer
            self.buffer = numpy.roll(self.buffer, -1)
            self.complex_buffer = numpy.roll(self.complex_buffer, -1)
            self.buffer_meta.pop(0)
            self.buffer_meta.append(dict())

            sig = in0[i*self.M:(i+1)*self.M]
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
            
            input_len = len(in0)
            self.signal_buffer=numpy.roll(self.signal_buffer, -input_len)
            self.signal_buffer[-input_len:] = in0
            if(self.detect_preamble()):
                print("detect preamble")
                self.detection_flag=1

            if(self.detection_flag):
                self.chunk_index+=1
                #print(self.chunk_index)
                out0[:] = self.signal_buffer[:input_len]
                print(out0)
                if(self.chunk_index>=self.maximum_index_size):
                    self.chunk_index=0
                    self.detection_flag=0
            else:
                out0[:] = 0
        
        #return len(output_items[0])
        return len(out0[:])

