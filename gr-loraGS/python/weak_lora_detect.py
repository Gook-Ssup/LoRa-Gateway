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
    def __init__(self, sf,threshold,preamble_len):
        gr.sync_block.__init__(self,
            name="weak_lora_detect",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

        self.M = int(2**sf)
        self.preamble_len = preamble_len
        self.thres = threshold

        # for sending
        self.sending_mode = 0
        self.maximum_index_size = 10
        self.chunk_index = 0
        self.chunk_size = 2048
        self.signal_buffer = numpy.zeros(self.chunk_size * self.maximum_index_size, dtype=numpy.complex64)

        #Buffers are initially set to -1
        self.conj_buffer = numpy.zeros(2, dtype=numpy.int) - 1
        self.conj_complex_buffer = numpy.zeros(2, dtype=numpy.complex64)

        self.buffer = numpy.zeros(preamble_len + 2, dtype=numpy.int) - 1
        self.complex_buffer = numpy.zeros(preamble_len + 2, dtype=numpy.complex64)
        self.buffer_meta = [dict() for i in range(0, preamble_len + 2)]

        self.set_output_multiple(self.M)

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.dechirp = numpy.exp(1j*numpy.pi*k/self.M*k)
        self.dechirp_8 = numpy.tile(self.dechirp, self.preamble_len)


    def work(self, input_items, output_items):
        # in0 = input_items[0]
        n_syms = len(in0)//self.M

        for i in range(0, n_syms):
            #Demod and shift buffer
            self.buffer = numpy.roll(self.buffer, -1)
            self.complex_buffer = numpy.roll(self.complex_buffer, -1)
            self.buffer_meta.pop(0)
            self.buffer_meta.append(dict())

            sig = in0[i*self.M:(i+1)*self.M]
        
        
        output_items[0][:] = self.dechirp_8[:len(output_items[0])]

        return len(output_items[0])

