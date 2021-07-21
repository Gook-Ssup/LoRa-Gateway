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

class dechirp_source(gr.sync_block):
    """
    docstring for block dechirp_source
    """
    def __init__(self, sf):
        gr.sync_block.__init__(self,
            name="dechirp_source",
            in_sig=None,
            out_sig=[numpy.complex64])
        
        self.M = 2**sf

    def work(self, input_items, output_items):
        output_len = len(output_items[0][:])
        
        self.M = output_len
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.conj_chirp = numpy.exp(1j*numpy.pi*k/self.M*k)

        output_items[0][:] = self.conj_chirp[:output_len]
        return len(output_items[0])

