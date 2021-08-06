#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 weak_test1.
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
import matplotlib.pyplot as plt

class weak_test1(gr.sync_block):
    """
    docstring for block weak_test1
    """
    def __init__(self, sf, threshold, preamble_len):
        gr.sync_block.__init__(self,
            name="weak_test1",
            in_sig=[numpy.complex64],
            out_sig=[numpy.complex64])

        self.M = int(2**sf)
        self.preamble_len = preamble_len
        self.thres = threshold

         # for charm
        self.max_chunk_count = 32
        self.signal_buffer = numpy.zeros(self.M * self.max_chunk_count, dtype=numpy.complex64)

        #
        self.save_buffer = numpy.zeros(self.M * self.max_chunk_count * self.preamble_len , dtype=numpy.complex64)
        self.image_count = 0
        #
        self.signal_index = 0
        self.energe_buffer = numpy.zeros(self.max_chunk_count, dtype=numpy.float) - 1
        self.bin_buffer = numpy.zeros(self.max_chunk_count, dtype=numpy.int) - 1
        self.increase_count = 0
        self.index_num = 0
        self.result = 0
        #Buffers are initially set to -1
        self.buffer_meta = [dict() for i in range(0, self.max_chunk_count)]

        # ?
        self.set_output_multiple(self.M)

        # dechirp
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.dechirp = numpy.exp(1j*numpy.pi*k/self.M*k)
        self.dechirp_8 = numpy.tile(self.dechirp, self.preamble_len)

        # for sending
        self.sending_mode = False
        self.sending_size = 15 * self.M
        self.sending_index = 0

    def find_maximum(self):
        max_index = 0
        print("%10s\t\t\t%10s\t\t\t%10s" %("index","bin","energe"))
        for k in range(self.max_chunk_count):
            if(self.energe_buffer[k] > self.energe_buffer[max_index]):
                max_index = k
            print("%d\t\t\t%d\t\t\t%.2f" %(k, self.bin_buffer[k], self.energe_buffer[k]))
        self.index_num = max_index
        print("index_num : ", self.index_num)
        return (self.bin_buffer[max_index], self.energe_buffer[max_index], self.index_num)
    
    def find_index(self, max_index):
        if(max_index <= 23):
            result_signals = self.save_buffer[(max_index)*self.M*self.preamble_len:(max_index+1)*self.M*self.preamble_len]
            #print("length of result_signals: ", len(result_signals))
            result_signals_dechirp = result_signals*self.dechirp_8
            result_signals_fft = numpy.fft.fftshift(numpy.fft.fft(result_signals_dechirp))
            self.result = numpy.max(numpy.abs(result_signals_fft))
        
        return self.result

    def draw_graph(self, buffer):
        #x = numpy.linspace(-512, len(buffer) -1, 1)
        #y = buffer[x]
        plt.plot(buffer)
        
        #plt.savefig('./image/test%d.png' %(self.image_count)/test%d.png' %(self.image_count))
        plt.savefig("/home/yun/LoRa-Gateway/gr-loraGS/python/image/image%d.png" %(self.image_count))
        plt.clf()
        self.image_count += 1
        
    def find_max_magnitude(self, max_index):
        arr = []
        if(max_index <= 24):
            for i in range(0,1024):
                dechirp_max = self.save_buffer[(max_index)*self.M*8 + (i+1): (max_index+1)*self.M*8 + (i+1)]
                dechirp_max_signal = dechirp_max * self.dechirp_8
                dechirp_max_fft = numpy.fft.fftshift(numpy.fft.fft(dechirp_max_signal))
                #print("Max : ", numpy.max(numpy.abs(dechirp_max_fft)))
                arr.append(numpy.max(numpy.abs(dechirp_max_fft)))
            self.draw_graph(arr)
    

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
        #print("signal_size: ", signal_size)
        check_index = self.preamble_len - 1
 
        # for i in range(0, n_syms): 2 1
        for i in range(n_syms, 0, -1):
            # dechirped_signals = self.signal_buffer[(self.max_chunk_count - n_syms + i - 7)*self.M:(self.max_chunk_count - n_syms + i + 1)*self.M] * self.dechirp_8
            # dechirped_signals = input_items[0][i*self.M : (i+8)*self.M] * self.dechirp_8
            dechirped_signals = numpy.roll(self.signal_buffer, (i-1)*self.M)[-8*self.M:]*self.dechirp_8    
            dechirped_signals_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_signals))
            self.save_buffer = numpy.roll(self.save_buffer, -8*self.M)
            self.save_buffer[-8*self.M:] = numpy.roll(self.signal_buffer, (i-1)*self.M)[-8*self.M:]

            self.energe_buffer = numpy.roll(self.energe_buffer, -1)
            self.energe_buffer[-1] = numpy.max(numpy.abs(dechirped_signals_fft))
            self.bin_buffer = numpy.roll(self.bin_buffer, -1)
            self.bin_buffer[-1] = numpy.argmax(numpy.abs(dechirped_signals_fft))
            
            ## Step 4
            check_index = 15
            if(self.energe_buffer[check_index-1] > self.energe_buffer[check_index-2]):
                self.increase_count += 1
                
            elif(self.energe_buffer[check_index-1] == self.energe_buffer[check_index-2]):
                pass
            else:
                if(self.increase_count >= 6):
                    # print("detect lora preamble (with charm)")
                    max_bin, energe, max_index = self.find_maximum()
                    print("max bin:", max_bin)
                    print("max mag:", energe)
                    self.sending_mode = True
                    print("======================")
                    ans = self.find_index(max_index)
                    print("answer : ", ans) 
                    print("======================")
                    self.find_max_magnitude(max_index)
                    print("sending")
                self.increase_count = 0
        
        # send
        if(self.sending_mode):
            self.sending_index += signal_size
            output_items[0][:] = self.signal_buffer[:signal_size]
            if(self.sending_index >= self.sending_size):
                self.sending_index = 0
                self.sending_mode = False
        else:
            output_items[0][:] = numpy.random.normal(size=signal_size)

        return len(output_items[0])
