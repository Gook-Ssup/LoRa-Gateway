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
from influxdb import InfluxDBClient
from copy import deepcopy
import time

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
        self.max_chunk_count = 30
        self.signal_buffer = numpy.zeros(self.M * self.max_chunk_count, dtype=numpy.complex64)

        #
        self.save_buffer = numpy.zeros(self.M * self.max_chunk_count * (self.preamble_len + 2) , dtype=numpy.complex64)
        self.image_count = 0
        self.image_count2 = 0
        #

        self.signal_index = 0
        self.energe_buffer = numpy.zeros(self.max_chunk_count, dtype=numpy.float) - 1
        self.bin_buffer = numpy.zeros(self.max_chunk_count, dtype=numpy.int) - 1
        self.max_mag = 0
        self.increase_count = 0
        self.decrease_count = 0
        self.enough_increase = False
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

        # InfluxDB
        self.host = 'localhost'
        self.port = 8086
        self.username = 'lorauser'
        self.password = '1234'
        self.database = 'test'

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
        # self.print_buffers()
        max_index = 0
        for k in range(30):
            if(self.energe_buffer[k] > self.energe_buffer[max_index]):
                max_index = k
            #print("%d\t\t\t%d\t\t\t%.2f" %(k, self.bin_buffer[k], self.energe_buffer[k]))
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
        x = numpy.linspace(-1024, 1023, 2048)
        #y = buffer[x]
        plt.plot(x,buffer)
        #plt.show()
        #plt.savefig('./image/test%d.png' %(self.image_count)/test%d.png' %(self.image_count))
        plt.savefig("/home/yun/LoRa-Gateway/gr-loraGS/python/image/image%d.png" %(self.image_count))
        plt.clf()
        # plt.cla()
        # plt.close()
        self.image_count += 1
    
    def insert_max(self, index_num, mag):
        client = InfluxDBClient(host=self.host,
                                port=self.port,
                                username=self.username,
                                password=self.password,
                                database=self.database)
        json_payload = []
        point = {
            "measurement": "max_info",
            "tags": {
                "signal": "lora"
            },
            "fields": {
                "max_index": 1,
                "max_mag": 1.0
            }
        }
        p = deepcopy(point)
        p['fields']['max_index'] = index_num
        p['fields']['max_mag'] = mag
        json_payload.append(p)
        client.write_points(json_payload)
        client.close()

    def draw_two_graph(self, buffer1, max_index, max_value, max_mag):
        x = numpy.linspace(-1024, 1023, 2048)
        plt.subplot(2, 1, 1)    #nrows = 2, ncols = 1, index = 1
        plt.plot(buffer1)
        plt.title('Energe Buffer\n%.2fWWW%.2f' %(max_value,max_mag))

        arr = []
        if(max_index <= 27):
            for i in range(0,2048):
                dechirp_max = self.save_buffer[(max_index)*self.M*(self.preamble_len + 2) + (i+1): (max_index)*self.M*(self.preamble_len + 2) + (self.M)*8+ (i+1)]
                dechirp_max_signal = dechirp_max * self.dechirp_8
                dechirp_max_fft = numpy.fft.fftshift(numpy.fft.fft(dechirp_max_signal))
                #print("Max : ", numpy.max(numpy.abs(dechirp_max_fft)))
                arr.append(numpy.max(numpy.abs(dechirp_max_fft)))
        
        Ans_mag = numpy.max(arr)
        Ans_index = numpy.argmax(arr)
        self.insert_max(Ans_index,Ans_mag)
        plt.subplot(2, 1, 2)
        plt.plot(arr)
        plt.title('Origin: %.2f   Ans: %.2f   index: %d' %(max_mag, Ans_mag, Ans_index))
        plt.tight_layout()
        plt.savefig("/home/yun/LoRa-Gateway/gr-loraGS/python/image/image_max%d.png" %(self.image_count2))
        self.image_count2 += 1
        plt.clf()


    def draw_max_magnitude(self,buffer):
        plt.plot(buffer)
        plt.savefig("/home/yun/LoRa-Gateway/gr-loraGS/python/image/image_max%d.png" %(self.image_count2))
        plt.clf()
        # plt.cla()
        self.image_count2 += 1
        
    def find_max_magnitude(self, max_index):
        arr = []
        if(max_index <= 28 and max_index >= 1):
            for i in range(0,2048):
                dechirp_max = self.save_buffer[(max_index)*self.M*8 + (i+1): (max_index+1)*self.M*8 + (i+1)]
                dechirp_max_signal = dechirp_max * self.dechirp_8
                dechirp_max_fft = numpy.fft.fftshift(numpy.fft.fft(dechirp_max_signal))
                #print("Max : ", numpy.max(numpy.abs(dechirp_max_fft)))
                arr.append(numpy.max(numpy.abs(dechirp_max_fft)))
            # self.draw_max_magnitude(arr)

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
        check_index = 15
 
        # for i in range(0, n_syms): 2 1
        for i in range(0, n_syms):
        # for i in range(n_syms, 0, -1):
            # dechirped_signals = numpy.roll(self.signal_buffer, (i-1)*self.M)[-8*self.M:]*self.dechirp_8
            dechirped_signals = numpy.roll(self.signal_buffer, (n_syms - i - 1)*self.M)[-8*self.M:]*self.dechirp_8
            dechirped_signals_fft = numpy.fft.fftshift(numpy.fft.fft(dechirped_signals))
            self.save_buffer = numpy.roll(self.save_buffer, -(self.preamble_len + 2)*self.M)
            self.save_buffer[-(self.preamble_len+2)*self.M:] = numpy.roll(self.signal_buffer, (n_syms - i -1)*self.M)[-(self.preamble_len+2)*self.M:]

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
                            # self.draw_graph(self.save_buffer)
                            print("detect lora preamble (with charm)")
                            
                            max_bin, energe, max_index = self.find_maximum()
                            ######
                            # self.find_max_magnitude(max_index)
                            # self.draw_graph(self.energe_buffer)
                            self.draw_two_graph(self.energe_buffer, max_index, self.energe_buffer[-15],self.max_mag)
                            ######
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
                #self.print_buffers()
                print(self.buffer)
            # --------------------------- Checking End ---------------------------

        # send
        if(self.sending_mode):
            output_items[0][:] = self.signal_buffer[-self.M * 15 :]
        else:
            output_items[0][:] = numpy.random.normal(size=self.sending_size)
        self.sending_mode = False
        return len(output_items[0])