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

import datetime
from pymongo import MongoClient


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

        # adjusted_signal : Frequnecy Offset
        # adjutsed_signal2 : CFO_fine & Phase Offset
        # channel_est : Channel Estimation
        self.adjusted_signal = numpy.zeros(self.M * self.preamble_len, dtype=numpy.complex64)
        self.adjusted_signal2 = numpy.zeros(self.M * self.preamble_len, dtype=numpy.complex64)
        self.channel_est = numpy.zeros(self.M * self.preamble_len, dtype=numpy.complex64)
        
        self.max_mag = 0
        self.increase_count = 0
        self.decrease_count = 0
        self.enough_increase = False
        self.description = "/home/gnuradio-inc/Yun/image/"

        # dechirp method 1
        k = numpy.linspace(0.0, self.M-1.0, self.M)
        self.dechirp = numpy.exp(-1j*numpy.pi*k/self.M*k)
        self.dechirp_8 = numpy.tile(self.dechirp, 8)

        # dechirp method 2
        # self.dechirp = numpy.exp((1j*numpy.pi*(k*k)/self.M))
        # self.dechirp = numpy.conj(self.dechirp)
        # self.dechirp_8 = numpy.tile(self.dechirp, 8)
        
        # dechirp method 3
        # self.dechirp = numpy.exp((1j*2*numpy.pi*((k*k)/(2*self.M))))
        # self.dechirp = self.dechirp*numpy.exp(1j*2*numpy.pi*(-k/2)) 
        # self.dechirp = numpy.conj(self.dechirp)
        # self.dechirp_8 = numpy.tile(self.dechirp, 8)


        # for sending
        self.sending_mode = False
        self.sending_size = 8 * self.M
        self.sending_signal = numpy.zeros(self.M*8, dtype=numpy.complex64) -1

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
        BW = 125000
        fft_interval = BW / (self.M * self.preamble_len)
        frequency_offset_bin = bin_number - (self.M * (self.preamble_len / 2))
        frequency_offset = frequency_offset_bin * fft_interval

        # k = numpy.linspace(0.0, self.M * self.preamble_len - 1.0, self.M * self.preamble_len) / BW
        k = numpy.linspace(0.0, self.M - 1.0, self.M) / BW
        offset_signal = numpy.exp(1j*2*numpy.pi*(-frequency_offset)*k)

        # adjust CFO
        for i in range(8):
            # --------Correct frequency offset--------
            self.adjusted_signal[i*self.M:(i+1)*self.M] = self.signal_buffer[self.M*i + signal_index : self.M*(i+1) + signal_index]
            self.adjusted_signal[i*self.M:(i+1)*self.M] = self.adjusted_signal[i*self.M:(i+1)*self.M] * offset_signal

            # -------------show bin----------------
            # adjusted_dechirp = self.adjusted_signal[i*self.M:(i+1)*self.M] * self.dechirp
            # bin=numpy.fft.fft(adjusted_dechirp)
            # bin_abs = numpy.abs(bin)
            # result_bin = numpy.argmax(bin_abs)
            # print("bin : ",result_bin)
            # --------------!show bin----------------

    def angdiff(self,a,b):
        d = a - b
        d = numpy.mod(d + numpy.pi, 2*numpy.pi) - numpy.pi
        return d

    def set_phase_offset(self):
        symbol_mag = numpy.zeros(self.preamble_len, dtype=numpy.complex64)
        symbol_bin = numpy.zeros(self.preamble_len, dtype=numpy.int)
        for i in range(8):
            symbol_fft = numpy.fft.fft(self.adjusted_signal[i*self.M : (i+1)*self.M] * self.dechirp)
            symbol_fft_abs = numpy.abs(symbol_fft)
            symbol_mag[i] = numpy.max(symbol_fft_abs)
            symbol_bin[i] = numpy.argmax(symbol_fft_abs)
        # print("-----------------------------")
        # print("bin : ", symbol_bin)
        # print("mag : ", symbol_mag)
        phase_angle = numpy.zeros(self.preamble_len, dtype=numpy.float)
        for i in range(8):
            phase_fft = numpy.fft.fft(self.adjusted_signal[i*self.M : (i+1)*self.M] * self.dechirp)
            phase_angle[i] = numpy.angle(phase_fft[0])
        # print("phase_angle : ", phase_angle)
        phase_diff = numpy.zeros(self.preamble_len - 1, dtype=numpy.float)
        for i in range(self.preamble_len - 1):
            phase_diff[i] = self.angdiff(phase_angle[i], phase_angle[i + 1])

        # print("----------Angle----------")
        # print(phase_angle)
        # print()
        # print("-----------Diff--------------")
        # print(phase_diff)
        # print()
        # print("--------Symbol index----------")
        # print(symbol_bin)
        # print()
        phase_mean = numpy.mean(phase_diff[2:7])
        # print("----------phase_mean----------")
        # print(phase_mean)
        # print()
        CFO_fine=(phase_mean*125000)/(1024*2*numpy.pi)

        # print("----------CFO_fine-------------")
        # print(CFO_fine)
        # print()
        k = numpy.linspace(0.0, self.M - 1.0, self.M) / 125000
        for i in range(8):
            self.adjusted_signal2[i*self.M : (i+1)*self.M] = self.adjusted_signal[i*self.M : (i+1)*self.M] * numpy.exp(2j*numpy.pi*(-CFO_fine)*k) * numpy.exp(1j*(phase_mean)*i)

        phase_angle2 = numpy.zeros(8, dtype=numpy.float)
        phase_diff2 = numpy.zeros(7, dtype=numpy.float)
        for i in range(8):
            phase_fft2 = numpy.fft.fft(self.adjusted_signal2[i*self.M : (i+1)*self.M] * self.dechirp)
            phase_angle2[i] = numpy.angle(phase_fft2[0])
        for i in range(7):
            phase_diff2[i] = self.angdiff(phase_angle2[i], phase_angle2[i+1])

        # print("-----------PHASE DIFF--------------")
        # print(phase_diff2)

    def channel_estimation(self):
        self.upchirp = numpy.conj(self.dechirp)
        self.upchirp_8 = numpy.tile(self.upchirp, self.preamble_len)
        for i in range(8):
            self.channel_est[i*self.M : (i+1)*self.M] = self.adjusted_signal2[i*self.M : (i+1)*self.M] / self.upchirp

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

    def draw_plot_specto(self, description):
        fig = plt.figure()
        ax = fig.add_subplot(3,1,1)
        ax.plot(self.energe_buffer)
        ax = fig.add_subplot(3,1,2)
        ax.specgram(self.signal_buffer, Fs=1)
        ax = fig.add_subplot(3,1,3)
        sending_fft = numpy.fft.fftshift(self.sending_signal*self.dechirp_8)
        sending_fft_abs = numpy.abs(sending_fft)
        ax.plot(sending_fft_abs)
        fig.savefig(description)
        fig.clf()
    
    def draw_energe_long_preamble(self, description, timing_index):
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.plot(self.energe_buffer)
        ax = fig.add_subplot(2,1,2)
        interval = self.M * 2
        # result_signal = self.signal_buffer[timing_index - interval: timing_index + (self.M * self.preamble_len) + interval]
        result_signal = self.signal_buffer
        ax.specgram(result_signal, Fs=1)
        fig.savefig(description)
        fig.clf()

    def draw_plot_specto_bin(self, description, graph):
        sending_fft = numpy.fft.fft(graph*self.dechirp_8)
        sending_fft_shift = numpy.fft.fftshift(sending_fft)
        sending_fft_abs = numpy.abs(sending_fft_shift)
        graph_bin = numpy.argmax(sending_fft_abs)
        graph_mag = numpy.max(sending_fft_abs)
        fig = plt.figure()
        ax = fig.add_subplot(3,1,1)
        ax.specgram(graph,Fs=125000)
        ax.set_title("bin : %d    mag : %.2f"%(graph_bin, graph_mag))

        ax = fig.add_subplot(3,1,2)
        ax.plot(self.adjusted_signal2 * self.dechirp_8)

        ax = fig.add_subplot(3,1,3)
        ax.plot(self.channel_est)
        
        fig.tight_layout()
        fig.savefig(description)

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
        
        for i in range(0, n_syms):
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
                            print("detect lora preamble (with charm)")
                            # self.draw_plot_specto(self.description + "signal-energe-bin-%d" %self.image_count)
                            max_index, energe = self.find_maximum()
                            # for buf_index, buf_mag in enumerate(self.energe_buffer):
                            #     print(buf_index, ":", buf_mag)
                            max_index_detail, self.max_bin_detail = self.find_maximum_detail(max_index)
                            self.signal_timing_index = self.M * (max_index - 8) + max_index_detail
                            self.sending_signal = self.signal_buffer[self.signal_timing_index: self.signal_timing_index + self.sending_size].copy()
                            self.set_frequencyOffset(self.signal_timing_index, self.max_bin_detail)
                            self.set_phase_offset()
                            self.channel_estimation()
                            # draw graph
                            # self.draw_plot_specto_bin(self.description + "signal-spec-bin-%d"%self.image_count, self.sending_signal)
                            
                            self.sending_mode = True
                            # self.save_signal_to_db()
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
            # output_items[0][:] = self.sending_signal
            output_items[0][:] = self.adjusted_signal2
            # output_items[0][0:self.M * self.preamble_len] = self.adjusted_signal[0:self.sending_size]
        else:
            output_items[0][:] = numpy.random.normal(size=self.sending_size)
        self.sending_mode = False
        return len(output_items[0])