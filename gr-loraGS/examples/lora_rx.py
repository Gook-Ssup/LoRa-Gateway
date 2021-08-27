#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: LoRa RX
# GNU Radio version: v3.8.3.1-6-g896b7138

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import filter
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
import loraGS
import numpy
import osmosdr
import time

from gnuradio import qtgui

class lora_rx(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "LoRa RX")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("LoRa RX")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "lora_rx")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.SF = SF = 7
        self.sig_name = sig_name = 'zeros_SF' + str(SF) + '_CR4.raw'
        self.sig_dir = sig_dir = '/home/amarquet/Documents/postdoc-imt/enseignements/projets-S5/2018/sdr/emetteur-recepteur-lora-sdr/scripts/signals'
        self.interp = interp = 1
        self.chan_bw = chan_bw = 125000
        self.RF_samp_rate = RF_samp_rate = 1500000
        self.M = M = 2**SF
        self.samp_rate = samp_rate = chan_bw
        self.radio_offset = radio_offset = 0.0
        self.n_syms = n_syms = 100
        self.freq_offset = freq_offset = 4*chan_bw
        self.frac_decim = frac_decim = RF_samp_rate/((RF_samp_rate//(interp*chan_bw))*(interp*chan_bw))
        self.file = file = sig_dir+'/'+sig_name
        self.decim = decim = RF_samp_rate//(interp*chan_bw)
        self.chan_margin = chan_margin = 75000
        self.chan_freq = chan_freq = 868e6
        self.attenuation = attenuation = 20
        self.T = T = float(M)/chan_bw

        ##################################################
        # Blocks
        ##################################################
        self.rtlsdr_source_0 = osmosdr.source(
            args="numchan=" + str(2) + " " + ""
        )
        self.rtlsdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.rtlsdr_source_0.set_sample_rate(RF_samp_rate)
        self.rtlsdr_source_0.set_center_freq(915e6, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_gain(10, 0)
        self.rtlsdr_source_0.set_if_gain(20, 0)
        self.rtlsdr_source_0.set_bb_gain(20, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
        self.rtlsdr_source_0.set_center_freq(915e6, 1)
        self.rtlsdr_source_0.set_freq_corr(0, 1)
        self.rtlsdr_source_0.set_gain(10, 1)
        self.rtlsdr_source_0.set_if_gain(20, 1)
        self.rtlsdr_source_0.set_bb_gain(20, 1)
        self.rtlsdr_source_0.set_antenna('', 1)
        self.rtlsdr_source_0.set_bandwidth(0, 1)
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, #fftsize
            firdes.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            RF_samp_rate, #bw
            "", #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            True #plotconst
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.pyqwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(False)

        self.top_layout.addWidget(self._qtgui_sink_x_0_win)
        self.mmse_resampler_xx_0_0 = filter.mmse_resampler_cc(0.0, frac_decim)
        self.mmse_resampler_xx_0 = filter.mmse_resampler_cc(0.0, frac_decim)
        self.low_pass_filter_0_0 = filter.fir_filter_ccf(
            decim,
            firdes.low_pass(
                200,
                RF_samp_rate,
                (chan_bw + chan_margin)/2,
                (chan_bw + chan_margin)/8,
                firdes.WIN_HAMMING,
                6.76))
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            decim,
            firdes.low_pass(
                200,
                RF_samp_rate,
                (chan_bw + chan_margin)/2,
                (chan_bw + chan_margin)/8,
                firdes.WIN_HAMMING,
                6.76))
        self.loraGS_weak_index_0_0 = loraGS.weak_index(10, 1e-4, 8)
        self.loraGS_weak_index_0 = loraGS.weak_index(10, 1e-4, 8)
        self.loraGS_add2_0 = loraGS.add2(0)
        self.freq_xlating_fir_filter_xxx_0_0 = filter.freq_xlating_fir_filter_ccc(1, [1], 0, RF_samp_rate)
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccc(1, [1], 0, RF_samp_rate)
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self._attenuation_range = Range(0.0, 89.75, 0.25, 20, 200)
        self._attenuation_win = RangeWidget(self._attenuation_range, self.set_attenuation, 'Attenuation', "counter_slider", float)
        self.top_layout.addWidget(self._attenuation_win)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_throttle_0, 0), (self.loraGS_weak_index_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.loraGS_weak_index_0_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0_0, 0), (self.low_pass_filter_0_0, 0))
        self.connect((self.loraGS_add2_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.loraGS_weak_index_0, 0), (self.loraGS_add2_0, 0))
        self.connect((self.loraGS_weak_index_0_0, 0), (self.loraGS_add2_0, 1))
        self.connect((self.low_pass_filter_0, 0), (self.mmse_resampler_xx_0, 0))
        self.connect((self.low_pass_filter_0_0, 0), (self.mmse_resampler_xx_0_0, 0))
        self.connect((self.mmse_resampler_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.mmse_resampler_xx_0_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.rtlsdr_source_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))
        self.connect((self.rtlsdr_source_0, 1), (self.freq_xlating_fir_filter_xxx_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "lora_rx")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_SF(self):
        return self.SF

    def set_SF(self, SF):
        self.SF = SF
        self.set_M(2**self.SF)
        self.set_sig_name('zeros_SF' + str(self.SF) + '_CR4.raw')

    def get_sig_name(self):
        return self.sig_name

    def set_sig_name(self, sig_name):
        self.sig_name = sig_name
        self.set_file(self.sig_dir+'/'+self.sig_name)

    def get_sig_dir(self):
        return self.sig_dir

    def set_sig_dir(self, sig_dir):
        self.sig_dir = sig_dir
        self.set_file(self.sig_dir+'/'+self.sig_name)

    def get_interp(self):
        return self.interp

    def set_interp(self, interp):
        self.interp = interp
        self.set_decim(self.RF_samp_rate//(self.interp*self.chan_bw))
        self.set_frac_decim(self.RF_samp_rate/((self.RF_samp_rate//(self.interp*self.chan_bw))*(self.interp*self.chan_bw)))

    def get_chan_bw(self):
        return self.chan_bw

    def set_chan_bw(self, chan_bw):
        self.chan_bw = chan_bw
        self.set_T(float(self.M)/self.chan_bw)
        self.set_decim(self.RF_samp_rate//(self.interp*self.chan_bw))
        self.set_frac_decim(self.RF_samp_rate/((self.RF_samp_rate//(self.interp*self.chan_bw))*(self.interp*self.chan_bw)))
        self.set_freq_offset(4*self.chan_bw)
        self.set_samp_rate(self.chan_bw)
        self.low_pass_filter_0.set_taps(firdes.low_pass(200, self.RF_samp_rate, (self.chan_bw + self.chan_margin)/2, (self.chan_bw + self.chan_margin)/8, firdes.WIN_HAMMING, 6.76))
        self.low_pass_filter_0_0.set_taps(firdes.low_pass(200, self.RF_samp_rate, (self.chan_bw + self.chan_margin)/2, (self.chan_bw + self.chan_margin)/8, firdes.WIN_HAMMING, 6.76))

    def get_RF_samp_rate(self):
        return self.RF_samp_rate

    def set_RF_samp_rate(self, RF_samp_rate):
        self.RF_samp_rate = RF_samp_rate
        self.set_decim(self.RF_samp_rate//(self.interp*self.chan_bw))
        self.set_frac_decim(self.RF_samp_rate/((self.RF_samp_rate//(self.interp*self.chan_bw))*(self.interp*self.chan_bw)))
        self.low_pass_filter_0.set_taps(firdes.low_pass(200, self.RF_samp_rate, (self.chan_bw + self.chan_margin)/2, (self.chan_bw + self.chan_margin)/8, firdes.WIN_HAMMING, 6.76))
        self.low_pass_filter_0_0.set_taps(firdes.low_pass(200, self.RF_samp_rate, (self.chan_bw + self.chan_margin)/2, (self.chan_bw + self.chan_margin)/8, firdes.WIN_HAMMING, 6.76))
        self.qtgui_sink_x_0.set_frequency_range(0, self.RF_samp_rate)
        self.rtlsdr_source_0.set_sample_rate(self.RF_samp_rate)

    def get_M(self):
        return self.M

    def set_M(self, M):
        self.M = M
        self.set_T(float(self.M)/self.chan_bw)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)

    def get_radio_offset(self):
        return self.radio_offset

    def set_radio_offset(self, radio_offset):
        self.radio_offset = radio_offset

    def get_n_syms(self):
        return self.n_syms

    def set_n_syms(self, n_syms):
        self.n_syms = n_syms

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset

    def get_frac_decim(self):
        return self.frac_decim

    def set_frac_decim(self, frac_decim):
        self.frac_decim = frac_decim
        self.mmse_resampler_xx_0.set_resamp_ratio(self.frac_decim)
        self.mmse_resampler_xx_0_0.set_resamp_ratio(self.frac_decim)

    def get_file(self):
        return self.file

    def set_file(self, file):
        self.file = file

    def get_decim(self):
        return self.decim

    def set_decim(self, decim):
        self.decim = decim

    def get_chan_margin(self):
        return self.chan_margin

    def set_chan_margin(self, chan_margin):
        self.chan_margin = chan_margin
        self.low_pass_filter_0.set_taps(firdes.low_pass(200, self.RF_samp_rate, (self.chan_bw + self.chan_margin)/2, (self.chan_bw + self.chan_margin)/8, firdes.WIN_HAMMING, 6.76))
        self.low_pass_filter_0_0.set_taps(firdes.low_pass(200, self.RF_samp_rate, (self.chan_bw + self.chan_margin)/2, (self.chan_bw + self.chan_margin)/8, firdes.WIN_HAMMING, 6.76))

    def get_chan_freq(self):
        return self.chan_freq

    def set_chan_freq(self, chan_freq):
        self.chan_freq = chan_freq

    def get_attenuation(self):
        return self.attenuation

    def set_attenuation(self, attenuation):
        self.attenuation = attenuation

    def get_T(self):
        return self.T

    def set_T(self, T):
        self.T = T





def main(top_block_cls=lora_rx, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()

    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()

if __name__ == '__main__':
    main()
