# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:40:03 2016
use poirazi's idea for pca on gpu
@author: soyunkope
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

def bpbutFFilt(sign, sr, high, low, order=3):    
    nyq = 0.5 * sr
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    res = filtfilt(b, a, signal)
    return res


def pinkster(res, sr, samps):
    nyq = .5*sr
sr = 30000
x = np.arange(0,2,1/sr)
sA = np.sin(x*3*2*np.pi)
sB = np.sin(x*200*2*np.pi)
sC = np.sin(x*25*2*np.pi)*np.cos(x*3*2*np.pi)
signal = sA+sB+sC

fSA = bpbutFFilt(signal, )

ftsig = np.fft.fft(signal)
fqsig = np.fft.fftfreq(signal.size, d=1/sr)