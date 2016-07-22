# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:52:59 2016

Crude implementation of the proposal by:
    Panagiotis C. Petrantonakis and Panayiota Poirazi, 2015 
For this to work you'll need to  have installed all necesary python
libraries, look at the import statements and change the paths in the functions

@author: imakoopa
INDP2015
Champalimaud Centre for the Unknown

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import scipy.io as sio


def Dw(sig, w=5, s=1, norm=0):
    '''
        Goes through the signal summing the euclidean distance of each
        w points window in s steps. Making each signal more informative
        about the underlying content.
    '''
    if norm == 1:
        sig = np.array((sig-np.mean(sig))/np.std(sig))
    f = np.array([sig[i*s:(i*s)+w] for i in range(int((len(sig)-w)/s)+1)])
    dw = np.sqrt(np.sum(np.diff(f, 1, axis=0)**2, 1))
    return dw


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    ph = np.arctan2(y, x)
    return(r, ph)


def cart3d2sphe(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arccos(z/r)
    ph = np.arctan2(y, x)
    return r, th, ph


def spkDet(signal, method='QQ', T=3, ori='neg', lim=10):
    '''
    Detect events on signal depending on method. Can be:
    MAD:
        Median Absolute Deviation:
            median(absolute(signal - median(signal)))
    QQ:
        Quian-Quiroga's .75 Gaussian CDF of the median of the signal:
            median(|signal|/.6745)
    Then uses that method times T to detect depending on orientation ori
    Detect crossings and keep only min point
    '''
    if method == 'QQ':
        QQ = np.median(np.abs(signal)/0.6745)
        thresh = QQ*T  # High THRSH
    elif method == 'MAD':
        MAD = np.median(np.abs(signal - np.median(signal)))
        thresh = MAD*T
    if ori == 'neg':
        thresh = -thresh
    elif ori == 'pos':
        thresh = thresh
    crX = np.nonzero(signal < thresh)[0]  # Threshold crossings
    suX = np.nonzero(np.diff(crX) > lim)[0]  # only first crossing
    spX = crX[suX[1:]]
    spkIX = [np.argmin(signal[spX[i]-lim:spX[i]+lim])+spX[i]-lim
             for i in range(len(spX))]
    return spkIX


def spkAlign(spX, signal, d=10, D=[12, 20], method='MIN'):
    '''
    Align detected spikes and return the waveform from D[0] to D[1]
    MIN: Use the argmin from d+\- spX
    '''
    spkIX = [None]*len(spX)
    if method == 'MIN':
        for i in range(len(spX)):
            spkIX[i] = np.argmin(signal[spX[i]-d:spX[i]+d])+(spX[i]-d)
    waves = [signal[spkIX[j]-D[0]:spkIX[j]+D[1]]
             for j in range(len(spkIX))]
    return waves


def loadAK(filename, numChannels, dtype):
    '''
    Load electrodes data into memory, filename should be
    fpa+fpb+ele
    Where:
        fpa = '/Users/soyunkope/Documents/scriptkidd/git/spikeSorting/'
        For test with Kampf's:
            fpb = 'data/kampfLab/'
            ele = 'adc2014-11-25T23_00_08.bin' 

    Filename:  String that points to the raw data file
    numChannels:  Number of channels on the probe
    dtype:  DType of the record
    '''
    fdata = np.fromfile(filename,dtype=dtype)
    numsamples = int(len(fdata) / numChannels)
    data = np.reshape(fdata,(numsamples,numChannels))
    return (np.transpose(data))

def loadJP(uarray, electrode, dur):
    '''
    For test with Joe's electrodes
        uarray:  str(Array) ['F32', 'L32']
        GT02 = '/Users/soyunkope/Documents/INDP2015/2016_S02/R01_JoePaton/00_AutoSpikeSorting/TestDAta/GT02/GroundTruth6ch.dat'
    '''
    fpa = '/Users/soyunkope/Documents/scriptkidd/git/spikeSorting/'
    fpb = 'data/patonLab/'
    fur = str(uarray)+'/'+str(electrode)+'/'
    fn = 'ele'+str(electrode)+'.dat'
    filename = fpa+fpb+fur+fn
    signal = np.fromfile(filename, dtype='int', count=dur)
    return signal


def loadQQ(N):
    fPA = '/Users/soyunkope'
    fPB = '/Documents/DataSets/Ephys/Simulator/'
    fSig = ['C_Burst_Easy2_noise015.mat', 'C_Difficult1_noise005.mat',
            'C_Difficult1_noise01.mat', 'C_Difficult1_noise015.mat',
            'C_Difficult1_noise02.mat', 'C_Difficult2_noise005.mat',
            'C_Difficult2_noise01.mat', 'C_Difficult2_noise015.mat',
            'C_Difficult2_noise02.mat', 'C_Drift_Easy2_noise015.mat',
            'C_Easy1_noise005.mat', 'C_Easy1_noise01.mat', 'C_Easy1_noise015.mat',
            'C_Easy1_noise01_short.mat', 'C_Easy1_noise02.mat',
            'C_Easy1_noise025.mat', 'C_Easy1_noise03.mat',
            'C_Easy1_noise035.mat', 'C_Easy1_noise04.mat',
            'C_Easy2_noise005.mat', 'C_Easy2_noise01.mat',
            'C_Easy2_noise015.mat', 'C_Easy2_noise02.mat',
            'C_Test_LFPcorr_Easy2_noise015.mat',  'times_C_Difficult1_noise015.mat'
            ]
    fname = fPA+fPB+fSig[N]
    fdata = sio.loadmat(fname, struct_as_record=0, squeeze_me=1)
    signal = -fdata['data']
    return signal


def main(N):
    '''
        Makes this work for you by changing the paths to your need,
        Read the functions, some of them are sort of documented
    '''
    sig = loadQQ(N)
    dF = Dw(sig, w=5, s=1)
    dQ = Dw(sig, w=15, s=1)
    spIX = spkDet(-dF, method='QQ', T=4, ori='neg')
    waves = np.array(spkAlign(spIX, sig, D=[12, 12]))
    x = np.array([dF[i] for i in spIX[:-10]])
    y = np.array([dQ[i] for i in spIX[:-10]])
    r = np.array([cart2pol(i, j) for i, j in zip(x, y)])
    # weights = 'uniform'
    cols = 'rgbc'
    n_clusters = 4
    algo = cluster.KMeans(n_clusters, max_iter=500, n_init=300)
    #  cluster.MeanShift(bandwidth=bandwidth,bin_seeding=0)
    cls = algo.fit(r)
    fa = plt.figure()
    ax = [fa.add_subplot(3, 1, i) for i in [1, 2]]
    axb = [fa.add_subplot(3, 4, j) for j in np.arange(9, 13)]
    trans = .25
    for i in range(4):
        ndx = np.nonzero(cls.labels_ == i)[0]
        samp = np.random.randint(0, np.size(ndx), 20)
        ax[0].plot(x[ndx], y[ndx], 'o'+cols[i], alpha=trans, ms=3)
        ax[1].plot(r[ndx, 0], r[ndx, 1], 'o'+cols[i], alpha=trans, ms=3)
        wav = [np.mean(waves[ndx], 0), np.std(waves[ndx], 0)]
        axb[i].plot(wav[0], cols[i], alpha=.9, lw=2)
        axb[i].plot(wav[0]+wav[1], 'k--', alpha=.25)
        axb[i].plot(wav[0]-wav[1], 'k--', alpha=.25)
        [axb[i].plot(waves[j], cols[i], alpha=.15) for j in ndx[samp]]
    [ax[j].set_xticks([]) for j in [0, 1]]
    [ax[j].set_yticks([]) for j in [0, 1]]
    [axb[j].set_ylim([-2, 2]) for j in range(4)]
    [axb[j].set_xlim([-1, 25]) for j in range(4)]
    [axb[j].set_xticks([]) for j in range(4)]
    [axb[j].set_yticks([]) for j in range(4)]
    fa.suptitle(fSig[N], va='top', ha='center', x=.5, y=.95)
    plt.tight_layout()

main(5)