# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:52:59 2016

@author: soyunkope
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import scipy.io as sio


def Dwestim(sig, w=5, step=1):
    '''
        Goes through the signal meassuring the eclidean distance of each
        w points window in step steps. Making each signal more informative
        about the underlying content.
        TODO: Use this a set of some 3 Dw conversions to PCA the signal.
        TODO: Think on this new representation of the signal as a new feature
              space for the signals. 
    '''
    #sig = np.array((sig-np.mean(sig))/np.std(sig))
    f = np.array([sig[i:i+w:step] for i in range(len(sig)-w)])
    Dw = np.sqrt(np.sum(np.diff(f)**2, 1))
    return np.array(Dw)


def spkDet(signal, method='MAD', T=3, ori='neg'):
    '''
    Detect events on signal depending on method. Can be:
    MAD:
        Median Absolute Deviation:
            median(absolute(signal - median(signal)))
    QQ:
        Quian-Quiroga's .75 Gaussian CDF of the median of the signal:
            median(|signal|/.6745)
    Then uses that method times T to detect dependind on orientation ori
    Detect crossings and keep only first point
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
    suX = np.nonzero(np.diff(crX) > 15)[0]  # only first crossing
    spX = crX[suX[1:]]
    spkIX = [np.argmin(signal[spX[i]-30:spX[i]+30])+spX[i]-30
                for i in range(len(spX))]
    return spkIX


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan(y/x)
    return(phi, rho)



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


def main(N=3):
    fname = fPA+fPB+fSig[N]
    sio.loadmat(fname, struct_as_record=0, squeeze_me=1)
    sig = -fdata['data']
    dF = Dwestim(sig, w=5, step=1)
    dQ = Dwestim(sig, w=15, step=1)
    spIX = spkDet(-dF, method='QQ', T=4.5, ori='neg')
    waves = np.array(spkAlign(spIX, sig, D=[12, 12]))
    x = np.array([dF[i] for i in spIX[:-10]])
    y = np.array([dQ[i] for i in spIX[:-10]])
    r = np.array([cart2pol(i, j) for i, j in zip(x, y)])
    # weights = 'uniform'
    cols = 'rgbc'
    # TODO: Look for hierrchical clustering
    n_clusters = 3
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

if __name__ == main():
    main(N=5)
