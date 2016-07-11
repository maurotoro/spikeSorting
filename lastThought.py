# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:00:33 2016

@author: soyunkope
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import scipy.io as sio
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def Dwestim(sig, w=5, step=1):
    '''
        Goes through the signal summing the euclidean distance of each
        w points window in step steps. Making each signal more informative
        about the underlying content.
        TODO: Use this a set of some 3 Dw conversions to PCA the signal.
        TODO: Think on this new representation of the signal as a new feature
              space for the signals. 
        TODO: Rethink the signal into ecludianogram
    '''
    sig = np.array((sig-np.mean(sig))/np.std(sig))
    f = np.array([sig[i:i+w:step] for i in range(len(sig)-w)])
    Dw = np.sqrt(np.sum(np.diff(f)**2, 1))
    return Dw


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan(y/x)
    return(phi, rho)

def loadAK(filename,numChannels,dtype):
    '''
    Load electrodes data into memory, filename should be
    fpa+fpb+ele
    Where:
        fpa = '/Users/soyunkope/Documents/scriptkidd/git/spikeSorting/'
        For test with Kampf's:
            fpb = 'data/kampfLab/'
            ele = 'adc2014-11-25T23_00_08.bin' 
        To test with Joe's:
            fpb = 'data/PatonLab/'
            ele = Q_Ref_S16_first32_2016-04-15T11_30_43.bin
            ele = Q_Ref_S16_last32_2016-04-15T11_30_43.bin
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
        fpa = '/Users/soyunkope/Documents/'
        fpb = 'INDP2015/2016_S02/R01_JoePaton/00_AutoSpikeSorting/realData/F32/'
        ele = Q_Ref_S16_first32_2016-04-15T11_30_43.bin
    uarray:  str(Array) ['F32', 'L32']
    GT02 = '/Users/soyunkope/Documents/INDP2015/2016_S02/R01_JoePaton/00_AutoSpikeSorting/TestDAta/GT02/GroundTruth6ch.dat'

    '''
    fpa = '/Users/soyunkope/Documents/'
    fpb = 'INDP2015/2016_S02/R01_JoePaton/00_AutoSpikeSorting/realData/'
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


def testUP2noise(signal):
    '''
    Takes the signal and creates a noise signal with the same size
    Then plots 3 information dimensions of the signal
    Asumming sr=30k, nyquist = 15k, maximun "information dimension"
    '''
    span = len(signal)
    #randSig = np.random.random_sample(span)
    res = 150, 1500, 15000
    sampD = span-res[-1]
    ele8 = np.array([Dwestim(signal, w=x, step=1) for x in res])
    ele8N = np.array([ele8[x]/sum(ele8[x]) for x in range(len(res))])
    
    #ranB = np.array([Dwestim(randSig, w=x, step=1) for x in res])
    #ranBN= np.array([ranB[x] / sum(ranB[x]) for x in range(len(res))])
    index = [1,2]
    #np.random.shuffle(index)
    fig = plt.figure(figsize=[20,7])
    ax = [fig.add_subplot(1, 2, i, projection='3d') for i in index]
    
    ax[0].plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], 'g-', alpha=.71)
    ax[0].plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], '*r-', alpha=.004)
    ax[1].plot(np.arange(len(signal)), signal)
    #ax[1].plot3D(ranBN[0][:sampD],ranBN[1][:sampD], ranBN[2][:sampD], 'g-', alpha=.71)
    #ax[1].plot3D(ranBN[0][:sampD],ranBN[1][:sampD], ranBN[2][:sampD], '*r-', alpha=.004)


def testUP2Self(signal, sr):
    '''
    Takes the signal and creates a noise signal with the same size
    Then plots 3 information dimensions of the signal
    Asumming sr=30k, nyquist = 15k, maximun "information dimension"
    '''
    span = len(signal)
    res = [50, 150, 12000]
    sampD = span-res[-1]
    ele8 = np.array([Dwestim(signal, w=x, step=1) for x in res])
    ele8N = np.array([ele8[x]/sum(ele8[x]) for x in range(len(res))])
    fig = plt.figure(figsize=[20, 7])
    axA = fig.add_subplot(1, 2, 1, projection='3d')
    axB = fig.add_subplot(1, 2, 2)
    axA.plot3D(ele8N[0][:sampD], ele8N[1][:sampD], ele8N[2][:sampD],
               'g-', alpha=.71)
    axA.plot3D(ele8N[0][:sampD], ele8N[1][:sampD], ele8N[2][:sampD],
               '*r-', alpha=.004)
    axB.plot(signal[:sampD])

def testUP2GT(signal, sr):
    '''
    2 seconds of data from QQ and the 6 GT from Joe's roborat
    sig = [loadQQ(x)[:40000] for x in [3,4,-3,-4]]
    gt6 = loadAK(GT02, 6, 'int')
    gt = gt6[:,:60000]
    Example Guess the Pairs:
    test = [testUP2GT(sig[x], 20000) for x in range(4)]
    test.append(testUP2GT(sig[0], 20000))    
    indx = np.arange(4)
    np.random.shuffle(indx)
    fig = plt.figure(figsize=[20,20])
    ax = [fig.add_subplot(2,2,x+1, projection='3d') for x in range(4)]
    [ax[i].plot3D(test[indx[i]][0][:28000], test[indx[i]][1][:28000],
     test[indx[i]][2][:28000], '*r-', alpha=.004) for i in range(4)]
    figB = plt.figure(figsize=(20,20))
    ax = [figB.add_subplot(4,1,i+1) for i in range(4)]
    [ax[i].plot(sig[i][:28000]) for i in range(4)]
    ax[5].plot(sig[0][:28000])
    
    Example Odd one out:
    test = [testUP2GT(gt[x], 30000) for x in range(5)]
    test.append(testUP2GT(sig[0], 20000))
    
    ax = [fig.add_subplot(6,1,i+1) for i in range(6)]
    [ax[i].plot(gt[i][:28000]) for i in range(5)]
    figB = plt.figure(figsize=(20,20))
    ax = [figB.add_subplot(6,1,i+1) for i in range(6)]
    [ax[i].plot(gt[i][:28000]) for i in range(5)]
    ax[5].plot(sig[0][:28000])
    '''
    span = len(signal)
    res = [50, 150, 12000]
    sampD = span-res[-1]
    ele8 = np.array([Dwestim(signal, w=x, step=1) for x in res])
    ele8N = np.array([ele8[x]/sum(ele8[x]) for x in range(len(res))])
    #fig = plt.figure(figsize=[5, 5])
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.plot3D(ele8N[0][:sampD], ele8N[1][:sampD], ele8N[2][:sampD],
    #           'g-', alpha=.71)
    #ax.plot3D(ele8N[0][:sampD], ele8N[1][:sampD], ele8N[2][:sampD],
    #           '*r-', alpha=.004)
    return ele8N

def testUP2Spikes(signal, sr):
    '''
    Takes the signal and creates a noise signal with the same size
    Then plots 3 information dimensions of the signal
    Asumming sr=30k, nyquist = 15k, maximun "information dimension"
    '''
    span = len(signal)
    res = [50, 150, 12000]
    sampD = span-res[-1]
    ele8 = np.array([Dwestim(signal, w=x, step=1) for x in res])
    ele8N = np.array([ele8[x]/sum(ele8[x]) for x in range(len(res))])
    d5 = Dwestim(signal)
    spX = np.array(spkDet(signal, T=4.5))
    spk = spX[spX<sampD]
    index = [1,2]
    fig = plt.figure(figsize=[20,7])
    axA = fig.add_subplot(1, 2, 1, projection='3d')
    axB = fig.add_subplot(1, 2, 2)
    axA.plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], 'g-', alpha=.71)
    axA.plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], '*r-', alpha=.004)
    axA.plot3D(ele8N[0][spk],ele8N[1][spk], ele8N[2][spk], 'ro', ms=10)
    axB.plot(signal[:sampD])
    axB.plot(spk,signal[spk], 'ro', ms=10)
    
    
def testUP_PCA(signal, sr, MaxRes):
    '''
    Takes the signal and creates a noise signal with the same size
    Then plots 3 information dimensions of the signal
    Asumming sr=30k, nyquist = 15k, maximun "information dimension"
    '''
    span = len(signal)
    res = np.arange(2,MaxRes)
    sampD = span-res[-1]
    ele8 = np.array([Dwestim(signal, w=x, step=1) for x in res])
    ele8N = np.array([ele8[x]/sum(ele8[x]) for x in range(len(res))])
    spX = np.array(spkDet(-ele8[4], T=4))
    spk = spX[spX<sampD]
    pca = PCA()
    pcaUP = PCA(n_components= 3)
    redUP = pcaUP.fit_transform(ele8N[:])

    fig = plt.figure(figsize=[20,7])
    axA = fig.add_subplot(1, 2, 1, projection='3d')
    axB = fig.add_subplot(1, 2, 2)
    axA.plot3D(redUP[0][:sampD],redUP[1][:sampD], redUP[2][:sampD], 'g-', alpha=.71)
    axA.plot3D(redUP[0][:sampD],redUP[1][:sampD], redUP[2][:sampD], '*r-', alpha=.004)
    axA.plot3D(redUP[0][spk],redUP[1][spk], redUP[2][spk], 'ro', ms=10)
    axB.plot(signal[:sampD])
    axB.plot(spk,signal[spk], 'ro', ms=10)


def spkDet(signal, method='QQ', T=3, ori='neg'):
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
    spkIX = [np.argmin(signal[spX[i]-5:spX[i]+5])+spX[i]-10
                for i in range(len(spX))]
    return spkIX




'''
    nteresting electrodes JP
    electrode(F,2) -> main('1')
    electrode(F,10) -> main('9')
    electrode(F,18) -> main('17')
    electro(F,26) -> main('25')    

sig = loadJP('F32', 9, 60000)
testUP(sig)
fig = plt.figure()
ax = [fig.add_subplot(4, 1, i, projection='3d') for i in [1,2,3,4]]
dw3 = np.array([Dwestim(sig, w=x, step=1) for x in [150,1500,15000]])
fig = plt.figure(figsize=[20,20])
for i in range(4):
    if i > 0:
        ax = fig.add_subplot(4,1,i+1, sharex=axA)
        ax.plot(dw3[i-1][:30000])
    else:
        axA = fig.add_subplot(4,1,1)
        axA.plot(sig[:30000])

S = 5
sig = [loadJP('F32', x, 30000) for x in np.random.randint(0,32, size=S)]
[testUP(sig[x]) for x in range(S)]

S = 5
sigAK = loadAK(filename, 32, 'int16')
sig = sigAK[:,:60000]
eles = np.arange(10,15)
[testUP(sig[x]) for x in eles]

fname ='/Users/soyunkope/Documents/INDP2015/2016_S02/R01_JoePaton/00_AutoSpikeSorting/TestDAta/GT01/GroundTruth_pseudoRat_1ch.dat'
sig= np.fromfile(fname, dtype='int', count=-1)
testUP2Self(sig[:90000], 30000)
'''
