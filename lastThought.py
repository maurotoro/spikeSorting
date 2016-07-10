# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 16:00:33 2016

@author: soyunkope
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import scipy.io as sio
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
    For test with Kampf's:
        fpa = '/Users/soyunkope/Documents/scriptkidd/git/spikeSorting/'
        fpb = 'data/kampfLab/'
        ele = 'adc2014-11-25T23_00_08.bin' 
    '''
    fdata = np.fromfile(filename,dtype=dtype)
    numsamples = len(fdata) / numChannels
    data = np.reshape(fdata,(numsamples,numChannels))
    return (np.transpose(data))

def loadJP(uarray, electrode, dur):
    '''
    For test with Joe's electrodes
        fpa = '/Users/soyunkope/Documents/'
        fpb = 'INDP2015/2016_S02/R01_JoePaton/00_AutoSpikeSorting/realData/F32/'
        ele = np.random.randint(0,31)
        filename = fpa+fpb+str(ele)+'/ele'+str(ele)+'.dat'
    '''
    fpa = '/Users/soyunkope/Documents/'
    fpb = 'INDP2015/2016_S02/R01_JoePaton/00_AutoSpikeSorting/realData/'
    fur = str(uarray)+'/'+str(electrode)+'/'
    fn = 'ele'+str(electrode)+'.dat'
    filename = fpa+fpb+fur+fn
    signal = np.fromfile(filename, dtype='int', count=dur)
    return signal
    
def testUP(signal):
    '''
    Takes the signal and creates a noise signal with the same size
    Then plots 3 information dimensions of the signal
    Asumming sr=30k, nyquist = 15k, maximun "information dimension"
    '''
    span = len(signal)
    randSig = np.random.random_sample(span)
    res = 150, 1500, 15000
    sampD = span-res[-1]
    ele8 = np.array([Dwestim(signal, w=x, step=1) for x in res])
    ele8N = np.array([ele8[x]/sum(ele8[x]) for x in range(len(res))])
    
    ranB = np.array([Dwestim(randSig, w=x, step=1) for x in res])
    ranBN= np.array([ranB[x] / sum(ranB[x]) for x in range(len(res))])
    index = [1,2]
    np.random.shuffle(index)
    fig = plt.figure(figsize=[20,7])
    ax = [fig.add_subplot(1, 2, i, projection='3d') for i in index]
    
    ax[0].plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], 'g-', alpha=.71)
    ax[0].plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], '*r-', alpha=.004)
    
    ax[1].plot3D(ranBN[0][:sampD],ranBN[1][:sampD], ranBN[2][:sampD], 'g-', alpha=.71)
    ax[1].plot3D(ranBN[0][:sampD],ranBN[1][:sampD], ranBN[2][:sampD], '*r-', alpha=.004)

'''
    nteresting electrodes JP
    electrode(F,2) -> main('1')
    electrode(F,10) -> main('9')
    electrode(F,18) -> main('17')
    electro(F,26) -> main('25')    
'''
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



