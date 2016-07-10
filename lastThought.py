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
        Goes through the signal meassuring the eclidean distance of each
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


fpa = '/Users/soyunkope/Documents/INDP2015/'
fpb = '2016_S02/R01_JoePaton/00_AutoSpikeSorting/realData/L32/'
ele = '8'
fname = fpa+fpb+ele+'/ele'+ele+'.dat'
fname
sr = 30000
dur = sr*1

signal = np.fromfile(fname, dtype='int', count=dur)
randSig = np.random.random_sample(dur)
res = 15, 1500, 15000
sampD = dur-res[-1]
ele8 = np.array([Dwestim(signal, w=x, step=1) for x in res])
ele8N = np.array([ele8[x]/sum(ele8[x]) for x in range(len(res))])

ranB = np.array([Dwestim(randSig, w=x, step=1) for x in res])
ranBN= np.array([ranB[x] / sum(ranB[x]) for x in range(len(res))])

fig = np.figure(figsize=[20,7])
ax = [fig.add_subplot(1, 2, i, projection='3d') for i in [1,2]]

ax[0].plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], 'g-', alpha=.71)
ax[0].plot3D(ele8N[0][:sampD],ele8N[1][:sampD], ele8N[2][:sampD], '*r-', alpha=.0041)

ax[1].plot3D(ranBN[0][:sampD],ranBN[1][:sampD], ranBN[2][:sampD], 'g-', alpha=.71)
ax[1].plot3D(ranBN[0][:sampD],ranBN[1][:sampD], ranBN[2][:sampD], '*r-', alpha=.0041)