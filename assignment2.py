#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:18:51 2020

@author: ernie
"""
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt

from numpy.core.numeric import (
    absolute, asanyarray, arange, zeros, greater_equal, multiply, ones,
    asarray, where, int8, int16, int32, int64, empty, promote_types, diagonal,
    nonzero
    )
import os
import matplotlib.image as mpimg


_range = range

def JointHist(I, J, bins=10):
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 1 and N != 2:
        xedges = yedges = asarray(bins)
        bins = [xedges, yedges]
      
    sample = np.atleast_2d([I, J]).T
    nbin = np.empty(2, int)  
    edges = 2*[None]    
    bins = 2*[bins]    
    
    for i in _range(2):                         # Create edge arrays
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            a = sample[:,i]
            first_edge, last_edge = a.min(), a.max()
            if first_edge == last_edge: # expand empty range to avoid divide by zero
                first_edge = first_edge - 0.5
                last_edge = last_edge + 0.5
            smin = first_edge
            smax = last_edge
            edges[i] = np.linspace(smin, smax, bins[i] + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                    .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))
            
        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
    Ncount = tuple(   
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(2)
    )
    for i in _range(2):
        on_edge = (sample[:, i] == edges[i][-1])
        Ncount[i][on_edge] -= 1
    xy = np.ravel_multi_index(Ncount, nbin)
    hist = np.bincount(xy, None, minlength=nbin.prod())
    hist = hist.reshape(nbin)
    hist = hist.astype(float, casting='safe')
    core = 2*(slice(1, -1),)
    hist = hist[core]
    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")    
    return hist, edges[0], edges[1]



def getSourceImage (imageDirectoryNfilename):
    directory = os.path.join(os.path.dirname(__file__))
    dataDirectory = os.path.join(directory, imageDirectoryNfilename)
    #img = nib.load(dataDirectory)
    #imgData = img.get_data()
    img  = mpimg.imread(dataDirectory)
    return img

img1 = getSourceImage("images/I1.png")
img2 = getSourceImage("images/I2.jpg")

img3 = getSourceImage("images/J1.png")
img4 = getSourceImage("images/J2.jpg")

'''
raveledX = np.ravel(img1[:,:,0])
raveledY = np.ravel(img3)
'''
raveledX = np.ravel(img2)
raveledY = np.ravel(img4)


print(raveledX.shape, raveledY.shape)
jh = JointHist(raveledX,raveledY,50)
#jh = np.histogram2d(raveledX,raveledY,50)
plt.imshow(np.log(jh[0]))
print(np.sum(jh[0]))
cbar = plt.colorbar()
cbar.ax.set_ylabel('Color')