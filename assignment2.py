#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:18:51 2020

@author: ernie
"""
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import functools

from numpy.core.numeric import (
    absolute, asanyarray, arange, zeros, greater_equal, multiply, ones,
    asarray, where, int8, int16, int32, int64, empty, promote_types, diagonal,
    nonzero
    )
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.core import iinfo, transpose
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import matplotlib.image as mpimg




_range = range

def JointHist(I, J, bins=10, range=None, weights=None):

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        xedges = yedges = asarray(bins)
        bins = [xedges, yedges]
        
    hist, edges = histogramddErnie([I, J], bins, range, weights)
    return hist, edges[0], edges[1]

def histogramddErnie(sample, bins=10, range=None,  weights=None):
    try:
        # Sample is an ND-array.
        N, D = sample.shape
        print(N,D)
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape
        
    #print(N,D)
    nbin = np.empty(D, int)
    edges = D*[None]
    dedges = D*[None]
    if weights is not None:
        weights = np.asarray(weights)

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]

    # normalize the range argument
    if range is None:
        range = (None,) * D
    elif len(range) != D:
        raise ValueError('range argument must have one entry per dimension')

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = get_outer_edges(sample[:,i], range[i])
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
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = np.bincount(xy, weights, minlength=nbin.prod())

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting='safe')

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D*(slice(1, -1),)
    hist = hist[core]

    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")
    return hist, edges


def get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge




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

raveledX = np.ravel(img1[:,:,0])
#raveledY = np.ravel(img2)

raveledY = np.ravel(img3)

print(raveledY.shape)
jh = JointHist(raveledX,raveledY,50)
plt.imshow(jh[0])