from __future__ import division
import os
import astropy.io.fits as fits
import numpy as np
import scipy.fftpack as fft
import numpy.ma as ma
from functools import reduce
import re
from scipy.ndimage.interpolation import shift

def file_helper(location):
    return lambda f, l: \
        [(location + os.sep + 'lmi.%04d.fits' % i) for i in range(f, l+1)]

def combine_flat(filelist, bias=None):
    flat = open_image(filelist, bias=bias, combine='median', medium_subtract=False)
    flat.data = flat.data.astype(np.float64)
    flat.data /= np.mean(flat.data)
    return flat

def combine_bias(filelist):
    return open_image(filelist, combine='median', medium_subtract=False)

def open_image(filelist, bias=None, flat=None, combine='offset_pad', overscan_region=None, medium_subtract=True):
    if hasattr(filelist, 'lower'):
        filelist = [filelist]

    if isinstance(bias, fits.HDUList):
        if len(bias) != 1:
            raise ValueError("bias argument to open_image has more then one HDU")
        bias = bias[0]
    if hasattr(bias, 'data'):
        bias = bias.data

    if isinstance(flat, fits.HDUList):
        if len(flat) != 1:
            raise ValueError("flat argument to open_image has more then one HDU")
        flat = flat[0]
    if hasattr(flat, 'data'):
        flat = flat.data

    VALID_COMBINE_VALUES = ['offset_trim', 'offset_pad', 'median', 'stacked']
    if combine not in VALID_COMBINE_VALUES:
        raise ValueError('combinue argument to open_image must be one of %s'
                %str(VALID_COMBINE_VALUES))

    file_data = [fits.getdata(f) for f in filelist]
    header = fits.getheader(filelist[0]).copy()

    if overscan_region is None and 'TRIMSEC' in header:
        overscan_match = re.match('^\[(\d*):(\d*),(\d*):(\d*)\]$', header['TRIMSEC'])
        minX, maxX, minY, maxY = map(int,overscan_match.group(3,4,1,2))
    elif overscan_region is not None:
        minX, maxX, minY, maxY = overscan_region
    else:
        minX, maxX, minY, maxY = 0, len(file_data[0]), 0, len(file_data[0][0])

    for i in range(len(file_data)):
        file_data[i] = file_data[i][minX:maxX, minY:maxY]
        if bias is not None:
            file_data[i] = file_data[i] - bias
        if flat is not None:
            file_data[i] = file_data[i] / flat
        if medium_subtract:
            file_data[i] -= np.median(file_data[i])
        file_data[i] = file_data[i].astype(np.float32)

    if combine == 'median':
        combined = np.median(file_data, axis=0)
    elif combine == 'stacked':
        combined = np.sum(file_data, axis=0)
    elif combine == 'offset_trim':
        combined = reduce(_combine_images, file_data)
    elif combine == 'offset_pad':
        combined = reduce(lambda x1, x2: _combine_images(x1, x2, clip=False), file_data)

    return fits.PrimaryHDU(combined, header)

def get_halpha(ha_on, ha_off):
    ON_OFF_RATIO =  1068.9235000000006 / 265.45335000000017
    newdata = ON_OFF_RATIO*ha_on.data - ha_off.data
    return fits.PrimaryHDU(newdata, ha_on.header.copy())

# from AsPyLib, Copyright under GPL3+ by Jerome Caron
def find_offset(image1, image2):
    im1 = fft.fft2(image1)
    im2 = fft.fft2(image2)
    crosscorr = fft.ifft2(im1 * np.ma.conjugate(im2))

    modulus = fft.fftshift(np.abs(crosscorr))
    dx, dy = np.where(modulus==np.max(modulus.flatten()))

    if len(dx) == 1:
        dx = dx[0]
        dy = dy[0]
    else:
        dy = dx[1]
        dx = dx[0]

    dx3 = image1.shape[0]/2. - np.float(dx)
    dy3 = image1.shape[1]/2. - np.float(dy)
    return [dx3, dy3]

def _zero_pad_to_same_size(a, b):
    [ay, ax], [by, bx] = a.shape, b.shape
    if ax < bx or ay < by:
        a = np.pad(a, ( (by-ay,0),(bx-ax,0) ), mode='constant')
    elif ax > bx or ay > by:
        b = np.pad(b, ( (ay-by,0),(ax-bx,0) ), mode='constant')
    return a, b, [ax-bx, ay-by]

def _combine_images(image1, image2, offset=None, clip=True):
    image1, image2, padding_offset = _zero_pad_to_same_size(image1, image2)

    if offset is None:
        offset = find_offset(image1, image2)

    padding_offset = np.asarray(padding_offset)
    offset = np.asarray(offset)

    offset += padding_offset * (-1 if clip else 1)
    offsetx, offsety = [int(x) for x in offset]

    image2 = shift(image2, [offsety, offsetx])
    image = image1 + image2

    keep_range = [ (offsety, None), (offsetx, None) ]
    if offsety < 0: keep_range[0] = (None, offsety)
    if offsetx < 0: keep_range[1] = (None, offsetx)

    if clip:
        image = image[keep_range[0][0]:keep_range[0][1], :]
        image = image[:, keep_range[1][0]:keep_range[1][1] ]
    else:
        image[keep_range[0][1]:keep_range[0][0], :] = 0
        image[:, keep_range[1][1]:keep_range[1][0] ] = 0

    return image
