"""
Low-level functions to work in frequency domain for n-dim arrays

Copied from https://github.com/int-brain-lab/ibl-neuropixel/ on 2/1/2024

"""
from math import pi

import numpy as np
import scipy.fft

def rms(x, axis=-1):
    """
    Root mean square of array along axis

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    return np.sqrt(np.mean(x ** 2, axis=axis))

def _fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y


def fcn_cosine(bounds, gpu=False):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :param gpu: bool
    :return: lambda function
    """
    if gpu:
        import cupy as gp
    else:
        gp = np

    def _cos(x):
        return (1 - gp.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * gp.pi)) / 2
    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func


def fscale(ns, si=1, one_sided=False):
    """
    numpy.fft.fftfreq returns Nyquist as a negative frequency so we propose this instead

    :param ns: number of samples
    :param si: sampling interval in seconds
    :param one_sided: if True, returns only positive frequencies
    :return: fscale: numpy vector containing frequencies in Hertz
    """
    fsc = np.arange(0, np.floor(ns / 2) + 1) / ns / si  # sample the frequency scale
    if one_sided:
        return fsc
    else:
        return np.concatenate((fsc, -fsc[slice(-2 + (ns % 2), 0, -1)]), axis=0)


def bp(ts, si, b, axis=None):
    """
    Band-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 4 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='bp')


def lp(ts, si, b, axis=None):
    """
    Low-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='lp')


def hp(ts, si, b, axis=None):
    """
    High-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='hp')


def _freq_filter(ts, si, b, axis=None, typ='lp'):
    """
        Wrapper for hp/lp/bp filters
    """
    if axis is None:
        axis = ts.ndim - 1
    ns = ts.shape[axis]
    f = fscale(ns, si=si, one_sided=True)
    if typ == 'bp':
        filc = _freq_vector(f, b[0:2], typ='hp') * _freq_vector(f, b[2:4], typ='lp')
    else:
        filc = _freq_vector(f, b, typ=typ)
    if axis < (ts.ndim - 1):
        filc = filc[:, np.newaxis]
    return np.real(np.fft.ifft(np.fft.fft(ts, axis=axis) * fexpand(filc, ns, axis=0), axis=axis))


def _freq_vector(f, b, typ='lp'):
    """
        Returns a frequency modulated vector for filtering

        :param f: frequency vector, uniform and monotonic
        :param b: 2 bounds array
        :return: amplitude modulated frequency vector
    """
    filc = fcn_cosine(b)(f)
    if typ.lower() in ['hp', 'highpass']:
        return filc
    elif typ.lower() in ['lp', 'lowpass']:
        return 1 - filc

    
def fexpand(x, ns=1, axis=None):
    """
    Reconstructs full spectrum from positive frequencies
    Works on the last dimension (contiguous in c-stored array)

    :param x: numpy.ndarray
    :param axis: axis along which to perform reduction (last axis by default)
    :return: numpy.ndarray
    """
    if axis is None:
        axis = x.ndim - 1
    # dec = int(ns % 2) * 2 - 1
    # xcomp = np.conj(np.flip(x[..., 1:x.shape[-1] + dec], axis=axis))
    ilast = int((ns + (ns % 2)) / 2)
    xcomp = np.conj(np.flip(np.take(x, np.arange(1, ilast), axis=axis), axis=axis))
    return np.concatenate((x, xcomp), axis=axis)


class WindowGenerator(object):
    """
    `wg = WindowGenerator(ns, nswin, overlap)`

    Provide sliding windows indices generator for signal processing applications.
    For straightforward spectrogram / periodogram implementation, prefer scipy methods !

    Example of implementations in test_dsp.py.
    """
    def __init__(self, ns, nswin, overlap):
        """
        :param ns: number of sample of the signal along the direction to be windowed
        :param nswin: number of samples of the window
        :return: dsp.WindowGenerator object:
        """
        self.ns = int(ns)
        self.nswin = int(nswin)
        self.overlap = int(overlap)
        self.nwin = int(np.ceil(float(ns - nswin) / float(nswin - overlap))) + 1
        self.iw = None

    @property
    def firstlast(self):
        """
        Generator that yields first and last index of windows

        :return: tuple of [first_index, last_index] of the window
        """
        self.iw = 0
        first = 0
        while True:
            last = first + self.nswin
            last = min(last, self.ns)
            yield (first, last)
            if last == self.ns:
                break
            first += self.nswin - self.overlap
            self.iw += 1

    @property
    def slice(self):
        """
        Generator that yields slices of windows

        :return: a slice of the window
        """
        for first, last in self.firstlast:
            yield slice(first, last)

    def slice_array(self, sig, axis=-1):
        """
        Provided an array or sliceable object, generator that yields
        slices corresponding to windows. Especially useful when working on memmpaps

        :param sig: array
        :param axis: (optional, -1) dimension along which to provide the slice
        :return: array slice Generator
        """
        for first, last in self.firstlast:
            yield np.take(sig, np.arange(first, last), axis=axis)

    def tscale(self, fs):
        """
        Returns the time scale associated with Window slicing (middle of window)
        :param fs: sampling frequency (Hz)
        :return: time axis scale
        """
        return np.array([(first + (last - first - 1) / 2) / fs for first, last in self.firstlast])
