#!/usr/bin/env python3

"""Compare fftconvolve with scipy and pyfftw backends
From https://github.com/pyFFTW/pyFFTW/issues/310#issuecomment-946342809
"""

import pyfftw
import multiprocessing
import scipy.signal
import scipy.fft
import numpy
from timeit import Timer

xsize = 1024
ysize = 1024
number_it = 10

a = pyfftw.empty_aligned((xsize, ysize), dtype=np.float64)
b = pyfftw.empty_aligned((xsize, ysize), dtype=np.float64)

a[:] = numpy.random.randn(xsize, ysize)
b[:] = numpy.random.randn(xsize, ysize)

t = Timer(lambda: scipy.signal.fftconvolve(a, b))

print(f"Time with scipy.fft default backend installed: {(t.timeit(number=number_it)):1.3f} seconds")

# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"

# Use the backend pyfftw.interfaces.scipy_fft
with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft, only=True):
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()

    # We cheat a bit by doing the planning first
    scipy.signal.fftconvolve(a, b)

    print(f"Time with pyfftw backend installed: {(t.timeit(number=number_it)):1.3f} seconds")

