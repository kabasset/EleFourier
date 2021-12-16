#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#


"""
:file: python/PyFFTPlayground/PyFFTWPlayground.py

:date: 12/15/21
:author: user

"""

import sys
import argparse
import logging as log
import multiprocessing
import numpy as np
from astropy.io import fits
import pyfftw
from scipy.fft import fft2, ifft2, set_backend
from Timer import Timer
from scipy import signal


def mainMethod(args):
    """
    @brief The "main" method.
    
    @details This method is the entry point to the program. In this sense, it is
    similar to a main (and it is why it is called mainMethod()).
    Taken from pyFFTW example in https://github.com/pyFFTW/pyFFTW/issues/310#issuecomment-946342809
    """

    log.basicConfig(stream=sys.stdout, level=log.INFO)
    logger = log.getLogger("PyFFTWPlayground")

    logger.info("#")
    logger.info("# Entering PyFFTWPlayground mainMethod()")
    logger.info("#")

    t = Timer(logger=logger.info)

    logger.info("Opening fits file...")
    t.start()
    input_file = args.input
    hdul = fits.open(input_file)
    t.stop()

    kernel_hdu = hdul[0].data.astype(np.float64)
    kernel_shape = kernel_hdu.shape

    kernel = pyfftw.empty_aligned(kernel_shape, dtype=np.float64)
    image = pyfftw.empty_aligned(kernel_shape, dtype=np.float64)

    # Configure PyFFTW to use all cores (the default is single-threaded)
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"

    logger.info("Applying DFT to all HDUs...")
    t.start()

    kernel[:] = kernel_hdu
    # Use the backend pyfftw.interfaces.scipy_fft
    with set_backend(pyfftw.interfaces.scipy_fft, only=True):
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()

        transform = []
        for hdu in hdul:
            # Copy in input buffers
            image[:] = hdu.data.astype(np.float64)
            transform.append(fft2(image))

    t.stop()

    logger.info("Perform convolution using signal.fftconvolve...")
    t.start()
    conv = []
    kernel[:] = kernel_hdu

    with set_backend(pyfftw.interfaces.scipy_fft, only=True):
        for hdu in hdul[1:]:
            image[:] = hdu.data.astype(np.float64)
            conv.append(signal.fftconvolve(image, kernel))

    t.stop()

    logger.info("#")
    logger.info("# Exiting PyFFTWPlayground mainMethod()")
    logger.info("#")

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="Input fits file to process")

    args = parser.parse_args()
    mainMethod(args)
