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
:file: python/PyFFTPlayground/PyFFTPlaygroundTutorial.py

:date: 12/15/21
:author: user

"""

import sys
import argparse
import logging as log
import numpy as np
from astropy.io import fits
from scipy.fft import fft2, ifft2, set_global_backend
from Timer import Timer
from scipy import signal


def mainMethod(args):
    """
    @brief The "main" method.

    @details This method is the entry point to the program. In this sense, it is
    similar to a main (and it is why it is called mainMethod()).
    """
    log.basicConfig(stream=sys.stdout, level=log.INFO)
    logger = log.getLogger("PyFFTPlaygroundTutorial")

    logger.info("#")
    logger.info("# Entering PyFFTPlaygroundTutorial mainMethod()")
    logger.info("#")

    t = Timer(logger=logger.info)

    # Set FFT backend$
    # From scipy benchmark source: https://github.com/scipy/scipy/blob/master/benchmarks/benchmarks/fft_basic.py#L245
    backend = args.backend

    if backend == "scipy":
        logger.info("Use default scipy as FFT backend (pocketfft?)...")
    elif backend == "numpy":
        from scipy.fft._debug_backends import NumPyBackend

        set_global_backend(NumPyBackend)
        logger.info("Use NumPy as backend (advertised to be slower)...")
    elif backend == "pyfftw":
        import pyfftw

        pyfftw.interfaces.cache.enable()
        set_global_backend(pyfftw.interfaces.scipy_fft)
        logger.info("Use FFTW as backend (advertised to be faster with plans)...")
    else:
        raise ValueError("Supported backends are scipy,numpy and pyfftw if pyfftw is available")

    logger.info("Opening fits file...")
    t.start()
    input_file = args.input
    hdul = fits.open(input_file)
    t.stop()

    logger.info("Applying DFT to all HDUs...")
    t.start()
    transform = []
    for hdu in hdul:
        # Need to convert ndarray to np.float64
        data = hdu.data.astype(np.float64)
        transform.append(fft2(data))
    t.stop()

    logger.info("Perform convolution using signal.fftconvolve...")
    t.start()
    conv = []
    kernel = hdul[0].data.astype(np.float64)
    for hdu in hdul[1:]:
        data = hdu.data.astype(np.float64)
        conv.append(signal.fftconvolve(data, kernel, mode="same"))
    t.stop()

    logger.info("Applying inverse DFTs...")
    t.start()
    inverse = []
    for hdu in hdul:
        data = hdu.data.astype(np.float64)
        inverse.append(ifft2(data))
    t.stop()

    logger.info("#")
    logger.info("# Exiting PyFFTPlaygroundTutorial mainMethod()")
    logger.info("#")

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="Input fits file to process")
    parser.add_argument("--backend", type=str, help="Set the FFT backend (can be numpy, scipy, pyfftw", default="scipy")

    args = parser.parse_args()
    mainMethod(args)
