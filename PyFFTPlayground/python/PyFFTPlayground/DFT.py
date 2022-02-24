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

import pyfftw


class DFT:
    def __init__(self, plan):
        self.plan = plan


def create_complex_plan(shape, flags):
    i = pyfftw.empty_aligned(shape, dtype="complex128")
    o = pyfftw.empty_aligned(shape, dtype="complex128")
    return DFT(pyfftw.FFTW(i, o, direction="FFTW_BACKWARD", flags=flags))


def create_real_plan(shape, flags):
    i = pyfftw.empty_aligned(shape, dtype="float64")
    output_shape = (i.shape[0], i.shape[-1] // 2 + 1)
    o = pyfftw.empty_aligned(output_shape, dtype="complex128")

    return DFT(pyfftw.FFTW(i, o, direction="FFTW_FORWARD", flags=flags))


def create_real_plan_backward(shape, flags):
    i = pyfftw.empty_aligned(shape, dtype="float64")
    output_shape = (i.shape[0], i.shape[-1] // 2 + 1)
    o = pyfftw.empty_aligned(output_shape, dtype="complex128")

    return DFT(pyfftw.FFTW(o, i, direction="FFTW_BACKWARD", flags=flags))
