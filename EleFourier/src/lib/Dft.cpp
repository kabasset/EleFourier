/**
 * @copyright (C) 2012-2020 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 3.0 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 */

#include "EleFourier/Dft.h"

namespace Euclid {
namespace Fourier {

namespace Internal {

template <>
long coefsWidth<ComplexToComplex>(long imageWidth) {
  return imageWidth;
}

template <>
fftw_plan makeFftwTransform<ComplexToComplex>(
    Fits::Raster<std::complex<double>, 3>& image,
    Fits::Raster<std::complex<double>, 3>& coefs) {
  const int width = static_cast<int>(image.shape()[0]);
  const int height = static_cast<int>(image.shape()[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft(
      2, // rank
      n,
      image.shape()[2], // howmany,
      reinterpret_cast<fftw_complex*>(image.data()), // in
      nullptr, // inembed
      1, // istride
      width * height, // idist
      reinterpret_cast<fftw_complex*>(coefs.data()), // out
      nullptr, // onembed
      1, // ostride
      width * height, // odist
      FFTW_FORWARD, // sign
      FFTW_MEASURE); // FIXME other flags?
}

template <>
fftw_plan makeFftwInverse<ComplexToComplex>(
    Fits::Raster<std::complex<double>, 3>& image,
    Fits::Raster<std::complex<double>, 3>& coefs) {
  const int width = static_cast<int>(image.shape()[0]);
  const int height = static_cast<int>(image.shape()[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft(
      2, // rank
      n,
      image.shape()[2], // howmany,
      reinterpret_cast<fftw_complex*>(image.data()), // in
      nullptr, // inembed
      1, // istride
      width * height, // idist
      reinterpret_cast<fftw_complex*>(coefs.data()), // out
      nullptr, // onembed
      1, // ostride
      width * height, // odist
      FFTW_BACKWARD, // sign
      FFTW_MEASURE); // FIXME other flags?
}

template <>
long coefsWidth<RealToComplex>(long imageWidth) {
  return imageWidth / 2 + 1;
}

template <>
fftw_plan
makeFftwTransform<RealToComplex>(Fits::Raster<double, 3>& image, Fits::Raster<std::complex<double>, 3>& coefs) {
  const int width = static_cast<int>(image.shape()[0]);
  const int height = static_cast<int>(image.shape()[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft_r2c(
      2, // rank
      n, // n
      image.shape()[2], // howmany
      reinterpret_cast<double*>(image.data()), // in
      nullptr, // inembed
      1, // istride
      width * height, // idist
      reinterpret_cast<fftw_complex*>(coefs.data()), // out
      nullptr, // onembed
      1, // ostride
      (width / 2 + 1) * height, // odist
      FFTW_MEASURE); // FIXME other flags?
}

template <>
fftw_plan makeFftwInverse<RealToComplex>(Fits::Raster<double, 3>& image, Fits::Raster<std::complex<double>, 3>& coefs) {
  const int width = static_cast<int>(image.shape()[0]);
  const int height = static_cast<int>(image.shape()[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft_c2r(
      2, // rank
      n, // n
      image.shape()[2], // howmany
      reinterpret_cast<fftw_complex*>(coefs.data()), // in
      nullptr, // inembed
      1, // istride
      (width / 2 + 1) * height, // idist
      reinterpret_cast<double*>(image.data()), // out
      nullptr, // onembed
      1, // ostride
      width * height, // odist
      FFTW_MEASURE); // FIXME other flags?
}

} // namespace Internal
} // namespace Fourier
} // namespace Euclid
