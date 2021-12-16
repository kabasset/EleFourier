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

Fits::Position<2> RealForwardDftType::outShape(const Fits::Position<2>& shape) {
  return {shape[0] / 2 + 1, shape[1]};
}

Fits::Position<2> RealBackwardDftType::outShape(const Fits::Position<2>& shape) {
  return shape;
}

Fits::Position<2> ComplexForwardDftType::outShape(const Fits::Position<2>& shape) {
  return shape;
}

Fits::Position<2> ComplexBackwardDftType::outShape(const Fits::Position<2>& shape) {
  return shape;
}

fftw_plan RealForwardDftType::initFftwPlan(Fits::Raster<double, 3>& in, Fits::Raster<std::complex<double>, 3>& out) {
  const auto& shape = in.shape();
  const int width = static_cast<int>(shape[0]);
  const int height = static_cast<int>(shape[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft_r2c(
      2, // rank
      n, // n
      in.shape()[2], // howmany
      reinterpret_cast<double*>(in.data()), // in
      nullptr, // inembed
      1, // istride
      width * height, // idist
      reinterpret_cast<fftw_complex*>(out.data()), // out
      nullptr, // onembed
      1, // ostride
      (width / 2 + 1) * height, // odist
      FFTW_MEASURE); // FIXME other flags?
}

fftw_plan RealBackwardDftType::initFftwPlan(Fits::Raster<std::complex<double>, 3>& in, Fits::Raster<double, 3>& out) {
  const auto& shape = out.shape();
  const int width = static_cast<int>(shape[0]);
  const int height = static_cast<int>(shape[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft_c2r(
      2, // rank
      n, // n
      shape[2], // howmany
      reinterpret_cast<fftw_complex*>(in.data()), // in
      nullptr, // inembed
      1, // istride
      (width / 2 + 1) * height, // idist
      reinterpret_cast<double*>(out.data()), // out
      nullptr, // onembed
      1, // ostride
      width * height, // odist
      FFTW_MEASURE); // FIXME other flags?
}

fftw_plan ComplexForwardDftType::initFftwPlan(
    Fits::Raster<std::complex<double>, 3>& in,
    Fits::Raster<std::complex<double>, 3>& out) {
  const auto& shape = in.shape();
  const int width = static_cast<int>(shape[0]);
  const int height = static_cast<int>(shape[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft(
      2, // rank
      n,
      shape[2], // howmany,
      reinterpret_cast<fftw_complex*>(in.data()), // in
      nullptr, // inembed
      1, // istride
      width * height, // idist
      reinterpret_cast<fftw_complex*>(out.data()), // out
      nullptr, // onembed
      1, // ostride
      width * height, // odist
      FFTW_FORWARD, // sign
      FFTW_MEASURE); // FIXME other flags?
}

fftw_plan ComplexBackwardDftType::initFftwPlan(
    Fits::Raster<std::complex<double>, 3>& in,
    Fits::Raster<std::complex<double>, 3>& out) {
  const auto& shape = out.shape();
  const int width = static_cast<int>(shape[0]);
  const int height = static_cast<int>(shape[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft(
      2, // rank
      n,
      shape[2], // howmany,
      reinterpret_cast<fftw_complex*>(in.data()), // in
      nullptr, // inembed
      1, // istride
      width * height, // idist
      reinterpret_cast<fftw_complex*>(out.data()), // out
      nullptr, // onembed
      1, // ostride
      width * height, // odist
      FFTW_BACKWARD, // sign
      FFTW_MEASURE); // FIXME other flags?
}

} // namespace Fourier
} // namespace Euclid
