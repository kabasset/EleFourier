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

#include "EleFourier/DftType.h"

namespace Euclid {
namespace Fourier {

template <>
Fits::Position<2> RealDftType::Parent::outShape(const Fits::Position<2>& shape) {
  return {shape[0] / 2 + 1, shape[1]};
}

template <>
fftw_plan initFftwPlan<RealDftType>(Fits::PtrRaster<double, 3>& in, Fits::PtrRaster<std::complex<double>, 3>& out) {
  const auto& shape = in.shape();
  const int width = static_cast<int>(shape[0]);
  const int height = static_cast<int>(shape[1]);
  int n[] = {height, width}; // FFTW ordering
  return fftw_plan_many_dft_r2c(
      2, // rank
      n, // n
      shape[2], // howmany
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

template <>
fftw_plan
initFftwPlan<Inverse<RealDftType>>(Fits::PtrRaster<std::complex<double>, 3>& in, Fits::PtrRaster<double, 3>& out) {
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

template <>
fftw_plan initFftwPlan<ComplexDftType>(
    Fits::PtrRaster<std::complex<double>, 3>& in,
    Fits::PtrRaster<std::complex<double>, 3>& out) {
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

template <>
fftw_plan initFftwPlan<Inverse<ComplexDftType>>(
    Fits::PtrRaster<std::complex<double>, 3>& in,
    Fits::PtrRaster<std::complex<double>, 3>& out) {
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

template <>
Fits::Position<2> HermitianComplexDftType::Parent::inShape(const Fits::Position<2>& shape) {
  return {shape[0] / 2 + 1, shape[1]};
}

template <>
Fits::Position<2> HermitianComplexDftType::Parent::outShape(const Fits::Position<2>& shape) {
  return {shape[0] / 2 + 1, shape[1]};
}

template <>
fftw_plan initFftwPlan<HermitianComplexDftType>(
    Fits::PtrRaster<std::complex<double>, 3>& in,
    Fits::PtrRaster<std::complex<double>, 3>& out) {
  return initFftwPlan<ComplexDftType>(in, out);
}

template <>
fftw_plan initFftwPlan<Inverse<HermitianComplexDftType>>(
    Fits::PtrRaster<std::complex<double>, 3>& in,
    Fits::PtrRaster<std::complex<double>, 3>& out) {
  return initFftwPlan<Inverse<ComplexDftType>>(in, out);
}

} // namespace Fourier
} // namespace Euclid