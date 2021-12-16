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

#ifndef _ELEFOURIER_DFT_H
#define _ELEFOURIER_DFT_H

#include "EleFitsData/Raster.h"
#include "EleFourier/TransformPlan.h"

#include <complex>
#include <fftw3.h>

namespace Euclid {
namespace Fourier {

struct ComplexToComplex {
  using Value = std::complex<double>;
  using Coefficient = std::complex<double>;
};

struct RealToComplex {
  using Value = double;
  using Coefficient = std::complex<double>;
};

namespace Internal {

template <typename T>
Fits::PtrRaster<T, 3> makeFftwRaster(long width, long height, long count, T* data = nullptr) {
  T* d = data ? data : (T*)fftw_malloc(sizeof(T) * width * height * count);
  if (data) {
    printf("Data already allocated at %p\n", (void*)d);
  } else {
    printf("Allocated %li values at %p\n", width * height * count, (void*)d);
  }
  return {{width, height, count}, d};
}

template <typename TDomains>
fftw_plan makeFftwTransform(
    Fits::Raster<typename TDomains::Value, 3>& signal,
    Fits::Raster<typename TDomains::Coefficient, 3>& fourier);

template <typename TDomains>
fftw_plan makeFftwInverse(
    Fits::Raster<typename TDomains::Value, 3>& signal,
    Fits::Raster<typename TDomains::Coefficient, 3>& fourier);

template <typename TDomains>
long fourierWidth(long signalWidth);

} // namespace Internal

/**
 * @brief Compute the Fourier coefficients magnitude.
 */
Fits::VecRaster<double> evalMagnitude(const Fits::Raster<std::complex<double>>& coefficients) {
  Fits::VecRaster<double> res(coefficients.shape());
  std::transform(coefficients.begin(), coefficients.end(), res.begin(), [](auto c) {
    return std::norm(c);
  });
  return res;
}

struct RealForwardDftType;
struct RealBackwardDftType;
struct ComplexForwardDftType;
struct ComplexBackwardDftType;

struct RealForwardDftType {
  using InValue = double;
  using OutValue = std::complex<double>;
  using Inverse = RealBackwardDftType;
  static Fits::Position<2> outShape(const Fits::Position<2>& shape);
  static fftw_plan initFftwPlan(Fits::Raster<InValue, 3>& in, Fits::Raster<OutValue, 3>& fourier);
};

struct RealBackwardDftType {
  using InValue = std::complex<double>;
  using OutValue = double;
  using Inverse = RealForwardDftType;
  static Fits::Position<2> outShape(const Fits::Position<2>& shape);
  static fftw_plan initFftwPlan(Fits::Raster<InValue, 3>& in, Fits::Raster<OutValue, 3>& fourier);
};

struct ComplexForwardDftType {
  using InValue = std::complex<double>;
  using OutValue = std::complex<double>;
  using Inverse = ComplexBackwardDftType;
  static Fits::Position<2> outShape(const Fits::Position<2>& shape);
  static fftw_plan initFftwPlan(Fits::Raster<InValue, 3>& in, Fits::Raster<OutValue, 3>& fourier);
};

struct ComplexBackwardDftType {
  using InValue = std::complex<double>;
  using OutValue = std::complex<double>;
  using Inverse = ComplexForwardDftType;
  static Fits::Position<2> outShape(const Fits::Position<2>& shape);
  static fftw_plan initFftwPlan(Fits::Raster<InValue, 3>& in, Fits::Raster<OutValue, 3>& fourier);
};

using RealForwardDft = TransformPlan<RealForwardDftType, false, false>;
using RealBackwardDft = TransformPlan<RealBackwardDftType, false, false>;
using ComplexForwardDft = TransformPlan<ComplexForwardDftType, false, false>;
using ComplexBackwardDft = TransformPlan<ComplexBackwardDftType, false, false>;

} // namespace Fourier
} // namespace Euclid

#endif
