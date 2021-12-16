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
