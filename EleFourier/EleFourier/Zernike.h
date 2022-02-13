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

#ifndef _ELEFOURIER_ZERNIKE_H
#define _ELEFOURIER_ZERNIKE_H

#include <algorithm> // copy_n
#include <array>
#include <limits> // NaN
#include <utility> // index_sequence

namespace Euclid {
namespace Fourier {

struct LocalZernikeSeries {

  static constexpr long JMax = 20;

  double x1, x2, x3, x4, x5, x6, x7, x8, x9;
  double y1, y2, y3, y4, y5, y6, y7, y8, y9;
  double nan;

  LocalZernikeSeries(double u, double v, double radius, double blank = std::numeric_limits<double>::quiet_NaN()) :
      x1((u - radius) / radius), // scaling to [-1, 1]
      x2(x1 * x1), x3(x1 * x2), x4(x1 * x3), x5(x1 * x4), x6(x1 * x5), x7(x1 * x6), x8(x1 * x7), x9(x1 * x8),
      y1((v - radius) / radius), // scaling to [-1, 1]
      y2(y1 * y1), y3(y1 * y2), y4(y1 * y3), y5(y1 * y4), y6(y1 * y5), y7(y1 * y6), y8(y1 * y7), y9(y1 * y8),
      nan(blank) {}

  template <long N, long M>
  double orders() const {
    return ansi<(N * (N + 2) + M) / 2>();
  }

  template <long J>
  double ansi() const;

  template <std::size_t... Js>
  std::array<double, sizeof...(Js)> ansiList(std::index_sequence<Js...>) {
    return {ansi<Js>()...};
  }

  template <std::size_t Size>
  std::array<double, Size> ansiSeq() {
    return ansiList(std::make_index_sequence<Size>());
  }

  void ansiSeq(double* dst, long count = JMax + 1) {
    const auto array = ansiSeq<JMax + 1>();
    std::copy_n(array.begin(), count, dst); // FIXME throw if count > JMax + 1
  }
};

#define DEF_ZERNIKE(J, expr) \
  template <> \
  double LocalZernikeSeries::ansi<J>() const { \
    if (x2 + y2 > 1) { \
      return nan; \
    } \
    return expr; \
  }

// https://jumar.lowell.edu/confluence/download/attachments/20546539/2011.JMOp.58.545L.pdf?version=2&modificationDate=1610084878000&api=v2

DEF_ZERNIKE(0, 1)
DEF_ZERNIKE(1, x1)
DEF_ZERNIKE(2, y1)
DEF_ZERNIKE(3, 2 * x1 * y1)
DEF_ZERNIKE(4, -1 + 2 * x2 + 2 * y2)
DEF_ZERNIKE(5, -x2 + y2)
DEF_ZERNIKE(6, -x3 + 3 * x1 * y2)
DEF_ZERNIKE(7, -2 * x1 + 3 * x3 + 3 * x1 * y2)
DEF_ZERNIKE(8, -2 * y1 + 3 * y3 + 3 * x2 * y1)
DEF_ZERNIKE(9, y3 - 3 * x2 * y1)
DEF_ZERNIKE(10, -4 * x3 * y1 + 4 * x1 * y3)
DEF_ZERNIKE(11, -6 * x1 * y1 + 8 * x3 * y1 + 8 * x1 * y3)
DEF_ZERNIKE(12, 1 - 6 * x2 - 6 * y2 + 6 * x4 + 12 * x2 * y2 + 6 * y4)
DEF_ZERNIKE(13, 3 * x2 - 3 * y2 - 4 * x4 + 4 * y4)
DEF_ZERNIKE(14, x4 - 6 * x2 * y2 + y4)
DEF_ZERNIKE(15, x5 - 10 * x3 * y2 + 5 * x1 * y4)
DEF_ZERNIKE(16, 4 * x3 - 12 * x1 * y2 - 5 * x5 + 10 * x3 * y2 + 15 * x1 * y4)
DEF_ZERNIKE(17, 3 * x1 - 12 * x3 - 12 * x1 * y2 + 10 * x5 + 20 * x3 * y2 + 10 * x1 * y4)
DEF_ZERNIKE(18, 3 * y1 - 12 * y3 - 12 * x2 * y1 + 10 * y5 + 20 * x2 * y3 - 15 * x4 * y1)
DEF_ZERNIKE(19, -4 * y3 + 12 * x2 * y1 + 5 * y5 - 10 * x2 * y3 - 15 * x4 * y1)
DEF_ZERNIKE(20, y5 - 10 * x2 * y3 + 5 * x4 * y1)

} // namespace Fourier
} // namespace Euclid

#endif // _ELEFOURIER_ZERNIKE_H
