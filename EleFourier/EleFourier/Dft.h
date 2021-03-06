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

#include "EleFourier/DftPlan.h"

namespace Euclid {
namespace Fourier {

/**
 * @brief Real DFT plan.
 */
using RealDft = DftPlan<RealDftType>;

/**
 * @brief Complex DFT plan.
 */
using ComplexDft = DftPlan<ComplexDftType>;

/**
 * @brief Complex DFT plan with Hermitian symmetry.
 */
using HermitianComplexDft = DftPlan<HermitianComplexDftType>;

} // namespace Fourier
} // namespace Euclid

#endif
