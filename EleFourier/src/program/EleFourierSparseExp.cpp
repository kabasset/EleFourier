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

#include "EleFits/SifFile.h"
#include "EleFitsData/Raster.h"
#include "EleFitsUtils/ProgramOptions.h"
#include "EleFitsValidation/Chronometer.h"
#include "ElementsKernel/ProgramHeaders.h"

#include <map>
#include <string>

using boost::program_options::value;

using namespace Euclid;

Fits::VecRaster<double> generatePupil(long maskSide, long pupilRadius) {
  Fits::VecRaster<double> pupil({maskSide, maskSide});
  for (const auto& p : pupil.domain()) {
    const auto x = double(p[0] - maskSide / 2) / pupilRadius;
    const auto y = double(p[1] - maskSide / 2) / pupilRadius;
    if (std::abs(x * x + y * y) < 1) {
      pupil[p] = 1.;
    }
  }
  return pupil;
}

template <typename TRaster>
void completeExp(TRaster& raster) {
  raster.apply([](auto& v) {
    return std::exp(v);
  });
}

template <typename TRaster>
void sparseExp(TRaster& raster) {
  raster.apply([](auto& v) {
    if (v != 0.) {
      return std::exp(v);
    } else {
      return v;
    }
  });
}

template <typename TRaster>
void saveSif(const TRaster& raster, const std::string& filename) {
  Fits::SifFile f(filename, Fits::FileMode::Overwrite);
  f.writeRaster(raster);
}

class EleFourierSparseExp : public Elements::Program {

public:
  std::pair<OptionsDescription, PositionalOptionsDescription> defineProgramArguments() override {
    Fits::ProgramOptions options("Compare complete and sparse exponentiations.");
    options.named("side", value<long>()->default_value(1024), "Pupil mask side");
    options.named("radius", value<long>()->default_value(512), "Pupil radius");
    // FIXME options.named("radius", 1024L, "Pupil radius");
    options.named("pupil", value<std::string>()->default_value("/tmp/pupil.fits"), "Pupil mask input or output file");
    // FIXME options.named("pupil", "/tmp/pupil.fits"s, "Pupil mask input or output file");
    options.flag("sparse", "Compute exp only where mask is not null");
    return options.asPair();
  }

  ExitCode mainMethod(std::map<std::string, VariableValue>& args) override {
    Logging logger = Logging::getLogger("EleFourierSparseExp");

    const auto maskSide = args["side"].as<long>();
    const auto pupilRadius = args["radius"].as<long>();
    const auto pupilFilename = args["pupil"].as<std::string>();
    const auto sparse = args["sparse"].as<bool>();
    // FIXME load if exists, generate and save otherwise
    auto pupil = generatePupil(maskSide, pupilRadius);
    if (sparse) {
      sparseExp(pupil);
    } else {
      completeExp(pupil);
    }
    saveSif(pupil, pupilFilename);

    return ExitCode::OK;
  }
};

MAIN_FOR(EleFourierSparseExp)
