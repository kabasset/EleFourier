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

#include "EleFits/MefFile.h"
#include "EleFitsUtils/ProgramOptions.h"
#include "EleFitsValidation/Chronometer.h"
#include "EleFourier/Dft.h"
#include "ElementsKernel/ProgramHeaders.h"

#include <functional>
#include <map>
#include <string>

using boost::program_options::value;

using namespace Euclid;
using namespace Fourier;

class EleFourierTutorial : public Elements::Program {

public:
  std::pair<OptionsDescription, PositionalOptionsDescription> defineProgramArguments() override {
    Fits::ProgramOptions options("Convolve via DFT.");
    options.positional("filename", value<std::string>()->default_value("/tmp/data.fits"), "File name");
    return options.asPair();
  }

  Elements::ExitCode mainMethod(std::map<std::string, VariableValue>& args) override {

    Elements::Logging logger = Elements::Logging::getLogger("EleFourierTutorial");
    const auto filename = args["filename"].as<std::string>();
    Fits::Test::Chronometer<std::chrono::milliseconds> chrono;

    // Open Fits file
    logger.info() << "Opening file: " << filename;
    chrono.start();
    Fits::MefFile f(filename, Fits::FileMode::Edit); // Open file
    const auto& primary = f.primary().raster();
    const auto shape = primary.readShape();
    const auto count = f.hduCount() - 1;
    chrono.stop();
    logger.info() << "  Image size: " << shape[0] << "x" << shape[1];
    logger.info() << "  Image count: " << count;
    logger.info() << "  Done in " << chrono.last().count() << "ms";

    // Initialize DFT plans
    logger.info() << "Initializing plans...";
    chrono.start();
    RealDft filter(shape);
    RealDft images(shape, count);
    chrono.stop();
    logger.info() << "  Done in " << chrono.last().count() << "ms";

    // Read filter and images
    logger.info() << "Reading filter and images...";
    chrono.start();
    auto tempImage = filter.image();
    primary.readTo(tempImage);
    for (long i = 0; i < count; ++i) {
      tempImage = images.image(i);
      f.access<Fits::ImageHdu>(i + 1).raster().readTo(tempImage); // FIXME access<ImageRaster>()
    }
    chrono.stop();
    logger.info() << "  Done in " << chrono.last().count() << "ms";

    // Fourier transform
    logger.info() << "Applying DFT to filter...";
    chrono.start();
    filter.transform();
    chrono.stop();
    logger.info() << "  Done in " << chrono.last().count() << "ms";
    logger.info() << "Applying DFT to images...";
    chrono.start();
    images.transform();
    chrono.stop();
    logger.info() << "  Done in " << chrono.last().count() << "ms";

    // Perform convolution (frequency-domain multiplication into dft0 and dft1)
    logger.info() << "Convolving...";
    chrono.start();
    auto kernel = filter.coefs();
    for (long i = 0; i < count; ++i) {
      auto dft = images.coefs(i);
      auto begin = dft.data();
      auto end = begin + dft.size();
      std::transform(begin, end, kernel.data(), begin, std::multiplies<>()); // FIXME dft.begin()/end()...
    }
    chrono.stop();
    logger.info() << "  Done in " << chrono.last().count() << "ms";

    // Inverse Fourier transform (in-place, overwrites space-domain data)
    logger.info() << "Applying inverse DFTs...";
    chrono.start();
    images.inverse();
    chrono.stop();
    logger.info() << "  Done in " << chrono.last().count() << "ms";

    logger.info() << "Writing images...";
    chrono.start();
    for (long i = 0; i < count; ++i) {
      f.access<Fits::ImageHdu>(i + 1).raster().write(images.image(i));
    }
    chrono.stop();
    logger.info() << "  Done in " << chrono.last().count() << "ms";

    return Elements::ExitCode::OK;
  }
};

MAIN_FOR(EleFourierTutorial)
