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

#include "EleFitsUtils/ProgramOptions.h"
#include "EleFitsValidation/Chronometer.h"
#include "EleFourier/Dft.h"
#include "ElementsKernel/ProgramHeaders.h"

#include <chrono>
#include <functional>
#include <map>
#include <omp.h>
#include <random>
#include <string>

using boost::program_options::value;

using namespace Euclid;
using namespace Fourier;

/**
 * @brief The set of DFT plans of each parallel branch.
 * @details
 * There are 3 plans, connected together without copies as follows:
 * - A real DFT `r`;
 * - A complex DFT `c` whose input buffer is `r`'s output buffer;
 * - The inverse of `c` whose input (resp. output) buffer is `c`'s output (resp input) buffer.
 * 
 * Additionally, a chronometer is provided to measure branch-wise transform times.
 * @warning
 * Plans should be initialized in a single thread, as this writes in global variables.
 */
struct BranchDfts {

  /** Real DFT. */
  RealDft r;

  /** Complex DFT piped to the real DFT's output. */
  DftPlan<ComplexDftType, true, false> c; // FIXME ugly

  /** In-place inverse complex DFT. */
  decltype(c.inverse()) i; // FIXME ugly

  /** Branch-wise chronometer. */
  Fits::Validation::Chronometer<std::chrono::milliseconds> chrono;

  /** Constructor. */
  BranchDfts(const Fits::Position<2>& shape) :
      r(shape), c(r.compose<ComplexDft>(r.outShape())), // FIXME HermitianComplexDft
      i(c.inverse()), chrono() {}
};

class EleFourierParallelizationTutorial : public Elements::Program {

public:
  std::pair<OptionsDescription, PositionalOptionsDescription> defineProgramArguments() override {
    Fits::ProgramOptions options("Demonstrates multi-thread usage of the library.");
    options.named("branches", value<long>()->default_value(1), "Number of branches");
    options.named("inputs", value<long>()->default_value(10), "Number of inputs per branch");
    options.named("side", value<long>()->default_value(1024), "Input width and height");
    return options.asPair();
  }

  Elements::ExitCode mainMethod(std::map<std::string, VariableValue>& args) override {

    Elements::Logging logger = Elements::Logging::getLogger("EleFourierParallelizationTutorial");

    const auto branches = args["branches"].as<long>();
    const auto inputs = args["inputs"].as<long>();
    const auto side = args["side"].as<long>();
    const Fits::Position<2> shape {side, side};
    Fits::Validation::Chronometer<std::chrono::milliseconds> programChrono;

    // Initialize DFT plans sequentially - NOT THREAD SAFE!
    std::vector<BranchDfts> dfts;
    dfts.reserve(branches);
    for (long b = 0; b < branches; ++b) {
      logger.info() << "Initializing branch #" << b << "'s plans...";
      programChrono.start();
      dfts.emplace_back(shape);
      programChrono.stop();
      logger.info() << "  Done in " << programChrono.last().count() << " ms.";
    }

    // Use plans in parallel
    logger.info() << "Executing plans in parallel...";
    logger.info() << "  Number of branches: " << branches;
    logger.info() << "  Available number of threads: " << omp_get_max_threads();
    omp_set_num_threads(branches);
    programChrono.start();
#pragma omp parallel for
    for (long b = 0; b < branches; ++b) {

      // Shortcuts
      auto& real = dfts[b].r;
      auto& complex = dfts[b].c;
      auto& inverse = dfts[b].i;
      auto& chrono = dfts[b].chrono;
      auto inBegin = real.inBuffer().begin();
      auto inEnd = real.inBuffer().end();

      // Random number generator
      std::default_random_engine engine;
      std::uniform_real_distribution<double> distribution(0., 1.);

      // Loop over branch's inputs
      for (long i = 0; i < inputs; ++i) {

        // Generate input
        std::generate(inBegin, inEnd, [&]() {
          return distribution(engine);
        });

        // Perform and time transforms
        chrono.start();
        real.transform();
        chrono.stop();
        chrono.start();
        complex.transform();
        chrono.stop();
        chrono.start();
        inverse.transform();
        chrono.stop();
        chrono.stop();
      }
    }
    programChrono.stop();
    logger.info() << "  Done in " << programChrono.last().count() << " ms.";

    // Print aggregate times
    logger.info() << "Branch-wise timings:";
    for (long b = 0; b < branches; ++b) {
      logger.info() << "  Branch #" << b << " transforms lasted " << dfts[b].chrono.mean() << " ms in average.";
    }

    return Elements::ExitCode::OK;
  }
};

MAIN_FOR(EleFourierParallelizationTutorial)
