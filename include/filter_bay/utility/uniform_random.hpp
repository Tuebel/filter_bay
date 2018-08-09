#pragma once
#include <random>

namespace filter_bay
{
/*!
Simplifies drawing from a uniform distribution by supplying a true random seed.
Uses mt19937 random number generator with this seed.
*/
class UniformRandom
{
public:
  UniformRandom()
  {
    std::random_device device;
    generator.seed(device());
  }

  /*!
  Generates a float random value \f$ x \f$ with \f$ lower<=x<upper \f$
  */
  double generate(double lower, double upper)
  {
    std::uniform_real_distribution<double> dist(lower, upper);
    return dist(generator);
  }

private:
  std::mt19937 generator;
};
} // namespace filter_bay