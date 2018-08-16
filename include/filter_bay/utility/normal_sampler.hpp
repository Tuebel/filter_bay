#pragma once
#include <functional>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace filter_bay
{
/*!
Simplifies drawing from a multivariant normal distribution.
*/
class NormalSampler
{
  // Drawing from multivariate normal distribution:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
  // Sample by projecting standard normal distributed values with L, where
  // L L^T = covarianc
public:
  // typedef Eigen::Matrix<double, dim, dim> MatrixDim;
  // typedef Eigen::Matrix<double, dim, 1> VectorDim;

  /*!
  Create the sampler using a random_device to create the seed.
  */
  NormalSampler()
  {
    std::random_device random_device;
    uniform_generator = std::mt19937(random_device());
  }

  /*!
  Create the sampler via a seed. For testing it should always be the same seed.
  In real world use the random_device class to generate the seed.
  */
  NormalSampler(unsigned int seed) : uniform_generator(seed) {}

  /*!
  Samples a random vector from a normal distribution using a robust 
  decomposition algorithm.
  \param mean the mean of the distribution
  \param covariance the covariance of the distribution
  */
  template <typename scalar, int dim>
  auto sample_robust(const Eigen::Matrix<scalar, dim, 1> &mean,
                     const Eigen::Matrix<scalar, dim, dim> &covariance)
      -> Eigen::Matrix<scalar, dim, 1>
  {
    // Use LDLT decomposition which works with semi definite covariances:
    auto ldlt = covariance.ldlt();
    // L = P^T L sqrt(D), where D is a diagonal matrix.
    // https://stats.stackexchange.com/questions/48749/how-to-sample-from-a-multivariate-normal-given-the-pt-ldlt-p-decomposition-o
    auto P_T = ldlt.transpositionsP().transpose();
    auto L = ldlt.matrixL().toDenseMatrix();
    // Root for diagonal matrix can be calculated element wise
    auto sqrt_D = ldlt.vectorD().cwiseSqrt().asDiagonal();
    auto decomposed = P_T * L * sqrt_D;
    return sample<scalar, dim>(mean, decomposed);
  }

  /*!
  Samples a random vector from a normal distribution using the Cholesky
  decomposition. It might fail, if the covariance is not positive definite.
  \param mean the mean of the distribution
  \param covariance the covariance of the distribution
  */
  template <typename scalar, int dim>
  auto sample_cholesky(const Eigen::Matrix<scalar, dim, 1> &mean,
                       const Eigen::Matrix<scalar, dim, dim> &covariance)
      -> Eigen::Matrix<scalar, dim, 1>
  {
    auto decomposed = covariance.llt().matrixL();
    return sample<scalar, dim>(mean, decomposed);
  }

  /*!
  Draws random values from a normal distribution parametrized by the mean and
  standard deviation (NOT variance as in the covariance matrix version).
  */
  template <typename scalar>
  scalar draw_normal(scalar mean, scalar standard_deviation)
  {
    std::normal_distribution<scalar> normal_dist;
    return normal_dist(uniform_generator) * standard_deviation + mean;
  }

private:
  // Algorithm that provides uniform distributed pseudo-random numbers.
  // Real randomness is imposed via a random seed.
  std::mt19937 uniform_generator;

  /*!
  Samples with the given mean and decomposed covariance matrix.
  */
  template <typename scalar, int dim>
  auto sample(const Eigen::Matrix<scalar, dim, 1> &mean,
              const Eigen::Matrix<scalar, dim, dim> &decomposed)
      -> Eigen::Matrix<scalar, dim, 1>
  {
    auto norm_dist = create_normal_dist_vector<scalar, dim>();
    return mean + decomposed * norm_dist;
  }

  /*!
  Creates a vector with all elements drawn randlomly from a standard normal
  distribution.
  */
  template <typename scalar, int dim>
  Eigen::Matrix<scalar, dim, 1> create_normal_dist_vector()
  {
    std::normal_distribution<scalar> normal_dist;
    Eigen::Matrix<scalar, dim, 1> result;
    for (int i = 0; i < dim; i++)
    {
      result(i) = normal_dist(uniform_generator);
    }
    return result;
  }
};
} // namespace filter_bay