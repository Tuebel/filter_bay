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
template <int dim>
class NormalSampler
{
  // Drawing from multivariate normal distribution:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
  // Sample by projecting standard normal distributed values with L, where
  // L L^T = covarianc
public:
  typedef Eigen::Matrix<double, dim, dim> MatrixDim;
  typedef Eigen::Matrix<double, dim, 1> VectorDim;

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
  VectorDim sample_robust(const VectorDim &mean, const MatrixDim &covariance)
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
    return sample(mean, decomposed);
  }

  /*!
  Samples a random vector from a normal distribution using the Cholesky
  decomposition. It might fail, if the covariance is not positive definite.
  \param mean the mean of the distribution
  \param covariance the covariance of the distribution
  */
  VectorDim sample_cholesky(const VectorDim &mean, const MatrixDim &covariance)
  {
    auto decomposed = covariance.llt().matrixL();
    return sample(mean, decomposed);
  }

private:
  // Algorithm that provides uniform distributed pseudo-random numbers.
  // Real randomness is imposed via a random seed.
  std::mt19937 uniform_generator;

  /*!
  Samples with the given mean and decomposed covariance matrix.
  */
  VectorDim sample(const VectorDim &mean, const MatrixDim &decomposed)
  {
    auto normal_vec = create_normal_dist_vector();
    return mean + decomposed * normal_vec;
  }

  /*!
  Creates a vector with all elements drawn randlomly from a standard normal
  distribution.
  */
  Eigen::Matrix<double, dim, 1> create_normal_dist_vector()
  {
    std::normal_distribution<> normal_dist;
    Eigen::Matrix<double, dim, 1> normal_vec;
    for (int i = 0; i < dim; i++)
    {
      normal_vec(i, 0) = normal_dist(uniform_generator);
    }
    return normal_vec;
  }
};
} // namespace filter_bay