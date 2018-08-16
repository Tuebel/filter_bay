#include <array>
#include <filter_bay/utility/normal_sampler.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix2d;
using Eigen::Vector2d;
// The more samples are used the better the precision.
// So for less samples increse EPS
const int SAMPLES_COUNT = 5000;
const double EPS = 0.1;
using SamplerType = filter_bay::NormalSampler;
// Use the same seed for repeatability of the test
const unsigned int SEED = 85013;

TEST(NormalSamplerTest, TestRobust)
{
  // Bootstrap the sampler with const seed for
  SamplerType sampler(SEED);
  Vector2d mean;
  mean << 42.0, 66.6;
  Matrix2d covariance;
  covariance << 6.6, 1.3, 1.3, 4.2;
  // Store samples
  std::array<Vector2d, SAMPLES_COUNT> samples;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    samples[i] = sampler.sample_robust(mean, covariance);
  }
  // Calculate mean of the samples
  Vector2d new_mean;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    new_mean(0) += samples[i](0);
    new_mean(1) += samples[i](1);
  }
  new_mean = new_mean / SAMPLES_COUNT;
  // Calculate covariance of the samples
  Matrix2d new_covariance;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    for (int r = 0; r < 2; r++)
    {
      for (int c = 0; c < 2; c++)
      {
        new_covariance(r, c) += (samples[i](r) - mean(r)) *
                                (samples[i](c) - mean(c));
      }
    }
  }
  new_covariance = new_covariance / SAMPLES_COUNT;
  // Test results
  ASSERT_TRUE(mean.isApprox(new_mean, EPS)) << mean << "\n\n"
                                            << new_mean << "\n";
  ASSERT_TRUE(covariance.isApprox(new_covariance, EPS)) << covariance << "\n\n"
                                                        << new_covariance << "\n";
}

TEST(NormalSamplerTest, TestCholesky)
{
  // Bootstrap the sampler
  SamplerType sampler(SEED);
  Vector2d mean;
  mean << 42.0, 66.6;
  Matrix2d covariance;
  covariance << 6.6, 1.3, 1.3, 4.2;
  // Store samples

  std::array<Vector2d, SAMPLES_COUNT> samples;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    samples[i] = sampler.sample_cholesky(mean, covariance);
  }
  // Calculate mean of the samples
  Vector2d new_mean;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    new_mean(0) += samples[i](0);
    new_mean(1) += samples[i](1);
  }
  new_mean = new_mean / SAMPLES_COUNT;
  // Calculate covariance of the samples
  Matrix2d new_covariance;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    for (int r = 0; r < 2; r++)
    {
      for (int c = 0; c < 2; c++)
      {
        new_covariance(r, c) += (samples[i](r) - mean(r)) *
                                (samples[i](c) - mean(c));
      }
    }
  }
  new_covariance = new_covariance / SAMPLES_COUNT;
  // Test results
  ASSERT_TRUE(mean.isApprox(new_mean, EPS)) << mean << "\n\n"
                                            << new_mean << "\n";
  ASSERT_TRUE(covariance.isApprox(new_covariance, EPS)) << covariance << "\n\n"
                                                        << new_covariance << "\n";
}

TEST(NormalSamplerTest, TestScalar)
{
  // Bootstrap the sampler
  SamplerType sampler(SEED);
  const double MEAN = 42;
  const double STANDARD_DEVIATION = 2.5;
  const double VARIANCE = STANDARD_DEVIATION * STANDARD_DEVIATION;
  std::array<double, SAMPLES_COUNT> samples;
  for (auto &current : samples)
  {
    current = sampler.draw_normal(MEAN, STANDARD_DEVIATION);
  }
  double new_mean = 0;
  double new_variance = 0;
  for (auto current : samples)
  {
    new_mean += current / SAMPLES_COUNT;
    new_variance += (current - MEAN) * (current - MEAN) / SAMPLES_COUNT;
  }
  ASSERT_NEAR(MEAN, new_mean, EPS);
  ASSERT_NEAR(VARIANCE, new_variance, EPS);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
