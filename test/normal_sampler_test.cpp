#include <array>
#include <filter_bay/utility/normal_sampler.hpp>
#include <gtest/gtest.h>

// The more samples are used the better the precision.
// So for less samples increse EPS
const int SAMPLES_COUNT = 1000;
const double EPS = 0.1;
using SamplerType = filter_bay::NormalSampler<2>;
// Use the same seed for repeatability of the test
const unsigned int SEED = 85013;

TEST(NormalSamplerTest, TestRobust)
{
  // Bootstrap the sampler with const seed for
  SamplerType sampler(SEED);
  SamplerType::VectorDim mean;
  mean << 42.0, 66.6;
  SamplerType::MatrixDim covariance;
  covariance << 6.6, 1.3, 1.3, 4.2;
  // Store samples

  std::array<SamplerType::VectorDim, SAMPLES_COUNT> samples;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    samples[i] = sampler.sample_robust(mean, covariance);
  }
  // Calculate mean of the samples
  SamplerType::VectorDim new_mean;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    new_mean(0) += samples[i](0);
    new_mean(1) += samples[i](1);
  }
  new_mean = new_mean / SAMPLES_COUNT;
  // Calculate covariance of the samples
  SamplerType::MatrixDim new_covariance;
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
  SamplerType::VectorDim mean;
  mean << 42.0, 66.6;
  SamplerType::MatrixDim covariance;
  covariance << 6.6, 1.3, 1.3, 4.2;
  // Store samples

  std::array<SamplerType::VectorDim, SAMPLES_COUNT> samples;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    samples[i] = sampler.sample_cholesky(mean, covariance);
  }
  // Calculate mean of the samples
  SamplerType::VectorDim new_mean;
  for (int i = 0; i < SAMPLES_COUNT; i++)
  {
    new_mean(0) += samples[i](0);
    new_mean(1) += samples[i](1);
  }
  new_mean = new_mean / SAMPLES_COUNT;
  // Calculate covariance of the samples
  SamplerType::MatrixDim new_covariance;
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

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
