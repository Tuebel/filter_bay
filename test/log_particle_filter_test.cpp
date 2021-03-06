#include <filter_bay/particle_filter/log_particle_filter.hpp>
#include <filter_bay/utility/normal_sampler.hpp>
#include <gtest/gtest.h>
#include <cmath>

// Simple 1D state input & observation
using MyFilter = filter_bay::LogParticleFilter<double, double, double>;
const double MEAN = 42;
const double VARIANCE = 1;
const double OBSERVATION = 42.5;

// For testability a simple transition without noise
double predict(const double &state)
{
  return state + MEAN;
}

// Probability in [0,1], decreasing with distance = exponential distribution
// of the square error. The logarithm of exp is just the squared diff
double log_likelihood(const double &state, const double &observation)
{
  double diff = observation - state;
  return -(diff * diff);
}

MyFilter create_filter()
{
  // Create filter
  return MyFilter(10);
}

TEST(LogParticleFilterTest, TestInitialization)
{
  auto filter = create_filter();
  auto states = filter.get_states();
  double avg_log_weight = log(1.0 / states.size());
  for (size_t i = 0; i < states.size(); i++)
  {
    states[i] = i;
  }
  // Test initializing
  filter.initialize(states);
  for (size_t i = 0; i < filter.get_particle_count(); i++)
  {
    ASSERT_DOUBLE_EQ(i, filter.get_states()[i]);
    ASSERT_DOUBLE_EQ(avg_log_weight, filter.get_log_weights()[i]);
  }
  for (size_t i = 0; i < states.size(); i++)
  {
    states[i] = 0;
  }
  // Again with 0 as value. The weights must be the avg.
  filter.initialize(states);
  for (size_t i = 0; i < filter.get_particle_count(); i++)
  {
    ASSERT_DOUBLE_EQ(0, filter.get_states()[i]);
    ASSERT_DOUBLE_EQ(avg_log_weight, filter.get_log_weights()[i]);
  }
}

TEST(LogParticleFilterTest, TestFilterStep)
{

  auto filter = create_filter();
  auto states = filter.get_states();
  for (size_t i = 0; i < states.size(); i++)
  {
    states[i] = i;
  }
  filter.initialize(states);
  // Test prediction
  filter.predict(predict);
  double avg_log_weight = log(1.0 / states.size());
  for (size_t i = 0; i < filter.get_particle_count(); i++)
  {
    ASSERT_DOUBLE_EQ(i + MEAN, filter.get_states()[i]);
    ASSERT_DOUBLE_EQ(avg_log_weight, filter.get_log_weights()[i]);
  }
  // Test filter step using a gaussian
  filter_bay::NormalSampler normal_sampler;
  Eigen::Matrix<double, 1, 1> mean;
  mean << MEAN;
  Eigen::Matrix<double, 1, 1> variance;
  variance << VARIANCE;
  for (size_t i = 0; i < states.size(); i++)
  {
    states[i] = normal_sampler.sample_robust(mean, variance)(0);
  }
  filter.initialize(states);
  for (size_t i = 0; i < filter.get_particle_count(); i++)
  {
    std::cout << filter.get_states()[i] << "\t"
              << filter.get_log_weights()[i] << "\n";
  }
  for (int n = 0; n < 5; n++)
  {
    std::cout << "updat no " << n << "\n";
    filter.update(OBSERVATION, log_likelihood);
    auto updated_states = filter.get_states();
    auto updated_weights = filter.get_log_weights();
    for (size_t i = 0; i < filter.get_particle_count(); i++)
    {
      std::cout << updated_states[i] << "\t"
                << updated_weights[i] << "\n";
    }
    std::cout << "MAP " << filter.get_map_state() << "\n";
  }
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
