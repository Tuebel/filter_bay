#include <filter_bay/particle_filter/particle_filter.hpp>
#include <filter_bay/utility/normal_sampler.hpp>
#include <gtest/gtest.h>
#include <cmath>

// Simple 1D state input & observation
using MyFilter = filter_bay::ParticleFilter<5, double, double, double>;

// For testability a simple transition without noise
double predict(const double &state, const double &input)
{
  return state + input;
}

// Probability in [0,1], decreasing with distance = exponential distribution
// of the square error
double likelihood(const double &state, const double &observation)
{
  double diff = observation - state;
  return exp(-(diff * diff));
}

MyFilter create_filter()
{
  // Create model
  MyFilter::Model model;
  model.predict = predict;
  model.likelihood = likelihood;
  // Create filter
  return MyFilter(std::move(model));
}

TEST(ParticleFilterTest, TestInitialization)
{
  auto filter = create_filter();
  MyFilter::Belief belief;
  double avg_weight = (double)1.0 / belief.size();
  for (size_t i = 0; i < belief.size(); i++)
  {
    belief[i].state = i;
  }
  // Test initializing
  filter.initialize(belief);
  for (size_t i = 0; i < filter.get_belief().size(); i++)
  {
    ASSERT_DOUBLE_EQ(i, filter.get_belief()[i].state);
    ASSERT_DOUBLE_EQ(avg_weight, filter.get_belief()[i].weight);
  }
  for (size_t i = 0; i < belief.size(); i++)
  {
    belief[i].state = 0;
    belief[i].weight = 0;
  }
  // Again with 0 as value. The weights must be the avg.
  filter.initialize(belief);
  for (size_t i = 0; i < filter.get_belief().size(); i++)
  {
    ASSERT_DOUBLE_EQ(0, filter.get_belief()[i].state);
    ASSERT_DOUBLE_EQ(avg_weight, filter.get_belief()[i].weight);
  }
}

TEST(ParticleFilterTest, TestFilterStep)
{

  auto filter = create_filter();
  MyFilter::Belief belief;
  for (size_t i = 0; i < belief.size(); i++)
  {
    belief[i].state = i;
  }
  filter.initialize(belief);
  // Test prediction
  filter.predict(42);
  double avg_weight = (double)1.0 / belief.size();
  for (size_t i = 0; i < filter.get_belief().size(); i++)
  {
    ASSERT_DOUBLE_EQ(i + 42, filter.get_belief()[i].state);
    ASSERT_DOUBLE_EQ(avg_weight, filter.get_belief()[i].weight);
  }
  // Test filter step using a gaussian
  filter_bay::NormalSampler<1> normal_sampler(42);
  Eigen::Matrix<double, 1, 1> mean;
  mean << 42;
  Eigen::Matrix<double, 1, 1> variance;
  variance << 1;
  for (size_t i = 0; i < belief.size(); i++)
  {
    belief[i].state = normal_sampler.sample_robust(mean, variance)(0);
  }
  filter.initialize(belief);
  for (size_t i = 0; i < filter.get_belief().size(); i++)
  {
    std::cout << filter.get_belief()[i].state << "\t"
              << filter.get_belief()[i].weight << "\n";
  }
  for (int n = 0; n < 5; n++)
  {
    filter.update(43);
    for (size_t i = 0; i < filter.get_belief().size(); i++)
    {
      std::cout << filter.get_belief()[i].state << "\t"
                << filter.get_belief()[i].weight << "\n";
    }
  }
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
