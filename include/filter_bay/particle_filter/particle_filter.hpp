#pragma once
#include <filter_bay/utility/uniform_random.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>

namespace filter_bay
{
/*!
Simple implementation of a particle filter which uses a 
*/
template <size_t particle_count, typename StateType, typename InputType,
          typename ObservationType>
class ParticleFilter
{
public:
  using States = typename std::array<StateType, particle_count>;
  using Weights = typename std::array<double, particle_count>;
  /*! Predicts the state transition */
  using PredictFunction = std::function<StateType(const StateType &, const InputType &)>;
  /*! Calculates the likelihood from an observation */
  using LikelihoodFunction = std::function<double(const StateType &state, const ObservationType &observation)>;

  ParticleFilter(PredictFunction predict_function,
                 LikelihoodFunction likelihood_function)
      : predict_fn(predict_function), likelihood_fn(likelihood_function)
  {
  }

  /*!
  Set the initial belief. Makes sure that the initial weights are distributed
  uniformally.
  */
  void initialize(States initial_states)
  {
    states = std::move(initial_states);
    double avg_weight = 1.0 / particle_count;
    for (double &current : weights)
    {
      current = avg_weight;
    }
  }

  /*!
  Calculates the prediction for every particle without updating the weights.
  \param u the control input
  */
  void predict(const InputType &u)
  {
    for (StateType &current : states)
    {
      current = predict_fn(current, u);
    }
  }

  /*!
  Updates the particle weights by incorporating the observation.
  The update step is done in a SIR bootstrap filter fashion.
  Resampling is performed with the low variance resampling method.
  \param z the current observation
  \param resample_threshold threshold of the effective sample size(ESS). 
  Resampling is performed if ESS < resample_threshold. N/2 is a typical value. 
  */
  void update(const ObservationType &z,
              size_t resample_threshold = particle_count / 2)
  {
    double weight_sum = 0;
    // weights as posterior of observation
    for (size_t i = 0; i < particle_count; i++)
    {
      // Using the prior as proposal so the weight recursion is simply:
      weights[i] *= likelihood_fn(states[i], z);
      weight_sum += weights[i];
    }
    // Normalize weights
    double normalize_const = 1 / weight_sum;
    double square_sum = 0;
    for (auto &current : weights)
    {
      current *= normalize_const;
      square_sum += current * current;
    }
    // Perform resampling? Calc effective sample size
    if (1 / square_sum < resample_threshold)
    {
      resample_systematic();
    }
  }

  /*!
  Sampling systematically and thus keeping the sample variance lower than pure
  random sampling. It is also faster O(m) instead of O(m logm)
  */
  void resample_systematic()
  {
    // Make copy of old belief
    States old_states = states;
    Weights old_weights = weights;
    // Start at random value within average weight
    double cumulative = old_weights[0];
    double avg_weight = 1.0 / particle_count;
    double start_weight = uniform_random.generate(0, avg_weight);
    // indices: o in old, n in new
    size_t o = 0;
    for (size_t n = 0; n < particle_count; n++)
    {
      double U = start_weight + n * avg_weight;
      while (U > cumulative)
      {
        o++;
        cumulative += old_weights[o];
      }
      // Resample this particle and reset the weight
      states[n] = old_states[o];
      weights[n] = avg_weight;
    }
  }

  /*!
  Returns the maximum a posteriori state (largest weight).
  */
  StateType get_map_state() const
  {
    auto index = std::distance(weights,
                               std::max_element(weights.begin(),
                                                weights.end()));
    return states[index];
  }

  States get_states() const
  {
    return states;
  }

  Weights get_weights() const
  {
    return weights;
  }

  size_t get_particle_count()
  {
    return particle_count;
  }

private:
  // corrsponding states and weights
  Weights weights;
  States states;
  // state transition and measurement functions
  PredictFunction predict_fn;
  LikelihoodFunction likelihood_fn;
  // sample from uniform distribution
  UniformRandom uniform_random;
}; // namespace filter_bay
} // namespace filter_bay