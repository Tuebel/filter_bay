#pragma once
#include <filter_bay/utility/uniform_random.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>

namespace filter_bay
{
/*!
Simple implementation of a particle filter which uses systematic resampling.
*/
template <size_t particle_count, typename StateType, typename InputType,
          typename ObservationType>
class ParticleFilter
{
public:
  using States = typename std::array<StateType, particle_count>;
  using Weights = typename std::array<double, particle_count>;
  using Likelihoods = typename std::array<double, particle_count>;
  /*! Predicts the state transition */
  using TransitionFunction = std::function<StateType(const StateType &, const InputType &)>;
  /*! Calculates the likelihood from an observation */
  using LikelihoodFunction = std::function<double(const StateType &state, const ObservationType &observation)>;
  /*! Calculates the likelihoods for a batch of states. Can optimize the
  calculating for limited resources, async calculations, etc. */
  using BatchLikelihoodFunction = std::function<Likelihoods(const States &states, const ObservationType &observation)>;

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
  \param transition the function for an arbitrary state transition.
  */
  void predict(const InputType &u, const TransitionFunction &transition)
  {
    for (StateType &current : states)
    {
      current = transition(current, u);
    }
  }

  /*!
  Updates the particle weights by incorporating the observation.
  The update step is done in a SIR bootstrap filter fashion.
  Resampling is performed with the low variance resampling method.
  \param z the current observation
  \param likelihood the function to estimate the likelihood for a given state
  and observation
  \param resample_threshold threshold of the effective sample size(ESS). 
  Resampling is performed if ESS < resample_threshold. N/2 is a typical value. 
  */
  void update(const ObservationType &z, const LikelihoodFunction &likelihood,
              double resample_threshold = particle_count / 2.0)
  {
    Likelihoods likelihoods;
    for (size_t i = 0; i < particle_count; i++)
    {
      likelihoods[i] = likelihood(states[i], z);
    }
    update_by_likelihoods(likelihoods, resample_threshold);
  }

  /*!
  Updates the particle weights by incorporating the observation as a batch.
  The update step is done in a SIR bootstrap filter fashion.
  Resampling is performed with the low variance resampling method.
  \param z the current observation
  \param likelihood the function to estimate the likelihood for a given state
  and observation
  \param resample_threshold threshold of the effective sample size(ESS). 
  Resampling is performed if ESS < resample_threshold. N/2 is a typical value. 
  */
  void update_batch(const ObservationType &z,
                    const BatchLikelihoodFunction &batch_likelihood,
                    double resample_threshold = particle_count / 2.0)
  {
    update_by_likelihoods(batch_likelihood(states, z), resample_threshold);
  }

  /*!
  Returns the maximum-a-posteriori state from the last update step.
  */
  StateType get_map_state() const
  {
    return map_state;
  }

  /*! 
  Returns the state of the particles. Use get_weights to obtain the full belief.
  */
  States get_states() const
  {
    return states;
  }

  /*! 
  Returns the weights of the particles. Use get_states to obtain the full
  belief.
  Warning: after resampling the weights are worthless.
  */
  Weights get_weights() const
  {
    return weights;
  }

  /*! Returns the number of particles beeing simulated */
  size_t get_particle_count()
  {
    return particle_count;
  }

private:
  // corrsponding states and weights
  Weights weights;
  States states;
  // maximum-a-posteriori state;
  StateType map_state;
  // sample from uniform distribution
  UniformRandom uniform_random;

  /*!
  Updates the weights, the MAP estimate by given likelihoods.
  Resamples the particles if the effective sample size is smaller than the
  threshold
  */
  void update_by_likelihoods(const Likelihoods &likelihoods,
                             double resample_threshold)
  {
    double weight_sum = 0;
    double max_weight = -std::numeric_limits<double>::infinity();
    // weights as posterior of observation
    for (size_t i = 0; i < particle_count; i++)
    {
      // Using the prior as proposal so the weight recursion is simply:
      weights[i] *= likelihoods[i];
      // check for MAP here, after resampling the weights are all equal
      if (weights[i] > max_weight)
      {
        map_state = states[i];
        max_weight = weights[i];
      }
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
};
} // namespace filter_bay