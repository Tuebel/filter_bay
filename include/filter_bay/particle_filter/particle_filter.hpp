#pragma once
#include <filter_bay/utility/uniform_random.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace filter_bay
{
/*!
Simple implementation of a particle filter which uses systematic resampling.
*/
template <typename StateType, typename InputType, typename ObservationType>
class ParticleFilter
{
public:
  using States = typename std::vector<StateType>;
  using Weights = typename std::vector<double>;
  using Likelihoods = typename std::vector<double>;
  /*! Predicts the state transition */
  using TransitionFunction = std::function<StateType(StateType, const InputType &)>;
  /*! Calculates the likelihood from an observation */
  using LikelihoodFunction = std::function<double(const StateType &state, const ObservationType &observation)>;
  /*! Calculates the likelihoods for a batch of states. Can optimize the
  calculating for limited resources, async calculations, etc. */
  using BatchLikelihoodFunction = std::function<Likelihoods(const States &states, const ObservationType &observation)>;

  /*!
  Create a particle with the given number of particles.
  */
  ParticleFilter(size_t particle_count)
      : weights(particle_count), states(particle_count)
  {
    this->particle_count = particle_count;
  }

  /*!
  Set the initial belief. Must have the size particle_count. Makes sure that the
  initial weights are distributed uniformally.
  */
  void initialize(States initial_states)
  {
    if (initial_states.size() == particle_count)
    {
      states = std::move(initial_states);
      double avg_weight = 1.0 / particle_count;
      for (double &current : weights)
      {
        current = avg_weight;
      }
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
  */
  void update(const ObservationType &z, const LikelihoodFunction &likelihood)
  {
    Likelihoods likelihoods(particle_count);
    for (size_t i = 0; i < particle_count; i++)
    {
      likelihoods[i] = likelihood(states[i], z);
    }
    update_by_likelihoods(likelihoods);
  }

  /*!
  Updates the particle weights by incorporating the observation as a batch.
  The update step is done in a SIR bootstrap filter fashion.
  Resampling is performed with the low variance resampling method.
  \param z the current observation
  \param likelihood the function to estimate the likelihood for a given state
  and observation 
  */
  void update_batch(const ObservationType &z,
                    const BatchLikelihoodFunction &batch_likelihood)
  {
    update_by_likelihoods(batch_likelihood(states, z));
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
  size_t get_particle_count() const
  {
    return particle_count;
  }

  /*! 
  Set the number of simulated particles. The particles are resized and it is
  advised to call initialize after setting the particle count.
  */
  void set_particle_count(size_t count)
  {
    particle_count = count;
    states.resize(particle_count);
    weights.resize(particle_count);
  }

private:
  // corrsponding states and weights
  Weights weights;
  States states;
  // parameters
  size_t particle_count;
  // maximum-a-posteriori state;
  StateType map_state;
  // sample from uniform distribution
  UniformRandom uniform_random;

  /*!
  Updates the weights, the MAP estimate by given likelihoods.
  Resamples the particles if the effective sample size is smaller than half
  of the particle count
  */
  void update_by_likelihoods(const Likelihoods &likelihoods)
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
    // resample if effective sample size is smaller than the half of particcle
    // count
    if (1.0 / square_sum < particle_count / 2.0)
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