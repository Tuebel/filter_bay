#pragma once
#include <filter_bay/particle_filter/particle_model.hpp>
#include <filter_bay/utility/uniform_random.hpp>

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
  using Model = typename filter_bay::ParticleModel<StateType, InputType, ObservationType>;

  ParticleFilter(Model particle_model)
      : model(std::move(particle_model))
  {
  }

  /*!
  Set the initial belief. Makes sure that the initial weights are distributed
  uniformally.
  */
  void initialize(Belief initial_belief)
  {
    belief = std::move(initial_belief);
    double avg_weight = (double)1.0 / belief.size();
    for (Sample &sample : belief)
    {
      sample.weight = avg_weight;
    }
  }

  /*!
  Calculates the prediction for every particle without updating the weights.
  \param u the control input
  */
  void predict(const InputType &u)
  {
    for (Sample &sample : belief)
    {
      sample.state = model.predict(sample.state, u);
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
    for (auto &sample : belief)
    {
      // Using the prior as proposal so the weight recursion is simply:
      sample.weight = sample.weight *
                      model.likelihood(sample.state, z);
      weight_sum += sample.weight;
    }
    // Normalize weights
    double normalize_const = 1 / weight_sum;
    double square_sum = 0;
    for (auto &sample : belief)
    {
      sample.weight = normalize_const * sample.weight;
      square_sum += sample.weight * sample.weight;
    }
    // Perform resampling?
    if (1 / square_sum < resample_threshold)
    {
      belief = resample_low_var(belief);
    }
  }

  /*!
  Sampling systematically and thus keeping the sample variance lower than pure
  random sampling. It is also faster O(m) instead of O(m logm)
  */
  Belief resample_low_var(const Belief &old_belief)
  {
    Belief new_belief;
    if (old_belief.size() == 0)
    {
      return new_belief;
    }
    double avg_weight = 1.0 / belief.size();
    double r = uniform_random.generate(0, avg_weight);
    // init with weight of first particle
    double c = old_belief[0].weight;
    size_t i = 0;
    double U = 0;
    for (size_t m = 0; m < old_belief.size(); m++)
    {
      // find the particle for the given U
      U = r + (m * avg_weight);
      while (U > c)
      {
        i++;
        c += old_belief[i].weight;
      }
      // Resample this particle and reset the weight
      new_belief[m] = old_belief[i];
      new_belief[m].weight = avg_weight;
    }
    return new_belief;
  }

  Belief get_belief() const
  {
    return belief;
  }

private:
  // corrsponding states and weights
  std::array<double, particle_count> log_weights;
  std::array<StateType, particle_count> states;
  // state transition and observation model
  Model model;
  // sample from uniform distribution
  UniformRandom uniform_random;
}; // namespace filter_bay
} // namespace filter_bay