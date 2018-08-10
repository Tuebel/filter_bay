#pragma once
#include <filter_bay/particle_filter/particle_model.hpp>
#include <filter_bay/utility/log_arithmetics.h>
#include <filter_bay/utility/uniform_random.hpp>
#include <array>

namespace filter_bay
{
/*!
Particle filter which operates in logarithmic probability domain, see:

Christian Gentner, Siwei Zhang, and Thomas Jost, “Log-PF: Particle Filtering in
Logarithm Domain,” Journal of Electrical and Computer Engineering, vol. 2018, 
Article ID 5763461, 11 pages, 2018. https://doi.org/10.1155/2018/5763461.

I simplified the systematic resampling and use the log_exp_sum larger sums
instead of the iterative jacobian logarithms.
*/
template <size_t particle_count, typename StateType, typename InputType,
          typename ObservationType>
class LogParticleFilter
{
public:
  using Model = typename filter_bay::ParticleModel<StateType, InputType, ObservationType>;

  ParticleFilter(Model particle_model) : model(std::move(particle_model))
  {
  }

  /*!
  Set the initial state belief. Distributes the weights uniformly
  */
  void initialize(std::array<StateType, particle_count> state_belief)
  {
    states = std::move(state_belief);
    // log(1/belief_size) = log(1)-log(belief_size)=-log(belief_size)
    double log_avg = -log(belief.size());
    for (double &current : log_weight)
    {
      current = log_avg;
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
      current = model.predict(current, u);
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
    // weights as posterior of observation, prior is proposal density
    for (double &current : log_weights)
    {
      // log(weight*likelihood) = log(weight) + log_likelihood
      current += model.log_likelihood(sample.state, z);
    }
    // Normalize weights
    log_weights = normalized_logs(log_weights);
    // Perform resampling?
    if (ess_log(log_weights) < resample_threshold)
    {
      belief = resample_systematic(belief);
    }
  }

  /*!
  Sampling systematically and thus keeping the sample variance lower than pure
  random sampling. It is also faster O(m) instead of O(m logm)
  */
  Belief resample_systematic(
      const std::array<double, particle_count> &old_weights)
  {
    auto new_weights = old_weights;
    if (old_belief.size() == 0)
    {
      return new_weights;
    }
    double log_avg = -log(particle_count);
    // init with weight of first particle
    double cumulative = old_weights[0];
    double cumulative_min = uniform_random.generate(0, avg_weight);
    // o in old_weights, n in new_weigts
    size_t o = 0;
    for (size_t n = 0; n < old_belief.size(); n++)
    {
      // find the particle for the given U
      while (cumulative_min > c)
      {
        o++;
        cumulative = jacobi_logarithm(cumulative, old_weights[o]);
      }
      // Resample this particle and reset the weight
      new_belief[n] = old_belief[o];
      new_belief[n].log_weight = log_avg;
      // increase for next draw
      avg_cumulative = jacobi_logarithm(avg_cumulative, log_avg);
    }
    return new_belief;
  }

  StateType get_best() const
  {
    auto index = std::distance(log_weights,
                               std::max_element(log_weights.begin(),
                                                log_weights.end()));
    return states[index];
  }

  std::array<StateType, particle_count> get_states() const
  {
    return states;
  }

  std::array<double, particle_count> get_log_weights() const
  {
    return log_weights;
  }

private:
  // corrsponding states and weights
  std::array<double, particle_count> log_weights;
  std::array<StateType, particle_count> states;
  // Transition and observation
  Model model;
  // Uniform random number generator
  UniformRandom uniform_random;
};
} // namespace filter_bay