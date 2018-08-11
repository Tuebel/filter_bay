#pragma once
#include <filter_bay/utility/log_arithmetics.hpp>
#include <filter_bay/utility/uniform_random.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>

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
  using States = typename std::array<StateType, particle_count>;
  using LogWeights = typename std::array<double, particle_count>;
  /*! Predicts the state transition */
  using PredictFunction = std::function<StateType(const StateType &, const InputType &)>;
  /*! Calculates the logarithmic likelihood from an observation */
  using LogLikelihoodFunction = std::function<double(const StateType &state, const ObservationType &observation)>;

  LogParticleFilter(PredictFunction predict_function,
                    LogLikelihoodFunction log_likelihood_function)
      : predict_fn(predict_function), log_likelihood_fn(log_likelihood_function)
  {
  }

  /*!
  Set the initial state belief. Distributes the weights uniformly
  */
  void initialize(States initial_states)
  {
    states = std::move(initial_states);
    // log(1/belief_size) = -log(belief_size)
    double log_avg = -log(particle_count);
    for (double &current : log_weights)
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
      current = predict_fn(current, u);
    }
  }

  /*!
  Updates the particle weights by incorporating the observation.
  The update step is done in a SIR bootstrap filter fashion.
  Resampling is performed with the low variance resampling method.
  \param z the current observation
  \param log_resample_threshold threshold of the effective sample size(ESS). 
  Resampling is performed if ESS < resample_threshold. log(N/2) is a typical
  value. 
  */
  void update(const ObservationType &z,
              double log_resample_threshold = log(particle_count / 2.0))
  {
    // weights as posterior of observation, prior is proposal density
    for (size_t i = 0; i < particle_count; i++)
    {
      // log(weight*likelihood) = log(weight) + log_likelihood
      log_weights[i] += log_likelihood_fn(states[i], z);
    }
    // Normalize weights
    log_weights = normalized_logs(log_weights);
    // Perform resampling? Effective sample_size < threshold
    if (ess_log(log_weights) < log_resample_threshold)
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
    LogWeights old_weights = log_weights;
    // Start at random value within average weight
    double cumulative = old_weights[0];
    double avg_weight = 1.0 / particle_count;
    double start_weight = uniform_random.generate(0, avg_weight);
    double log_avg = log(avg_weight);
    // o in old_weights, n in new_weigts
    size_t o = 0;
    for (size_t n = 0; n < particle_count; n++)
    {
      double U = log(start_weight + n * avg_weight);
      while (U > cumulative)
      {
        o++;
        cumulative = jacobi_logarithm(cumulative, old_weights[o]);
      }
      // Resample this particle and reset the weight
      states[n] = old_states[o];
      log_weights[n] = log_avg;
    }
  }

  /*! 
  Returns the state of the particles. Use get_log_weights to obtain the full
  belief.
  */
  States get_states() const
  {
    return states;
  }

  /*! 
  Returns the logarithmic weights of the particles. Use get_states to obtain the
  full belief.
  */
  LogWeights get_log_weights() const
  {
    return log_weights;
  }

  /*! Returns the number of particles beeing simulated */
  size_t get_particle_count()
  {
    return particle_count;
  }

private:
  // corrsponding states and weights
  LogWeights log_weights;
  States states;
  // state transition and measurement functions
  PredictFunction predict_fn;
  LogLikelihoodFunction log_likelihood_fn;
  // Uniform random number generator
  UniformRandom uniform_random;
};
} // namespace filter_bay