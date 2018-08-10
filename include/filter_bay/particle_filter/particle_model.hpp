#pragma once
#include <functional>

namespace filter_bay
{
/*!
Generic formulation of a transition + observation model.
Passing in the generic functions is easier than subclassing this template.
*/
template <typename StateType, typename InputType, typename ObservationType>
struct ParticleModel
{
  /*!
  Predicts the next particle state given the current state and input.
  \param state the current state
  \param input the control input
  */
  std::function<StateType(const StateType &, const InputType &)> predict;

  /*!
  Calculates the logarithmic likelihood of the particle state given the current 
  state and observation.
  \param state the current state
  \param observation the current measurement
  */
  std::function<double(const StateType &state,
                       const ObservationType &observation)>
      log_likelihood;
};
} // namespace filter_bay