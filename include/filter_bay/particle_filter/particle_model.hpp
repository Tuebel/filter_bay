#pragma once
#include <functional>

namespace filter_bay
{
/*!
A particle as a sample of the target distribution consists of the state and the
weight.
*/
template <typename StateType>
struct Particle
{
  /*! The state of the sample */
  StateType state;
  /*! The weight of the sample */
  double weight;
};

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
  Calculates the calculate of the particle state given the current state and
  observation.
  \param state the current state
  \param observation the current measurement
  */
  std::function<double(const StateType &state,
                       const ObservationType &observation)>
      likelihood;
};
} // namespace filter_bay