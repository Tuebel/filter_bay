#pragma once
#include<Eigen/Core>

namespace filter_bay
{
template <typename StateType>
struct Particle
{
  /*! The state of the sample */
  StateType state;
  /*! The weight of the sample */
  double weight;
};
} // namespace filter_bay