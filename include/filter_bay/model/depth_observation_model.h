#pragma once
#include <filter_bay/model/movement_state.h>

namespace filter_bay
{
/*!
Nonlinear observation model for a depth sensor.
\param height in pixels of the downsampled image
\param width in pixels of the downsampled image
\param DOF degrees of freedom of the pose
*/
template <size_t height, size_t width, size_t DOF>
class DepthObservationModel
{
public:
  /*! Float matrix of pixels depths. */
  using Observation = typename Eigen::Matrix<float, height, width>;
  using State = typename filter_bay::MovementState<DOF>;

  /*! 
  Calculates the likelihood of the observation given the prior state.
  \param state the prior state
  \param observation the current observation (camera image)
  */
  double calculate_likelihood(const State &state, const Observation &observation)
  {
    // Only the pose is measured
    State::Pose = state.get_pose();
  }
};
} // namespace filter_bay