#pragma once
#include <Eigen/Dense>

namespace filter_bay
{
/*!
The movement state consists of a pose and a velocity.
The state is stored as a Eigen::Matrix so it can be used directly int the
systems model.
The pose is stored in the first half of the state, the velocity is stored in
the second half.
*/
template <size_t DOF>
struct MovementState
{
  using Pose = Eigen::Matrix<double, DOF, 1>;
  using Velocity = Eigen::Matrix<double, DOF, 1>;
  using State = Eigen::Matrix<double, 2 * DOF, 1>;
  
  State state;

  Pose get_pose() const
  {
    return state.block<DOF, 1>(0, 0);
  }

  Velocity get_velocity() const
  {
    return state.block<DOF, 1>(DOF, 0);
  }

  void set_pose(Pose pose)
  {
    state.block<DOF, 1>(0, 0) = std::move(pose);
  }

  void set_velocity(Velocity velocity)
  {
    state.block<DOF, 1>(DOF, 0) = std::move(velocity);
  }
};
} // namespace filter_bay