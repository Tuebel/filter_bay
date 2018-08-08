#pragma once
#include <filter_bay/kalman_filter/kalman_filter.hpp>
#include <math.h>

namespace filter_bay
{
/*!
This class tracks a robot frame which is impedance controlled. 
It is assumed that all forces are applied in the base of the frame and the
robot acts against the force.
*/
template <size_t DOF>
class ImpedanceTracker
{
public:
  /*!
  An array for raw values (raw in a sense of no Eigen::Matrix).
  */
  typedef std::array<double, DOF> ValueArray;
  /*!
  The covariance of a state component.
  */
  typedef Eigen::Matrix<double, 2, 2> Covariance;
  /*!
  An array of covariances of the state.
  */
  typedef std::array<Covariance, DOF> CovarianceArray;
  /*!
  Initialize tracking for an impedance controlled frame.
  \param stiffness the stiffness [N/m] or [Nm] per coordinate.
  \param damping the damping [N*s/m] or [Nm*s] per coordinate.
  \param mass the mass/intertia per coordinate. Deviation moments are neglected.
  \param measurement_noise the covariance of the noise of the measurements
  \param T the sampling inerval [s].
  */
  ImpedanceTracker(const ValueArray &stiffness,
                   const ValueArray &damping,
                   const ValueArray &mass,
                   const ValueArray &measurement_noise,
                   double T)
  {
    for (size_t i = 0; i < filters.size(); i++)
    {
      filters[i] = ImpedanceFilter(
          create_transition_model(stiffness[i], damping[i], mass[i], T),
          create_observation_model(measurement_noise[i]));
    }
  }

  /*!
  Initializes the tracker with the given pose.
  It is assumed that the pose has been measured, the covariance of the state is
  set according the measurement noise.
  The velocities are initialized 0.
  \param pose measured pose of the frame.
  */
  void initialize(const ValueArray &pose)
  {
    for (size_t i = 0; i < filters.size(); i++)
    {
      ImpedanceFilter::State state;
      state << pose[i], 0;
      ImpedanceFilter::Covariance covariance;
      auto R = filters[i].observation_model.R;
      covariance << R, 0, 0, 3 * R;
      filters[i].initialize(state, covariance);
    }
  }

  /*!
  Executes one filter prediction step.
  \param target_pose the control input pose
  \param target_velocity the control input velocity
  \param force the force vector
  \param variance the component wiese variance of the force vector.
  */
  void predict(
      const ValueArray &target_pose,
      const ValueArray &target_velocity,
      const ValueArray &force,
      const ValueArray &variance)
  {
    for (size_t i = 0; i < filters.size(); i++)
    {
      ImpedanceFilter::Input u;
      u << target_pose[i], target_velocity[i], force[i];
      ImpedanceFilter::ProcessNoise Q;
      Q << variance[i];
      filters[i].predict(u, Q);
    }
  }

  /*!
  Executes one filter update step.
  \param measured_pose the current measured pose.
  */
  void update(const ValueArray &measured_pose)
  {
    for (size_t i = 0; i < filters.size(); i++)
    {
      ImpedanceFilter::Observation y;
      y << measured_pose[i];
      filters[i].update(y);
    }
  }

  /*!
  The mean of the current pose belief.
  */
  ValueArray get_pose() const
  {
    ValueArray pose;
    for (size_t i = 0; i < filters.size(); i++)
    {
      pose[i] = filters[i].get_state()(0);
    }
    return pose;
  }

  /*!
  The mean of the current velocity belief.
  */
  ValueArray get_velocity() const
  {
    ValueArray velocity;
    for (size_t i = 0; i < filters.size(); i++)
    {
      velocity[i] = filters[i].get_state()(1);
    }
    return velocity;
  }

  /*!
  The component wise covariance cov(pose,velocity) of the current belief.
  */
  CovarianceArray get_covariance() const
  {
    CovarianceArray covariance;
    for (size_t i = 0; i < filters.size(); i++)
    {
      covariance[i] = filters[i].get_covariance();
    }
    return covariance;
  }

private:
  typedef filter_bay::KalmanFilter<2, 3, 1, 1> ImpedanceFilter;
  typedef ImpedanceFilter::ObservationModel ObservationModel;
  typedef ImpedanceFilter::TransitionModel TransitionModel;
  std::array<ImpedanceFilter, DOF> filters;

  /*!
  Creates the discrete transition model for the impedance tracker.
  \param c stiffness
  \param d damping
  \param m mass/ moment of inertia
  \param T sampling interval
  */
  TransitionModel create_transition_model(double c, double d, double m,
                                          double T)
  {
    return TransitionModel(create_transition_matrix(c, d, m, T),
                           create_input_matrix(c, d, m, T),
                           create_process_noise_matrix(m, T));
  }

  /*!
  Creates the observation model for the impedance tracker.
  \param r the covariance of the observation noise
  */
  ObservationModel create_observation_model(double r)
  {
    return ObservationModel(create_observation_matrix(),
                            create_observation_covariance(r));
  }

  /*!
  Creates the time discrete transition matrix for a impendance.
  \param c stiffness
  \param d damping
  \param m mass/moment of inertia
  \param T sampling interval
  */
  TransitionModel::TransitionMatrix create_transition_matrix(
      double c, double d, double m, double T)
  {
    TransitionModel::TransitionMatrix F;
    F << 1 - c * pow(T, 2) / (2 * m),
        T - d * pow(T, 2) / (2 * m),
        -c * T / m,
        1 - d * T / m;
    return F;
  }

  /*!
  Creates the time discrete control input matrix for a impendance.
  \param c stiffness
  \param d damping
  \param m mass/ moment of inertia
  \param T sampling interval
  */
  TransitionModel::InputMatrix create_input_matrix(
      double c, double d, double m, double T)
  {
    TransitionModel::InputMatrix B;
    B << c * pow(T, 2) / (2 * m),
        d * pow(T, 2) / (2 * m),
        pow(T, 2) / (2 * m),
        c * T / m,
        d * T / m,
        T / m;
    return B;
  }

  /*!
  Creates the time discrete process noise matrix.
  \param m mass/ moment of inertia
  \param T sampling interval
  */
  TransitionModel::NoiseMatrix create_process_noise_matrix(
      double m, double T)
  {
    TransitionModel::NoiseMatrix G;
    G << pow(T, 2) / (2 * m), T / m;
    return G;
  }

  /*!
  Creates the measurement matrix of the system. The pose (first item of
  the state) is measured.
  */
  ObservationModel::ObservationMatrix create_observation_matrix()
  {
    ObservationModel::ObservationMatrix H;
    H << 1, 0;
    return H;
  }

  /*!
  Creates the matrix of the measurement covariance.
  \param r covariance for the given measurement.
  */
  ObservationModel::NoiseCovariance create_observation_covariance(double r)
  {
    ObservationModel::NoiseCovariance R;
    R << r;
    return R;
  }
};
} // namespace filter_bay