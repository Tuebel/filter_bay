#pragma once
#include <filter_bay/model/linear_transition_model.hpp>
#include <filter_bay/model/gaussian_observation_model.hpp>

namespace filter_bay
{

/*!
A regular Kalman Filter with a explicit process noise model.
This means the state transition equation is 
\f[
  x_{k+1}=Ax_k + Bu_k + Gz_k
\f]
and the Matrix \f$G\f$ must be provided additionally to the common matrices.
The process noise \f%z_k\f$ must be provided at each prediction step.
*/
template <size_t state_size, size_t input_size, size_t process_noise_size,
          size_t observation_size>
class KalmanFilter
{
public:
  using ObservationModel = typename filter_bay::GaussianObservationModel<state_size, observation_size>;
  using TransitionModel = typename filter_bay::LinearTransitionModel<state_size, input_size, process_noise_size>;
  using State = typename TransitionModel::State;
  using Covariance = typename TransitionModel::StateCovariance;
  using Input = typename TransitionModel::Input;
  using ProcessNoise = typename TransitionModel::NoiseCovariance;
  using Observation = typename ObservationModel::Observation;

  /*! Model for predicition the state transition */
  TransitionModel transition_model;
  /*! Model for calculating the measurement from the state */
  ObservationModel observation_model;

  KalmanFilter() {}

  KalmanFilter(TransitionModel t_model, ObservationModel o_model)
      : transition_model(std::move(t_model)), 
      observation_model(std::move(o_model)) {}

  /*!
  Initializes the filter with the state x and covariance P.
  */
  void initialize(State x, Covariance P)
  {
    this->x = std::move(x);
    this->P = std::move(P);
  }

  /*!
  Returns the current state.
  */
  State get_state() const
  {
    return x;
  }

  /*!
  Returns the current covariance.
  */
  Covariance get_covariance() const
  {
    return P;
  }

  /*!
  Updates the state & covariance with a prediction of the model.
  \param u control input u_{t-1}
  \param Q covariance of the process noise
  */
  void predict(const Input &u, const ProcessNoise &Q)
  {
    x = transition_model.predict_state(x, u);
    P = transition_model.predict_covariance(P, Q);
  }

  /*!
  Updates the state & covariance to the posterior via the observation model.
  \param y measurement y_{t}
  */
  void update(const Observation &y)
  {
    auto K = observation_model.calculate_gain(P);
    // Update
    x = observation_model.update_state(x, y, K);
    P = observation_model.update_covariance(P, K);
  }

private:
  // State of the filter
  State x;
  Covariance P;
};
} // namespace filter_bay