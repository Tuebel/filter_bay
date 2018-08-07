#pragma once
#include <tuple>
#include <Eigen/Dense>

namespace filter_bay
{
/*!
Model of a gaussian measurement.
The equation ist:
\f[
  y_k = H x_k + v_k
\f]
Where \f$x_k\f$ is the state and \f$v_k\f$ the measurement noise at step k.
The measurement noise is assumed to be a constant white noise.

See https://en.wikipedia.org/wiki/Kalman_filter#Update for further explanation.
*/
template <size_t stateSize, size_t observationSize>
struct GaussianObservationModel
{
  /*! \f$x_k\f$ */
  using State = typename Eigen::Matrix<double, stateSize, 1>;
  /*! \f$P_k\f$ */
  using StateCovariance = typename Eigen::Matrix<double, stateSize, stateSize>;
  /*! \f$y_k\f$ */
  using Observation = typename Eigen::Matrix<double, observationSize, 1>;
  /*! \f$H\f$ */
  using ObservationMatrix = typename Eigen::Matrix<double, observationSize, stateSize>;
  /*! \f$R\f$ */
  using NoiseCovariance = typename Eigen::Matrix<double, observationSize, observationSize>;
  /*! \f$K\f$ */
  using KalmanGain = typename Eigen::Matrix<double, stateSize, observationSize>;

  /*! Observatin matrix of the state */
  ObservationMatrix H;
  /*! Measurement covariance matrix */
  NoiseCovariance R;

  GaussianObservationModel() {}

  GaussianObservationModel(ObservationMatrix h, NoiseCovariance r)
      : H(std::move(h)), R(std::move(r)) {}

  /*!
  Calculates the expected measurement for the given state.
  \param x current state
  */
  Observation calculate_measurement(const State &x) const
  {
    return H * x;
  }

  /*!
  Calculates the kalman gain via the observation model.
  \param P the current covariance of the state (serves as confidence)
  */
  KalmanGain calculate_gain(const StateCovariance &P) const
  {
    // Calculate P * H^T only once
    Eigen::Matrix<double, stateSize, observationSize> PH_T = P * H.transpose();
    Eigen::Matrix<double, observationSize, observationSize> S = R + H * PH_T;
    return PH_T * S.inverse();
  }

  /*!
  Calculates the update of the state with the given observation model.
  \param x current (predicted) state
  \param y current measurement
  \param K kalman gain for the (predicted) state covariance
  */
  State update_state(const State &x, const Observation &y, KalmanGain &K) const
  {
    Observation residual = y - calculate_measurement(x);
    return x + K * residual;
  }

  StateCovariance update_covariance(const StateCovariance &P,
                                    const KalmanGain &K) const
  {
    // P = (I-KH)P(I-KH)^T + KRK^T
    // This is more numerically stable and works for non-optimal K vs the equation
    // P = (I - KH) P usually seen in the literature.
    // https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py
    StateCovariance I = StateCovariance::Identity();
    auto I_KH = I - K * H;
    return I_KH * P * I_KH.transpose() + K * R * K.transpose();
  }
};
} // namespace filter_bay