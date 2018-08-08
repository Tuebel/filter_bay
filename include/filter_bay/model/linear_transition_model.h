#pragma once

#include <Eigen/Dense>

namespace filter_bay
{
/*!
A linear state space model for transitioning between states with explicit noise.
The equation ist:
\f[
  x_{k+1} = F x_k + B u_k + G w_k
\f]
Where \f$x_k\f$ is the state, \f$u_k\f$ the input and \f$w_k\f$ the process
noise at step k.
The noise is not actually used in the state prediction but in the state 
covariance prediction:
\f[
  P_{k+1} = F P_k F^T + G Q_k G^T
\f]
The noise *\f$w_k\f$ is described via its covariance *\f$Q_k\f$.

See https://en.wikipedia.org/wiki/Kalman_filter#Example_application,_technical
for an example how \f$G\f% might be introduced into the state equation.
*/
template <size_t state_size, size_t input_size, size_t noise_size>
struct LinearTransitionModel
{
  /*! \f$F\f$ */
  using TransitionMatrix = Eigen::Matrix<double, state_size, state_size>;
  /*! \f$B\f$ */
  using InputMatrix = Eigen::Matrix<double, state_size, input_size>;
  /*! \f$G\f$ */
  using NoiseMatrix = Eigen::Matrix<double, state_size, noise_size>;
  /*! \f$x_k\f$ */
  using State = Eigen::Matrix<double, state_size, 1>;
  /*! \f$U_k\f$ */
  using Input = Eigen::Matrix<double, input_size, 1>;
  /*! \f$P_k\f$ */
  using StateCovariance = Eigen::Matrix<double, state_size, state_size>;
  /*! \f$Q_k\f$ */
  using NoiseCovariance = Eigen::Matrix<double, noise_size, noise_size>;

  /*! Transition matrix of the model. */
  TransitionMatrix F;
  /*! Control input matrix of the model. */
  InputMatrix B;
  /*! Explicit noise input matrix of the model. */
  NoiseMatrix G;

  LinearTransitionModel() {}

  LinearTransitionModel(TransitionMatrix f, InputMatrix b,
                        NoiseMatrix g) : F(std::move(f)), B(std::move(b)),
                                         G(std::move(g)) {}

  /*!
  Predict the new state via this model.
  \param x the last state
  \param u the last control input
  */
  State predict_state(const State &x, const Input &u) const
  {
    return F * x + B * u;
  }

  /*!
  Predict the new covariance of the state via this model.
  \param x the last state
  \param u the last control input
  */
  StateCovariance predict_covariance(const StateCovariance &P,
                                     const NoiseCovariance &Q) const
  {
    return F * P * F.transpose() + G * Q * G.transpose();
  }
};
} // namespace filter_bay