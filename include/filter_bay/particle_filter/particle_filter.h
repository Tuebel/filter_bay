#pragma once
#include <filter_bay/model/linear_transition_model.h>
#include <filter_bay/model/depth_observation_model.h>
#include <filter_bay/particle_filter/particle.h>
#include <filter_bay/utility/normal_sampler.h>

namespace filter_bay
{
/*!
Simple implementation of a particle filter which uses a 
*/
template <size_t stateSize, size_t inputSize, size_t processNoiseSize,
          size_t observationHeight, size_t observationWidth>
class DepthParticleFilter
{
public:
  using ObservationModel = typename filter_bay::DepthObservationModel<observationHeight, observationWidth, DOF>;
  using TransitionModel = typename filter_bay::LinearTransitionModel<stateSize, inputSize, processNoiseSize>;
  using State = typename TransitionModel::State;
  using Input = typename TransitionModel::Input;
  using ProcessNoise = typename TransitionModel::NoiseCovariance;
  using Observation = typename ObservationModel::Observation;

  DepthParticleFilter() : random_device(),
                          normal_sampler(random_device())
  {
  }

  /*! Linear 6DOF transition model */
  TransitionModel transition_model;

  /*!
  Calculates the prediction for every particle.
  \param u the control input
  \param Q the process noise
  */
  void predict(const Input &u, const ProcessNoise &Q)
  {
    // Calculate the process noise once
    auto covariance = Eigen::Matrix<double, stateSize, stateSize>::Zero();
    covariance = transition_model.predict_covariance(covariance, Q);
    // Empty matrix
    for (size_t i = 0; i < particles.size(); i++)
    {
      // Predict the state
      particles[i].state = transition_model.predict_state(particles[i].state,
                                                          u);
      // Apply the noise
      particles[i].state = normal_sampler.sample_robust(particles[i].state,
                                                        covariance);
    }
  }

  void update(const Observation &observation)
  {
  }

private:
  using Particle = typename filter_bay::Particle<stateSize>;
  std::vector<Particle, 
  std::array<Particle, particleCount> particles;
  std::random_device random_device;
  NormalSampler<stateSize> normal_sampler;
};
} // namespace filter_bay