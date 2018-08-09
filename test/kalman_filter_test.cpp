#include <filter_bay/kalman_filter/kalman_filter.hpp>
#include <math.h>
#include <gtest/gtest.h>

// Typedef the filter to get correct Matrices
using MyFilter = typename filter_bay::KalmanFilter<2, 3, 1, 1>;
using TransitionModel = typename MyFilter::TransitionModel;
using ObservationModel = typename MyFilter::ObservationModel;
// Threshold for positiv floating point test
const double EPS = 0.01;
const int LOOP_COUNT = 10000;
// Parameters of model
const double T = 1;
const double c = 5;
const double d = 2;
const double m = 10;
const double var_p = 0.3;
const double var_m = 0.1;

/*!
Creates the kalman filter for the test methods.
*/
MyFilter create_filter()
{
  // transition model
  TransitionModel::TransitionMatrix F;
  F << 1 - c * pow(T, 2) / (2 * m),
      T - d * pow(T, 2) / (2 * m),
      -c * T / m,
      1 - d * T / m;
  TransitionModel::InputMatrix B;
  B << c * pow(T, 2) / (2 * m),
      d * pow(T, 2) / (2 * m),
      pow(T, 2) / (2 * m),
      c * T / m,
      d * T / m,
      T / m;
  TransitionModel::NoiseMatrix G;
  G << pow(T, 2) / (2 * m),
      T / m;
  TransitionModel transition_model(std::move(F),
                                   std::move(B), std::move(G));
  // observation model
  ObservationModel::ObservationMatrix H;
  H << 1, 0;
  ObservationModel::NoiseCovariance R;
  R << var_m;
  ObservationModel observation_model(std::move(H), std::move(R));
  // filter
  return MyFilter(std::move(transition_model), std::move(observation_model));
}

TEST(KalmanFilterTest, TestInitialization)
{
  auto filter = create_filter();
  MyFilter::State x_0;
  x_0 << 0, 0;
  MyFilter::Covariance P_0;
  P_0 << 10, 0, 0, 10;
  filter.initialize(x_0, P_0);
  ASSERT_TRUE(x_0.isApprox(filter.get_state(), EPS));
  ASSERT_TRUE(P_0.isApprox(filter.get_covariance(), EPS));
}

TEST(KalmanFilterTest, TestFilterStep)
{
  auto filter = create_filter();
  MyFilter::State x_0;
  x_0 << 1, 0;
  MyFilter::Covariance P_0;
  P_0 << var_m, 0, 0, 3 * var_m;
  filter.initialize(x_0, P_0);
  // test predicition
  MyFilter::Input u_0;
  u_0 << 0, 0, 3;
  MyFilter::ProcessNoise Q_0;
  Q_0 << var_p;
  filter.predict(u_0, Q_0);
  MyFilter::State x_10;
  x_10 << 0.9, -0.2;
  MyFilter::Covariance P_10;
  P_10 << 0.3, 0.18, 0.18, 0.22;
  ASSERT_TRUE(x_10.isApprox(filter.get_state(), EPS));
  ASSERT_TRUE(P_10.isApprox(filter.get_covariance(), EPS));
  // test update
  MyFilter::Observation z_1;
  z_1 << 0.75;
  filter.update(z_1);
  MyFilter::State x_1;
  x_1 << 0.7875, -0.2675;
  MyFilter::Covariance P_1;
  P_1 << 0.075, 0.045, 0.045, 0.139;
  ASSERT_TRUE(x_1.isApprox(filter.get_state(), EPS)) << filter.get_state();
  ASSERT_TRUE(P_1.isApprox(filter.get_covariance(), EPS)) << filter.get_covariance();
  // Performance measure
  for (int i = 0; i < LOOP_COUNT; i++)
  {
    filter.predict(u_0, Q_0);
    filter.update(z_1);
  }
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
