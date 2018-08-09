#include <filter_bay/model/gaussian_observation_model.hpp>
#include <math.h>
#include <gtest/gtest.h>

using ObservationModel = typename filter_bay::GaussianObservationModel<2, 1>;
// Threshold for positiv floating point test
const double EPS = 0.01;
// Parameters of model
const double var_m = 0.1;

ObservationModel create_model()
{
  ObservationModel::ObservationMatrix H;
  H << 1, 0;
  ObservationModel::NoiseCovariance R;
  R << var_m;
  return ObservationModel(std::move(H), std::move(R));
}

TEST(GaussianObservationTest, TestInitialization)
{
  auto model = create_model();
  ObservationModel::ObservationMatrix H;
  H << 1, 0;
  ASSERT_TRUE(H.isApprox(model.H, EPS));
  ObservationModel::NoiseCovariance R;
  R << var_m;
  ASSERT_TRUE(R.isApprox(model.R, EPS));
}

TEST(GaussianObservationTest, TestUpdate)
{
  auto model = create_model();
  ObservationModel::State x_10;
  x_10 << 0.9, -0.2;
  ObservationModel::Observation y_10;
  y_10 << 0.9;
  ASSERT_TRUE(y_10.isApprox(model.calculate_measurement(x_10)));
  ObservationModel::StateCovariance P_10;
  P_10 << 0.3, 0.18, 0.18, 0.22;
  ObservationModel::KalmanGain K;
  K << 0.75, 0.45;
  ASSERT_TRUE(K.isApprox(model.calculate_gain(P_10), EPS));
  ObservationModel::State x_1;
  x_1 << 0.7875, -0.2675;
  ObservationModel::Observation y_1;
  y_1 << 0.75;
  ASSERT_TRUE(x_1.isApprox(model.update_state(x_10, y_1, K), EPS));
  ObservationModel::StateCovariance P_1;
  P_1 << 0.075, 0.045, 0.045, 0.139;
  ASSERT_TRUE(P_1.isApprox(model.update_covariance(P_10, K), EPS));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
