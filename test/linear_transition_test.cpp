#include <filter_bay/model/linear_transition_model.hpp>
#include <math.h>
#include <gtest/gtest.h>

using TransitionModel = typename filter_bay::LinearTransitionModel<2, 3, 1>;
// Threshold for positiv floating point test
const double EPS = 0.01;
// Parameters of model
const double T = 1;
const double c = 5;
const double d = 2;
const double m = 10;
const double var_p = 0.3;
const double var_m = 0.1;

TransitionModel create_model()
{
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
  return TransitionModel(std::move(F),
                         std::move(B), std::move(G));
}

TEST(LinearTransitionTest, TestInitialization)
{
  auto model = create_model();
  TransitionModel::TransitionMatrix F;
  F << 0.75, 0.9, -0.5, 0.8;
  ASSERT_TRUE(F.isApprox(model.F, EPS));
  TransitionModel::InputMatrix B;
  B << 0.25, 0.1, 0.05, 0.5, 0.2, 0.1;
  ASSERT_TRUE(B.isApprox(model.B, EPS));
  TransitionModel::NoiseMatrix G;
  G << 0.05, 0.1;
  ASSERT_TRUE(G.isApprox(model.G, EPS));
}

TEST(LinearTransitionTest, TestTransition)
{
  auto model = create_model();
  TransitionModel::State x_0;
  x_0 << 1, 0;
  TransitionModel::StateCovariance P_0;
  P_0 << var_m, 0, 0, 3 * var_m;
  TransitionModel::Input u_0;
  u_0 << 0, 0, 3;
  TransitionModel::NoiseCovariance Q_0;
  Q_0 << var_p;
  TransitionModel::State x_10;
  x_10 << 0.9, -0.2;
  TransitionModel::StateCovariance P_10;
  P_10 << 0.3, 0.18, 0.18, 0.22;
  ASSERT_TRUE(x_10.isApprox(model.predict_state(x_0, u_0), EPS))
   << model.predict_state(x_0, u_0);
  ASSERT_TRUE(P_10.isApprox(model.predict_covariance(P_0, Q_0), EPS))
   << model.predict_covariance(P_0, Q_0);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
