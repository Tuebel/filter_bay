#include <filter_bay/tracker/impedance_tracker.hpp>
#include <math.h>
#include <gtest/gtest.h>

typedef filter_bay::ImpedanceTracker<2> MyTracker;
typedef MyTracker::ValueArray ValueArray;
// Threshold for positiv floating point test
const double EPS = 0.01;
// Parameters of model
const double T = 1;
const double c = 5;
const double d = 2;
const double m = 10;
const double var_p = 0.3;
const double var_m = 0.1;

MyTracker create_tracker()
{
  ValueArray stiffness{c, 1};
  ValueArray damping{d, 1};
  ValueArray mass{m, 1};
  ValueArray measurement_noise{var_m, 1};
  return MyTracker(stiffness, damping, mass, measurement_noise, T);
}

TEST(ImpedanceTrackerTest, TestInitialization)
{
  auto filter = create_tracker();
  ValueArray x_0{1, 0};
  filter.initialize(x_0);
  ASSERT_DOUBLE_EQ(filter.get_pose()[0], 1);
  ASSERT_DOUBLE_EQ(filter.get_pose()[1], 0);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[0], 0);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[1], 0);
  MyTracker::Covariance P_0_0;
  P_0_0 << var_m, 0, 0, 3 * var_m;
  ASSERT_TRUE(filter.get_covariance()[0].isApprox(P_0_0, EPS));
  MyTracker::Covariance P_0_1;
  P_0_1 << 1, 0, 0, 3;
  ASSERT_TRUE(filter.get_covariance()[1].isApprox(P_0_1, EPS));
}

TEST(ImpedanceTrackerTest, TestFilterStep)
{
  auto filter = create_tracker();
  ValueArray x_0{1, 0};
  filter.initialize(x_0);
  ASSERT_DOUBLE_EQ(filter.get_pose()[0], 1);
  ASSERT_DOUBLE_EQ(filter.get_pose()[1], 0);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[0], 0);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[1], 0);
  // test predicition
  ValueArray target_pose{0, 0};
  ValueArray target_velocity{0, 0};
  ValueArray force{3, 0};
  ValueArray variance{var_p, 3};
  filter.predict(target_pose, target_velocity, force, variance);
  ASSERT_DOUBLE_EQ(filter.get_pose()[0], 0.9);
  ASSERT_DOUBLE_EQ(filter.get_pose()[1], 0);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[0], -0.2);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[1], 0);
  MyTracker::Covariance P_10_0;
  P_10_0 << 0.3, 0.18, 0.18, 0.22;
  MyTracker::Covariance P_10_1;
  P_10_1 << 1.75, 1.0, 1.0, 4.0;
  ASSERT_TRUE(filter.get_covariance()[0].isApprox(P_10_0, EPS))
      << filter.get_covariance()[0] << "\n\n";
  ASSERT_TRUE(filter.get_covariance()[1].isApprox(P_10_1, EPS))
      << filter.get_covariance()[0] << "\n\n"
      << filter.get_covariance()[1] << "\n\n";
  // Test update
  ValueArray measurement{0.75, 0};
  filter.update(measurement);
  ASSERT_DOUBLE_EQ(filter.get_pose()[0], 0.7875);
  ASSERT_DOUBLE_EQ(filter.get_pose()[1], 0);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[0], -0.2675);
  ASSERT_DOUBLE_EQ(filter.get_velocity()[1], 0);
  MyTracker::Covariance P_1_0;
  P_1_0 << 0.075, 0.045, 0.045, 0.139;
  MyTracker::Covariance P_1_1;
  P_1_1 << 0.636, 0.363, 0.363, 3.636;
  ASSERT_TRUE(filter.get_covariance()[0].isApprox(P_1_0, EPS));
  ASSERT_TRUE(filter.get_covariance()[1].isApprox(P_1_1, EPS));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
