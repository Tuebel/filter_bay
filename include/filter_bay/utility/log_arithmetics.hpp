#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace filter_bay
{
/*!
Calculates ln(exp(a) + exp(b)) while avoiding overflow and underflow issues.
*/
inline double jacobi_logarithm(double log_a, double log_b)
{
  return std::max(log_a, log_b) + std::log(1 +
                                           std::exp(-std::abs(log_a - log_b)));
}

/*!
Calculates the  logarithmic sum of exponentials ln(sum(exp(x_i))) while avoiding 
overflow and underflow issues.
*/
template <size_t N>
inline double log_sum_exp(const std::array<double, N> &log_values)
{
  double max = *std::max_element(log_values.begin(), log_values.end());
  double exp_sum = 0;
  for (double current : log_values)
  {
    exp_sum += std::exp(current - max);
  }
  return max + std::log(exp_sum);
}

/*!
Calculates the  logarithmic sum of exponentials ln(sum(exp(x_i))) while avoiding 
overflow and underflow issues.
*/
inline double log_sum_exp(const std::vector<double> &log_values)
{
  double max = *std::max_element(log_values.begin(), log_values.end());
  double exp_sum = 0;
  for (double current : log_values)
  {
    exp_sum += std::exp(current - max);
  }
  return max + std::log(exp_sum);
}

/*!
Calculates the normalized values in logarithmic domain:
norm_val = val - normalization_const . Instead of norm_val = val / norm_const
\param log_values the unnormalizedvalues in logarithmic domain
*/
template <size_t N>
inline std::array<double, N> normalized_logs(std::array<double, N> log_values)
{
  double norm_const = log_sum_exp(log_values);
  for (double &value : log_values)
  {
    // log(x/N) = log(x) - log(N)
    value -= norm_const;
  }
  return log_values;
}

/*!
Calculates the normalized values in logarithmic domain:
norm_val = val - normalization_const . Instead of norm_val = val / norm_const
\param log_values the unnormalizedvalues in logarithmic domain
*/
inline std::vector<double> normalized_logs(std::vector<double> log_values)
{
  double norm_const = log_sum_exp(log_values);
  for (double &value : log_values)
  {
    // log(x/N) = log(x) - log(N)
    value -= norm_const;
  }
  return log_values;
}

/*!
Calculates the logarithmic effective sample size.
\param norm_log_weights the normalized log weights
*/
inline double ess_log(std::vector<double> norm_log_weights)
{
  for (double &current : norm_log_weights)
  {
    // log(x^2) = 2log(x)
    current *= 2;
  }
  // log(1/x)=-log(x)
  return -log_sum_exp(norm_log_weights);
}

/*!
Calculates the logarithmic effective sample size.
\param norm_log_weights the normalized log weights
*/
template <size_t N>
inline double ess_log(std::array<double, N> norm_log_weights)
{
  for (double &current : norm_log_weights)
  {
    // log(x^2)=2*log(x)
    current *= 2;
  }
  // log(1/sum)=-log(sum)
  return -log_sum_exp(norm_log_weights);
}

} // namespace filter_bay