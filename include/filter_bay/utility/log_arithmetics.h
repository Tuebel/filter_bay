#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace filter_bay
{
/*!
Calculates ln(exp(a) + exp(b)) while avoiding overflow and underflow issues.
*/
double jacobi_logarithm(double log_a, double log_b)
{
  return std::max(log_a, log_b) + log(1 + exp(-abs(log_a - log_b)));
}

/*!
Calculates the  logarithmic sum of exponentials ln(sum(exp(x_i))) while avoiding 
overflow and underflow issues.
*/
template <size_t N>
double log_sum_exp(const std::array<double, N> &log_values)
{
  double max = *std::max_element(log_values.begin(), log_values.end());
  double exp_sum = 0;
  for (double current : log_values)
  {
    exp_sum += exp(current - max);
  }
  return max + log(exp_sum);
}

/*!
Calculates the  logarithmic sum of exponentials ln(sum(exp(x_i))) while avoiding 
overflow and underflow issues.
*/
double log_sum_exp(const std::vector<double> &log_values)
{
  double max = *std::max_element(log_values.begin(), log_values.end());
  double exp_sum = 0;
  for (double current : log_values)
  {
    exp_sum += exp(current - max);
  }
  return max + log(exp_sum);
}

/*!
Calculates the normalized values in logarithmic domain:
norm_val = val - normalization_const . Instead of norm_val = val / norm_const
\param log_values the unnormalizedvalues in logarithmic domain
*/
template <size_t N>
std::array<double, N> normalized_logs(std::array<double, N> log_values)
{
  double norm_const = jacobi_logarithm(log_values);
  for (double &value : log_values)
  {
    value -= norm_const;
  }
  return log_values;
}

/*!
Calculates the normalized values in logarithmic domain:
norm_val = val - normalization_const . Instead of norm_val = val / norm_const
\param log_values the unnormalizedvalues in logarithmic domain
*/
std::vector<double> normalized_logs(std::vector<double> log_values)
{
  double norm_const = log_sum_exp(log_values);
  for (double &value : log_values)
  {
    value -= norm_const;
  }
  return log_values;
}

/*!
Calculates the logarithmic effective sample size.
\param norm_log_weights the normalized log weights
*/
double ess_log(std::vector<double> norm_log_weights)
{
  for (double &current : norm_log_weights)
  {
    current *= 2;
  }
  return -log_sum_exp(norm_log_weights);
}

/*!
Calculates the logarithmic effective sample size.
\param norm_log_weights the normalized log weights
*/
template <size_t N>
double ess_log(std::array<double, N> norm_log_weights)
{
  for (double &current : norm_log_weights)
  {
    current *= 2;
  }
  return -log_sum_exp(norm_log_weights);
}

} // namespace filter_bay