#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace filter_bay
{
/*!
Calculates ln(a + b) = ln(exp(log_a) + exp(log_b))
*/
double jacobi_logarithm(double log_a, double log_b)
{
  return std::max(log_a, log_b) + log(1 + exp(-abs(log_a - log_b)));
}

/*! 
Calculates the logarithm over the sum of exponentials of the values:
ln(exp(x_1) + exp(x_2) + ... + exp(x_n))
\param log_values in logarithmic domain, e.g.: log(weight_i). Must have at least 
two values.
*/
template <size_t N>
double jacobi_logarithm(const std::array<double, N> &log_values)
{
  double cumulative = 0;
  for (double current : log_values)
  {
    cumulative = jacobi_logarithm(current, cumulative);
  }
  return cumulative;
}

/*! 
Calculates the logarithm over the sum of exponentials of the values:
ln(exp(x_1) + exp(x_2) + ... + exp(x_n))
\param log_values in logarithmic domain, e.g.: log(weight_i). Must have at least 
two values.
*/
double jacobi_logarithm(const std::vector<double> &log_values)
{
  double cumulative = 0;
  for (double current : log_values)
  {
    cumulative = jacobi_logarithm(current, cumulative);
  }
  return cumulative;
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
  double norm_const = jacobi_logarithm(log_values);
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
  return -jacobi_logarithm(norm_log_weights);
}

/*!
Calculates the logarithmic effective sample size.
\param norm_log_weights the normalized log weights
*/
template<size_t N>
double ess_log(std::array<double, N> norm_log_weights)
{
  for (double &current : norm_log_weights)
  {
    current *= 2;
  }
  return -jacobi_logarithm(norm_log_weights);
}

} // namespace filter_bay