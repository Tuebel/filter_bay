# filter_bay
A small filter library for BAYesian filtering. It does not try to be an "Eierlegende Wollmilchsau" as it accepts that the filter models are just too different.
  
# dependencies
Depends on:

- Eigen 3.3 (http://eigen.tuxfamily.org/)

Build erverything with cmake.

# usage
The classes are all templated. While the motivation for the Kalman filter is to verify the model dimensions at compile time, the particle filter can actually use generic transition and observation models.

The typical workflow is to define the generic filter and then retrieving the concrete model types from it. See the test classes on how to create a filter instance.
