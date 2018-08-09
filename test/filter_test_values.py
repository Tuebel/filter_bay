# This file is part of the bayesian fusion object tracker (bfot).
# https://git.rwth-aachen.de/tim.uebelhoer/ma_catkin_ws/tree/master/src/bfot

# bfot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# bfot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with bfot.  If not, see <http://www.gnu.org/licenses/>.

'''
This file creates the test values for the cpp implementation
It uses the FilterPY library (https://github.com/rlabbe/filterpy,
use Python3 !).
'''

import numpy as np
from filterpy.kalman import KalmanFilter

# init a 2D Filter
my_filter = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
# parameters
T = 1.0
c = 1.0
d = 1.0
m = 1.0
var_p = 3.0    # Variance of process
var_m = 1.0   # Variance of measurement
# model
my_filter.F = np.array([[1. - (c * T ** 2) / (2 * m),
                         T - (d * T ** 2) / (2 * m)],
                        [-(c * T) / m,
                         1. - (d * T) / m]])
G = np.array([[T ** 2 / (2 * m)],
              [T / m]])
my_filter.Q = G * var_p * G.T
my_filter.B = np.array([[(c * T ** 2) / (2 * m),
                         (d * T ** 2) / (2 * m),
                         (T ** 2) / (2 * m)],
                        [(c * T) / m,
                         (d * T) / m,
                         T / m]])
my_filter.H = np.array([[1., 0.]])
my_filter.R = np.array([[var_m]])
# initial belief
my_filter.x = np.array([[0.0],
                        [0.0]])
my_filter.P = np.array([[var_m, 0.0],
                        [0.0, 3 * var_m]])
print(my_filter)
# input & measurement
u_0 = np.array([[0.0], [0.0], [0.0]])
z_1 = np.array([0.0])
# Predict
my_filter.predict(u_0)
print(my_filter)
my_filter.update(z_1)
print(my_filter)
