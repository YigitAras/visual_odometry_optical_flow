/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 3> xi_hat;
  xi_hat << 0, -xi(2), xi(1), xi(2), 0, -xi(0), -xi(1), xi(0), 0;
  std::cout << xi_hat << std::endl;
  double theta =
      xi.norm() +
      std::numeric_limits<double>::epsilon();  // add infitesmal to avoid
                                               // divison by zero

  return Eigen::Matrix<T, 3, 3>::Identity(3, 3) +
         (sin(theta) / theta) * xi_hat +
         +((1 - cos(theta)) / (theta * theta)) * (xi_hat * xi_hat);
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  double theta = acos((mat.trace() - 1) / 2.0);
  Eigen::Matrix<T, 3, 1> w;
  w << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0), mat(1, 0) - mat(0, 1);
  w = w * theta / (2 * sin(theta) + std::numeric_limits<double>::epsilon());

  return w;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> v = xi.head(3);
  Eigen::Matrix<T, 3, 1> w = xi.tail(3);
  Eigen::Matrix<T, 4, 4> zeta;

  zeta.block(0, 0, 3, 3) = user_implemented_expmap(w);
  zeta(3, 0) = 0;
  zeta(3, 1) = 0;
  zeta(3, 2) = 0;
  zeta(3, 3) = 1.0;

  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  double theta = w.norm() + std::numeric_limits<double>::epsilon();

  Eigen::Matrix<T, 3, 3> J =
      Eigen::Matrix<T, 3, 3>::Identity(3, 3) +
      ((1 - cos(theta)) / (pow(theta, 2))) * w_hat +
      ((theta - sin(theta)) / pow(theta, 3)) * w_hat * w_hat;

  zeta.block(0, 3, 3, 1) = (J * v);
  return zeta;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 6, 1> res;
  Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.block(0, 3, 3, 1);
  res.tail(3) = user_implemented_logmap(R);
  double theta = res.tail(3).norm() + std::numeric_limits<double>::epsilon();
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat << 0, -res(5), res(4), res(5), 0, -res(3), -res(4), res(3), 0;

  Eigen::Matrix<T, 3, 3> J_inv =
      Eigen::Matrix<T, 3, 3>::Identity(3, 3) - w_hat / 2 +
      ((1 / pow(theta, 2) - (1 + cos(theta)) / (2 * theta * sin(theta))) *
       w_hat * w_hat);

  res.head(3) = J_inv * t;
  return res;
}

}  // namespace visnav
