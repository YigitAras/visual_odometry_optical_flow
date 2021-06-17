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

#include <memory>

#include <Eigen/Dense>
#include <sophus/se2.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <visnav/common_types.h>

namespace visnav {
// template <class T>
// class AbstractCamera;

struct TransformedFunctor {
  TransformedFunctor(
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& _img)
      : img(_img) {}
  bool operator()(const double* _x, const double* _y, double* _value) const {
    _value[0] = img(_x[0], _y[0]);
    return true;
  }
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> img;
};

struct MotionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MotionCostFunctor(
      const Eigen::Vector2d& p_2d,
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> img1,
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> img2)
      : p_2d(p_2d), img1(img1), img2(img2) {
    // get_transformed.reset(new ceres::CostFunctionToFunctor<1, 1, 1>(
    //     new ceres::NumericDiffCostFunction<TransformedFunctor,
    //     ceres::CENTRAL,
    //                                        1, 1, 1>(
    //         new TransformedFunctor(img2))));
  }
  // template <class T>
  bool operator()(double const* const sT_se2, double* sResidual) const {
    // map inputs
    Eigen::Map<Sophus::SE2<double> const> const T_se2(sT_se2);
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_img1;
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_img2;

    // cv2eigen(img1, eigen_img1);
    // cv2eigen(img2, eigen_img2);

    // std::cout << "IMG MATRIX: \n" << eigen_img1.cast<T>() << std::endl;
    double* residual(sResidual);
    /**************************************************
    *** Total patch size 29 for HALF_PATCH_SIZE = 3 ***
    **************************************************/
    int HALF_PATCH_SIZE = 5;
    std::vector<double> res1;
    std::vector<double> res2;
    double patch_sum1(0);
    double patch_sum2(0);
    for (int x = -HALF_PATCH_SIZE; x < HALF_PATCH_SIZE + 1; x++) {
      const int y_bound = sqrt(HALF_PATCH_SIZE * HALF_PATCH_SIZE - x * x);
      for (int y = -y_bound; y < y_bound + 1; y++) {
        Eigen::Matrix<double, 2, 1> T_p2(p_2d.x(), p_2d.y());
        T_p2.x() += x;
        T_p2.y() += y;

        // std::cout << "WITH A SEG FAULT HERE: \n" << std::endl;
        double val1 = img1(T_p2.x(), T_p2.y());
        res1.push_back(val1);
        patch_sum1 += val1;
        T_p2 = (T_se2 * T_p2);
        // double _x = (T_se2 * T_p2).x();
        // double _y = (T_se2 * T_p2).y();
        // _T_p2 = (T_se2 * T_p2.cast<T>());
        double val2;
        // Sophus::SE2<T> sumt(T_se2);
        val2 = img2(T_p2.x(), T_p2.y());
        // (*get_transformed)(&_x, &_y, &val2);
        // std::cout << "PRINTING VAL2: \n" << val2 << std::endl;
        patch_sum2 += val2;
        res2.push_back(val2);
      }
    }
    patch_sum1 /= res1.size();
    patch_sum2 /= res1.size();
    for (size_t i = 0; i < res1.size(); i++) {
      res1[i] /= patch_sum1;
      res2[i] /= patch_sum2;
    }
    // std::cout << "PRINT RES SIZE: \n" << res1.size() << std::endl;
    residual[0] = 0;
    for (size_t i = 0; i < res1.size(); i++) {
      residual[0] += (res1[i] - res2[i]) * (res1[i] - res2[i]);
    }
    // std::cout << "PRINT RESIDUAL: \n" << residual[0] << std::endl;
    return true;
  }

  Eigen::Vector2d p_2d;
  // std::unique_ptr<ceres::CostFunctionToFunctor<1, 1, 1>> get_transformed;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> img1;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> img2;
};

}  // namespace visnav
