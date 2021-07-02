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

#include <fstream>

#include <tbb/task_scheduler_init.h>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
// #include <visnav/motion_cost.h>
#include <sophus/se2.hpp>
#include <visnav/patch.h>
#include <visnav/patterns.h>
#include <visnav/image/image.h>
#include <visnav/image/image_pyr.h>
#include <visnav/local_parameterization_se3.hpp>

#include <visnav/tracks.h>

namespace visnav {

typedef OpticalFlowPatch<float, Pattern50<float>> PatchT;

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId& fcid0,
                                   const FrameCamId& fcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map

  opengv::bearingVectors_t bear1;
  opengv::bearingVectors_t bear2;

  for (auto sid : shared_track_ids) {
    if (landmarks.find(sid) != landmarks.end()) {
      continue;
    }
    auto kp0 = feature_corners.at(fcid0);
    auto kp1 = feature_corners.at(fcid1);

    auto feat_track = feature_tracks.at(sid);

    auto fid0 = feat_track.at(fcid0);
    auto fid1 = feat_track.at(fcid1);

    bear1.push_back(
        calib_cam.intrinsics[fcid0.cam_id]->unproject(kp0.corners[fid0]));
    bear2.push_back(
        calib_cam.intrinsics[fcid1.cam_id]->unproject(kp1.corners[fid1]));
  }
  // Calculate the transformation from second camera to the first camera frame

  auto se3_trans = cameras.at(fcid0).T_w_c.inverse() * cameras.at(fcid1).T_w_c;
  opengv::relative_pose::CentralRelativeAdapter adapter(
      bear1, bear2, se3_trans.translation(), se3_trans.rotationMatrix());
  int ind = 0;
  for (auto sid : shared_track_ids) {
    if (landmarks.find(sid) != landmarks.end()) {
      continue;
    }
    Landmark lm;
    new_track_ids.push_back(sid);
    opengv::point_t pp = cameras.at(fcid0).T_w_c *
                         opengv::triangulation::triangulate(adapter, ind);
    ind++;
    lm.p = pp;
    for (auto kv : cameras) {
      auto sfcamid = kv.first;
      auto strack = feature_tracks.at(sid);
      if (strack.find(sfcamid) != strack.end())
        lm.obs.insert(std::make_pair(sfcamid, strack.at(sfcamid)));
    }
    landmarks.insert(std::make_pair(sid, lm));
  }

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId& fcid0,
                                       const FrameCamId& fcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)

  cameras[fcid0].T_w_c = Sophus::SE3d(Eigen::Matrix4d::Identity());
  cameras[fcid1].T_w_c =
      calib_cam.T_i_c[fcid0.cam_id].inverse() * calib_cam.T_i_c[fcid1.cam_id];
  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const FrameCamId& fcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map
  opengv::bearingVectors_t bear;
  opengv::points_t kuma;
  auto kp = feature_corners.at(fcid);

  for (auto tid : shared_track_ids) {
    auto track = feature_tracks.at(tid);
    auto fid = track.at(fcid);
    bear.push_back(
        calib_cam.intrinsics[fcid.cam_id]->unproject(kp.corners[fid]));
    kuma.push_back(landmarks.at(tid).p);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bear, kuma);

  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      cenposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  // 0.92 > 0.99
  ransac.sac_model_ = cenposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.computeModel();

  adapter.setR(ransac.model_coefficients_.block<3, 3>(0, 0));
  adapter.sett(ransac.model_coefficients_.block<3, 1>(0, 3));

  opengv::transformation_t optimized =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
  ransac.sac_model_->selectWithinDistance(optimized, ransac.threshold_,
                                          ransac.inliers_);
  Eigen::Matrix4d res;
  res.block<3, 3>(0, 0) = optimized.block<3, 3>(0, 0);
  res.block<3, 1>(0, 3) = optimized.block<3, 1>(0, 3);
  res.block<1, 4>(3, 0) = Eigen::Vector4d(0, 0, 0, 1);
  T_w_c = Sophus::SE3d(res);
  for (auto i : ransac.inliers_) {
    inlier_track_ids.push_back(shared_track_ids[i]);
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

bool trackPointAtLevel(const visnav::Image<const uint8_t>& img_2,
                       const PatchT& dp, Eigen::AffineCompact2f& transform) {
  bool patch_valid = true;
  int max_iterations = 20;

  for (int iteration = 0; patch_valid && iteration < max_iterations;
       iteration++) {
    typename PatchT::VectorP res;

    typename PatchT::Matrix2P transformed_pat =
        transform.linear().matrix() * dp.pattern2;
    transformed_pat.colwise() += transform.translation();

    bool valid = dp.residual(img_2, transformed_pat, res);

    if (valid) {
      typename PatchT::Vector3 inc = -dp.H_se2_inv_J_se2_T * res;
      // std::cout << "Here..." << dp.H_se2_inv_J_se2_T << std::endl;
      // std::cout << "Here..." << res << std::endl;
      transform *= Sophus::SE2f::exp(inc).matrix();
      // std::cout << "... And there" << std::endl;
      const int filter_margin = 3;

      if (!img_2.InBounds(transform.translation(), filter_margin))
        patch_valid = false;
    } else {
      patch_valid = false;
    }
    // std::cout << "PATCH VALID: " << patch_valid << std::endl;
  }

  return patch_valid;
}

bool trackPoint(const visnav::ManagedImagePyr<uint8_t>& old_pyr,
                const visnav::ManagedImagePyr<uint8_t>& pyr,
                const size_t& num_levels,
                const Eigen::AffineCompact2f& old_transform,
                Eigen::AffineCompact2f& transform) {
  bool patch_valid = true;

  transform.linear().setIdentity();

  for (int level = num_levels; level >= 0 && patch_valid; level--) {
    const float scale = 1 << level;  // bit shift

    transform.translation() /= scale;

    PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

    // Perform tracking on current level
    patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);

    transform.translation() *= scale;
  }

  transform.linear() = old_transform.linear() * transform.linear();

  return patch_valid;
}

void find_motion_consec(
    const KeypointsData& kd, const visnav::ManagedImagePyr<uint8_t>& old_pyr,
    const visnav::ManagedImagePyr<uint8_t>& pyr, const size_t& num_levels,
    std::unordered_map<FeatureId, Eigen::AffineCompact2f>& transforms,
    bool prop, std::unordered_map<FeatureId, TrackId>& prop_tracks) {
  float optical_flow_max_recovered_dist2 = 0.04;
  // KeypointsData kdl = feature_corners.at(fcidl2);

  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_img1;
  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen_img2;

  // cv2eigen(cv::Mat(img1.h, img1.w, CV_8U, img1.ptr), eigen_img1);
  // cv2eigen(cv::Mat(img2.h, img2.w, CV_8U, img2.ptr), eigen_img2);
  // int ind = 0;
  // const int HALF_PATCH_SIZE = 3;
  // for (int x = -HALF_PATCH_SIZE; x < HALF_PATCH_SIZE + 1; x++) {
  //   const int y_bound = sqrt(HALF_PATCH_SIZE * HALF_PATCH_SIZE - x * x);
  //   for (int y = -y_bound; y < y_bound + 1; y++) {
  //     ind++;
  //   }
  //   ind++;
  // }
  // std::cout << "INDIEEEE: " << ind << std::endl;

  for (size_t i = 0; i < kd.corners.size(); i++) {
    const Eigen::Vector2d p2d = kd.corners[i];
    Eigen::AffineCompact2f transform_1 = transforms[i];
    // transform_1.setIdentity();
    // transform_1.translation() += p2d.cast<float>();
    Eigen::AffineCompact2f transform_2 = transform_1;
    // std::cout << "BEFORE:\n" << i << std::endl;
    // PatchT patch(img1, transform_1.translation());

    transform_2.linear().setIdentity();
    bool valid = trackPoint(old_pyr, pyr, num_levels, transform_1, transform_2);
    bool flag = false;
    transform_2.linear() = transform_1.linear() * transform_2.linear();
    // std::cout << "AFTER:\n" << i << std::endl;
    // std::cout << "NEW TRANSFORMS:\n" << transforms[i].matrix() << std::endl;
    
    if (valid) {
      Eigen::AffineCompact2f transform_1_recovered = transform_2;
      // PatchT patch2(img2, transform_2.translation());

      transform_1_recovered.linear().setIdentity();
      valid = trackPoint(pyr, old_pyr, num_levels, transform_2,
                         transform_1_recovered);
      transform_1_recovered.linear() =
          transform_2.linear() * transform_1_recovered.linear();

      if (valid) {
        float dist2 =
            (transform_1.translation() - transform_1_recovered.translation())
                .squaredNorm();

        if (dist2 < optical_flow_max_recovered_dist2) {
          transforms[i] = transform_2;
          flag = true;
        }

        // problem.AddParameterBlock(transforms[i].data(),
        //                           Sophus::SE2d::num_parameters);

        // ceres::CostFunction* cost_function =
        //     new ceres::NumericDiffCostFunction<MotionCostFunctor,
        //     ceres::CENTRAL, 1,
        //                                        Sophus::SE2d::num_parameters>(
        //         new MotionCostFunctor(p2d, eigen_img1, eigen_img2));
        // // problem.AddResidualBlock(cost_function, NULL,
        // transforms[i].data()); problem.AddResidualBlock(cost_function, (new
        // ceres::HuberLoss(1.0)),
        //                          transforms[i].data());
      }
    }
    if (!flag) {
      transforms.erase(i);
      if (prop) prop_tracks.erase(i);
    }
    // ceres::Problem problem;
    // int HALF_PATCH_SIZE = 3;
    // size_t patch_size = 0;
    // for (int x = -HALF_PATCH_SIZE; x < HALF_PATCH_SIZE + 1; x++) {
    //   const int y_bound = sqrt(HALF_PATCH_SIZE * HALF_PATCH_SIZE - x * x);
    //   for (int y = -y_bound; y < y_bound + 1; y++) {
    //     patch_size++;
    //   }
    // }

    // for (size_t i = 0; i < kd.corners.size(); i++) {
    //   const Eigen::Vector2d p2d = kd.corners[i];
    //   transforms[i] = Sophus::SE2d();

    //   problem.AddParameterBlock(transforms[i].data(),
    //                             Sophus::SE2d::num_parameters);

    //   ceres::CostFunction* cost_function =
    //       new ceres::NumericDiffCostFunction<MotionCostFunctor,
    //       ceres::CENTRAL, 1,
    //                                          Sophus::SE2d::num_parameters>(
    //           new MotionCostFunctor(p2d, eigen_img1, eigen_img2));
    //   // problem.AddResidualBlock(cost_function, NULL, transforms[i].data());
    //   problem.AddResidualBlock(cost_function, (new ceres::HuberLoss(1.0)),
    //                            transforms[i].data());
    // }
    // // std::cout << "PATCH SIZE: " << patch_size << std::endl;
    // // std::cout << "-----------" << std::endl;

    // // Solve

    // ceres::Solver::Options ceres_options;
    // ceres_options.max_num_iterations = 30;
    // ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
    // ceres_options.num_threads =
    // tbb::task_scheduler_init::default_num_threads(); ceres::Solver::Summary
    // summary; Solve(ceres_options, &problem, &summary); switch (1) {
    //   // 0: silent
    //   case 1:
    //     std::cout << summary.BriefReport() << std::endl;
    //     break;
    //   case 2:
    //     std::cout << summary.FullReport() << std::endl;
    //     break;
    // }
  }
}
// Run bundle adjustment to optimize cameras, points, and optionally
// intrinsics
void bundle_adjustment(const Corners& feature_corners,
                       const BundleAdjustmentOptions& options,
                       const std::set<FrameCamId>& fixed_cameras,
                       Calibration& calib_cam, Cameras& cameras,
                       Landmarks& landmarks) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  for (auto& cam : cameras) {
    for (auto& lm : landmarks) {
      if (lm.second.obs.find(cam.first) == lm.second.obs.end()) {
        continue;
      }

      problem.AddParameterBlock(cam.second.T_w_c.data(),
                                Sophus::SE3d::num_parameters,
                                new Sophus::test::LocalParameterizationSE3);
      if (fixed_cameras.find(cam.first) != fixed_cameras.end()) {
        problem.SetParameterBlockConstant(cam.second.T_w_c.data());
      }

      problem.AddParameterBlock(lm.second.p.data(), 3);
      double* params = calib_cam.intrinsics.at(cam.first.cam_id)->data();
      problem.AddParameterBlock(params, 8);
      if (!options.optimize_intrinsics) {
        problem.SetParameterBlockConstant(params);
      }
      auto cam_model = calib_cam.intrinsics.at(cam.first.cam_id)->name();
      auto kp = feature_corners.at(cam.first);
      auto fid = lm.second.obs.at(cam.first);
      auto p_2d = kp.corners[fid];

      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, 3, 8>(
          new BundleAdjustmentReprojectionCostFunctor(p_2d, cam_model));
      if (options.use_huber) {
        problem.AddResidualBlock(
            cost_function, (new ceres::HuberLoss(options.huber_parameter)),
            cam.second.T_w_c.data(), lm.second.p.data(), params);
      } else {
        problem.AddResidualBlock(cost_function, NULL, cam.second.T_w_c.data(),
                                 lm.second.p.data(), params);
      }
    }
  }
  // Solve

  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = tbb::task_scheduler_init::default_num_threads();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
