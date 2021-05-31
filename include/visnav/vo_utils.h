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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.
  for (auto loc : landmarks) {
    auto aligned = current_pose.inverse() * loc.second.p;
    if (aligned[2] > cam_z_threshold) {
      auto proj = cam->project(aligned);
      if (proj[0] < cam->width() && proj[1] < cam->height() && proj[0] >= 0 &&
          proj[1] >= 0) {
        projected_points.push_back(proj);
        projected_track_ids.push_back(loc.first);
      }
    }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.
  for (long unsigned int i = 0; i < kdl.corners.size(); i++) {
    auto pcr = kdl.corners[i];
    int dist1 = 257;
    int dist2 = 257;

    auto best_pind = 0;

    for (long unsigned int j = 0; j < projected_points.size(); j++) {
      auto pt = projected_points[j];
      if ((pt - pcr).norm() <= match_max_dist_2d) {
        auto p_ind = projected_track_ids[j];
        auto lm = landmarks.at(p_ind);
        auto desc1 = kdl.corner_descriptors[i];
        auto lm_dist = 257;
        for (auto el : lm.obs) {
          auto fcid = el.first;
          auto fid = el.second;
          auto desc2 = feature_corners.at(fcid).corner_descriptors[fid];
          int curr_dist = (desc1 ^ desc2).count();
          if (curr_dist < lm_dist) {
            lm_dist = curr_dist;
          }
        }
        if (lm_dist < dist1) {
          dist2 = dist1;
          dist1 = lm_dist;
          best_pind = projected_track_ids[j];
        } else if (lm_dist < dist2)
          dist2 = lm_dist;
      }
    }
    if ((dist1 >= feature_match_threshold) ||
        (dist2 < feature_match_dist_2_best * dist1)) {
      continue;
    }
    md.matches.push_back(std::make_pair(i, best_pind));
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly
  // have tracks.
  opengv::bearingVectors_t bear;
  opengv::points_t kuma;

  // MD --> inliears ===> featureID, TrackID
  for (auto match : md.matches) {
    auto feat_id = match.first;
    auto track_id = match.second;
    auto ptr = cam->unproject(kdl.corners[feat_id]);
    bear.push_back(ptr);
    kuma.push_back(landmarks.at(track_id).p);
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
  md.T_w_c = Sophus::SE3d(res);

  for (auto i : ransac.inliers_) {
    md.inliers.push_back(md.matches[i]);
  }
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to
  // landmark matches add the observations to the existing landmarks. If the
  // left camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to
  // the existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a
  // new landmark you should always increase next_landmark_id by 1.

  // binary array for checking
  std::set<std::pair<FeatureId, FeatureId>> checker;

  for (auto elem : md.inliers) {
    auto feat_id = elem.first;
    auto track_id = elem.second;
    landmarks.at(track_id).obs.insert(std::make_pair(fcidl, feat_id));
    for (auto el : md_stereo.inliers) {
      if (el.first == feat_id) {
        landmarks.at(track_id).obs.insert(std::make_pair(fcidr, el.second));
        // to check whetver the couple was inserted
        checker.insert(el);
      }
    }
  }
  opengv::bearingVectors_t bear1;
  opengv::bearingVectors_t bear2;
  for (auto el : md_stereo.inliers) {
    bear1.push_back(
        calib_cam.intrinsics[fcidl.cam_id]->unproject(kdl.corners[el.first]));
    bear2.push_back(
        calib_cam.intrinsics[fcidr.cam_id]->unproject(kdr.corners[el.second]));
  }
  opengv::relative_pose::CentralRelativeAdapter adapter(bear1, bear2, t_0_1,
                                                        R_0_1);

  auto ind = 0;
  for (auto el : md_stereo.inliers) {
    if (checker.find(el) == checker.end()) {
      opengv::point_t pp =
          md.T_w_c * opengv::triangulation::triangulate(adapter, ind);

      Landmark lm;
      lm.p = pp;
      lm.obs.insert(std::make_pair(fcidl, el.first));
      lm.obs.insert(std::make_pair(fcidr, el.second));
      landmarks.insert(std::make_pair(next_landmark_id, lm));
      next_landmark_id++;
    }
    ind++;
  }
}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  // TODO SHEET 5: Remove old cameras and observations if the number of
  // keyframe pairs (left and right image is a pair) is larger than
  // max_num_kfs. The ids of all the keyframes that are currently in the
  // optimization should be stored in kf_frames. Removed keyframes should be
  // removed from cameras and landmarks with no left observations should be
  // moved to old_landmarks.

  std::vector<FrameCamId> cam_erased;
  std::vector<TrackId> lm_erased;
  while ((int)kf_frames.size() > max_num_kfs) {
    auto next = *kf_frames.begin();
    for (auto cam : cameras) {
      if (cam.first.frame_id == next) {
        // remove the camera
        for (auto& elm : landmarks) {
          elm.second.obs.erase(cam.first);

          if (elm.second.obs.empty()) {
            old_landmarks.insert(std::make_pair(elm.first, elm.second));
            lm_erased.push_back(elm.first);
          }
        }
        cam_erased.push_back(cam.first);
      }
    }
    kf_frames.erase(next);
    for (auto c_er : cam_erased) {
      cameras.erase(c_er);
    }
    for (auto lm_er : lm_erased) {
      landmarks.erase(lm_er);
    }
    cam_erased.clear();
    lm_erased.clear();
  }
}
}  // namespace visnav
