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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <fstream>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tbb.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/video/tracking.hpp>

#include <CLI/CLI.hpp>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/vo_utils.h>
#include <visnav/image/image.h>
#include <visnav/image/image_pyr.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>

using namespace visnav;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t view_id);
void change_display_to_image(const FrameCamId& fcid);
void draw_scene();
void load_data(const std::string& path, const std::string& calib_path);
bool next_step();
void optimize();
void compute_projections();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

int current_frame = 0;
Sophus::SE3d current_pose;
bool take_keyframe = true;
TrackId next_landmark_id = 0;
std::unordered_map<FeatureId, TrackId> propagate_tracks;

std::atomic<bool> opt_running{false};
std::atomic<bool> opt_finished{false};

std::set<FrameId> kf_frames;

std::shared_ptr<std::thread> opt_thread;

/// intrinsic calibration
Calibration calib_cam;
Calibration calib_cam_opt;

/// loaded images
tbb::concurrent_unordered_map<FrameCamId, std::string> images;

/// timestamps for all stereo pairs
std::vector<Timestamp> timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// camera poses in the current map
Cameras cameras;
Cameras all_cameras;

/// copy of cameras for optimization in parallel thread
Cameras cameras_opt;

/// landmark positions and feature observations in current map
Landmarks landmarks;

/// copy of landmarks for optimization in parallel thread
Landmarks landmarks_opt;

/// landmark positions that were removed from the current map
Landmarks old_landmarks;

// KD-tree for Old landmarks

// SE2 transforms between consecutive frames
std::unordered_map<FrameCamId, std::unordered_map<TrackId, Sophus::SE2d>>
    transforms;

std::unordered_map<FrameCamId, std::vector<std::pair<FeatureId, FeatureId>>>
    matches;

visnav::ManagedImagePyr<uint8_t> old_pyramid, pyramid_r, pyramid_n;

size_t num_levels = 3;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images
ImageProjections image_projections;

///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel by
// switching the prefix from "ui" to "hidden" or vice verca. This way you can
// show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);

//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);

pangolin::Var<double> match_max_dist_2d("hidden.match_max_dist_2d", 20.0, 1.0,
                                        50);

pangolin::Var<int> new_kf_min_inliers("hidden.new_kf_min_inliers", 80, 1, 200);

pangolin::Var<int> max_num_kfs("hidden.max_num_kfs", 10, 5, 20);

pangolin::Var<double> cam_z_threshold("hidden.cam_z_threshold", 0.1, 1.0, 0.0);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

///////////////////////////////////////////////////////////////////////////////
//// OpenCV Optical Flow
/*
void opencv_opt_flow_LR(const pangolin::ManagedImage<uint8_t>& imgl,
                        const pangolin::ManagedImage<uint8_t>& imgr,
                        KeypointsData& kdl, KeypointsData& kdr,
                        MatchData& md_stereo) {
  std::vector<cv::Point2f> currPointsL, currPointsR, currPointsTemp;
  for (auto& c : kdl.corners) {
    currPointsL.push_back(cv::Point2f(c.x(), c.y()));
  }

  std::vector<float> err;
  std::vector<uchar> status;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.003);
  cv::Mat imageL(imgl.h, imgl.w, CV_8U, imgl.ptr);
  cv::Mat imageR(imgr.h, imgr.w, CV_8U, imgr.ptr);
  kdr.corners.clear();
  cv::calcOpticalFlowPyrLK(imageL, imageR, currPointsL, currPointsR, status,
                           err, cv::Size(5, 5), 2, criteria);
  cv::calcOpticalFlowPyrLK(imageR, imageL, currPointsR, currPointsTemp, status,
                           err, cv::Size(5, 5), 2, criteria);

  float threshold = 1;

  for (int i = 0; i < currPointsTemp.size(); i++) {
    Eigen::Vector2d diff(std::abs((currPointsTemp[i] - currPointsL[i]).x),
                         std::abs((currPointsTemp[i] - currPointsL[i]).y));
    Eigen::Vector2d tempy(currPointsL[i].x, currPointsL[i].y);
    //  Eigen::Vector3d tt = md_stereo.T_i_j *tempy3.cast<double>();
    // Eigen::Vector2f tempy(round(tt.x()), round(tt.y()));
    kdr.corners.push_back(tempy);
    if ((diff.allFinite()) && (diff.norm() < threshold)) {
      md_stereo.matches.push_back(std::make_pair(i, i));
    }
  }
  std::cout << kdr.corners.size() << " " << matches.size() << std::endl;
}

void opencv_opt_flow_ij(pangolin::ManagedImage<uint8_t>& imgc,
                        pangolin::ManagedImage<uint8_t>& imgn,
                        KeypointsData& kdl, KeypointsData& kdr,
                        std::unordered_map<FeatureId, TrackId>& prop_tracks) {
  std::vector<cv::Point2f> currPointsL, currPointsR, currPointsTemp;
  for (auto& c : kdl.corners) {
    currPointsL.push_back(cv::Point2f(c.x(), c.y()));
  }
  std::unordered_map<FeatureId, TrackId> updated_tracks;

  std::vector<float> err;
  std::vector<uchar> status;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.003);
  cv::Mat imageL(imgc.h, imgc.w, CV_8U, imgc.ptr);
  cv::Mat imageR(imgn.h, imgn.w, CV_8U, imgn.ptr);
  kdr.corners.clear();
  cv::calcOpticalFlowPyrLK(imageL, imageR, currPointsL, currPointsR, status,
                           err, cv::Size(7, 7), 2, criteria);
  cv::calcOpticalFlowPyrLK(imageR, imageL, currPointsR, currPointsTemp, status,
                           err, cv::Size(7, 7), 2, criteria);

  float threshold = 1;
  int j = 0;
  for (int i = 0; i < currPointsTemp.size(); i++) {
    Eigen::Vector2d diff(std::abs((currPointsTemp[i] - currPointsL[i]).x),
                         std::abs((currPointsTemp[i] - currPointsL[i]).y));
    Eigen::Vector2d tempy(currPointsR[i].x, currPointsR[i].y);
    //  Eigen::Vector3d tt = md_stereo.T_i_j *tempy3.cast<double>();
    // Eigen::Vector2f tempy(round(tt.x()), round(tt.y()));

    if ((diff.allFinite()) && (diff.squaredNorm() < threshold)) {
      // md_stereo.matches.push_back(std::make_pair(i, i));
      if (prop_tracks.find(i) != prop_tracks.end()) {
        kdr.corners.push_back(tempy);
        updated_tracks.insert(std::make_pair(j, prop_tracks[i]));
        j++;
      }
    }
  }
  prop_tracks = updated_tracks;

  std::cout << kdr.corners.size() << " " << matches.size() << std::endl;
}
// ///////////////////////////////////////////////////////////////////////////////
*/
///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path = "data/V1_01_easy/mav0";
  std::string cam_calib = "opt_calib.json";

  CLI::App app{"Visual odometry."};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: " + cam_calib);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(dataset_path, cam_calib);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      draw_scene();

      img_view_display.Activate();

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame1, 0));
          change_display_to_image(FrameCamId(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame2, 0));
          change_display_to_image(FrameCamId(show_frame2, 1));
        }
      }

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame1);
        auto cam_id = static_cast<CamId>(show_cam1);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[0]->SetImage(img);
        } else {
          img_view[0]->Clear();
        }
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame2);
        auto cam_id = static_cast<CamId>(show_cam2);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[1]->SetImage(img);
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();

      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  } else {
    // non-gui mode: Process all frames, then exit
    while (next_step()) {
      // nop
    }
  }

  return 0;
}

// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  auto frame_id =
      static_cast<FrameId>(view_id == 0 ? show_frame1 : show_frame2);
  auto cam_id = static_cast<CamId>(view_id == 0 ? show_cam1 : show_cam2);

  FrameCamId fcid(frame_id, cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(fcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(fcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, text_row);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }

  if (show_reprojections) {
    if (image_projections.count(fcid) > 0) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);  // red
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      const size_t num_points = image_projections.at(fcid).obs.size();
      double error_sum = 0;
      size_t num_outliers = 0;

      // count up and draw all inlier projections
      for (const auto& lm_proj : image_projections.at(fcid).obs) {
        error_sum += lm_proj->reprojection_error;

        if (lm_proj->outlier_flags != OutlierNone) {
          // outlier point
          glColor3f(1.0, 0.0, 0.0);  // red
          ++num_outliers;
        } else if (lm_proj->reprojection_error >
                   reprojection_error_huber_pixel) {
          // close to outlier point
          glColor3f(1.0, 0.5, 0.0);  // orange
        } else {
          // clear inlier point
          glColor3f(1.0, 1.0, 0.0);  // yellow
        }
        pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
        pangolin::glDrawLine(lm_proj->point_measured,
                             lm_proj->point_reprojected);
      }

      // only draw outlier projections
      if (show_outlier_observations) {
        glColor3f(1.0, 0.0, 0.0);  // red
        for (const auto& lm_proj : image_projections.at(fcid).outlier_obs) {
          pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
          pangolin::glDrawLine(lm_proj->point_measured,
                               lm_proj->point_reprojected);
        }
      }

      glColor3f(1.0, 0.0, 0.0);  // red
      pangolin::GlFont::I()
          .Text("Average repr. error (%u points, %u new outliers): %.2f",
                num_points, num_outliers, error_sum / num_points)
          .Draw(5, text_row);
      text_row += 20;
    }
  }

  if (show_epipolar) {
    glLineWidth(1.0);
    glColor3f(0.0, 1.0, 1.0);  // bright teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && it->second.inliers.size() > 20) {
      Sophus::SE3d T_this_other =
          idx == 0 ? it->second.T_i_j : it->second.T_i_j.inverse();

      Eigen::Vector3d p0 = T_this_other.translation().normalized();

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector3d p1(0, sin(i), cos(i));

        if (idx == 0) p1 = it->second.T_i_j * p1;

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          line.emplace_back(calib_cam.intrinsics[cam_id]->project(
              p0 * j + (1 - std::abs(j)) * p1));
        }

        Eigen::Vector2d c = calib_cam.intrinsics[cam_id]->project(p1);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }
}

// Update the image views to a given image id
void change_display_to_image(const FrameCamId& fcid) {
  if (0 == fcid.cam_id) {
    // left view
    show_cam1 = 0;
    show_frame1 = fcid.frame_id;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = fcid.cam_id;
    show_frame2 = fcid.frame_id;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {
  const FrameCamId fcid1(show_frame1, show_cam1);
  const FrameCamId fcid2(show_frame2, show_cam2);

  const u_int8_t color_camera_current[3]{255, 0, 0};         // red
  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_old_points[3]{170, 170, 170};         // gray
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

  // render cameras
  if (show_cameras3d) {
    int cnt = 0;
    for (const auto& cam : cameras) {
      // if (cnt % 5 == 0) {
      if (cam.first == fcid1) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                      0.1f);
      } else if (cam.first == fcid2) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_right,
                      0.1f);
      } else if (cam.first.cam_id == 0) {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left, 0.1f);
      } else {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_right,
                      0.1f);
      }
      // }
      // cnt++;
    }

    render_camera(current_pose.matrix(), 2.0f, color_camera_current, 0.1f);
  }

  // render points
  if (show_points3d && landmarks.size() > 0) {
    // std::cout << "Landmarks to be drawn: " <<  " " << landmarks.size() <<
    // std::endl;
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& kv_lm : landmarks) {
      const bool in_cam_1 = kv_lm.second.obs.count(fcid1) > 0;
      const bool in_cam_2 = kv_lm.second.obs.count(fcid2) > 0;

      const bool outlier_in_cam_1 = kv_lm.second.outlier_obs.count(fcid1) > 0;
      const bool outlier_in_cam_2 = kv_lm.second.outlier_obs.count(fcid2) > 0;

      if (in_cam_1 && in_cam_2) {
        glColor3ubv(color_selected_both);
      } else if (in_cam_1) {
        glColor3ubv(color_selected_left);
      } else if (in_cam_2) {
        glColor3ubv(color_selected_right);
      } else if (outlier_in_cam_1 || outlier_in_cam_2) {
        glColor3ubv(color_outlier_observation);
      } else {
        glColor3ubv(color_points);
      }

      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }

  // render points
  if (show_old_points3d && old_landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);

    for (const auto& kv_lm : old_landmarks) {
      glColor3ubv(color_old_points);
      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }
}

// Load images, calibration, and features / matches if available
void load_data(const std::string& dataset_path, const std::string& calib_path) {
  const std::string timestams_path = dataset_path + "/cam0/data.csv";

  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;

      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        timestamps.push_back(timestamp);
      }

      std::string img_name = line.substr(20, line.size() - 21);

      for (int i = 0; i < NUM_CAMS; i++) {
        FrameCamId fcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[fcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);
      std::cout << "Loaded camera from " << calib_path << " with models ";
      for (const auto& cam : calib_cam.intrinsics) {
        std::cout << cam->name() << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Execute next step in the overall odometry pipeline. Call this repeatedly
// until it returns false for automatic execution.

bool next_step() {
  if (current_frame >= (int(images.size() - 1) / NUM_CAMS)) {
    std::cout << "SEQUENCE IS OVER, PLEASE CLOSE THE PROGRAM \n";
    return false;
  }

  std::cout << "CHEKING OUT "
            << " " << current_frame << std::endl;

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_points;
  std::vector<TrackId> projected_track_ids;

  std::cout << "KF Projected " << projected_track_ids.size() << " points."
            << std::endl;

  MatchData md_stereo;
  LandmarkMatchData md;
  KeypointsData kdl, kdr, kdn;

  pangolin::ManagedImage<uint8_t> _imgl = pangolin::LoadImage(images[fcidl]);
  pangolin::ManagedImage<uint8_t> _imgr = pangolin::LoadImage(images[fcidr]);
  visnav::ManagedImage<uint8_t> imgl, imgr;
  imgl.CopyFrom(
      visnav::Image<uint8_t>(_imgl.ptr, _imgl.w, _imgl.h, _imgl.pitch));
  imgr.CopyFrom(
      visnav::Image<uint8_t>(_imgr.ptr, _imgr.w, _imgr.h, _imgr.pitch));

  old_pyramid.setFromImage(imgl, num_levels);
  pyramid_r.setFromImage(imgr, num_levels);

  if (feature_corners[fcidl].corners.size() < 150) {
    kdl = feature_corners[fcidl];
    KeypointsData dummy;

    detectKeypointsAndDescriptors(imgl, dummy, num_features_per_image,
                                  rotate_features);

    // code block
    std::vector<Eigen::Vector2d> ides;
    for (auto cc : dummy.corners) {
      bool found = false;
      ;
      for (auto ch : kdl.corners) {
        if ((cc - ch).norm() < 5) {
          found = true;
          break;
        }
      }
      if (found == false) ides.push_back(cc);
    }

    for (auto cc : ides) {
      kdl.corners.push_back(cc);
    }
    computeAngles(imgl, kdl, rotate_features);
    computeDescriptors(imgl, kdl);

    // code block ==//

    // propagate_tracks.clear();
    // md.matches.clear();
    // projected_track_ids.clear();
    // projected_points.clear();

    std::cout << "Detected " << kdl.corners.size() << " new keypoints."
              << std::endl;
  } else {
    kdl = feature_corners[fcidl];
    std::cout << "Retrieved " << kdl.corners.size() << " optimized keypoints."
              << std::endl;
  }

  md_stereo.T_i_j = T_0_1;

  Eigen::Matrix3d E;
  computeEssential(T_0_1, E);

  std::unordered_map<FeatureId, Eigen::AffineCompact2f> l_r_transforms;
  initialize_transforms(md, kdl, projected_points, projected_track_ids, true,
                        l_r_transforms, propagate_tracks);
  find_motion_consec(kdl, old_pyramid, pyramid_r, num_levels, l_r_transforms,
                     true, propagate_tracks);
  match_optical(kdr, l_r_transforms, md_stereo.matches, true, propagate_tracks);
  // opencv_opt_flow_LR(_imgl,_imgr,kdl,kdr,md_stereo);
  computeAngles(imgr, kdr, rotate_features);
  computeDescriptors(imgr, kdr);

  findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                       calib_cam.intrinsics[1], E, 1e-3, md_stereo);

  std::cout << "Found " << md_stereo.inliers.size() << " stereo-matches."
            << std::endl;

  feature_corners[fcidl] = kdl;
  feature_corners[fcidr] = kdr;
  feature_matches[std::make_pair(fcidl, fcidr)] = md_stereo;

  find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                         projected_track_ids, match_max_dist_2d,
                         feature_match_max_dist, feature_match_test_next_best,
                         md, propagate_tracks);

  std::cout << "Found " << md.matches.size() << " matches." << std::endl;

  localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                  reprojection_error_pnp_inlier_threshold_pixel, md);

  current_pose = md.T_w_c;

  cameras[fcidl].T_w_c = current_pose;
  cameras[fcidr].T_w_c = current_pose * T_0_1;

  all_cameras[fcidl].T_w_c = current_pose;
  all_cameras[fcidr].T_w_c = current_pose * T_0_1;
  add_new_landmarks(fcidl, fcidr, kdl, kdr, calib_cam, md_stereo, md, landmarks,
                    next_landmark_id, propagate_tracks);

  remove_old_keyframes(fcidl, 5, cameras, landmarks, old_landmarks, kf_frames);
  optimize();
  current_pose = cameras[fcidl].T_w_c;

  project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                    cam_z_threshold, projected_points, projected_track_ids);

  std::unordered_map<FeatureId, Eigen::AffineCompact2f> i_j_transforms;
  initialize_transforms(md, kdl, projected_points, projected_track_ids, false,
                        i_j_transforms, propagate_tracks);
  // std::unordered_map<FeatureId, Eigen::AffineCompact2f> j_i_transforms;
  if (current_frame < int(images.size() - 1)) {
    FrameCamId n_fcidl(fcidl.frame_id + 1, 0);
    pangolin::ManagedImage<uint8_t> _n_imgl =
        pangolin::LoadImage(images[n_fcidl]);

    visnav::ManagedImage<uint8_t> n_imgl;
    n_imgl.CopyFrom(visnav::Image<uint8_t>(_n_imgl.ptr, _n_imgl.w, _n_imgl.h,
                                           _n_imgl.pitch));

    pyramid_n.setFromImage(n_imgl, num_levels);

    find_motion_consec(kdl, old_pyramid, pyramid_n, num_levels, i_j_transforms,
                       true, propagate_tracks);

    match_optical(kdn, i_j_transforms, matches[fcidl], false, propagate_tracks);
    // opencv_opt_flow_ij(_imgl, _n_imgl, kdl, kdn, propagate_tracks);
    computeAngles(n_imgl, kdn, rotate_features);
    computeDescriptors(n_imgl, kdn);
    feature_corners[n_fcidl] = kdn;

    // visualize(_imgl, _n_imgl, kdl, kdn, matches[fcidl]);
  }

  // update image views
  change_display_to_image(fcidl);
  change_display_to_image(fcidr);

  compute_projections();

  std::ofstream outfile;

  // Save trajectory in EUROC format
  outfile.open("trajectory.txt",
               std::ios_base::app);  // append instead of overwrite
  outfile << timestamps[current_frame] << "," << current_pose.translation().x()
          << "," << current_pose.translation().y() << ','
          << current_pose.translation().z() << ',' << current_pose.data()[0]
          << "," << current_pose.data()[1] << "," << current_pose.data()[2]
          << "," << current_pose.data()[3] << "," << 0 << "," << 0 << std::endl;
  current_frame++;
  return true;
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();

  for (const auto& kv_lm : landmarks) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].obs.push_back(proj_lm);
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Optimize the active map with bundle adjustment
void optimize() {
  size_t num_obs = 0;
  for (const auto& kv : landmarks) {
    num_obs += kv.second.obs.size();
  }

  std::cerr << "Optimizing map with " << cameras.size() << " cameras, "
            << landmarks.size() << " points and " << num_obs << " observations."
            << std::endl;

  // Fix oldest two cameras to fix SE3 and scale gauge. Making the whole second
  // camera constant is a bit suboptimal, since we only need 1 DoF, but it's
  // simple and the initial poses should be good from calibration.
  FrameId fid = *(kf_frames.begin());

  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = ba_verbose;

  calib_cam_opt = calib_cam;
  cameras_opt = cameras;
  landmarks_opt = landmarks;

  std::set<FrameCamId> fixed_cameras = {{fid, 0}, {fid, 1}};
  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  compute_projections();
}
