// What do ?
// TODO: Fuck bitches Implement shit

#include <visnav/common_types.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
using namespace visnav;

extern cv::Mat prevImageR;
extern cv::Mat prevImageL;

extern KeypointsData prevKDL;
extern KeypointsData prevKDR;

void matchOptFlow(const pangolin::ManagedImage<uint8_t>& currImgL,
                  const pangolin::ManagedImage<uint8_t>& currImgR,
                  // for now no references consider referencing
                  KeypointsData currKDL, KeypointsData currKDR,
                  MatchData &md, int threshold,
                  double dist_2_best) {
  cv::Mat imageL(currImgL.h, currImgL.w, CV_8U, currImgL.ptr);
  cv::Mat imageR(currImgR.h, currImgR.w, CV_8U, currImgR.ptr);
  std::vector<uchar> status;
  md.matches.clear();
  md.inliers.clear();
  std::vector<float> err;
  currKDL.corners.clear();
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.003);
  std::vector<cv::Point2f> prevPointsL, currPointsL, testL;

  for (auto& c : prevKDL.corners) {
    prevPointsL.push_back(cv::Point2f(c.x(), c.y()));
  }

  cv::calcOpticalFlowPyrLK(prevImageL, imageL, prevPointsL, currPointsL, status,
                           err, cv::Size(5, 5), 2, criteria);
  cv::calcOpticalFlowPyrLK(imageL, prevImageL, currPointsL, testL, status, err,
                           cv::Size(5, 5), 2, criteria);
  cv::Mat out;
  cv::hconcat(prevImageL, imageL,
              out);  // Syntax-> hconcat(source1,source2,destination);
  int checkyboi = 0;
  for (int i = 0; i < currPointsL.size(); i++) {
    if(checkyboi % 5 == 0){
    cv::Point src((int)prevPointsL[i].x, (int)prevPointsL[i].y);
    cv::Point trgt(currPointsL[i].x + imageL.cols, currPointsL[i].y);

    int thickness = 1;
    int lineType = cv::LINE_8;
    cv::line(out, src, trgt, cv::Scalar(255, 0, 0), thickness, lineType);
    }
    checkyboi++;
  }

//   cv::imshow("ANAN", out);
//   cv::waitKey(0);
//   std::cout << prevPointsL.size() << " and " << testL.size() << std::endl;

  for (int i = 0; i < testL.size(); i++) {
    // std::cout << testL[i].x << " " << testL[i].y << std::endl;
    // std::cout << prevPointsL[i].x << " " << prevPointsL[i].y << std::endl;
    // std::cout << "------------------------" << std::endl;


    Eigen::Vector2f diff((testL[i] - prevPointsL[i]).x,(testL[i] - prevPointsL[i]).y);
    Eigen::Vector2d tempy(round(currPointsL[i].x),round(currPointsL[i].y));
    currKDL.corners.push_back(tempy);
    //NANLAR VAR CHECK ET
    if(diff.norm() < 5){

        md.matches.push_back(std::make_pair(i,i));
    }
    

    //if (i > 20) break;
  }
  std::cout << "NUM MATCHES " << md.matches.size()  << " TOTAL " << testL.size()<< std::endl;

//    md.T_i_j = Sophus::SE3d();

//   // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
//   // the final inlier indices in md.inliers and the final relative pose in
//   // md.T_i_j (normalize translation). If the number of inliers is smaller than
//   // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
//   // was successful, you should do non-linear refinement of the model parameters
//   // using all inliers, and then re-estimate the inlier set with the refined
//   // model parameters.
//   opengv::bearingVectors_t bear1;
//   opengv::bearingVectors_t bear2;

//   for (long unsigned int i = 0; i < md.matches.size(); i++) {
//     bear1.push_back(cam1->unproject(kd1.corners[md.matches[i].first]));
//     bear2.push_back(cam2->unproject(kd2.corners[md.matches[i].second]));
//   }
//   opengv::relative_pose::CentralRelativeAdapter adapter(bear1, bear2);
//   opengv::sac::Ransac<
//       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
//       ransac;
//   std::shared_ptr<
//       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
//       relposeproblem_ptr(
//           new opengv::sac_problems::relative_pose::
//               CentralRelativePoseSacProblem(
//                   adapter, opengv::sac_problems::relative_pose::
//                                CentralRelativePoseSacProblem::NISTER));
//   // 0.92 > 0.99
//   ransac.sac_model_ = relposeproblem_ptr;
//   ransac.threshold_ = ransac_thresh;
//   ransac.computeModel();

//   opengv::transformation_t optimized =
//       opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
//   ransac.sac_model_->selectWithinDistance(optimized, ransac_thresh,
//                                           ransac.inliers_);
//   Eigen::Matrix4d res;
//   res.block<3, 3>(0, 0) = optimized.block<3, 3>(0, 0);
//   res.block<3, 1>(0, 3) = optimized.block<3, 1>(0, 3).normalized();
//   res.block<1, 4>(3, 0) = Eigen::Vector4d(0, 0, 0, 1);
//   md.T_i_j = Sophus::SE3d(res);

//   if ((int)ransac.inliers_.size() > ransac_min_inliers) {
//     for (long unsigned int i = 0; i < ransac.inliers_.size(); i++) {
//       md.inliers.push_back(md.matches[ransac.inliers_[i]]);
//     }
//   }
}