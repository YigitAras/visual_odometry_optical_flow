// What do ?
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

extern Sophus::SE3d running_TWC;
void matchOptFlow(const pangolin::ManagedImage<uint8_t>& currImgL,
                  const pangolin::ManagedImage<uint8_t>& currImgR,
                  // for now no references consider referencing
                  KeypointsData &currKDL, KeypointsData &currKDR, MatchData& md,
                  double ransac_thresh, int ransac_min_inliers,
                  const Calibration& calib_cam, double threshold) {
  cv::Mat imageL(currImgL.h, currImgL.w, CV_8U, currImgL.ptr);
  cv::Mat imageR(currImgR.h, currImgR.w, CV_8U, currImgR.ptr);

  md.matches.clear();
  md.inliers.clear();
  std::vector<float> err;
  std::vector<uchar> status;
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
  //  cv::Mat out;
  //  cv::hconcat(prevImageL, imageL,
  //              out);  // Syntax-> hconcat(source1,source2,destination);
  //  int checkyboi = 0;
  //  for (int i = 0; i < currPointsL.size(); i++) {
  //    if (checkyboi % 5 == 0) {
  //      cv::Point src((int)prevPointsL[i].x, (int)prevPointsL[i].y);
  //      cv::Point trgt(currPointsL[i].x + imageL.cols, currPointsL[i].y);

  //      int thickness = 1;
  //      int lineType = cv::LINE_8;
  //      cv::line(out, src, trgt, cv::Scalar(255, 0, 0), thickness, lineType);
  //    }
  //    checkyboi++;
  //  }

  //   cv::imshow("ANAN", out);
  //   cv::waitKey(1);
  //   std::cout << prevPointsL.size() << " and " << testL.size() << std::endl;

  for (int i = 0; i < testL.size(); i++) {
    // std::cout << testL[i].x << " " << testL[i].y << std::endl;
    // std::cout << prevPointsL[i].x << " " << prevPointsL[i].y << std::endl;
    // std::cout << "------------------------" << std::endl;

    Eigen::Vector2f diff(abs((testL[i] - prevPointsL[i]).x),
                         abs((testL[i] - prevPointsL[i]).y));
    Eigen::Vector2d tempy(round(currPointsL[i].x), round(currPointsL[i].y));

    if ((diff.allFinite()) && (tempy.x() >= 0) && (tempy.y() >= 0) && ((tempy.y() < imageL.rows &&  tempy.x() < imageL.cols) )&&
        (diff.norm() < threshold)) {
      //      md.matches.push_back(std::make_pair(i, i));
      currKDL.corners.push_back(tempy);
    }
  }
  std::cout << "NUM KEYPOINTS T-1 to T " << currKDL.corners.size() << " TOTAL "
            << testL.size() << std::endl;
}

// TODO: Make this KD reference
void matchLeftRightOptFlow(const pangolin::ManagedImage<uint8_t>& currImgL,
                           const pangolin::ManagedImage<uint8_t>& currImgR,
                           KeypointsData &currKDL, KeypointsData &currKDR,
                           std::vector<std::pair<int, int>> &matches,
                           double threshold) {
  std::vector<cv::Point2f> currPointsL, currPointsR, currPointsTemp;
  for (auto& c : currKDL.corners) {
    currPointsL.push_back(cv::Point2f(c.x(), c.y()));
  }

  std::vector<float> err;
  std::vector<uchar> status;
  cv::TermCriteria criteria = cv::TermCriteria(
      (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.003);
  cv::Mat imageL(currImgL.h, currImgL.w, CV_8U, currImgL.ptr);
  cv::Mat imageR(currImgR.h, currImgR.w, CV_8U, currImgR.ptr);
  currKDR.corners.clear();
  matches.clear();
  cv::calcOpticalFlowPyrLK(imageL, imageR, currPointsL, currPointsR, status,
                           err, cv::Size(5, 5), 2, criteria);
  cv::calcOpticalFlowPyrLK(imageR, imageL, currPointsR, currPointsTemp,
                           status, err, cv::Size(5, 5), 2, criteria);

  for (int i = 0; i < currPointsTemp.size(); i++) {
    // std::cout << testL[i].x << " " << testL[i].y << std::endl;
    // std::cout << prevPointsL[i].x << " " << prevPointsL[i].y << std::endl;
    // std::cout << "------------------------" << std::endl;

    Eigen::Vector2f diff(abs((currPointsTemp[i] - currPointsL[i]).x),
                         abs((currPointsTemp[i] - currPointsL[i]).y));
    Eigen::Vector2d tempy(round(currPointsR[i].x), round(currPointsR[i].y));

    currKDR.corners.push_back(tempy);
    if ((diff.allFinite()) && (tempy.x() >= 0) && (tempy.y() >= 0) && ((tempy.y() < imageL.rows &&  tempy.x() < imageL.cols) )&&
        (diff.norm() < threshold)) {
      matches.push_back(std::make_pair(i, i));
    }
  }
  // cv::Mat out;
  // cv::hconcat(imageL, imageR,
  //             out);  // Syntax-> hconcat(source1,source2,destination);
  // int checkyboi = 0;
  // for (int i = 0; i < matches.size(); i++) {
  //   if (checkyboi % 20 == 0) { 
  //     cv::Point src((int)currPointsL[matches[i].first].x, (int)currPointsL[matches[i].first].y);
  //     cv::Point trgt(currPointsR[matches[i].second].x + imageL.cols, currPointsR[matches[i].second].y);

  //     int thickness = 1;
  //     int lineType = cv::LINE_8;
  //     cv::line(out, src, trgt, cv::Scalar(255, 0, 0), thickness, lineType);
  //   }
  //   checkyboi++;
  // }

  // cv::imshow("DEDEN", out);
  // cv::waitKey(1);
  //std::cout << currPointsR<< " and " << testL.size() << std::endl;

  // TODO Matching
  // TODO Profit
}
