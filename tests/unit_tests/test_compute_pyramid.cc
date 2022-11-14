#include "cam/orb_feature/orb_extractor.h"

int main(int argc, char **argv) {
  ORB_SLAM_FUSION::OrbExtractor* orb_extractor = new ORB_SLAM_FUSION::OrbExtractor(1000, 1.2, 6, 20, 7);
  cv::Mat img_in;
  img_in = cv::imread("/home/bynav/gj/code/feature_detection/park_1.JPG");
  cv::Size sz(1280, 960);
  cv::Mat img_out;
  cv::resize(img_in, img_out, sz);
  orb_extractor->ComputePyramid(img_out);
  cv::imshow("level 0", orb_extractor->img_pyramid_[0]);
  cv::imshow("level 1", orb_extractor->img_pyramid_[1]);
  cv::imshow("level 2", orb_extractor->img_pyramid_[2]);
  cv::imshow("level 3", orb_extractor->img_pyramid_[3]);
  cv::imshow("level 4", orb_extractor->img_pyramid_[4]);
  cv::imshow("level 5", orb_extractor->img_pyramid_[5]);
  cv::waitKey(0);
}