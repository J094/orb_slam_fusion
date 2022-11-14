/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez
 * Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ORB_SLAM_FUSION {

class ExtractorNode {
 public:
  ExtractorNode() : no_more_(false) {}

  void DivideNode(ExtractorNode &n_1, ExtractorNode &n_2, ExtractorNode &n_3,
                  ExtractorNode &n_4);

  std::vector<cv::KeyPoint> kps_;
  cv::Point2i UL_, UR_, BL_, BR_;
  std::list<ExtractorNode>::iterator lit_;
  bool no_more_;
};

class OrbExtractor {
 public:
  enum { kHarrisScore = 0, kFastScore = 1 };

  OrbExtractor(int num_feats, float scale_factor, int num_levs, int ini_th_fast,
               int min_th_fast);

  ~OrbExtractor() {}

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  int operator()(cv::InputArray img, cv::InputArray msk,
                 std::vector<cv::KeyPoint> &kps,
                 cv::OutputArray descs, std::vector<int> &lapping_areas);

  int inline GetLevels() { return num_levs_; }

  float inline GetScaleFactor() { return scale_factor_; }

  std::vector<float> inline GetScaleFactors() { return scale_factors_; }

  std::vector<float> inline GetInverseScaleFactors() {
    return inv_scale_factors_;
  }

  std::vector<float> inline GetScaleSigmaSquares() { return lev_sigma_2_; }

  std::vector<float> inline GetInverseScaleSigmaSquares() {
    return inv_lev_sigma_2_;
  }

  std::vector<cv::Mat> img_pyramid_;

  void ComputePyramid(cv::Mat img);
 protected:
  void ComputeKeyPointsOctTree(
      std::vector<std::vector<cv::KeyPoint> > &all_kps);
  std::vector<cv::KeyPoint> DistributeOctTree(
      const std::vector<cv::KeyPoint> &kps, const int &min_x, const int &max_x,
      const int &min_y, const int &max_y, const int &num_feats, const int &lev);

  void ComputeKeyPointsOld(
      std::vector<std::vector<cv::KeyPoint> > &all_kps);
  std::vector<cv::Point> patterns_;

  int num_feats_;
  double scale_factor_;
  int num_levs_;
  int ini_th_fast_;
  int min_th_fast_;

  std::vector<int> num_feats_per_lev_;

  std::vector<int> umax_;

  std::vector<float> scale_factors_;
  std::vector<float> inv_scale_factors_;
  std::vector<float> lev_sigma_2_;
  std::vector<float> inv_lev_sigma_2_;
};

}  // namespace ORB_SLAM_FUSION

#endif
