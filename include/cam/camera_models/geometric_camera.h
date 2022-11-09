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

#ifndef CAMERAMODELS_GEOMETRICCAMERA_H
#define CAMERAMODELS_GEOMETRICCAMERA_H

#include <Eigen/Geometry>
#include <boost/serialization/access.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sophus/se3.hpp>
#include <vector>

#include "utils/converter.h"
#include "utils/geometric_tools.h"

namespace ORB_SLAM_FUSION {
class GeometricCamera {
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& id_;
    ar& type_;
    ar& params_;
  }

 public:
  GeometricCamera() {}
  GeometricCamera(const std::vector<float>& params) : params_(params) {}
  ~GeometricCamera() {}

  virtual cv::Point2f Project(const cv::Point3f& p3d_cv) = 0;
  virtual Eigen::Vector2d Project(const Eigen::Vector3d& p3d_eig) = 0;
  virtual Eigen::Vector2f Project(const Eigen::Vector3f& p3d_eig) = 0;
  virtual Eigen::Vector2f ProjectMat(const cv::Point3f& p3d_cv) = 0;

  virtual float Uncertainty2(const Eigen::Matrix<double, 2, 1>& p2d_eig) = 0;

  virtual Eigen::Vector3f UnprojectEig(const cv::Point2f& p2d_cv) = 0;
  virtual cv::Point3f Unproject(const cv::Point2f& p2d_cv) = 0;

  virtual Eigen::Matrix<double, 2, 3> ProjectJac(
      const Eigen::Vector3d& p3d_eig) = 0;

  virtual bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& kps_1,
                                       const std::vector<cv::KeyPoint>& kps_2,
                                       const std::vector<int>& matches12,
                                       Sophus::SE3f& T21_so,
                                       std::vector<cv::Point3f>& p3ds_cv,
                                       std::vector<bool>& is_triangulated) = 0;

  virtual cv::Mat ToKCv() = 0;
  virtual Eigen::Matrix3f ToKEig() = 0;

  virtual bool EpipolarConstrain(GeometricCamera* other_cam,
                                 const cv::KeyPoint& kp_1,
                                 const cv::KeyPoint& kp_2,
                                 const Eigen::Matrix3f& R12_eig,
                                 const Eigen::Vector3f& t12_eig,
                                 const float sigma_lev, const float unc) = 0;

  float GetParameter(const size_t i) { return params_[i]; }
  void SetParameter(const float p, const size_t i) { params_[i] = p; }

  size_t Size() { return params_.size(); }

  virtual bool MatchAndTriangulate(
      const cv::KeyPoint& kp_1, const cv::KeyPoint& kp_2,
      GeometricCamera* outher_cam, Sophus::SE3f& Tcw_1_so,
      Sophus::SE3f& Tcw_2_so, const float sigma_lev_1, const float sigma_lev_2,
      Eigen::Vector3f& p3d_eig) = 0;

  unsigned int GetId() { return id_; }

  unsigned int GetType() { return type_; }

  const static unsigned int kCamPinhole = 0;
  const static unsigned int kCamFisheye = 1;

  static long unsigned int next_id_;

 protected:
  std::vector<float> params_;

  unsigned int id_;

  unsigned int type_;
};
}  // namespace ORB_SLAM_FUSION

#endif  // CAMERAMODELS_GEOMETRICCAMERA_H
