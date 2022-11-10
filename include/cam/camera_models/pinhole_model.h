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

#ifndef CAMERAMODELS_PINHOLE_H
#define CAMERAMODELS_PINHOLE_H

#include <assert.h>

#include "cam/camera_models/geometric_camera.h"
#include "cam/two_view_reconstruction.h"

namespace ORB_SLAM_FUSION {
class Pinhole : public GeometricCamera {
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& boost::serialization::base_object<GeometricCamera>(*this);
  }

 public:
  Pinhole() {
    params_.resize(4);
    id_ = next_id_++;
    type_ = kCamPinhole;
  }
  Pinhole(const std::vector<float> _vParameters)
      : GeometricCamera(_vParameters), tvr_(nullptr) {
    assert(params_.size() == 4);
    id_ = next_id_++;
    type_ = kCamPinhole;
  }

  Pinhole(Pinhole* pPinhole)
      : GeometricCamera(pPinhole->params_), tvr_(nullptr) {
    assert(params_.size() == 4);
    id_ = next_id_++;
    type_ = kCamPinhole;
  }

  ~Pinhole() {
    if (tvr_) delete tvr_;
  }

  cv::Point2f Project(const cv::Point3f& p3d_cv);
  Eigen::Vector2d Project(const Eigen::Vector3d& p3d_eig);
  Eigen::Vector2f Project(const Eigen::Vector3f& p3d_eig);
  Eigen::Vector2f ProjectMat(const cv::Point3f& p3d_cv);

  float Uncertainty2(const Eigen::Matrix<double, 2, 1>& p2d_eig);

  Eigen::Vector3f UnprojectEig(const cv::Point2f& p2d_cv);
  cv::Point3f Unproject(const cv::Point2f& p2d_cv);

  Eigen::Matrix<double, 2, 3> ProjectJac(const Eigen::Vector3d& p3d_eig);

  bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& kps_1,
                               const std::vector<cv::KeyPoint>& kps_2,
                               const std::vector<int>& matches,
                               Sophus::SE3f& T21_so,
                               std::vector<cv::Point3f>& p3ds_cv,
                               std::vector<bool>& is_triangulated);

  cv::Mat ToKCv();
  Eigen::Matrix3f ToKEig();

  bool EpipolarConstrain(GeometricCamera* cam_2, const cv::KeyPoint& kp_1,
                         const cv::KeyPoint& kp_2, const Eigen::Matrix3f& R12_eig,
                         const Eigen::Vector3f& t12_eig, const float sigma_lev,
                         const float unc);

  bool MatchAndTriangulate(const cv::KeyPoint& kp_1, const cv::KeyPoint& kp_2,
                           GeometricCamera* cam_2, Sophus::SE3f& Tcw_1_so,
                           Sophus::SE3f& Tcw_2_so, const float sigma_lev_1,
                           const float sigma_lev_2,
                           Eigen::Vector3f& p3d_eig) {
    return false;
  }

  friend std::ostream& operator<<(std::ostream& os, const Pinhole& ph);
  friend std::istream& operator>>(std::istream& os, Pinhole& ph);

  bool IsEqual(GeometricCamera* cam);

 private:
  // Parameters vector corresponds to
  //       [fx, fy, cx, cy]
  TwoViewReconstruction* tvr_;
};
}  // namespace ORB_SLAM_FUSION

// BOOST_CLASS_EXPORT_KEY(ORBSLAM2::Pinhole)

#endif  // CAMERAMODELS_PINHOLE_H
