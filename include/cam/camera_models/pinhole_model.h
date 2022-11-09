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
      : GeometricCamera(_vParameters), tvr(nullptr) {
    assert(params_.size() == 4);
    id_ = next_id_++;
    type_ = kCamPinhole;
  }

  Pinhole(Pinhole* pPinhole)
      : GeometricCamera(pPinhole->params_), tvr(nullptr) {
    assert(params_.size() == 4);
    id_ = next_id_++;
    type_ = kCamPinhole;
  }

  ~Pinhole() {
    if (tvr) delete tvr;
  }

  cv::Point2f Project(const cv::Point3f& p3d_cv);
  Eigen::Vector2d Project(const Eigen::Vector3d& p3d_eig);
  Eigen::Vector2f Project(const Eigen::Vector3f& p3d_eig);
  Eigen::Vector2f ProjectMat(const cv::Point3f& p3d_cv);

  float Uncertainty2(const Eigen::Matrix<double, 2, 1>& p2d_eig);

  Eigen::Vector3f UnprojectEig(const cv::Point2f& p2d_cv);
  cv::Point3f Unproject(const cv::Point2f& p2d_cv);

  Eigen::Matrix<double, 2, 3> ProjectJac(const Eigen::Vector3d& p3d_eig);

  bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1,
                               const std::vector<cv::KeyPoint>& vKeys2,
                               const std::vector<int>& vMatches12,
                               Sophus::SE3f& T21,
                               std::vector<cv::Point3f>& vP3D,
                               std::vector<bool>& vbTriangulated);

  cv::Mat toK();
  Eigen::Matrix3f toK_();

  bool epipolarConstrain(GeometricCamera* pCamera2, const cv::KeyPoint& kp1,
                         const cv::KeyPoint& kp2, const Eigen::Matrix3f& R12,
                         const Eigen::Vector3f& t12, const float sigmaLevel,
                         const float unc);

  bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
                           GeometricCamera* pOther, Sophus::SE3f& Tcw1,
                           Sophus::SE3f& Tcw2, const float sigmaLevel1,
                           const float sigmaLevel2,
                           Eigen::Vector3f& x3Dtriangulated) {
    return false;
  }

  friend std::ostream& operator<<(std::ostream& os, const Pinhole& ph);
  friend std::istream& operator>>(std::istream& os, Pinhole& ph);

  bool IsEqual(GeometricCamera* cam);

 private:
  // Parameters vector corresponds to
  //       [fx, fy, cx, cy]
  TwoViewReconstruction* tvr;
};
}  // namespace ORB_SLAM_FUSION

// BOOST_CLASS_EXPORT_KEY(ORBSLAM2::Pinhole)

#endif  // CAMERAMODELS_PINHOLE_H
