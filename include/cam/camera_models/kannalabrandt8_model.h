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

#ifndef CAMERAMODELS_KANNALABRANDT8_H
#define CAMERAMODELS_KANNALABRANDT8_H

#include <assert.h>

#include "cam/camera_models/geometric_camera.h"
#include "cam/two_view_reconstruction.h"

namespace ORB_SLAM_FUSION {
class KannalaBrandt8 : public GeometricCamera {
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& boost::serialization::base_object<GeometricCamera>(*this);
    ar& const_cast<float&>(precision_);
  }

 public:
  KannalaBrandt8() : precision_(1e-6) {
    params_.resize(8);
    id_ = next_id_++;
    type_ = kCamFisheye;
  }
  KannalaBrandt8(const std::vector<float> _vParameters)
      : GeometricCamera(_vParameters),
        precision_(1e-6),
        lapping_areas_(2, 0),
        tvr_(nullptr) {
    assert(params_.size() == 8);
    id_ = next_id_++;
    type_ = kCamFisheye;
  }

  KannalaBrandt8(const std::vector<float> _vParameters, const float _precision)
      : GeometricCamera(_vParameters),
        precision_(_precision),
        lapping_areas_(2, 0) {
    assert(params_.size() == 8);
    id_ = next_id_++;
    type_ = kCamFisheye;
  }
  KannalaBrandt8(KannalaBrandt8* pKannala)
      : GeometricCamera(pKannala->params_),
        precision_(pKannala->precision_),
        lapping_areas_(2, 0),
        tvr_(nullptr) {
    assert(params_.size() == 8);
    id_ = next_id_++;
    type_ = kCamFisheye;
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

  float TriangulateMatches(GeometricCamera* cam_2, const cv::KeyPoint& kp_1,
                           const cv::KeyPoint& kp_2, const Eigen::Matrix3f& R12_eig,
                           const Eigen::Vector3f& t12_eig, const float sigma_lev,
                           const float unc, Eigen::Vector3f& p3d_eig);

  std::vector<int> lapping_areas_;

  bool MatchAndTriangulate(const cv::KeyPoint& kp_1, const cv::KeyPoint& kp_2,
                           GeometricCamera* cam_2, Sophus::SE3f& Tcw_1_so,
                           Sophus::SE3f& Tcw_2_so, const float sigma_lev_1,
                           const float sigma_lev_2,
                           Eigen::Vector3f& p3d_eig);

  friend std::ostream& operator<<(std::ostream& os, const KannalaBrandt8& kb);
  friend std::istream& operator>>(std::istream& is, KannalaBrandt8& kb);

  float GetPrecision() { return precision_; }

  bool IsEqual(GeometricCamera* cam);

 private:
  const float precision_;

  // Parameters vector corresponds to
  //[fx, fy, cx, cy, k0, k1, k2, k3]

  TwoViewReconstruction* tvr_;

  void Triangulate(const cv::Point2f& p2d_1_cv, const cv::Point2f& p2d_2_cv,
                   const Eigen::Matrix<float, 3, 4>& Tcw_1_eig,
                   const Eigen::Matrix<float, 3, 4>& Tcw_2_eig,
                   Eigen::Vector3f& p3d_eig);
};
}  // namespace ORB_SLAM_FUSION

#endif  // CAMERAMODELS_KANNALABRANDT8_H
