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

#include "cam/camera_models/pinhole_model.h"

#include <boost/serialization/export.hpp>

// BOOST_CLASS_EXPORT_IMPLEMENT(ORB_SLAM_FUSION::Pinhole)

namespace ORB_SLAM_FUSION {
// BOOST_CLASS_EXPORT_GUID(Pinhole, "Pinhole")

long unsigned int GeometricCamera::next_id_ = 0;

cv::Point2f Pinhole::Project(const cv::Point3f &p3d_cv) {
  return cv::Point2f(params_[0] * p3d_cv.x / p3d_cv.z + params_[2],
                     params_[1] * p3d_cv.y / p3d_cv.z + params_[3]);
}

Eigen::Vector2d Pinhole::Project(const Eigen::Vector3d &p3d_eig) {
  Eigen::Vector2d res;
  res[0] = params_[0] * p3d_eig[0] / p3d_eig[2] + params_[2];
  res[1] = params_[1] * p3d_eig[1] / p3d_eig[2] + params_[3];
  return res;
}

Eigen::Vector2f Pinhole::Project(const Eigen::Vector3f &p3d_eig) {
  Eigen::Vector2f res;
  res[0] = params_[0] * p3d_eig[0] / p3d_eig[2] + params_[2];
  res[1] = params_[1] * p3d_eig[1] / p3d_eig[2] + params_[3];
  return res;
}

Eigen::Vector2f Pinhole::ProjectMat(const cv::Point3f &p3d_cv) {
  cv::Point2f point = this->Project(p3d_cv);
  return Eigen::Vector2f(point.x, point.y);
}

float Pinhole::Uncertainty2(const Eigen::Matrix<double, 2, 1> &p2d_eig) {
  return 1.0;
}

Eigen::Vector3f Pinhole::UnprojectEig(const cv::Point2f &p2d_cv) {
  return Eigen::Vector3f((p2d_cv.x - params_[2]) / params_[0],
                         (p2d_cv.y - params_[3]) / params_[1], 1.f);
}

cv::Point3f Pinhole::Unproject(const cv::Point2f &p2d_cv) {
  return cv::Point3f((p2d_cv.x - params_[2]) / params_[0],
                     (p2d_cv.y - params_[3]) / params_[1], 1.f);
}

Eigen::Matrix<double, 2, 3> Pinhole::ProjectJac(const Eigen::Vector3d &p3d_eig) {
  Eigen::Matrix<double, 2, 3> Jac;
  Jac(0, 0) = params_[0] / p3d_eig[2];
  Jac(0, 1) = 0.f;
  Jac(0, 2) = -params_[0] * p3d_eig[0] / (p3d_eig[2] * p3d_eig[2]);
  Jac(1, 0) = 0.f;
  Jac(1, 1) = params_[1] / p3d_eig[2];
  Jac(1, 2) = -params_[1] * p3d_eig[1] / (p3d_eig[2] * p3d_eig[2]);
  return Jac;
}

bool Pinhole::ReconstructWithTwoViews(const std::vector<cv::KeyPoint> &vKeys1,
                                      const std::vector<cv::KeyPoint> &vKeys2,
                                      const std::vector<int> &vMatches12,
                                      Sophus::SE3f &T21,
                                      std::vector<cv::Point3f> &vP3D,
                                      std::vector<bool> &vbTriangulated) {
  if (!tvr) {
    Eigen::Matrix3f K = this->toK_();
    tvr = new TwoViewReconstruction(K);
  }

  return tvr->Reconstruct(vKeys1, vKeys2, vMatches12, T21, vP3D,
                          vbTriangulated);
}

cv::Mat Pinhole::toK() {
  cv::Mat K = (cv::Mat_<float>(3, 3) << params_[0], 0.f, params_[2],
               0.f, params_[1], params_[3], 0.f, 0.f, 1.f);
  return K;
}

Eigen::Matrix3f Pinhole::toK_() {
  Eigen::Matrix3f K;
  K << params_[0], 0.f, params_[2], 0.f, params_[1],
      params_[3], 0.f, 0.f, 1.f;
  return K;
}

bool Pinhole::epipolarConstrain(GeometricCamera *pCamera2,
                                const cv::KeyPoint &kp1,
                                const cv::KeyPoint &kp2,
                                const Eigen::Matrix3f &R12,
                                const Eigen::Vector3f &t12,
                                const float sigmaLevel, const float unc) {
  // Compute Fundamental Matrix
  Eigen::Matrix3f t12x = Sophus::SO3f::hat(t12);
  Eigen::Matrix3f K1 = this->toK_();
  Eigen::Matrix3f K2 = pCamera2->toK_();
  Eigen::Matrix3f F12 = K1.transpose().inverse() * t12x * R12 * K2.inverse();

  // Epipolar line in second image l = x1'F12 = [a b c]
  const float a = kp1.pt.x * F12(0, 0) + kp1.pt.y * F12(1, 0) + F12(2, 0);
  const float b = kp1.pt.x * F12(0, 1) + kp1.pt.y * F12(1, 1) + F12(2, 1);
  const float c = kp1.pt.x * F12(0, 2) + kp1.pt.y * F12(1, 2) + F12(2, 2);

  const float num = a * kp2.pt.x + b * kp2.pt.y + c;

  const float den = a * a + b * b;

  if (den == 0) return false;

  const float dsqr = num * num / den;

  return dsqr < 3.84 * unc;
}

std::ostream &operator<<(std::ostream &os, const Pinhole &ph) {
  os << ph.params_[0] << " " << ph.params_[1] << " "
     << ph.params_[2] << " " << ph.params_[3];
  return os;
}

std::istream &operator>>(std::istream &is, Pinhole &ph) {
  float nextParam;
  for (size_t i = 0; i < 4; i++) {
    assert(is.good());  // Make sure the input stream is good
    is >> nextParam;
    ph.params_[i] = nextParam;
  }
  return is;
}

bool Pinhole::IsEqual(GeometricCamera *cam) {
  if (cam->GetType() != GeometricCamera::kCamPinhole) return false;

  Pinhole *pPinholeCam = (Pinhole *)cam;

  if (size() != pPinholeCam->size()) return false;

  bool is_same_camera = true;
  for (size_t i = 0; i < size(); ++i) {
    if (abs(params_[i] - pPinholeCam->getParameter(i)) > 1e-6) {
      is_same_camera = false;
      break;
    }
  }
  return is_same_camera;
}
}  // namespace ORB_SLAM_FUSION
