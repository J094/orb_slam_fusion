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

#include "cam/camera_models/kannalabrandt8_model.h"

#include <boost/serialization/export.hpp>

// BOOST_CLASS_EXPORT_IMPLEMENT(ORB_SLAM_FUSION::KannalaBrandt8)

namespace ORB_SLAM_FUSION {
// BOOST_CLASS_EXPORT_GUID(KannalaBrandt8, "KannalaBrandt8")

cv::Point2f KannalaBrandt8::Project(const cv::Point3f &p3d_cv) {
  const float x2_plus_y2 = p3d_cv.x * p3d_cv.x + p3d_cv.y * p3d_cv.y;
  const float theta = atan2f(sqrtf(x2_plus_y2), p3d_cv.z);
  const float psi = atan2f(p3d_cv.y, p3d_cv.x);

  const float theta2 = theta * theta;
  const float theta3 = theta * theta2;
  const float theta5 = theta3 * theta2;
  const float theta7 = theta5 * theta2;
  const float theta9 = theta7 * theta2;
  const float r = theta + params_[4] * theta3 + params_[5] * theta5 +
                  params_[6] * theta7 + params_[7] * theta9;
  return cv::Point2f(params_[0] * r * cos(psi) + params_[2],
                     params_[1] * r * sin(psi) + params_[3]);
}

Eigen::Vector2d KannalaBrandt8::Project(const Eigen::Vector3d &p3d_eig) {
  const double x2_plus_y2 = p3d_eig[0] * p3d_eig[0] + p3d_eig[1] * p3d_eig[1];
  const double theta = atan2f(sqrtf(x2_plus_y2), p3d_eig[2]);
  const double psi = atan2f(p3d_eig[1], p3d_eig[0]);

  const double theta2 = theta * theta;
  const double theta3 = theta * theta2;
  const double theta5 = theta3 * theta2;
  const double theta7 = theta5 * theta2;
  const double theta9 = theta7 * theta2;
  const double r = theta + params_[4] * theta3 + params_[5] * theta5 +
                   params_[6] * theta7 + params_[7] * theta9;
  Eigen::Vector2d res;
  res[0] = params_[0] * r * cos(psi) + params_[2];
  res[1] = params_[1] * r * sin(psi) + params_[3];
  return res;
}

Eigen::Vector2f KannalaBrandt8::Project(const Eigen::Vector3f &p3d_eig) {
  const float x2_plus_y2 = p3d_eig[0] * p3d_eig[0] + p3d_eig[1] * p3d_eig[1];
  const float theta = atan2f(sqrtf(x2_plus_y2), p3d_eig[2]);
  const float psi = atan2f(p3d_eig[1], p3d_eig[0]);

  const float theta2 = theta * theta;
  const float theta3 = theta * theta2;
  const float theta5 = theta3 * theta2;
  const float theta7 = theta5 * theta2;
  const float theta9 = theta7 * theta2;
  const float r = theta + params_[4] * theta3 + params_[5] * theta5 +
                  params_[6] * theta7 + params_[7] * theta9;
  Eigen::Vector2f res;
  res[0] = params_[0] * r * cos(psi) + params_[2];
  res[1] = params_[1] * r * sin(psi) + params_[3];
  return res;

  /*cv::Point2f cvres = this->Project(cv::Point3f(p3d_eig[0],p3d_eig[1],p3d_eig[2]));

  Eigen::Vector2d res;
  res[0] = cvres.x;
  res[1] = cvres.y;

  return res;*/
}

Eigen::Vector2f KannalaBrandt8::ProjectMat(const cv::Point3f &p3d_cv) {
  cv::Point2f point = this->Project(p3d_cv);
  return Eigen::Vector2f(point.x, point.y);
}

float KannalaBrandt8::Uncertainty2(const Eigen::Matrix<double, 2, 1> &p2d_eig) {
  /*Eigen::Matrix<double,2,1> c;
  c << params_[2], params_[3];
  if ((p2d_cv-c).squaredNorm()>57600) // 240*240 (256)
      return 100.f;
  else
      return 1.0f;*/
  return 1.f;
}

Eigen::Vector3f KannalaBrandt8::UnprojectEig(const cv::Point2f &p2d_cv) {
  cv::Point3f ray = this->Unproject(p2d_cv);
  return Eigen::Vector3f(ray.x, ray.y, ray.z);
}

cv::Point3f KannalaBrandt8::Unproject(const cv::Point2f &p2d_cv) {
  // Use Newthon method to solve for theta with good precision (err ~ e-6)
  cv::Point2f pw((p2d_cv.x - params_[2]) / params_[0],
                 (p2d_cv.y - params_[3]) / params_[1]);
  float scale = 1.f;
  float theta_d = sqrtf(pw.x * pw.x + pw.y * pw.y);
  theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);

  if (theta_d > 1e-8) {
    // Compensate distortion iteratively
    float theta = theta_d;

    for (int j = 0; j < 10; j++) {
      float theta2 = theta * theta, theta4 = theta2 * theta2,
            theta6 = theta4 * theta2, theta8 = theta4 * theta4;
      float k0_theta2 = params_[4] * theta2,
            k1_theta4 = params_[5] * theta4;
      float k2_theta6 = params_[6] * theta6,
            k3_theta8 = params_[7] * theta8;
      float theta_fix =
          (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) -
           theta_d) /
          (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8);
      theta = theta - theta_fix;
      if (fabsf(theta_fix) < precision_) break;
    }
    // scale = theta - theta_d;
    scale = std::tan(theta) / theta_d;
  }

  return cv::Point3f(pw.x * scale, pw.y * scale, 1.f);
}

Eigen::Matrix<double, 2, 3> KannalaBrandt8::ProjectJac(
    const Eigen::Vector3d &p3d_eig) {
  double x2 = p3d_eig[0] * p3d_eig[0], y2 = p3d_eig[1] * p3d_eig[1], z2 = p3d_eig[2] * p3d_eig[2];
  double r2 = x2 + y2;
  double r = sqrt(r2);
  double r3 = r2 * r;
  double theta = atan2(r, p3d_eig[2]);

  double theta2 = theta * theta, theta3 = theta2 * theta;
  double theta4 = theta2 * theta2, theta5 = theta4 * theta;
  double theta6 = theta2 * theta4, theta7 = theta6 * theta;
  double theta8 = theta4 * theta4, theta9 = theta8 * theta;

  double f = theta + theta3 * params_[4] + theta5 * params_[5] +
             theta7 * params_[6] + theta9 * params_[7];
  double fd = 1 + 3 * params_[4] * theta2 + 5 * params_[5] * theta4 +
              7 * params_[6] * theta6 + 9 * params_[7] * theta8;

  Eigen::Matrix<double, 2, 3> JacGood;
  JacGood(0, 0) =
      params_[0] * (fd * p3d_eig[2] * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
  JacGood(1, 0) =
      params_[1] * (fd * p3d_eig[2] * p3d_eig[1] * p3d_eig[0] / (r2 * (r2 + z2)) -
                         f * p3d_eig[1] * p3d_eig[0] / r3);

  JacGood(0, 1) =
      params_[0] * (fd * p3d_eig[2] * p3d_eig[1] * p3d_eig[0] / (r2 * (r2 + z2)) -
                         f * p3d_eig[1] * p3d_eig[0] / r3);
  JacGood(1, 1) =
      params_[1] * (fd * p3d_eig[2] * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

  JacGood(0, 2) = -params_[0] * fd * p3d_eig[0] / (r2 + z2);
  JacGood(1, 2) = -params_[1] * fd * p3d_eig[1] / (r2 + z2);

  return JacGood;
}

bool KannalaBrandt8::ReconstructWithTwoViews(
    const std::vector<cv::KeyPoint> &kps_1,
    const std::vector<cv::KeyPoint> &kps_2, const std::vector<int> &matches,
    Sophus::SE3f &T21_so, std::vector<cv::Point3f> &p3ds_cv,
    std::vector<bool> &is_triangulated) {
  if (!tvr_) {
    Eigen::Matrix3f K = this->ToKEig();
    tvr_ = new TwoViewReconstruction(K);
  }

  // Correct FishEye distortion
  std::vector<cv::KeyPoint> vKeysUn1 = kps_1, vKeysUn2 = kps_2;
  std::vector<cv::Point2f> vPts1(kps_1.size()), vPts2(kps_2.size());

  for (size_t i = 0; i < kps_1.size(); i++) vPts1[i] = kps_1[i].pt;
  for (size_t i = 0; i < kps_2.size(); i++) vPts2[i] = kps_2[i].pt;

  cv::Mat D =
      (cv::Mat_<float>(4, 1) << params_[4], params_[5], params_[6], params_[7]);
  cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat K = this->ToKCv();
  cv::fisheye::undistortPoints(vPts1, vPts1, K, D, R, K);
  cv::fisheye::undistortPoints(vPts2, vPts2, K, D, R, K);

  for (size_t i = 0; i < kps_1.size(); i++) vKeysUn1[i].pt = vPts1[i];
  for (size_t i = 0; i < kps_2.size(); i++) vKeysUn2[i].pt = vPts2[i];

  return tvr_->Reconstruct(kps_1, kps_2, matches, T21_so, p3ds_cv,
                          is_triangulated);
}

cv::Mat KannalaBrandt8::ToKCv() {
  cv::Mat K = (cv::Mat_<float>(3, 3) << params_[0], 0.f, params_[2],
               0.f, params_[1], params_[3], 0.f, 0.f, 1.f);
  return K;
}
Eigen::Matrix3f KannalaBrandt8::ToKEig() {
  Eigen::Matrix3f K;
  K << params_[0], 0.f, params_[2], 0.f, params_[1],
      params_[3], 0.f, 0.f, 1.f;
  return K;
}

bool KannalaBrandt8::EpipolarConstrain(
    GeometricCamera *cam_2, const cv::KeyPoint &kp_1, const cv::KeyPoint &kp_2,
    const Eigen::Matrix3f &R12_eig, const Eigen::Vector3f &t12_eig,
    const float sigma_lev, const float unc) {
  Eigen::Vector3f p3d_eig;
  return this->TriangulateMatches(cam_2, kp_1, kp_2, R12_eig, t12_eig, sigma_lev, unc,
                                  p3d_eig) > 0.0001f;
}

bool KannalaBrandt8::MatchAndTriangulate(
    const cv::KeyPoint &kp_1, const cv::KeyPoint &kp_2, GeometricCamera *cam_2,
    Sophus::SE3f &Tcw_1_so, Sophus::SE3f &Tcw_2_so, const float sigmaLevel1,
    const float sigmaLevel2, Eigen::Vector3f &x3Dtriangulated) {
  Eigen::Matrix<float, 3, 4> eigTcw1 = Tcw_1_so.matrix3x4();
  Eigen::Matrix3f Rcw1 = eigTcw1.block<3, 3>(0, 0);
  Eigen::Matrix3f Rwc1 = Rcw1.transpose();
  Eigen::Matrix<float, 3, 4> eigTcw2 = Tcw_2_so.matrix3x4();
  Eigen::Matrix3f Rcw2 = eigTcw2.block<3, 3>(0, 0);
  Eigen::Matrix3f Rwc2 = Rcw2.transpose();

  cv::Point3f ray1c = this->Unproject(kp_1.pt);
  cv::Point3f ray2c = cam_2->Unproject(kp_2.pt);

  Eigen::Vector3f r1(ray1c.x, ray1c.y, ray1c.z);
  Eigen::Vector3f r2(ray2c.x, ray2c.y, ray2c.z);

  // Check parallax between rays
  Eigen::Vector3f ray1 = Rwc1 * r1;
  Eigen::Vector3f ray2 = Rwc2 * r2;

  const float cosParallaxRays = ray1.dot(ray2) / (ray1.norm() * ray2.norm());

  // If parallax is lower than 0.9998, reject this match
  if (cosParallaxRays > 0.9998) {
    return false;
  }

  // Parallax is good, so we try to triangulate
  cv::Point2f p11, p22;

  p11.x = ray1c.x;
  p11.y = ray1c.y;

  p22.x = ray2c.x;
  p22.y = ray2c.y;

  Eigen::Vector3f x3D;

  Triangulate(p11, p22, eigTcw1, eigTcw2, x3D);

  // Check triangulation in front of cameras
  float z1 = Rcw1.row(2).dot(x3D) + Tcw_1_so.translation()(2);
  if (z1 <= 0) {  // Point is not in front of the first camera
    return false;
  }

  float z2 = Rcw2.row(2).dot(x3D) + Tcw_2_so.translation()(2);
  if (z2 <= 0) {  // Point is not in front of the first camera
    return false;
  }

  // Check reprojection error in first keyframe
  //   -Transform point into camera reference system
  Eigen::Vector3f x3D1 = Rcw1 * x3D + Tcw_1_so.translation();
  Eigen::Vector2f uv1 = this->Project(x3D1);

  float errX1 = uv1(0) - kp_1.pt.x;
  float errY1 = uv1(1) - kp_1.pt.y;

  if ((errX1 * errX1 + errY1 * errY1) >
      5.991 * sigmaLevel1) {  // Reprojection error is high
    return false;
  }

  // Check reprojection error in second keyframe;
  //   -Transform point into camera reference system
  Eigen::Vector3f x3D2 = Rcw2 * x3D + Tcw_2_so.translation();  // avoid using q
  Eigen::Vector2f uv2 = cam_2->Project(x3D2);

  float errX2 = uv2(0) - kp_2.pt.x;
  float errY2 = uv2(1) - kp_2.pt.y;

  if ((errX2 * errX2 + errY2 * errY2) >
      5.991 * sigmaLevel2) {  // Reprojection error is high
    return false;
  }

  // Since parallax is big enough and reprojection errors are low, this pair of
  // points can be considered as a match
  x3Dtriangulated = x3D;

  return true;
}

float KannalaBrandt8::TriangulateMatches(
    GeometricCamera *cam_2, const cv::KeyPoint &kp_1, const cv::KeyPoint &kp_2,
    const Eigen::Matrix3f &R12_eig, const Eigen::Vector3f &t12_eig,
    const float sigma_lev, const float unc, Eigen::Vector3f &p3d_eig) {
  Eigen::Vector3f r1 = this->UnprojectEig(kp_1.pt);
  Eigen::Vector3f r2 = cam_2->UnprojectEig(kp_2.pt);

  // Check parallax
  Eigen::Vector3f r21 = R12_eig * r2;

  const float cosParallaxRays = r1.dot(r21) / (r1.norm() * r21.norm());

  if (cosParallaxRays > 0.9998) {
    return -1;
  }

  // Parallax is good, so we try to triangulate
  cv::Point2f p11, p22;

  p11.x = r1[0];
  p11.y = r1[1];

  p22.x = r2[0];
  p22.y = r2[1];

  Eigen::Vector3f x3D;
  Eigen::Matrix<float, 3, 4> Tcw_1_eig;
  Tcw_1_eig << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

  Eigen::Matrix<float, 3, 4> Tcw_2_eig;

  Eigen::Matrix3f R21 = R12_eig.transpose();
  Tcw_2_eig << R21, -R21 * t12_eig;

  Triangulate(p11, p22, Tcw_1_eig, Tcw_2_eig, x3D);
  // cv::Mat x3Dt = x3D.t();

  float z1 = x3D(2);
  if (z1 <= 0) {
    return -2;
  }

  float z2 = R21.row(2).dot(x3D) + Tcw_2_eig(2, 3);
  if (z2 <= 0) {
    return -3;
  }

  // Check reprojection error
  Eigen::Vector2f uv1 = this->Project(x3D);

  float errX1 = uv1(0) - kp_1.pt.x;
  float errY1 = uv1(1) - kp_1.pt.y;

  if ((errX1 * errX1 + errY1 * errY1) >
      5.991 * sigma_lev) {  // Reprojection error is high
    return -4;
  }

  Eigen::Vector3f x3D2 = R21 * x3D + Tcw_2_eig.col(3);
  Eigen::Vector2f uv2 = cam_2->Project(x3D2);

  float errX2 = uv2(0) - kp_2.pt.x;
  float errY2 = uv2(1) - kp_2.pt.y;

  if ((errX2 * errX2 + errY2 * errY2) >
      5.991 * unc) {  // Reprojection error is high
    return -5;
  }

  p3d_eig = x3D;

  return z1;
}

std::ostream &operator<<(std::ostream &os, const KannalaBrandt8 &kb) {
  os << kb.params_[0] << " " << kb.params_[1] << " "
     << kb.params_[2] << " " << kb.params_[3] << " "
     << kb.params_[4] << " " << kb.params_[5] << " "
     << kb.params_[6] << " " << kb.params_[7];
  return os;
}

std::istream &operator>>(std::istream &is, KannalaBrandt8 &kb) {
  float nextParam;
  for (size_t i = 0; i < 8; i++) {
    assert(is.good());  // Make sure the input stream is good
    is >> nextParam;
    kb.params_[i] = nextParam;
  }
  return is;
}

void KannalaBrandt8::Triangulate(const cv::Point2f &p1, const cv::Point2f &p2,
                                 const Eigen::Matrix<float, 3, 4> &Tcw_1_eig,
                                 const Eigen::Matrix<float, 3, 4> &Tcw_2_eig,
                                 Eigen::Vector3f &x3D) {
  Eigen::Matrix<float, 4, 4> A;
  A.row(0) = p1.x * Tcw_1_eig.row(2) - Tcw_1_eig.row(0);
  A.row(1) = p1.y * Tcw_1_eig.row(2) - Tcw_1_eig.row(1);
  A.row(2) = p2.x * Tcw_2_eig.row(2) - Tcw_2_eig.row(0);
  A.row(3) = p2.y * Tcw_2_eig.row(2) - Tcw_2_eig.row(1);

  Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
  Eigen::Vector4f x3Dh = svd.matrixV().col(3);
  x3D = x3Dh.head(3) / x3Dh(3);
}

bool KannalaBrandt8::IsEqual(GeometricCamera *cam) {
  if (cam->GetType() != GeometricCamera::kCamFisheye) return false;

  KannalaBrandt8 *pKBCam = (KannalaBrandt8 *)cam;

  if (abs(precision_ - pKBCam->GetPrecision()) > 1e-6) return false;

  if (Size() != pKBCam->Size()) return false;

  bool is_same_camera = true;
  for (size_t i = 0; i < Size(); ++i) {
    if (abs(params_[i] - pKBCam->GetParameter(i)) > 1e-6) {
      is_same_camera = false;
      break;
    }
  }
  return is_same_camera;
}

}  // namespace ORB_SLAM_FUSION
