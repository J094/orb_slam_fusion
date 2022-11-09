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

#include "tracking.h"

#include <chrono>
#include <iostream>
#include <mutex>

#include "cam/camera_models/kannalabrandt8_model.h"
#include "cam/camera_models/pinhole_model.h"
#include "cam/orb_feature/orb_matcher.h"
#include "solver/g2o_solver/g2o_types.h"
#include "solver/g2o_solver/optimizer.h"
#include "solver/mlpnp_solver.h"
#include "utils/converter.h"
#include "utils/frame_drawer.h"
#include "utils/geometric_tools.h"

using namespace std;

namespace ORB_SLAM_FUSION {

Tracking::Tracking(System* sys, ORBVocabulary* voc, FrameDrawer* frame_drawer,
                   MapDrawer* map_drawer, Atlas* atlas,
                   KeyFrameDatabase* kf_database, const string& strSettingPath,
                   const int sensor, Settings* settings, const string& _nameSeq)
    : state_(NO_IMAGES_YET),
      sensor_(sensor),
      mTrackedFr(0),
      mbStep(false),
      mbOnlyTracking(false),
      mbMapUpdated(false),
      mbVO(false),
      orb_voc_(voc),
      mpKeyFrameDB(kf_database),
      mbReadyToInitializate(false),
      mpSystem(sys),
      mpViewer(NULL),
      bStepByStep(false),
      frame_drawer_(frame_drawer),
      mpMapDrawer(map_drawer),
      atlas_(atlas),
      mnLastRelocFrameId(0),
      time_recently_lost(5.0),
      mnInitialFrameId(0),
      mbCreatedMap(false),
      mnFirstFrameId(0),
      cam2_(nullptr),
      mpLastKeyFrame(static_cast<KeyFrame*>(NULL)) {
  // Load camera parameters from settings file
  if (settings) {
    newParameterLoader(settings);
  } else {
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    bool b_parse_cam = ParseCamParamFile(fSettings);
    if (!b_parse_cam) {
      std::cout << "*Error with the camera parameters in the config file*"
                << std::endl;
    }

    // Load ORB parameters
    bool b_parse_orb = ParseORBParamFile(fSettings);
    if (!b_parse_orb) {
      std::cout << "*Error with the ORB parameters in the config file*"
                << std::endl;
    }

    bool b_parse_imu = true;
    if (sensor == System::kImuMonocular || sensor == System::kImuStereo ||
        sensor == System::kImuRgbd) {
      b_parse_imu = ParseIMUParamFile(fSettings);
      if (!b_parse_imu) {
        std::cout << "*Error with the IMU parameters in the config file*"
                  << std::endl;
      }

      mnFramesToResetIMU = max_frames_;
    }

    if (!b_parse_cam || !b_parse_orb || !b_parse_imu) {
      std::cerr << "**ERROR in the config file, the format is not correct**"
                << std::endl;
      try {
        throw -1;
      } catch (exception& e) {
      }
    }
  }

  initID = 0;
  lastID = 0;
  mbInitWith3KFs = false;
  num_dataset_ = 0;

  vector<GeometricCamera*> cams = atlas_->GetAllCameras();
  std::cout << "There are " << cams.size() << " cameras in the atlas"
            << std::endl;
  for (GeometricCamera* cam : cams) {
    std::cout << "Camera " << cam->GetId();
    if (cam->GetType() == GeometricCamera::kCamPinhole) {
      std::cout << " is pinhole" << std::endl;
    } else if (cam->GetType() == GeometricCamera::kCamFisheye) {
      std::cout << " is fisheye" << std::endl;
    } else {
      std::cout << " is unknown" << std::endl;
    }
  }
}

Tracking::~Tracking() {
  // f_track_stats.close();
}

void Tracking::newParameterLoader(Settings* settings) {
  cam_ = settings->camera1();
  cam_ = atlas_->AddCamera(cam_);

  if (settings->needToUndistort()) {
    dist_coef_ = settings->camera1DistortionCoef();
  } else {
    dist_coef_ = cv::Mat::zeros(4, 1, CV_32F);
  }

  // TODO: missing image scaling and rectification
  img_scale_ = 1.0f;

  cv_K_ = cv::Mat::eye(3, 3, CV_32F);
  cv_K_.at<float>(0, 0) = cam_->getParameter(0);
  cv_K_.at<float>(1, 1) = cam_->getParameter(1);
  cv_K_.at<float>(0, 2) = cam_->getParameter(2);
  cv_K_.at<float>(1, 2) = cam_->getParameter(3);

  eig_K_.setIdentity();
  eig_K_(0, 0) = cam_->getParameter(0);
  eig_K_(1, 1) = cam_->getParameter(1);
  eig_K_(0, 2) = cam_->getParameter(2);
  eig_K_(1, 2) = cam_->getParameter(3);

  if ((sensor_ == System::kStereo || sensor_ == System::kImuStereo ||
       sensor_ == System::kImuRgbd) &&
      settings->cameraType() == Settings::kKannalaBrandt) {
    cam2_ = settings->camera2();
    cam2_ = atlas_->AddCamera(cam2_);

    so_Tlr_ = settings->Tlr();

    frame_drawer_->both = true;
  }

  if (sensor_ == System::kStereo || sensor_ == System::kRgbd ||
      sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
    bf_ = settings->bf();
    th_depth_ = settings->b() * settings->thDepth();
  }

  if (sensor_ == System::kRgbd || sensor_ == System::kImuRgbd) {
    depth_map_factor_ = settings->depthMapFactor();
    if (fabs(depth_map_factor_) < 1e-5)
      depth_map_factor_ = 1;
    else
      depth_map_factor_ = 1.0f / depth_map_factor_;
  }

  min_frames_ = 0;
  max_frames_ = settings->fps();
  is_Rgb_ = settings->rgb();

  // ORB parameters
  int nFeatures = settings->nFeatures();
  int nLevels = settings->nLevels();
  int fIniThFAST = settings->initThFAST();
  int fMinThFAST = settings->minThFAST();
  float fScaleFactor = settings->scaleFactor();

  orb_extractor_left_ = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                         fIniThFAST, fMinThFAST);

  if (sensor_ == System::kStereo || sensor_ == System::kImuStereo)
    orb_extractor_right_ = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                            fIniThFAST, fMinThFAST);

  if (sensor_ == System::kMonocular || sensor_ == System::kImuMonocular)
    mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels,
                                         fIniThFAST, fMinThFAST);

  // IMU parameters
  Sophus::SE3f Tbc = settings->Tbc();
  mInsertKFsLost = settings->insertKFsWhenLost();
  mImuFreq = settings->imuFrequency();
  // TODO: ESTO ESTA BIEN?
  //  1.0 / (double) mImuFreq;
  mImuPer = 0.001;
  float Ng = settings->noiseGyro();
  float Na = settings->noiseAcc();
  float Ngw = settings->gyroWalk();
  float Naw = settings->accWalk();

  const float sf = sqrt(mImuFreq);
  imu_calib_ = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

  imu_preint_kf_ = new IMU::Preintegrated(IMU::Bias(), *imu_calib_);
}

bool Tracking::ParseCamParamFile(cv::FileStorage& fSettings) {
  dist_coef_ = cv::Mat::zeros(4, 1, CV_32F);
  cout << endl << "Camera Parameters: " << endl;
  bool b_miss_params = false;

  string sCameraName = fSettings["Camera.type"];
  if (sCameraName == "PinHole") {
    float fx, fy, cx, cy;
    img_scale_ = 1.f;

    // Camera calibration parameters
    cv::FileNode node = fSettings["Camera.fx"];
    if (!node.empty() && node.isReal()) {
      fx = node.real();
    } else {
      std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.fy"];
    if (!node.empty() && node.isReal()) {
      fy = node.real();
    } else {
      std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.cx"];
    if (!node.empty() && node.isReal()) {
      cx = node.real();
    } else {
      std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.cy"];
    if (!node.empty() && node.isReal()) {
      cy = node.real();
    } else {
      std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    // Distortion parameters
    node = fSettings["Camera.k1"];
    if (!node.empty() && node.isReal()) {
      dist_coef_.at<float>(0) = node.real();
    } else {
      std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.k2"];
    if (!node.empty() && node.isReal()) {
      dist_coef_.at<float>(1) = node.real();
    } else {
      std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.p1"];
    if (!node.empty() && node.isReal()) {
      dist_coef_.at<float>(2) = node.real();
    } else {
      std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.p2"];
    if (!node.empty() && node.isReal()) {
      dist_coef_.at<float>(3) = node.real();
    } else {
      std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.k3"];
    if (!node.empty() && node.isReal()) {
      dist_coef_.resize(5);
      dist_coef_.at<float>(4) = node.real();
    }

    node = fSettings["Camera.imageScale"];
    if (!node.empty() && node.isReal()) {
      img_scale_ = node.real();
    }

    if (b_miss_params) {
      return false;
    }

    if (img_scale_ != 1.f) {
      // K matrix parameters must be scaled.
      fx = fx * img_scale_;
      fy = fy * img_scale_;
      cx = cx * img_scale_;
      cy = cy * img_scale_;
    }

    vector<float> vCamCalib{fx, fy, cx, cy};

    cam_ = new Pinhole(vCamCalib);

    cam_ = atlas_->AddCamera(cam_);

    std::cout << "- Camera: Pinhole" << std::endl;
    std::cout << "- Image scale: " << img_scale_ << std::endl;
    std::cout << "- fx: " << fx << std::endl;
    std::cout << "- fy: " << fy << std::endl;
    std::cout << "- cx: " << cx << std::endl;
    std::cout << "- cy: " << cy << std::endl;
    std::cout << "- k1: " << dist_coef_.at<float>(0) << std::endl;
    std::cout << "- k2: " << dist_coef_.at<float>(1) << std::endl;

    std::cout << "- p1: " << dist_coef_.at<float>(2) << std::endl;
    std::cout << "- p2: " << dist_coef_.at<float>(3) << std::endl;

    if (dist_coef_.rows == 5)
      std::cout << "- k3: " << dist_coef_.at<float>(4) << std::endl;

    cv_K_ = cv::Mat::eye(3, 3, CV_32F);
    cv_K_.at<float>(0, 0) = fx;
    cv_K_.at<float>(1, 1) = fy;
    cv_K_.at<float>(0, 2) = cx;
    cv_K_.at<float>(1, 2) = cy;

    eig_K_.setIdentity();
    eig_K_(0, 0) = fx;
    eig_K_(1, 1) = fy;
    eig_K_(0, 2) = cx;
    eig_K_(1, 2) = cy;
  } else if (sCameraName == "KannalaBrandt8") {
    float fx, fy, cx, cy;
    float k1, k2, k3, k4;
    img_scale_ = 1.f;

    // Camera calibration parameters
    cv::FileNode node = fSettings["Camera.fx"];
    if (!node.empty() && node.isReal()) {
      fx = node.real();
    } else {
      std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }
    node = fSettings["Camera.fy"];
    if (!node.empty() && node.isReal()) {
      fy = node.real();
    } else {
      std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.cx"];
    if (!node.empty() && node.isReal()) {
      cx = node.real();
    } else {
      std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.cy"];
    if (!node.empty() && node.isReal()) {
      cy = node.real();
    } else {
      std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    // Distortion parameters
    node = fSettings["Camera.k1"];
    if (!node.empty() && node.isReal()) {
      k1 = node.real();
    } else {
      std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }
    node = fSettings["Camera.k2"];
    if (!node.empty() && node.isReal()) {
      k2 = node.real();
    } else {
      std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.k3"];
    if (!node.empty() && node.isReal()) {
      k3 = node.real();
    } else {
      std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.k4"];
    if (!node.empty() && node.isReal()) {
      k4 = node.real();
    } else {
      std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }

    node = fSettings["Camera.imageScale"];
    if (!node.empty() && node.isReal()) {
      img_scale_ = node.real();
    }

    if (!b_miss_params) {
      if (img_scale_ != 1.f) {
        // K matrix parameters must be scaled.
        fx = fx * img_scale_;
        fy = fy * img_scale_;
        cx = cx * img_scale_;
        cy = cy * img_scale_;
      }

      vector<float> vCamCalib{fx, fy, cx, cy, k1, k2, k3, k4};
      cam_ = new KannalaBrandt8(vCamCalib);
      cam_ = atlas_->AddCamera(cam_);
      std::cout << "- Camera: Fisheye" << std::endl;
      std::cout << "- Image scale: " << img_scale_ << std::endl;
      std::cout << "- fx: " << fx << std::endl;
      std::cout << "- fy: " << fy << std::endl;
      std::cout << "- cx: " << cx << std::endl;
      std::cout << "- cy: " << cy << std::endl;
      std::cout << "- k1: " << k1 << std::endl;
      std::cout << "- k2: " << k2 << std::endl;
      std::cout << "- k3: " << k3 << std::endl;
      std::cout << "- k4: " << k4 << std::endl;

      cv_K_ = cv::Mat::eye(3, 3, CV_32F);
      cv_K_.at<float>(0, 0) = fx;
      cv_K_.at<float>(1, 1) = fy;
      cv_K_.at<float>(0, 2) = cx;
      cv_K_.at<float>(1, 2) = cy;

      eig_K_.setIdentity();
      eig_K_(0, 0) = fx;
      eig_K_(1, 1) = fy;
      eig_K_(0, 2) = cx;
      eig_K_(1, 2) = cy;
    }

    if (sensor_ == System::kStereo || sensor_ == System::kImuStereo ||
        sensor_ == System::kImuRgbd) {
      // Right camera
      // Camera calibration parameters
      cv::FileNode node = fSettings["Camera2.fx"];
      if (!node.empty() && node.isReal()) {
        fx = node.real();
      } else {
        std::cerr
            << "*Camera2.fx parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }
      node = fSettings["Camera2.fy"];
      if (!node.empty() && node.isReal()) {
        fy = node.real();
      } else {
        std::cerr
            << "*Camera2.fy parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }

      node = fSettings["Camera2.cx"];
      if (!node.empty() && node.isReal()) {
        cx = node.real();
      } else {
        std::cerr
            << "*Camera2.cx parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }

      node = fSettings["Camera2.cy"];
      if (!node.empty() && node.isReal()) {
        cy = node.real();
      } else {
        std::cerr
            << "*Camera2.cy parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }

      // Distortion parameters
      node = fSettings["Camera2.k1"];
      if (!node.empty() && node.isReal()) {
        k1 = node.real();
      } else {
        std::cerr
            << "*Camera2.k1 parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }
      node = fSettings["Camera2.k2"];
      if (!node.empty() && node.isReal()) {
        k2 = node.real();
      } else {
        std::cerr
            << "*Camera2.k2 parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }

      node = fSettings["Camera2.k3"];
      if (!node.empty() && node.isReal()) {
        k3 = node.real();
      } else {
        std::cerr
            << "*Camera2.k3 parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }

      node = fSettings["Camera2.k4"];
      if (!node.empty() && node.isReal()) {
        k4 = node.real();
      } else {
        std::cerr
            << "*Camera2.k4 parameter doesn't exist or is not a real number*"
            << std::endl;
        b_miss_params = true;
      }

      int leftLappingBegin = -1;
      int leftLappingEnd = -1;

      int rightLappingBegin = -1;
      int rightLappingEnd = -1;

      node = fSettings["Camera.lappingBegin"];
      if (!node.empty() && node.isInt()) {
        leftLappingBegin = node.operator int();
      } else {
        std::cout << "WARNING: Camera.lappingBegin not correctly defined"
                  << std::endl;
      }
      node = fSettings["Camera.lappingEnd"];
      if (!node.empty() && node.isInt()) {
        leftLappingEnd = node.operator int();
      } else {
        std::cout << "WARNING: Camera.lappingEnd not correctly defined"
                  << std::endl;
      }
      node = fSettings["Camera2.lappingBegin"];
      if (!node.empty() && node.isInt()) {
        rightLappingBegin = node.operator int();
      } else {
        std::cout << "WARNING: Camera2.lappingBegin not correctly defined"
                  << std::endl;
      }
      node = fSettings["Camera2.lappingEnd"];
      if (!node.empty() && node.isInt()) {
        rightLappingEnd = node.operator int();
      } else {
        std::cout << "WARNING: Camera2.lappingEnd not correctly defined"
                  << std::endl;
      }

      node = fSettings["Tlr"];
      cv::Mat cvTlr;
      if (!node.empty()) {
        cvTlr = node.mat();
        if (cvTlr.rows != 3 || cvTlr.cols != 4) {
          std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*"
                    << std::endl;
          b_miss_params = true;
        }
      } else {
        std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
        b_miss_params = true;
      }

      if (!b_miss_params) {
        if (img_scale_ != 1.f) {
          // K matrix parameters must be scaled.
          fx = fx * img_scale_;
          fy = fy * img_scale_;
          cx = cx * img_scale_;
          cy = cy * img_scale_;

          leftLappingBegin = leftLappingBegin * img_scale_;
          leftLappingEnd = leftLappingEnd * img_scale_;
          rightLappingBegin = rightLappingBegin * img_scale_;
          rightLappingEnd = rightLappingEnd * img_scale_;
        }

        static_cast<KannalaBrandt8*>(cam_)->mvLappingArea[0] = leftLappingBegin;
        static_cast<KannalaBrandt8*>(cam_)->mvLappingArea[1] = leftLappingEnd;

        frame_drawer_->both = true;

        vector<float> vCamCalib2{fx, fy, cx, cy, k1, k2, k3, k4};
        cam2_ = new KannalaBrandt8(vCamCalib2);
        cam2_ = atlas_->AddCamera(cam2_);

        so_Tlr_ = Converter::toSophus(cvTlr);

        static_cast<KannalaBrandt8*>(cam2_)->mvLappingArea[0] =
            rightLappingBegin;
        static_cast<KannalaBrandt8*>(cam2_)->mvLappingArea[1] = rightLappingEnd;

        std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", "
                  << leftLappingEnd << std::endl;

        std::cout << std::endl << "Camera2 Parameters:" << std::endl;
        std::cout << "- Camera: Fisheye" << std::endl;
        std::cout << "- Image scale: " << img_scale_ << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << k1 << std::endl;
        std::cout << "- k2: " << k2 << std::endl;
        std::cout << "- k3: " << k3 << std::endl;
        std::cout << "- k4: " << k4 << std::endl;

        std::cout << "- so_Tlr_: \n" << cvTlr << std::endl;

        std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", "
                  << rightLappingEnd << std::endl;
      }
    }

    if (b_miss_params) {
      return false;
    }

  } else {
    std::cerr << "*Not Supported Camera Sensor*" << std::endl;
    std::cerr << "Check an example configuration file with the desired sensor"
              << std::endl;
  }

  if (sensor_ == System::kStereo || sensor_ == System::kRgbd ||
      sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
    cv::FileNode node = fSettings["Camera.bf"];
    if (!node.empty() && node.isReal()) {
      bf_ = node.real();
      if (img_scale_ != 1.f) {
        bf_ *= img_scale_;
      }
    } else {
      std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }
  }

  float fps = fSettings["Camera.fps"];
  if (fps == 0) fps = 30;

  // Max/Min Frames to insert keyframes and to check relocalisation
  min_frames_ = 0;
  max_frames_ = fps;

  cout << "- fps: " << fps << endl;

  int nRGB = fSettings["Camera.RGB"];
  is_Rgb_ = nRGB;

  if (is_Rgb_)
    cout << "- color order: RGB (ignored if grayscale)" << endl;
  else
    cout << "- color order: BGR (ignored if grayscale)" << endl;

  if (sensor_ == System::kStereo || sensor_ == System::kRgbd ||
      sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
    float fx = cam_->getParameter(0);
    cv::FileNode node = fSettings["ThDepth"];
    if (!node.empty() && node.isReal()) {
      th_depth_ = node.real();
      th_depth_ = bf_ * th_depth_ / fx;
      cout << endl
           << "Depth Threshold (Close/Far Points): " << th_depth_ << endl;
    } else {
      std::cerr << "*ThDepth parameter doesn't exist or is not a real number*"
                << std::endl;
      b_miss_params = true;
    }
  }

  if (sensor_ == System::kRgbd || sensor_ == System::kImuRgbd) {
    cv::FileNode node = fSettings["DepthMapFactor"];
    if (!node.empty() && node.isReal()) {
      depth_map_factor_ = node.real();
      if (fabs(depth_map_factor_) < 1e-5)
        depth_map_factor_ = 1;
      else
        depth_map_factor_ = 1.0f / depth_map_factor_;
    } else {
      std::cerr
          << "*DepthMapFactor parameter doesn't exist or is not a real number*"
          << std::endl;
      b_miss_params = true;
    }
  }

  if (b_miss_params) {
    return false;
  }

  return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage& fSettings) {
  bool b_miss_params = false;
  int nFeatures, nLevels, fIniThFAST, fMinThFAST;
  float fScaleFactor;

  cv::FileNode node = fSettings["ORBextractor.nFeatures"];
  if (!node.empty() && node.isInt()) {
    nFeatures = node.operator int();
  } else {
    std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an "
                 "integer*"
              << std::endl;
    b_miss_params = true;
  }

  node = fSettings["ORBextractor.scaleFactor"];
  if (!node.empty() && node.isReal()) {
    fScaleFactor = node.real();
  } else {
    std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not "
                 "a real number*"
              << std::endl;
    b_miss_params = true;
  }

  node = fSettings["ORBextractor.nLevels"];
  if (!node.empty() && node.isInt()) {
    nLevels = node.operator int();
  } else {
    std::cerr
        << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*"
        << std::endl;
    b_miss_params = true;
  }

  node = fSettings["ORBextractor.iniThFAST"];
  if (!node.empty() && node.isInt()) {
    fIniThFAST = node.operator int();
  } else {
    std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an "
                 "integer*"
              << std::endl;
    b_miss_params = true;
  }

  node = fSettings["ORBextractor.minThFAST"];
  if (!node.empty() && node.isInt()) {
    fMinThFAST = node.operator int();
  } else {
    std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an "
                 "integer*"
              << std::endl;
    b_miss_params = true;
  }

  if (b_miss_params) {
    return false;
  }

  orb_extractor_left_ = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                         fIniThFAST, fMinThFAST);

  if (sensor_ == System::kStereo || sensor_ == System::kImuStereo)
    orb_extractor_right_ = new ORBextractor(nFeatures, fScaleFactor, nLevels,
                                            fIniThFAST, fMinThFAST);

  if (sensor_ == System::kMonocular || sensor_ == System::kImuMonocular)
    mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels,
                                         fIniThFAST, fMinThFAST);

  cout << endl << "ORB Extractor Parameters: " << endl;
  cout << "- Number of Features: " << nFeatures << endl;
  cout << "- Scale Levels: " << nLevels << endl;
  cout << "- Scale Factor: " << fScaleFactor << endl;
  cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
  cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

  return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage& fSettings) {
  bool b_miss_params = false;

  cv::Mat cvTbc;
  cv::FileNode node = fSettings["Tbc"];
  if (!node.empty()) {
    cvTbc = node.mat();
    if (cvTbc.rows != 4 || cvTbc.cols != 4) {
      std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*"
                << std::endl;
      b_miss_params = true;
    }
  } else {
    std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
    b_miss_params = true;
  }
  cout << endl;
  cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
  Sophus::SE3f Tbc(eigTbc);

  node = fSettings["InsertKFsWhenLost"];
  mInsertKFsLost = true;
  if (!node.empty() && node.isInt()) {
    mInsertKFsLost = (bool)node.operator int();
  }

  if (!mInsertKFsLost)
    cout << "Do not insert keyframes when lost visual tracking " << endl;

  float Ng, Na, Ngw, Naw;

  node = fSettings["IMU.Frequency"];
  if (!node.empty() && node.isInt()) {
    mImuFreq = node.operator int();
    mImuPer = 0.001;  // 1.0 / (double) mImuFreq;
  } else {
    std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*"
              << std::endl;
    b_miss_params = true;
  }

  node = fSettings["IMU.NoiseGyro"];
  if (!node.empty() && node.isReal()) {
    Ng = node.real();
  } else {
    std::cerr
        << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*"
        << std::endl;
    b_miss_params = true;
  }

  node = fSettings["IMU.NoiseAcc"];
  if (!node.empty() && node.isReal()) {
    Na = node.real();
  } else {
    std::cerr
        << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*"
        << std::endl;
    b_miss_params = true;
  }

  node = fSettings["IMU.GyroWalk"];
  if (!node.empty() && node.isReal()) {
    Ngw = node.real();
  } else {
    std::cerr
        << "*IMU.GyroWalk parameter doesn't exist or is not a real number*"
        << std::endl;
    b_miss_params = true;
  }

  node = fSettings["IMU.AccWalk"];
  if (!node.empty() && node.isReal()) {
    Naw = node.real();
  } else {
    std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*"
              << std::endl;
    b_miss_params = true;
  }

  node = fSettings["IMU.fastInit"];
  mFastInit = false;
  if (!node.empty()) {
    mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
  }

  if (mFastInit)
    cout << "Fast IMU initialization. Acceleration is not checked \n";

  if (b_miss_params) {
    return false;
  }

  const float sf = sqrt(mImuFreq);
  cout << endl;
  cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
  cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
  cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
  cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
  cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

  imu_calib_ = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

  imu_preint_kf_ = new IMU::Preintegrated(IMU::Bias(), *imu_calib_);

  return true;
}

void Tracking::SetLocalMapper(LocalMapping* pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing* pLoopClosing) {
  mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer* viewer) { mpViewer = viewer; }

void Tracking::SetStepByStep(bool bSet) { bStepByStep = bSet; }

bool Tracking::GetStepByStep() { return bStepByStep; }

Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat& img_rect_left,
                                       const cv::Mat& img_rect_right,
                                       const double& timestamp,
                                       string filename) {
  // cout << "GrabImageStereo" << endl;

  img_gray_ = img_rect_left;
  cv::Mat img_gray_right = img_rect_right;
  img_right_ = img_rect_right;

  if (img_gray_.channels() == 3) {
    // cout << "Image with 3 channels" << endl;
    if (is_Rgb_) {
      cvtColor(img_gray_, img_gray_, cv::COLOR_RGB2GRAY);
      cvtColor(img_gray_right, img_gray_right, cv::COLOR_RGB2GRAY);
    } else {
      cvtColor(img_gray_, img_gray_, cv::COLOR_BGR2GRAY);
      cvtColor(img_gray_right, img_gray_right, cv::COLOR_BGR2GRAY);
    }
  } else if (img_gray_.channels() == 4) {
    // cout << "Image with 4 channels" << endl;
    if (is_Rgb_) {
      cvtColor(img_gray_, img_gray_, cv::COLOR_RGBA2GRAY);
      cvtColor(img_gray_right, img_gray_right, cv::COLOR_RGBA2GRAY);
    } else {
      cvtColor(img_gray_, img_gray_, cv::COLOR_BGRA2GRAY);
      cvtColor(img_gray_right, img_gray_right, cv::COLOR_BGRA2GRAY);
    }
  }

  // cout << "Incoming frame creation" << endl;

  if (sensor_ == System::kStereo && !cam2_)
    curr_frame_ = Frame(img_gray_, img_gray_right, timestamp,
                        orb_extractor_left_, orb_extractor_right_, orb_voc_,
                        cv_K_, dist_coef_, bf_, th_depth_, cam_);
  else if (sensor_ == System::kStereo && cam2_)
    curr_frame_ =
        Frame(img_gray_, img_gray_right, timestamp, orb_extractor_left_,
              orb_extractor_right_, orb_voc_, cv_K_, dist_coef_, bf_, th_depth_,
              cam_, cam2_, so_Tlr_);
  else if (sensor_ == System::kImuStereo && !cam2_)
    curr_frame_ =
        Frame(img_gray_, img_gray_right, timestamp, orb_extractor_left_,
              orb_extractor_right_, orb_voc_, cv_K_, dist_coef_, bf_, th_depth_,
              cam_, &last_frame_, *imu_calib_);
  else if (sensor_ == System::kImuStereo && cam2_)
    curr_frame_ =
        Frame(img_gray_, img_gray_right, timestamp, orb_extractor_left_,
              orb_extractor_right_, orb_voc_, cv_K_, dist_coef_, bf_, th_depth_,
              cam_, cam2_, so_Tlr_, &last_frame_, *imu_calib_);

  // cout << "Incoming frame ended" << endl;

  curr_frame_.file_name_ = filename;
  curr_frame_.num_dataset_ = num_dataset_;

  // cout << "Tracking start" << endl;
  Track();
  // cout << "Tracking end" << endl;

  return curr_frame_.GetPose();
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                                     const double& timestamp, string filename) {
  img_gray_ = imRGB;
  cv::Mat imDepth = imD;

  if (img_gray_.channels() == 3) {
    if (is_Rgb_)
      cvtColor(img_gray_, img_gray_, cv::COLOR_RGB2GRAY);
    else
      cvtColor(img_gray_, img_gray_, cv::COLOR_BGR2GRAY);
  } else if (img_gray_.channels() == 4) {
    if (is_Rgb_)
      cvtColor(img_gray_, img_gray_, cv::COLOR_RGBA2GRAY);
    else
      cvtColor(img_gray_, img_gray_, cv::COLOR_BGRA2GRAY);
  }

  if ((fabs(depth_map_factor_ - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
    imDepth.convertTo(imDepth, CV_32F, depth_map_factor_);

  if (sensor_ == System::kRgbd)
    curr_frame_ = Frame(img_gray_, imDepth, timestamp, orb_extractor_left_,
                        orb_voc_, cv_K_, dist_coef_, bf_, th_depth_, cam_);
  else if (sensor_ == System::kImuRgbd)
    curr_frame_ = Frame(img_gray_, imDepth, timestamp, orb_extractor_left_,
                        orb_voc_, cv_K_, dist_coef_, bf_, th_depth_, cam_,
                        &last_frame_, *imu_calib_);

  curr_frame_.file_name_ = filename;
  curr_frame_.num_dataset_ = num_dataset_;

  Track();

  return curr_frame_.GetPose();
}

Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat& im,
                                          const double& timestamp,
                                          string filename) {
  img_gray_ = im;
  if (img_gray_.channels() == 3) {
    if (is_Rgb_)
      cvtColor(img_gray_, img_gray_, cv::COLOR_RGB2GRAY);
    else
      cvtColor(img_gray_, img_gray_, cv::COLOR_BGR2GRAY);
  } else if (img_gray_.channels() == 4) {
    if (is_Rgb_)
      cvtColor(img_gray_, img_gray_, cv::COLOR_RGBA2GRAY);
    else
      cvtColor(img_gray_, img_gray_, cv::COLOR_BGRA2GRAY);
  }

  if (sensor_ == System::kMonocular) {
    if (state_ == NOT_INITIALIZED || state_ == NO_IMAGES_YET ||
        (lastID - initID) < max_frames_)
      curr_frame_ = Frame(img_gray_, timestamp, mpIniORBextractor, orb_voc_,
                          cam_, dist_coef_, bf_, th_depth_);
    else
      curr_frame_ = Frame(img_gray_, timestamp, orb_extractor_left_, orb_voc_,
                          cam_, dist_coef_, bf_, th_depth_);
  } else if (sensor_ == System::kImuMonocular) {
    if (state_ == NOT_INITIALIZED || state_ == NO_IMAGES_YET) {
      curr_frame_ =
          Frame(img_gray_, timestamp, mpIniORBextractor, orb_voc_, cam_,
                dist_coef_, bf_, th_depth_, &last_frame_, *imu_calib_);
    } else
      curr_frame_ =
          Frame(img_gray_, timestamp, orb_extractor_left_, orb_voc_, cam_,
                dist_coef_, bf_, th_depth_, &last_frame_, *imu_calib_);
  }

  if (state_ == NO_IMAGES_YET) t0 = timestamp;

  curr_frame_.file_name_ = filename;
  curr_frame_.num_dataset_ = num_dataset_;

  lastID = curr_frame_.id_;
  Track();

  return curr_frame_.GetPose();
}

void Tracking::GrabImuData(const IMU::Point& imuMeasurement) {
  unique_lock<mutex> lock(mutex_imu_);
  mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU() {
  if (!curr_frame_.mpPrevFrame) {
    Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
    curr_frame_.setIntegrated();
    return;
  }

  mvImuFromLastFrame.clear();
  mvImuFromLastFrame.reserve(mlQueueImuData.size());
  if (mlQueueImuData.size() == 0) {
    Verbose::PrintMess("Not IMU data in mlQueueImuData!!",
                       Verbose::VERBOSITY_NORMAL);
    curr_frame_.setIntegrated();
    return;
  }

  while (true) {
    bool bSleep = false;
    {
      unique_lock<mutex> lock(mutex_imu_);
      if (!mlQueueImuData.empty()) {
        IMU::Point* m = &mlQueueImuData.front();
        cout.precision(17);
        if (m->t < curr_frame_.mpPrevFrame->timestamp_ - mImuPer) {
          mlQueueImuData.pop_front();
        } else if (m->t < curr_frame_.timestamp_ - mImuPer) {
          mvImuFromLastFrame.push_back(*m);
          mlQueueImuData.pop_front();
        } else {
          mvImuFromLastFrame.push_back(*m);
          break;
        }
      } else {
        break;
        bSleep = true;
      }
    }
    if (bSleep) usleep(500);
  }

  const int n = mvImuFromLastFrame.size() - 1;
  if (n == 0) {
    cout << "Empty IMU measurements vector!!!\n";
    return;
  }

  IMU::Preintegrated* imu_preint_f =
      new IMU::Preintegrated(last_frame_.mImuBias, curr_frame_.mImuCalib);

  for (int i = 0; i < n; i++) {
    float tstep;
    Eigen::Vector3f acc, angVel;
    if ((i == 0) && (i < (n - 1))) {
      float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
      float tini =
          mvImuFromLastFrame[i].t - curr_frame_.mpPrevFrame->timestamp_;
      acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
             (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) *
                 (tini / tab)) *
            0.5f;
      angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) *
                    (tini / tab)) *
               0.5f;
      tstep = mvImuFromLastFrame[i + 1].t - curr_frame_.mpPrevFrame->timestamp_;
    } else if (i < (n - 1)) {
      acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
      angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
      tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
    } else if ((i > 0) && (i == (n - 1))) {
      float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
      float tend = mvImuFromLastFrame[i + 1].t - curr_frame_.timestamp_;
      acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
             (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) *
                 (tend / tab)) *
            0.5f;
      angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) *
                    (tend / tab)) *
               0.5f;
      tstep = curr_frame_.timestamp_ - mvImuFromLastFrame[i].t;
    } else if ((i == 0) && (i == (n - 1))) {
      acc = mvImuFromLastFrame[i].a;
      angVel = mvImuFromLastFrame[i].w;
      tstep = curr_frame_.timestamp_ - curr_frame_.mpPrevFrame->timestamp_;
    }

    if (!imu_preint_kf_) cout << "imu_preint_kf_ does not exist" << endl;
    imu_preint_kf_->IntegrateNewMeasurement(acc, angVel, tstep);
    imu_preint_f->IntegrateNewMeasurement(acc, angVel, tstep);
  }

  curr_frame_.mpImuPreintegratedFrame = imu_preint_f;
  curr_frame_.mpImuPreintegrated = imu_preint_kf_;
  curr_frame_.mpLastKeyFrame = mpLastKeyFrame;

  curr_frame_.setIntegrated();

  // Verbose::PrintMess("Preintegration is finished!! ",
  // Verbose::VERBOSITY_DEBUG);
}

bool Tracking::PredictStateIMU() {
  if (!curr_frame_.mpPrevFrame) {
    Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
    return false;
  }

  if (mbMapUpdated && mpLastKeyFrame) {
    const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
    const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
    const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

    const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
    const float t12 = imu_preint_kf_->dT;

    Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(
        Rwb1 * imu_preint_kf_->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
    Eigen::Vector3f twb2 =
        twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
        Rwb1 * imu_preint_kf_->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
    Eigen::Vector3f Vwb2 =
        Vwb1 + t12 * Gz +
        Rwb1 * imu_preint_kf_->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
    curr_frame_.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

    curr_frame_.mImuBias = mpLastKeyFrame->GetImuBias();
    curr_frame_.mPredBias = curr_frame_.mImuBias;
    return true;
  } else if (!mbMapUpdated) {
    const Eigen::Vector3f twb1 = last_frame_.GetImuPosition();
    const Eigen::Matrix3f Rwb1 = last_frame_.GetImuRotation();
    const Eigen::Vector3f Vwb1 = last_frame_.GetVelocity();
    const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
    const float t12 = curr_frame_.mpImuPreintegratedFrame->dT;

    Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(
        Rwb1 * curr_frame_.mpImuPreintegratedFrame->GetDeltaRotation(
                   last_frame_.mImuBias));
    Eigen::Vector3f twb2 =
        twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
        Rwb1 * curr_frame_.mpImuPreintegratedFrame->GetDeltaPosition(
                   last_frame_.mImuBias);
    Eigen::Vector3f Vwb2 =
        Vwb1 + t12 * Gz +
        Rwb1 * curr_frame_.mpImuPreintegratedFrame->GetDeltaVelocity(
                   last_frame_.mImuBias);

    curr_frame_.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

    curr_frame_.mImuBias = last_frame_.mImuBias;
    curr_frame_.mPredBias = curr_frame_.mImuBias;
    return true;
  } else
    cout << "not IMU prediction!!" << endl;

  return false;
}

void Tracking::ResetFrameIMU() {
  // TODO To implement...
}

void Tracking::Track() {
  if (bStepByStep) {
    std::cout << "Tracking: Waiting to the next step" << std::endl;
    while (!mbStep && bStepByStep) usleep(500);
    mbStep = false;
  }

  if (mpLocalMapper->mbBadImu) {
    cout << "TRACK: Reset map because local mapper set the bad imu flag "
         << endl;
    mpSystem->ResetActiveMap();
    return;
  }

  Map* pCurrentMap = atlas_->GetCurrentMap();
  if (!pCurrentMap) {
    cout << "ERROR: There is not an active map in the atlas" << endl;
  }

  if (state_ != NO_IMAGES_YET) {
    if (last_frame_.timestamp_ > curr_frame_.timestamp_) {
      cerr
          << "ERROR: Frame with a timestamp older than previous frame detected!"
          << endl;
      unique_lock<mutex> lock(mutex_imu_);
      mlQueueImuData.clear();
      CreateMapInAtlas();
      return;
    } else if (curr_frame_.timestamp_ > last_frame_.timestamp_ + 1.0) {
      // cout << curr_frame_.timestamp_ << ", " << last_frame_.timestamp_ <<
      // endl; cout << "id last: " << last_frame_.id_ << "    id curr: " <<
      // curr_frame_.id_ << endl;
      if (atlas_->isInertial()) {
        if (atlas_->isImuInitialized()) {
          cout << "Timestamp jump detected. State set to LOST. Reseting IMU "
                  "integration..."
               << endl;
          if (!pCurrentMap->GetIniertialBA2()) {
            mpSystem->ResetActiveMap();
          } else {
            CreateMapInAtlas();
          }
        } else {
          cout << "Timestamp jump detected, before IMU initialization. "
                  "Reseting..."
               << endl;
          mpSystem->ResetActiveMap();
        }
        return;
      }
    }
  }

  if ((sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
       sensor_ == System::kImuRgbd) &&
      mpLastKeyFrame)
    curr_frame_.SetNewBias(mpLastKeyFrame->GetImuBias());

  if (state_ == NO_IMAGES_YET) {
    state_ = NOT_INITIALIZED;
  }

  last_processed_state_ = state_;

  if ((sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
       sensor_ == System::kImuRgbd) &&
      !mbCreatedMap) {
    PreintegrateIMU();
  }
  mbCreatedMap = false;

  // Get Map Mutex -> Map cannot be changed
  unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

  mbMapUpdated = false;

  int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
  int nMapChangeIndex = pCurrentMap->GetLastMapChange();
  if (nCurMapChangeIndex > nMapChangeIndex) {
    pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
    mbMapUpdated = true;
  }

  if (state_ == NOT_INITIALIZED) {
    if (sensor_ == System::kStereo || sensor_ == System::kRgbd ||
        sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
      StereoInitialization();
    } else {
      MonocularInitialization();
    }

    // frame_drawer_->Update(this);

    if (state_ != OK)  // If rightly initialized, state_=OK
    {
      last_frame_ = Frame(curr_frame_);
      return;
    }

    if (atlas_->GetAllMaps().size() == 1) {
      mnFirstFrameId = curr_frame_.id_;
    }
  } else {
    // System is initialized. Track Frame.
    bool bOK;
    // Initial camera pose estimation using motion model or relocalization (if
    // tracking is lost)
    if (!mbOnlyTracking) {
      // State OK
      // Local Mapping is activated. This is the normal behaviour, unless
      // you explicitly activate the "only tracking" mode.
      if (state_ == OK) {
        // Local Mapping might have changed some MapPoints tracked in last frame
        CheckReplacedInLastFrame();

        if ((!mbVelocity && !pCurrentMap->isImuInitialized()) ||
            curr_frame_.id_ < mnLastRelocFrameId + 2) {
          Verbose::PrintMess("TRACK: Track with respect to the reference KF ",
                             Verbose::VERBOSITY_DEBUG);
          bOK = TrackReferenceKeyFrame();
        } else {
          Verbose::PrintMess("TRACK: Track with motion model",
                             Verbose::VERBOSITY_DEBUG);
          bOK = TrackWithMotionModel();
          if (!bOK) bOK = TrackReferenceKeyFrame();
        }

        if (!bOK) {
          if (curr_frame_.id_ <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
              (sensor_ == System::kImuMonocular ||
               sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd)) {
            state_ = LOST;
          } else if (pCurrentMap->KeyFramesInMap() > 10) {
            // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
            state_ = RECENTLY_LOST;
            mTimeStampLost = curr_frame_.timestamp_;
          } else {
            state_ = LOST;
          }
        }
      } else {
        if (state_ == RECENTLY_LOST) {
          Verbose::PrintMess("Lost for a short time",
                             Verbose::VERBOSITY_NORMAL);

          bOK = true;
          if ((sensor_ == System::kImuMonocular ||
               sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd)) {
            if (pCurrentMap->isImuInitialized())
              PredictStateIMU();
            else
              bOK = false;

            if (curr_frame_.timestamp_ - mTimeStampLost > time_recently_lost) {
              state_ = LOST;
              Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
              bOK = false;
            }
          } else {
            // Relocalization
            bOK = Relocalization();
            // std::cout << "curr_frame_.timestamp_:" <<
            // to_string(curr_frame_.timestamp_) << std::endl; std::cout <<
            // "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
            if (curr_frame_.timestamp_ - mTimeStampLost > 3.0f && !bOK) {
              state_ = LOST;
              Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
              bOK = false;
            }
          }
        } else if (state_ == LOST) {
          Verbose::PrintMess("A new map is started...",
                             Verbose::VERBOSITY_NORMAL);

          if (pCurrentMap->KeyFramesInMap() < 10) {
            mpSystem->ResetActiveMap();
            Verbose::PrintMess("Reseting current map...",
                               Verbose::VERBOSITY_NORMAL);
          } else
            CreateMapInAtlas();

          if (mpLastKeyFrame) mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

          Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

          return;
        }
      }

    } else {
      // Localization Mode: Local Mapping is deactivated (TODO Not available in
      // inertial mode)
      if (state_ == LOST) {
        if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
            sensor_ == System::kImuRgbd)
          Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
        bOK = Relocalization();
      } else {
        if (!mbVO) {
          // In last frame we tracked enough MapPoints in the map
          if (mbVelocity) {
            bOK = TrackWithMotionModel();
          } else {
            bOK = TrackReferenceKeyFrame();
          }
        } else {
          // In last frame we tracked mainly "visual odometry" points.

          // We compute two camera poses, one from motion model and one doing
          // relocalization. If relocalization is sucessfull we choose that
          // solution, otherwise we retain the "visual odometry" solution.

          bool bOKMM = false;
          bool bOKReloc = false;
          vector<MapPoint*> vpMPsMM;
          vector<bool> vbOutMM;
          Sophus::SE3f TcwMM;
          if (mbVelocity) {
            bOKMM = TrackWithMotionModel();
            vpMPsMM = curr_frame_.mvpMapPoints;
            vbOutMM = curr_frame_.mvbOutlier;
            TcwMM = curr_frame_.GetPose();
          }
          bOKReloc = Relocalization();

          if (bOKMM && !bOKReloc) {
            curr_frame_.SetPose(TcwMM);
            curr_frame_.mvpMapPoints = vpMPsMM;
            curr_frame_.mvbOutlier = vbOutMM;

            if (mbVO) {
              for (int i = 0; i < curr_frame_.N; i++) {
                if (curr_frame_.mvpMapPoints[i] && !curr_frame_.mvbOutlier[i]) {
                  curr_frame_.mvpMapPoints[i]->IncreaseFound();
                }
              }
            }
          } else if (bOKReloc) {
            mbVO = false;
          }

          bOK = bOKReloc || bOKMM;
        }
      }
    }

    if (!curr_frame_.mpReferenceKF) curr_frame_.mpReferenceKF = mpReferenceKF;
    // If we have an initial estimation of the camera pose and matching. Track
    // the local map.
    if (!mbOnlyTracking) {
      if (bOK) {
        bOK = TrackLocalMap();
      }
      if (!bOK) cout << "Fail to track local map!" << endl;
    } else {
      // mbVO true means that there are few matches to MapPoints in the map. We
      // cannot retrieve a local map and therefore we do not perform
      // TrackLocalMap(). Once the system relocalizes the camera we will use the
      // local map again.
      if (bOK && !mbVO) bOK = TrackLocalMap();
    }

    if (bOK)
      state_ = OK;
    else if (state_ == OK) {
      if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
          sensor_ == System::kImuRgbd) {
        Verbose::PrintMess("Track lost for less than one second...",
                           Verbose::VERBOSITY_NORMAL);
        if (!pCurrentMap->isImuInitialized() ||
            !pCurrentMap->GetIniertialBA2()) {
          cout << "IMU is not or recently initialized. Reseting active map..."
               << endl;
          mpSystem->ResetActiveMap();
        }

        state_ = RECENTLY_LOST;
      } else
        state_ = RECENTLY_LOST;  // visual to lost

      /*if(curr_frame_.id_>mnLastRelocFrameId+max_frames_)
      {*/
      mTimeStampLost = curr_frame_.timestamp_;
      //}
    }

    // Save frame if recent relocalization, since they are used for IMU reset
    // (as we are making copy, it shluld be once mCurrFrame is completely
    // modified)
    if ((curr_frame_.id_ < (mnLastRelocFrameId + mnFramesToResetIMU)) &&
        (curr_frame_.id_ > mnFramesToResetIMU) &&
        (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
         sensor_ == System::kImuRgbd) &&
        pCurrentMap->isImuInitialized()) {
      // TODO check this situation
      Verbose::PrintMess("Saving pointer to frame. imu needs reset...",
                         Verbose::VERBOSITY_NORMAL);
      Frame* pF = new Frame(curr_frame_);
      pF->mpPrevFrame = new Frame(last_frame_);

      // Load preintegration
      pF->mpImuPreintegratedFrame =
          new IMU::Preintegrated(curr_frame_.mpImuPreintegratedFrame);
    }

    if (pCurrentMap->isImuInitialized()) {
      if (bOK) {
        if (curr_frame_.id_ == (mnLastRelocFrameId + mnFramesToResetIMU)) {
          cout << "RESETING FRAME!!!" << endl;
          ResetFrameIMU();
        } else if (curr_frame_.id_ > (mnLastRelocFrameId + 30))
          mLastBias = curr_frame_.mImuBias;
      }
    }

    // Update drawer
    frame_drawer_->Update(this);
    if (curr_frame_.isSet())
      mpMapDrawer->SetCurrentCameraPose(curr_frame_.GetPose());

    if (bOK || state_ == RECENTLY_LOST) {
      // Update motion model
      if (last_frame_.isSet() && curr_frame_.isSet()) {
        Sophus::SE3f LastTwc = last_frame_.GetPose().inverse();
        mVelocity = curr_frame_.GetPose() * LastTwc;
        mbVelocity = true;
      } else {
        mbVelocity = false;
      }

      if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
          sensor_ == System::kImuRgbd)
        mpMapDrawer->SetCurrentCameraPose(curr_frame_.GetPose());

      // Clean VO matches
      for (int i = 0; i < curr_frame_.N; i++) {
        MapPoint* pMP = curr_frame_.mvpMapPoints[i];
        if (pMP)
          if (pMP->Observations() < 1) {
            curr_frame_.mvbOutlier[i] = false;
            curr_frame_.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
          }
      }

      // Delete temporal MapPoints
      for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(),
                                     lend = mlpTemporalPoints.end();
           lit != lend; lit++) {
        MapPoint* pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();
      bool bNeedKF = NeedNewKeyFrame();

      // Check if we need to insert a new keyframe
      // if(bNeedKF && bOK)
      if (bNeedKF && (bOK || (mInsertKFsLost && state_ == RECENTLY_LOST &&
                              (sensor_ == System::kImuMonocular ||
                               sensor_ == System::kImuStereo ||
                               sensor_ == System::kImuRgbd))))
        CreateNewKeyFrame();

      // We allow points with high innovation (considererd outliers by the Huber
      // Function) pass to the new keyframe, so that bundle adjustment will
      // finally decide if they are outliers or not. We don't want next frame to
      // estimate its position with those points so we discard them in the
      // frame. Only has effect if lastframe is tracked
      for (int i = 0; i < curr_frame_.N; i++) {
        if (curr_frame_.mvpMapPoints[i] && curr_frame_.mvbOutlier[i])
          curr_frame_.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
      }
    }

    // Reset if the camera get lost soon after initialization
    if (state_ == LOST) {
      if (pCurrentMap->KeyFramesInMap() <= 10) {
        mpSystem->ResetActiveMap();
        return;
      }
      if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
          sensor_ == System::kImuRgbd)
        if (!pCurrentMap->isImuInitialized()) {
          Verbose::PrintMess(
              "Track lost before IMU initialisation, reseting...",
              Verbose::VERBOSITY_QUIET);
          mpSystem->ResetActiveMap();
          return;
        }

      CreateMapInAtlas();

      return;
    }

    if (!curr_frame_.mpReferenceKF) curr_frame_.mpReferenceKF = mpReferenceKF;

    last_frame_ = Frame(curr_frame_);
  }

  if (state_ == OK || state_ == RECENTLY_LOST) {
    // Store frame pose information to retrieve the complete camera trajectory
    // afterwards.
    if (curr_frame_.isSet()) {
      Sophus::SE3f Tcr_ =
          curr_frame_.GetPose() * curr_frame_.mpReferenceKF->GetPoseInverse();
      mlRelativeFramePoses.push_back(Tcr_);
      mlpReferences.push_back(curr_frame_.mpReferenceKF);
      mlFrameTimes.push_back(curr_frame_.timestamp_);
      mlbLost.push_back(state_ == LOST);
    } else {
      // This can happen if tracking is lost
      mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
      mlpReferences.push_back(mlpReferences.back());
      mlFrameTimes.push_back(mlFrameTimes.back());
      mlbLost.push_back(state_ == LOST);
    }
  }
}

void Tracking::StereoInitialization() {
  if (curr_frame_.N > 500) {
    if (sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
      if (!curr_frame_.mpImuPreintegrated || !last_frame_.mpImuPreintegrated) {
        cout << "not IMU meas" << endl;
        return;
      }

      if (!mFastInit && (curr_frame_.mpImuPreintegratedFrame->avgA -
                         last_frame_.mpImuPreintegratedFrame->avgA)
                                .norm() < 0.5) {
        cout << "not enough acceleration" << endl;
        return;
      }

      if (imu_preint_kf_) delete imu_preint_kf_;

      imu_preint_kf_ = new IMU::Preintegrated(IMU::Bias(), *imu_calib_);
      curr_frame_.mpImuPreintegrated = imu_preint_kf_;
    }

    // Set Frame pose to the origin (In case of inertial SLAM to imu)
    if (sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
      Eigen::Matrix3f Rwb0 = curr_frame_.mImuCalib.mTcb.rotationMatrix();
      Eigen::Vector3f twb0 = curr_frame_.mImuCalib.mTcb.translation();
      Eigen::Vector3f Vwb0;
      Vwb0.setZero();
      curr_frame_.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
    } else
      curr_frame_.SetPose(Sophus::SE3f());

    // Create KeyFrame
    KeyFrame* pKFini =
        new KeyFrame(curr_frame_, atlas_->GetCurrentMap(), mpKeyFrameDB);

    // Insert KeyFrame in the map
    atlas_->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    if (!cam2_) {
      for (int i = 0; i < curr_frame_.N; i++) {
        float z = curr_frame_.mvDepth[i];
        if (z > 0) {
          Eigen::Vector3f x3D;
          curr_frame_.UnprojectStereo(i, x3D);
          MapPoint* pNewMP = new MapPoint(x3D, pKFini, atlas_->GetCurrentMap());
          pNewMP->AddObservation(pKFini, i);
          pKFini->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          atlas_->AddMapPoint(pNewMP);

          curr_frame_.mvpMapPoints[i] = pNewMP;
        }
      }
    } else {
      for (int i = 0; i < curr_frame_.Nleft; i++) {
        int rightIndex = curr_frame_.mvLeftToRightMatch[i];
        if (rightIndex != -1) {
          Eigen::Vector3f x3D = curr_frame_.mvStereo3Dpoints[i];

          MapPoint* pNewMP = new MapPoint(x3D, pKFini, atlas_->GetCurrentMap());

          pNewMP->AddObservation(pKFini, i);
          pNewMP->AddObservation(pKFini, rightIndex + curr_frame_.Nleft);

          pKFini->AddMapPoint(pNewMP, i);
          pKFini->AddMapPoint(pNewMP, rightIndex + curr_frame_.Nleft);

          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          atlas_->AddMapPoint(pNewMP);

          curr_frame_.mvpMapPoints[i] = pNewMP;
          curr_frame_.mvpMapPoints[rightIndex + curr_frame_.Nleft] = pNewMP;
        }
      }
    }

    Verbose::PrintMess("New Map created with " +
                           to_string(atlas_->MapPointsInMap()) + " points",
                       Verbose::VERBOSITY_QUIET);

    // cout << "Active map: " << atlas_->GetCurrentMap()->GetId() << endl;

    mpLocalMapper->InsertKeyFrame(pKFini);

    last_frame_ = Frame(curr_frame_);
    mnLastKeyFrameId = curr_frame_.id_;
    mpLastKeyFrame = pKFini;
    // mnLastRelocFrameId = curr_frame_.id_;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = atlas_->GetAllMapPoints();
    mpReferenceKF = pKFini;
    curr_frame_.mpReferenceKF = pKFini;

    atlas_->SetReferenceMapPoints(mvpLocalMapPoints);

    atlas_->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mpMapDrawer->SetCurrentCameraPose(curr_frame_.GetPose());

    state_ = OK;
  }
}

void Tracking::MonocularInitialization() {
  if (!mbReadyToInitializate) {
    // Set Reference Frame
    if (curr_frame_.mvKeys.size() > 100) {
      mInitialFrame = Frame(curr_frame_);
      last_frame_ = Frame(curr_frame_);
      mvbPrevMatched.resize(curr_frame_.mvKeysUn.size());
      for (size_t i = 0; i < curr_frame_.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = curr_frame_.mvKeysUn[i].pt;

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      if (sensor_ == System::kImuMonocular) {
        if (imu_preint_kf_) {
          delete imu_preint_kf_;
        }
        imu_preint_kf_ = new IMU::Preintegrated(IMU::Bias(), *imu_calib_);
        curr_frame_.mpImuPreintegrated = imu_preint_kf_;
      }

      mbReadyToInitializate = true;

      return;
    }
  } else {
    if (((int)curr_frame_.mvKeys.size() <= 100) ||
        ((sensor_ == System::kImuMonocular) &&
         (last_frame_.timestamp_ - mInitialFrame.timestamp_ > 1.0))) {
      mbReadyToInitializate = false;

      return;
    }

    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(
        mInitialFrame, curr_frame_, mvbPrevMatched, mvIniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < 100) {
      mbReadyToInitializate = false;
      return;
    }

    Sophus::SE3f Tcw;
    vector<bool> vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

    if (cam_->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,
                                      curr_frame_.mvKeysUn, mvIniMatches, Tcw,
                                      mvIniP3D, vbTriangulated)) {
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      // Set Frame Poses
      mInitialFrame.SetPose(Sophus::SE3f());
      curr_frame_.SetPose(Tcw);

      CreateInitialMapMonocular();
    }
  }
}

void Tracking::CreateInitialMapMonocular() {
  // Create KeyFrames
  KeyFrame* pKFini =
      new KeyFrame(mInitialFrame, atlas_->GetCurrentMap(), mpKeyFrameDB);
  KeyFrame* pKFcur =
      new KeyFrame(curr_frame_, atlas_->GetCurrentMap(), mpKeyFrameDB);

  if (sensor_ == System::kImuMonocular)
    pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);

  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  atlas_->AddKeyFrame(pKFini);
  atlas_->AddKeyFrame(pKFcur);

  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    if (mvIniMatches[i] < 0) continue;

    // Create MapPoint.
    Eigen::Vector3f worldPos;
    worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
    MapPoint* pMP = new MapPoint(worldPos, pKFcur, atlas_->GetCurrentMap());

    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    curr_frame_.mvpMapPoints[mvIniMatches[i]] = pMP;
    curr_frame_.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    atlas_->AddMapPoint(pMP);
  }

  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  std::set<MapPoint*> sMPs;
  sMPs = pKFini->GetMapPoints();

  // Bundle Adjustment
  Verbose::PrintMess(
      "New Map created with " + to_string(atlas_->MapPointsInMap()) + " points",
      Verbose::VERBOSITY_QUIET);
  Optimizer::GlobalBundleAdjustemnt(atlas_->GetCurrentMap(), 20);

  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth;
  if (sensor_ == System::kImuMonocular)
    invMedianDepth = 4.0f / medianDepth;  // 4.0f
  else
    invMedianDepth = 1.0f / medianDepth;

  if (medianDepth < 0 ||
      // TODO Check, originally 100 tracks
      pKFcur->TrackedMapPoints(1) < 50) {
    Verbose::PrintMess("Wrong initialization, reseting...",
                       Verbose::VERBOSITY_QUIET);
    mpSystem->ResetActiveMap();
    return;
  }

  // Scale initial baseline
  Sophus::SE3f Tc2w = pKFcur->GetPose();
  Tc2w.translation() *= invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // Scale points
  vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
      pMP->UpdateNormalAndDepth();
    }
  }

  if (sensor_ == System::kImuMonocular) {
    pKFcur->mPrevKF = pKFini;
    pKFini->mNextKF = pKFcur;
    pKFcur->mpImuPreintegrated = imu_preint_kf_;

    imu_preint_kf_ = new IMU::Preintegrated(
        pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
  }

  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);
  mpLocalMapper->mFirstTs = pKFcur->timestamp_;

  curr_frame_.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = curr_frame_.id_;
  mpLastKeyFrame = pKFcur;
  // mnLastRelocFrameId = mInitialFrame.id_;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = atlas_->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  curr_frame_.mpReferenceKF = pKFcur;

  // Compute here initial velocity
  vector<KeyFrame*> vKFs = atlas_->GetAllKeyFrames();

  Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
  mbVelocity = false;
  Eigen::Vector3f phi = deltaT.so3().log();

  double aux = (curr_frame_.timestamp_ - last_frame_.timestamp_) /
               (curr_frame_.timestamp_ - mInitialFrame.timestamp_);
  phi *= aux;

  last_frame_ = Frame(curr_frame_);

  atlas_->SetReferenceMapPoints(mvpLocalMapPoints);

  mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

  atlas_->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

  state_ = OK;

  initID = pKFcur->id_;
}

void Tracking::CreateMapInAtlas() {
  mnLastInitFrameId = curr_frame_.id_;
  atlas_->CreateNewMap();
  if (sensor_ == System::kImuStereo || sensor_ == System::kImuMonocular ||
      sensor_ == System::kImuRgbd)
    atlas_->SetInertialSensor();
  mbSetInit = false;

  mnInitialFrameId = curr_frame_.id_ + 1;
  state_ = NO_IMAGES_YET;

  // Restart the variable with information about the last KF
  mbVelocity = false;
  // mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the
  // current id, because it is the new starting point for new map
  Verbose::PrintMess(
      "First frame id in map: " + to_string(mnLastInitFrameId + 1),
      Verbose::VERBOSITY_NORMAL);
  mbVO = false;  // Init value for know if there are enough MapPoints in the
                 // last KF
  if (sensor_ == System::kMonocular || sensor_ == System::kImuMonocular) {
    mbReadyToInitializate = false;
  }

  if ((sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
       sensor_ == System::kImuRgbd) &&
      imu_preint_kf_) {
    delete imu_preint_kf_;
    imu_preint_kf_ = new IMU::Preintegrated(IMU::Bias(), *imu_calib_);
  }

  if (mpLastKeyFrame) mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

  if (mpReferenceKF) mpReferenceKF = static_cast<KeyFrame*>(NULL);

  last_frame_ = Frame();
  curr_frame_ = Frame();
  mvIniMatches.clear();

  mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < last_frame_.N; i++) {
    MapPoint* pMP = last_frame_.mvpMapPoints[i];

    if (pMP) {
      MapPoint* pRep = pMP->GetReplaced();
      if (pRep) {
        last_frame_.mvpMapPoints[i] = pRep;
      }
    }
  }
}

bool Tracking::TrackReferenceKeyFrame() {
  // Compute Bag of Words vector
  curr_frame_.ComputeBoW();

  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.7, true);
  vector<MapPoint*> vpMapPointMatches;

  int nmatches =
      matcher.SearchByBoW(mpReferenceKF, curr_frame_, vpMapPointMatches);

  if (nmatches < 15) {
    cout << "TRACK_REF_KF: Less than 15 matches!!\n";
    return false;
  }

  curr_frame_.mvpMapPoints = vpMapPointMatches;
  curr_frame_.SetPose(last_frame_.GetPose());

  // curr_frame_.PrintPointDistribution();

  // cout << " TrackReferenceKeyFrame last_frame_.mTcw:  " << last_frame_.mTcw
  // << endl;
  Optimizer::PoseOptimization(&curr_frame_);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < curr_frame_.N; i++) {
    // if(i >= curr_frame_.Nleft) break;
    if (curr_frame_.mvpMapPoints[i]) {
      if (curr_frame_.mvbOutlier[i]) {
        MapPoint* pMP = curr_frame_.mvpMapPoints[i];

        curr_frame_.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        curr_frame_.mvbOutlier[i] = false;
        if (i < curr_frame_.Nleft) {
          pMP->mbTrackInView = false;
        } else {
          pMP->mbTrackInViewR = false;
        }
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = curr_frame_.id_;
        nmatches--;
      } else if (curr_frame_.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
      sensor_ == System::kImuRgbd)
    return true;
  else
    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame* pRef = last_frame_.mpReferenceKF;
  Sophus::SE3f Tlr = mlRelativeFramePoses.back();
  last_frame_.SetPose(Tlr * pRef->GetPose());

  if (mnLastKeyFrameId == last_frame_.id_ || sensor_ == System::kMonocular ||
      sensor_ == System::kImuMonocular || !mbOnlyTracking)
    return;

  // Create "visual odometry" MapPoints
  // We sort points according to their measured depth by the stereo/RGB-D sensor
  vector<pair<float, int>> vDepthIdx;
  const int Nfeat = last_frame_.Nleft == -1 ? last_frame_.N : last_frame_.Nleft;
  vDepthIdx.reserve(Nfeat);
  for (int i = 0; i < Nfeat; i++) {
    float z = last_frame_.mvDepth[i];
    if (z > 0) {
      vDepthIdx.push_back(make_pair(z, i));
    }
  }

  if (vDepthIdx.empty()) return;

  sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth<th_depth_)
  // If less than 100 close points, we insert the 100 closest ones.
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); j++) {
    int i = vDepthIdx[j].second;

    bool bCreateNew = false;

    MapPoint* pMP = last_frame_.mvpMapPoints[i];

    if (!pMP)
      bCreateNew = true;
    else if (pMP->Observations() < 1)
      bCreateNew = true;

    if (bCreateNew) {
      Eigen::Vector3f x3D;

      if (last_frame_.Nleft == -1) {
        last_frame_.UnprojectStereo(i, x3D);
      } else {
        x3D = last_frame_.UnprojectStereoFishEye(i);
      }

      MapPoint* pNewMP =
          new MapPoint(x3D, atlas_->GetCurrentMap(), &last_frame_, i);
      last_frame_.mvpMapPoints[i] = pNewMP;

      mlpTemporalPoints.push_back(pNewMP);
      nPoints++;
    } else {
      nPoints++;
    }

    if (vDepthIdx[j].first > th_depth_ && nPoints > 100) break;
  }
}

bool Tracking::TrackWithMotionModel() {
  ORBmatcher matcher(0.9, true);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();

  if (atlas_->isImuInitialized() &&
      (curr_frame_.id_ > mnLastRelocFrameId + mnFramesToResetIMU)) {
    // TODO: Check, if this change is good?
    //  Predict state with IMU if it is initialized and it doesnt need reset
    PredictStateIMU();
    if (!mpLocalMapper->IsInitializing() || !mpLocalMapper->isStopped())
      return true;
  } else {
    curr_frame_.SetPose(mVelocity * last_frame_.GetPose());
  }

  fill(curr_frame_.mvpMapPoints.begin(), curr_frame_.mvpMapPoints.end(),
       static_cast<MapPoint*>(NULL));

  // Project points seen in previous frame
  int th;

  if (sensor_ == System::kStereo)
    th = 7;
  else
    th = 15;

  int nmatches = matcher.SearchByProjection(
      curr_frame_, last_frame_, th,
      sensor_ == System::kMonocular || sensor_ == System::kImuMonocular);

  // If few matches, uses a wider window search
  if (nmatches < 20) {
    Verbose::PrintMess("Not enough matches, wider window search!!",
                       Verbose::VERBOSITY_NORMAL);
    fill(curr_frame_.mvpMapPoints.begin(), curr_frame_.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));

    nmatches = matcher.SearchByProjection(
        curr_frame_, last_frame_, 2 * th,
        sensor_ == System::kMonocular || sensor_ == System::kImuMonocular);
    Verbose::PrintMess("Matches with wider search: " + to_string(nmatches),
                       Verbose::VERBOSITY_NORMAL);
  }

  if (nmatches < 20) {
    Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
    if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
        sensor_ == System::kImuRgbd)
      return true;
    else
      return false;
  }

  // Optimize frame pose with all matches
  Optimizer::PoseOptimization(&curr_frame_);

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < curr_frame_.N; i++) {
    if (curr_frame_.mvpMapPoints[i]) {
      if (curr_frame_.mvbOutlier[i]) {
        MapPoint* pMP = curr_frame_.mvpMapPoints[i];

        curr_frame_.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        curr_frame_.mvbOutlier[i] = false;
        if (i < curr_frame_.Nleft) {
          pMP->mbTrackInView = false;
        } else {
          pMP->mbTrackInViewR = false;
        }
        pMP->mnLastFrameSeen = curr_frame_.id_;
        nmatches--;
      } else if (curr_frame_.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > 20;
  }

  if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
      sensor_ == System::kImuRgbd)
    return true;
  else
    return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the
  // frame. We retrieve the local map and try to find matches to points in the
  // local map.
  mTrackedFr++;

  UpdateLocalMap();
  SearchLocalPoints();

  // TOO check outliers before PO
  int aux1 = 0, aux2 = 0;
  for (int i = 0; i < curr_frame_.N; i++)
    if (curr_frame_.mvpMapPoints[i]) {
      aux1++;
      if (curr_frame_.mvbOutlier[i]) aux2++;
    }

  int inliers;
  if (!atlas_->isImuInitialized())
    Optimizer::PoseOptimization(&curr_frame_);
  else {
    if (curr_frame_.id_ <= mnLastRelocFrameId + mnFramesToResetIMU) {
      Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
      Optimizer::PoseOptimization(&curr_frame_);
    } else {
      // if(!mbMapUpdated && state_ == OK) //  && (mnMatchesInliers>30))
      if (!mbMapUpdated)  //  && (mnMatchesInliers>30))
      {
        Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ",
                           Verbose::VERBOSITY_DEBUG);
        inliers = Optimizer::PoseInertialOptimizationLastFrame(
            &curr_frame_);  // ,
                            // !mpLastKeyFrame->GetMap()->GetIniertialBA1());
      } else {
        Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ",
                           Verbose::VERBOSITY_DEBUG);
        inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(
            &curr_frame_);  // ,
                            // !mpLastKeyFrame->GetMap()->GetIniertialBA1());
      }
    }
  }

  aux1 = 0, aux2 = 0;
  for (int i = 0; i < curr_frame_.N; i++)
    if (curr_frame_.mvpMapPoints[i]) {
      aux1++;
      if (curr_frame_.mvbOutlier[i]) aux2++;
    }

  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < curr_frame_.N; i++) {
    if (curr_frame_.mvpMapPoints[i]) {
      if (!curr_frame_.mvbOutlier[i]) {
        curr_frame_.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (curr_frame_.mvpMapPoints[i]->Observations() > 0)
            mnMatchesInliers++;
        } else
          mnMatchesInliers++;
      } else if (sensor_ == System::kStereo)
        curr_frame_.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }
  }

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
  if (curr_frame_.id_ < mnLastRelocFrameId + max_frames_ &&
      mnMatchesInliers < 50)
    return false;

  if ((mnMatchesInliers > 10) && (state_ == RECENTLY_LOST)) return true;

  if (sensor_ == System::kImuMonocular) {
    if ((mnMatchesInliers < 15 && atlas_->isImuInitialized()) ||
        (mnMatchesInliers < 50 && !atlas_->isImuInitialized())) {
      return false;
    } else
      return true;
  } else if (sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
    if (mnMatchesInliers < 15) {
      return false;
    } else
      return true;
  } else {
    if (mnMatchesInliers < 30)
      return false;
    else
      return true;
  }
}

bool Tracking::NeedNewKeyFrame() {
  if ((sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
       sensor_ == System::kImuRgbd) &&
      !atlas_->GetCurrentMap()->isImuInitialized()) {
    if (sensor_ == System::kImuMonocular &&
        (curr_frame_.timestamp_ - mpLastKeyFrame->timestamp_) >= 0.25)
      return true;
    else if ((sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) &&
             (curr_frame_.timestamp_ - mpLastKeyFrame->timestamp_) >= 0.25)
      return true;
    else
      return false;
  }

  if (mbOnlyTracking) return false;

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
    /*if(sensor_ == System::kMonocular)
    {
        std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
    }*/
    return false;
  }

  const int nKFs = atlas_->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last
  // relocalisation
  if (curr_frame_.id_ < mnLastRelocFrameId + max_frames_ &&
      nKFs > max_frames_) {
    return false;
  }

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 3;
  if (nKFs <= 2) nMinObs = 2;
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be
  // potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose = 0;

  if (sensor_ != System::kMonocular && sensor_ != System::kImuMonocular) {
    int N = (curr_frame_.Nleft == -1) ? curr_frame_.N : curr_frame_.Nleft;
    for (int i = 0; i < N; i++) {
      if (curr_frame_.mvDepth[i] > 0 && curr_frame_.mvDepth[i] < th_depth_) {
        if (curr_frame_.mvpMapPoints[i] && !curr_frame_.mvbOutlier[i])
          nTrackedClose++;
        else
          nNonTrackedClose++;
      }
    }
    // Verbose::PrintMess("[NEEDNEWKF]-> closed points: " +
    // to_string(nTrackedClose) + "; non tracked closed points: " +
    // to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);//
    // Verbose::VERBOSITY_DEBUG);
  }

  bool bNeedToInsertClose;
  bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

  // Thresholds
  float thRefRatio = 0.75f;
  if (nKFs < 2) thRefRatio = 0.4f;

  /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
  const int thStereoClosedPoints = 15;
  if(nClosedPoints < thStereoClosedPoints && (sensor_==System::kStereo ||
  sensor_==System::kImuStereo))
  {
      //Pseudo-monocular, there are not enough close points to be confident
  about the stereo observations. thRefRatio = 0.9f;
  }*/

  if (sensor_ == System::kMonocular) thRefRatio = 0.9f;

  if (cam2_) thRefRatio = 0.75f;

  if (sensor_ == System::kImuMonocular) {
    if (mnMatchesInliers > 350)  // Points tracked from the local map
      thRefRatio = 0.75f;
    else
      thRefRatio = 0.90f;
  }

  // Condition 1a: More than "MaxFrames" have passed from last keyframe
  // insertion
  const bool c1a = curr_frame_.id_ >= mnLastKeyFrameId + max_frames_;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b =
      ((curr_frame_.id_ >= mnLastKeyFrameId + min_frames_) &&
       bLocalMappingIdle);  // mpLocalMapper->KeyframesInQueue() < 2);
  // Condition 1c: tracking is weak
  const bool c1c =
      sensor_ != System::kMonocular && sensor_ != System::kImuMonocular &&
      sensor_ != System::kImuStereo && sensor_ != System::kImuRgbd &&
      (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of
  // visual odometry compared to map matches.
  const bool c2 =
      (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) &&
       mnMatchesInliers > 15);

  // std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c
  // << "; c2=" << c2 << std::endl;
  //  Temporal condition for Inertial cases
  bool c3 = false;
  if (mpLastKeyFrame) {
    if (sensor_ == System::kImuMonocular) {
      if ((curr_frame_.timestamp_ - mpLastKeyFrame->timestamp_) >= 0.5)
        c3 = true;
    } else if (sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd) {
      if ((curr_frame_.timestamp_ - mpLastKeyFrame->timestamp_) >= 0.5)
        c3 = true;
    }
  }

  bool c4 = false;
  if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) ||
       state_ == RECENTLY_LOST) &&
      (sensor_ ==
       System::kImuMonocular))  // MODIFICATION_2, originally
                                // ((((mnMatchesInliers<75) &&
                                // (mnMatchesInliers>15)) ||
                                // state_==RECENTLY_LOST) && ((sensor_ ==
                                // System::kImuMonocular)))
    c4 = true;
  else
    c4 = false;

  if (((c1a || c1b || c1c) && c2) || c3 || c4) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle || mpLocalMapper->IsInitializing()) {
      return true;
    } else {
      mpLocalMapper->InterruptBA();
      if (sensor_ != System::kMonocular && sensor_ != System::kImuMonocular) {
        if (mpLocalMapper->KeyframesInQueue() < 3)
          return true;
        else
          return false;
      } else {
        // std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
        return false;
      }
    }
  } else
    return false;
}

void Tracking::CreateNewKeyFrame() {
  if (mpLocalMapper->IsInitializing() && !atlas_->isImuInitialized()) return;

  if (!mpLocalMapper->SetNotStop(true)) return;

  KeyFrame* pKF =
      new KeyFrame(curr_frame_, atlas_->GetCurrentMap(), mpKeyFrameDB);

  if (atlas_->isImuInitialized())  //  || mpLocalMapper->IsInitializing())
    pKF->bImu = true;

  pKF->SetNewBias(curr_frame_.mImuBias);
  mpReferenceKF = pKF;
  curr_frame_.mpReferenceKF = pKF;

  if (mpLastKeyFrame) {
    pKF->mPrevKF = mpLastKeyFrame;
    mpLastKeyFrame->mNextKF = pKF;
  } else
    Verbose::PrintMess("No last KF in KF creation!!",
                       Verbose::VERBOSITY_NORMAL);

  // Reset preintegration from last KF (Create new object)
  if (sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
      sensor_ == System::kImuRgbd) {
    imu_preint_kf_ = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
  }

  if (sensor_ != System::kMonocular &&
      // TODO check if incluide imu_stereo
      sensor_ != System::kImuMonocular) {
    curr_frame_.UpdatePoseMatrices();
    // cout << "create new MPs" << endl;
    // We sort points by the measured depth by the stereo/kRgbd sensor.
    // We create all those MapPoints whose depth < th_depth_.
    // If there are less than 100 close points we create the 100 closest.
    int maxPoint = 100;
    if (sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd)
      maxPoint = 100;

    vector<pair<float, int>> vDepthIdx;
    int N = (curr_frame_.Nleft != -1) ? curr_frame_.Nleft : curr_frame_.N;
    vDepthIdx.reserve(curr_frame_.N);
    for (int i = 0; i < N; i++) {
      float z = curr_frame_.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(make_pair(z, i));
      }
    }

    if (!vDepthIdx.empty()) {
      sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;
      for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = curr_frame_.mvpMapPoints[i];
        if (!pMP)
          bCreateNew = true;
        else if (pMP->Observations() < 1) {
          bCreateNew = true;
          curr_frame_.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        if (bCreateNew) {
          Eigen::Vector3f x3D;

          if (curr_frame_.Nleft == -1) {
            curr_frame_.UnprojectStereo(i, x3D);
          } else {
            x3D = curr_frame_.UnprojectStereoFishEye(i);
          }

          MapPoint* pNewMP = new MapPoint(x3D, pKF, atlas_->GetCurrentMap());
          pNewMP->AddObservation(pKF, i);

          // Check if it is a stereo observation in order to not
          // duplicate mappoints
          if (curr_frame_.Nleft != -1 &&
              curr_frame_.mvLeftToRightMatch[i] >= 0) {
            curr_frame_.mvpMapPoints[curr_frame_.Nleft +
                                     curr_frame_.mvLeftToRightMatch[i]] =
                pNewMP;
            pNewMP->AddObservation(
                pKF, curr_frame_.Nleft + curr_frame_.mvLeftToRightMatch[i]);
            pKF->AddMapPoint(
                pNewMP, curr_frame_.Nleft + curr_frame_.mvLeftToRightMatch[i]);
          }

          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          atlas_->AddMapPoint(pNewMP);

          curr_frame_.mvpMapPoints[i] = pNewMP;
          nPoints++;
        } else {
          nPoints++;
        }

        if (vDepthIdx[j].first > th_depth_ && nPoints > maxPoint) {
          break;
        }
      }
      // Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints),
      // Verbose::VERBOSITY_NORMAL);
    }
  }

  mpLocalMapper->InsertKeyFrame(pKF);

  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = curr_frame_.id_;
  mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  for (vector<MapPoint*>::iterator vit = curr_frame_.mvpMapPoints.begin(),
                                   vend = curr_frame_.mvpMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint*>(NULL);
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = curr_frame_.id_;
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
      }
    }
  }

  int nToMatch = 0;

  // Project points in frame and check its visibility
  for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(),
                                   vend = mvpLocalMapPoints.end();
       vit != vend; vit++) {
    MapPoint* pMP = *vit;

    if (pMP->mnLastFrameSeen == curr_frame_.id_) continue;
    if (pMP->isBad()) continue;
    // Project (this fills MapPoint variables for matching)
    if (curr_frame_.isInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      nToMatch++;
    }
    if (pMP->mbTrackInView) {
      curr_frame_.mmProjectPoints[pMP->id_] =
          cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
    }
  }

  if (nToMatch > 0) {
    ORBmatcher matcher(0.8);
    int th = 1;
    if (sensor_ == System::kRgbd || sensor_ == System::kImuRgbd) th = 3;
    if (atlas_->isImuInitialized()) {
      if (atlas_->GetCurrentMap()->GetIniertialBA2())
        th = 2;
      else
        th = 6;
    } else if (!atlas_->isImuInitialized() &&
               (sensor_ == System::kImuMonocular ||
                sensor_ == System::kImuStereo || sensor_ == System::kImuRgbd)) {
      th = 10;
    }

    // If the camera has been relocalised recently, perform a coarser search
    if (curr_frame_.id_ < mnLastRelocFrameId + 2) th = 5;

    if (state_ == LOST ||
        state_ == RECENTLY_LOST)  // Lost for less than 1 second
      th = 15;                    // 15

    int matches = matcher.SearchByProjection(curr_frame_, mvpLocalMapPoints, th,
                                             mpLocalMapper->mbFarPoints,
                                             mpLocalMapper->mThFarPoints);
  }
}

void Tracking::UpdateLocalMap() {
  // This is for visualization
  atlas_->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
  mvpLocalMapPoints.clear();

  int count_pts = 0;

  for (vector<KeyFrame*>::const_reverse_iterator
           itKF = mvpLocalKeyFrames.rbegin(),
           itEndKF = mvpLocalKeyFrames.rend();
       itKF != itEndKF; ++itKF) {
    KeyFrame* pKF = *itKF;
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(),
                                           itEndMP = vpMPs.end();
         itMP != itEndMP; itMP++) {
      MapPoint* pMP = *itMP;
      if (!pMP) continue;
      if (pMP->mnTrackReferenceForFrame == curr_frame_.id_) continue;
      if (!pMP->isBad()) {
        count_pts++;
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = curr_frame_.id_;
      }
    }
  }
}

void Tracking::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  map<KeyFrame*, int> keyframeCounter;
  if (!atlas_->isImuInitialized() ||
      (curr_frame_.id_ < mnLastRelocFrameId + 2)) {
    for (int i = 0; i < curr_frame_.N; i++) {
      MapPoint* pMP = curr_frame_.mvpMapPoints[i];
      if (pMP) {
        if (!pMP->isBad()) {
          const map<KeyFrame*, tuple<int, int>> observations =
              pMP->GetObservations();
          for (map<KeyFrame*, tuple<int, int>>::const_iterator
                   it = observations.begin(),
                   itend = observations.end();
               it != itend; it++)
            keyframeCounter[it->first]++;
        } else {
          curr_frame_.mvpMapPoints[i] = NULL;
        }
      }
    }
  } else {
    for (int i = 0; i < last_frame_.N; i++) {
      // Using lastframe since current frame has not matches yet
      if (last_frame_.mvpMapPoints[i]) {
        MapPoint* pMP = last_frame_.mvpMapPoints[i];
        if (!pMP) continue;
        if (!pMP->isBad()) {
          const map<KeyFrame*, tuple<int, int>> observations =
              pMP->GetObservations();
          for (map<KeyFrame*, tuple<int, int>>::const_iterator
                   it = observations.begin(),
                   itend = observations.end();
               it != itend; it++)
            keyframeCounter[it->first]++;
        } else {
          // MODIFICATION
          last_frame_.mvpMapPoints[i] = NULL;
        }
      }
    }
  }

  int max = 0;
  KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also
  // check which keyframe shares most points
  for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(),
                                           itEnd = keyframeCounter.end();
       it != itEnd; it++) {
    KeyFrame* pKF = it->first;

    if (pKF->isBad()) continue;

    if (it->second > max) {
      max = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(pKF);
    pKF->mnTrackReferenceForFrame = curr_frame_.id_;
  }

  // Include also some not-already-included keyframes that are neighbors to
  // already-included keyframes
  for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                         itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF; itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80)  // 80
      break;

    KeyFrame* pKF = *itKF;

    const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

    for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(),
                                           itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF; itNeighKF++) {
      KeyFrame* pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != curr_frame_.id_) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = curr_frame_.id_;
          break;
        }
      }
    }

    const set<KeyFrame*> spChilds = pKF->GetChilds();
    for (set<KeyFrame*>::const_iterator sit = spChilds.begin(),
                                        send = spChilds.end();
         sit != send; sit++) {
      KeyFrame* pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != curr_frame_.id_) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = curr_frame_.id_;
          break;
        }
      }
    }

    KeyFrame* pParent = pKF->GetParent();
    if (pParent) {
      if (pParent->mnTrackReferenceForFrame != curr_frame_.id_) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = curr_frame_.id_;
        break;
      }
    }
  }

  // Add 10 last temporal KFs (mainly for IMU)
  if ((sensor_ == System::kImuMonocular || sensor_ == System::kImuStereo ||
       sensor_ == System::kImuRgbd) &&
      mvpLocalKeyFrames.size() < 80) {
    KeyFrame* tempKeyFrame = curr_frame_.mpLastKeyFrame;

    const int Nd = 20;
    for (int i = 0; i < Nd; i++) {
      if (!tempKeyFrame) break;
      if (tempKeyFrame->mnTrackReferenceForFrame != curr_frame_.id_) {
        mvpLocalKeyFrames.push_back(tempKeyFrame);
        tempKeyFrame->mnTrackReferenceForFrame = curr_frame_.id_;
        tempKeyFrame = tempKeyFrame->mPrevKF;
      }
    }
  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    curr_frame_.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracking::Relocalization() {
  Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
  // Compute Bag of Words Vector
  curr_frame_.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for
  // relocalisation
  vector<KeyFrame*> vpCandidateKFs =
      mpKeyFrameDB->DetectRelocalizationCandidates(&curr_frame_,
                                                   atlas_->GetCurrentMap());

  if (vpCandidateKFs.empty()) {
    Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
    return false;
  }

  const int nKFs = vpCandidateKFs.size();

  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.75, true);

  vector<MLPnPsolver*> vpMLPnPsolvers;
  vpMLPnPsolvers.resize(nKFs);

  vector<vector<MapPoint*>> vvpMapPointMatches;
  vvpMapPointMatches.resize(nKFs);

  vector<bool> vbDiscarded;
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;

  for (int i = 0; i < nKFs; i++) {
    KeyFrame* pKF = vpCandidateKFs[i];
    if (pKF->isBad())
      vbDiscarded[i] = true;
    else {
      int nmatches =
          matcher.SearchByBoW(pKF, curr_frame_, vvpMapPointMatches[i]);
      if (nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        MLPnPsolver* pSolver =
            new MLPnPsolver(curr_frame_, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(
            0.99, 10, 300, 6, 0.5,
            5.991);  // This solver needs at least 6 points
        vpMLPnPsolvers[i] = pSolver;
        nCandidates++;
      }
    }
  }

  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;
  ORBmatcher matcher2(0.9, true);

  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nKFs; i++) {
      if (vbDiscarded[i]) continue;

      // Perform 5 Ransac Iterations
      vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;

      MLPnPsolver* pSolver = vpMLPnPsolvers[i];
      Eigen::Matrix4f eigTcw;
      bool bTcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers, eigTcw);

      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      // If a Camera Pose is computed, optimize
      if (bTcw) {
        Sophus::SE3f Tcw(eigTcw);
        curr_frame_.SetPose(Tcw);
        // Tcw.copyTo(curr_frame_.mTcw);

        set<MapPoint*> sFound;

        const int np = vbInliers.size();

        for (int j = 0; j < np; j++) {
          if (vbInliers[j]) {
            curr_frame_.mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else
            curr_frame_.mvpMapPoints[j] = NULL;
        }

        int nGood = Optimizer::PoseOptimization(&curr_frame_);

        if (nGood < 10) continue;

        for (int io = 0; io < curr_frame_.N; io++)
          if (curr_frame_.mvbOutlier[io])
            curr_frame_.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

        // If few inliers, search by projection in a coarse window and optimize
        // again
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(
              curr_frame_, vpCandidateKFs[i], sFound, 10, 100);

          if (nadditional + nGood >= 50) {
            nGood = Optimizer::PoseOptimization(&curr_frame_);

            // If many inliers but still not enough, search by projection again
            // in a narrower window the camera has been already optimized with
            // many points
            if (nGood > 30 && nGood < 50) {
              sFound.clear();
              for (int ip = 0; ip < curr_frame_.N; ip++)
                if (curr_frame_.mvpMapPoints[ip])
                  sFound.insert(curr_frame_.mvpMapPoints[ip]);
              nadditional = matcher2.SearchByProjection(
                  curr_frame_, vpCandidateKFs[i], sFound, 3, 64);

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = Optimizer::PoseOptimization(&curr_frame_);

                for (int io = 0; io < curr_frame_.N; io++)
                  if (curr_frame_.mvbOutlier[io])
                    curr_frame_.mvpMapPoints[io] = NULL;
              }
            }
          }
        }

        // If the pose is supported by enough inliers stop ransacs and continue
        if (nGood >= 50) {
          bMatch = true;
          break;
        }
      }
    }
  }

  if (!bMatch) {
    return false;
  } else {
    mnLastRelocFrameId = curr_frame_.id_;
    cout << "Relocalized!!" << endl;
    return true;
  }
}

void Tracking::Reset(bool bLocMap) {
  Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped()) usleep(3000);
  }

  // Reset Local Mapping
  if (!bLocMap) {
    Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
    mpLocalMapper->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
  }

  // Reset Loop Closing
  Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
  mpLoopClosing->RequestReset();
  Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

  // Clear BoW Database
  Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
  mpKeyFrameDB->clear();
  Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

  // Clear Map (this erase MapPoints and KeyFrames)
  atlas_->clearAtlas();
  atlas_->CreateNewMap();
  if (sensor_ == System::kImuStereo || sensor_ == System::kImuMonocular ||
      sensor_ == System::kImuRgbd)
    atlas_->SetInertialSensor();
  mnInitialFrameId = 0;

  KeyFrame::next_id_ = 0;
  Frame::next_id_ = 0;
  state_ = NO_IMAGES_YET;

  mbReadyToInitializate = false;
  mbSetInit = false;

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();
  curr_frame_ = Frame();
  mnLastRelocFrameId = 0;
  last_frame_ = Frame();
  mpReferenceKF = static_cast<KeyFrame*>(NULL);
  mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
  mvIniMatches.clear();

  if (mpViewer) mpViewer->Release();

  Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap) {
  Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped()) usleep(3000);
  }

  Map* pMap = atlas_->GetCurrentMap();

  if (!bLocMap) {
    Verbose::PrintMess("Reseting Local Mapper...",
                       Verbose::VERBOSITY_VERY_VERBOSE);
    mpLocalMapper->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
  }

  // Reset Loop Closing
  Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
  mpLoopClosing->RequestResetActiveMap(pMap);
  Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

  // Clear BoW Database
  Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
  mpKeyFrameDB->clearMap(pMap);  // Only clear the active map references
  Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

  // Clear Map (this erase MapPoints and KeyFrames)
  atlas_->clearMap();

  // KeyFrame::next_id_ = atlas_->GetLastInitKFid();
  // Frame::next_id_ = mnLastInitFrameId;
  mnLastInitFrameId = Frame::next_id_;
  // mnLastRelocFrameId = mnLastInitFrameId;
  state_ = NO_IMAGES_YET;  // NOT_INITIALIZED;

  mbReadyToInitializate = false;

  list<bool> lbLost;
  // lbLost.reserve(mlbLost.size());
  unsigned int index = mnFirstFrameId;
  cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
  for (Map* pMap : atlas_->GetAllMaps()) {
    if (pMap->GetAllKeyFrames().size() > 0) {
      if (index > pMap->GetLowerKFID()) index = pMap->GetLowerKFID();
    }
  }

  // cout << "First Frame id: " << index << endl;
  int num_lost = 0;
  cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

  for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end();
       ilbL++) {
    if (index < mnInitialFrameId)
      lbLost.push_back(*ilbL);
    else {
      lbLost.push_back(true);
      num_lost += 1;
    }

    index++;
  }
  cout << num_lost << " Frames set to lost" << endl;

  mlbLost = lbLost;

  mnInitialFrameId = curr_frame_.id_;
  mnLastRelocFrameId = curr_frame_.id_;

  curr_frame_ = Frame();
  last_frame_ = Frame();
  mpReferenceKF = static_cast<KeyFrame*>(NULL);
  mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
  mvIniMatches.clear();

  mbVelocity = false;

  if (mpViewer) mpViewer->Release();

  Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS() { return mvpLocalMapPoints; }

void Tracking::ChangeCalibration(const string& strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  eig_K_.setIdentity();
  eig_K_(0, 0) = fx;
  eig_K_(1, 1) = fy;
  eig_K_(0, 2) = cx;
  eig_K_(1, 2) = cy;

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(cv_K_);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(dist_coef_);

  bf_ = fSettings["Camera.bf"];

  Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool& flag) { mbOnlyTracking = flag; }

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias& b,
                              KeyFrame* pCurrentKeyFrame) {
  Map* pMap = pCurrentKeyFrame->GetMap();
  unsigned int index = mnFirstFrameId;
  list<ORB_SLAM_FUSION::KeyFrame*>::iterator lRit = mlpReferences.begin();
  list<bool>::iterator lbL = mlbLost.begin();
  for (auto lit = mlRelativeFramePoses.begin(),
            lend = mlRelativeFramePoses.end();
       lit != lend; lit++, lRit++, lbL++) {
    if (*lbL) continue;

    KeyFrame* pKF = *lRit;

    while (pKF->isBad()) {
      pKF = pKF->GetParent();
    }

    if (pKF->GetMap() == pMap) {
      (*lit).translation() *= s;
    }
  }

  mLastBias = b;

  mpLastKeyFrame = pCurrentKeyFrame;

  last_frame_.SetNewBias(mLastBias);
  curr_frame_.SetNewBias(mLastBias);

  while (!curr_frame_.imuIsPreintegrated()) {
    usleep(500);
  }

  if (last_frame_.id_ == last_frame_.mpLastKeyFrame->mnFrameId) {
    last_frame_.SetImuPoseVelocity(last_frame_.mpLastKeyFrame->GetImuRotation(),
                                   last_frame_.mpLastKeyFrame->GetImuPosition(),
                                   last_frame_.mpLastKeyFrame->GetVelocity());
  } else {
    const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
    const Eigen::Vector3f twb1 = last_frame_.mpLastKeyFrame->GetImuPosition();
    const Eigen::Matrix3f Rwb1 = last_frame_.mpLastKeyFrame->GetImuRotation();
    const Eigen::Vector3f Vwb1 = last_frame_.mpLastKeyFrame->GetVelocity();
    float t12 = last_frame_.mpImuPreintegrated->dT;

    last_frame_.SetImuPoseVelocity(
        IMU::NormalizeRotation(
            Rwb1 * last_frame_.mpImuPreintegrated->GetUpdatedDeltaRotation()),
        twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
            Rwb1 * last_frame_.mpImuPreintegrated->GetUpdatedDeltaPosition(),
        Vwb1 + Gz * t12 +
            Rwb1 * last_frame_.mpImuPreintegrated->GetUpdatedDeltaVelocity());
  }

  if (curr_frame_.mpImuPreintegrated) {
    const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

    const Eigen::Vector3f twb1 = curr_frame_.mpLastKeyFrame->GetImuPosition();
    const Eigen::Matrix3f Rwb1 = curr_frame_.mpLastKeyFrame->GetImuRotation();
    const Eigen::Vector3f Vwb1 = curr_frame_.mpLastKeyFrame->GetVelocity();
    float t12 = curr_frame_.mpImuPreintegrated->dT;

    curr_frame_.SetImuPoseVelocity(
        IMU::NormalizeRotation(
            Rwb1 * curr_frame_.mpImuPreintegrated->GetUpdatedDeltaRotation()),
        twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
            Rwb1 * curr_frame_.mpImuPreintegrated->GetUpdatedDeltaPosition(),
        Vwb1 + Gz * t12 +
            Rwb1 * curr_frame_.mpImuPreintegrated->GetUpdatedDeltaVelocity());
  }

  mnFirstImuFrameId = curr_frame_.id_;
}

void Tracking::NewDataset() { num_dataset_++; }

int Tracking::GetNumberDataset() { return num_dataset_; }

int Tracking::GetMatchesInliers() { return mnMatchesInliers; }

void Tracking::SaveSubTrajectory(string strNameFile_frames,
                                 string strNameFile_kf, string strFolder) {
  mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
  // mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames,
                                 string strNameFile_kf, Map* pMap) {
  mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
  if (!strNameFile_kf.empty())
    mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale() { return img_scale_; }

}  // namespace ORB_SLAM_FUSION
