#include "cam_processing.h"

#include "system.h"

using namespace std;

namespace ORB_SLAM_FUSION {

CamProcessing::CamProcessing(System* sys, ORBVocabulary* voc,
                             FrameDrawer* frame_drawer, MapDrawer* map_drawer,
                             Atlas* atlas, KeyFrameDatabase* kf_database,
                             const int sensor, Settings* settings)
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
  CamParamLoader(settings);
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

void CamProcessing::CamParamLoader(Settings* settings) {
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
  int num_features = settings->nFeatures();
  int num_levels = settings->nLevels();
  int init_th_fast = settings->initThFAST();
  int min_th_fast = settings->minThFAST();
  float scale_factor = settings->scaleFactor();

  orb_extractor_left_ = new ORBextractor(num_features, scale_factor, num_levels,
                                         init_th_fast, min_th_fast);

  if (sensor_ == System::kStereo || sensor_ == System::kImuStereo)
    orb_extractor_right_ = new ORBextractor(
        num_features, scale_factor, num_levels, init_th_fast, min_th_fast);

  if (sensor_ == System::kMonocular || sensor_ == System::kImuMonocular)
    mpIniORBextractor = new ORBextractor(5 * num_features, scale_factor,
                                         num_levels, init_th_fast, min_th_fast);
}

Sophus::SE3f CamProcessing::GrabImageStereo(const cv::Mat& img_rect_left,
                                            const cv::Mat& img_rect_right,
                                            const double& timestamp,
                                            string filename) {
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

bool CamProcessing::PredictStateIMU() {
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

void CamProcessing::Track() {
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
      CreateMapInAtlas();
      return;
    } else if (curr_frame_.timestamp_ > last_frame_.timestamp_ + 1.0) {
      // cout << curr_frame_.timestamp_ << ", " << last_frame_.timestamp_ <<
      // endl; cout << "id last: " << last_frame_.mnId << "    id curr: " <<
      // curr_frame_.mnId << endl;
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

  //TODO: Get imu preintegrated from imu processing
  PreintegrateIMU();

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
    StereoInitialization();

    // frame_drawer_->Update(this);

    if (state_ != OK)  // If rightly initialized, state_=OK
    {
      last_frame_ = Frame(curr_frame_);
      return;
    }

    if (atlas_->GetAllMaps().size() == 1) {
      mnFirstFrameId = curr_frame_.mnId;
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
            curr_frame_.mnId < mnLastRelocFrameId + 2) {
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
          if (curr_frame_.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
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

      /*if(curr_frame_.mnId>mnLastRelocFrameId+max_frames_)
      {*/
      mTimeStampLost = curr_frame_.timestamp_;
      //}
    }

    // Save frame if recent relocalization, since they are used for IMU reset
    // (as we are making copy, it shluld be once mCurrFrame is completely
    // modified)
    if ((curr_frame_.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) &&
        (curr_frame_.mnId > mnFramesToResetIMU) &&
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
        if (curr_frame_.mnId == (mnLastRelocFrameId + mnFramesToResetIMU)) {
          cout << "RESETING FRAME!!!" << endl;
          ResetFrameIMU();
        } else if (curr_frame_.mnId > (mnLastRelocFrameId + 30))
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

}  // namespace ORB_SLAM_FUSION