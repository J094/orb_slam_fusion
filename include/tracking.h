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

#ifndef TRACKING_H
#define TRACKING_H

#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <unordered_set>

#include "map/atlas.h"
#include "map/frame.h"
#include "utils/frame_drawer.h"
#include "cam/camera_models/geometric_camera.h"
#include "imu/imu_types.h"
#include "map/keyframe_database.h"
#include "localmapping.h"
#include "loopclosing.h"
#include "utils/map_drawer.h"
#include "cam/orb_feature/orb_vocabulary.h"
#include "cam/orb_feature/orb_extractor.h"
#include "config/settings.h"
#include "system.h"
#include "viewer.h"

namespace ORB_SLAM_FUSION {

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

class Tracking {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Tracking(System* sys, ORBVocabulary* voc, FrameDrawer* frame_drawer,
           MapDrawer* map_drawer, Atlas* atlas, KeyFrameDatabase* kf_database,
           const string& strSettingPath, const int sensor, Settings* settings,
           const string& _nameSeq = std::string());

  ~Tracking();

  // Parse the config file
  bool ParseCamParamFile(cv::FileStorage& fSettings);
  bool ParseORBParamFile(cv::FileStorage& fSettings);
  bool ParseIMUParamFile(cv::FileStorage& fSettings);

  // Preprocess the input and call Track(). Extract features and performs stereo
  // matching.
  Sophus::SE3f GrabImageStereo(const cv::Mat& img_rect_left,
                               const cv::Mat& img_rect_right,
                               const double& timestamp, string filename);
  Sophus::SE3f GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD,
                             const double& timestamp, string filename);
  Sophus::SE3f GrabImageMonocular(const cv::Mat& im, const double& timestamp,
                                  string filename);

  void GrabImuData(const IMU::Point& imuMeasurement);

  void SetLocalMapper(LocalMapping* pLocalMapper);
  void SetLoopClosing(LoopClosing* pLoopClosing);
  void SetViewer(Viewer* viewer);
  void SetStepByStep(bool bSet);
  bool GetStepByStep();

  // Load new settings
  // The focal lenght should be similar or scale prediction will fail when
  // projecting points
  void ChangeCalibration(const string& strSettingPath);

  // Use this function if you have deactivated local mapping and you only want
  // to localize the camera.
  void InformOnlyTracking(const bool& flag);

  void UpdateFrameIMU(const float s, const IMU::Bias& b,
                      KeyFrame* pCurrentKeyFrame);
  KeyFrame* GetLastKeyFrame() { return mpLastKeyFrame; }

  void CreateMapInAtlas();
  // std::mutex mMutexTracks;

  //--
  void NewDataset();
  int GetNumberDataset();
  int GetMatchesInliers();

  // DEBUG
  void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf,
                         string strFolder = "");
  void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf,
                         Map* pMap);

  float GetImageScale();

 public:
  // Tracking states
  enum eTrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    RECENTLY_LOST = 3,
    LOST = 4,
    OK_KLT = 5
  };

  eTrackingState state_;
  eTrackingState last_processed_state_;

  // Input sensor
  int sensor_;

  // Current Frame
  Frame curr_frame_;
  Frame last_frame_;

  cv::Mat img_gray_;

  // Initialization Variables (Monocular)
  std::vector<int> mvIniLastMatches;
  std::vector<int> mvIniMatches;
  std::vector<cv::Point2f> mvbPrevMatched;
  std::vector<cv::Point3f> mvIniP3D;
  Frame mInitialFrame;

  // Lists used to recover the full camera trajectory at the end of the
  // execution. Basically we store the reference keyframe for each frame and its
  // relative transformation
  list<Sophus::SE3f> mlRelativeFramePoses;
  list<KeyFrame*> mlpReferences;
  list<double> mlFrameTimes;
  list<bool> mlbLost;

  // frames with estimated pose
  int mTrackedFr;
  bool mbStep;

  // True if local mapping is deactivated and we are performing only
  // localization
  bool mbOnlyTracking;

  void Reset(bool bLocMap = false);
  void ResetActiveMap(bool bLocMap = false);

  float mMeanTrack;
  bool mbInitWith3KFs;
  double t0;     // time-stamp of first read frame
  double t0vis;  // time-stamp of first inserted keyframe
  double t0IMU;  // time-stamp of IMU initialization
  bool mFastInit = false;

  vector<MapPoint*> GetLocalMapMPS();

  bool mbWriteStats;

 protected:
  // Main tracking function. It is independent of the input sensor.
  void Track();

  // Map initialization for stereo and RGB-D
  void StereoInitialization();

  // Map initialization for monocular
  void MonocularInitialization();
  // void CreateNewMapPoints();
  void CreateInitialMapMonocular();

  void CheckReplacedInLastFrame();
  bool TrackReferenceKeyFrame();
  void UpdateLastFrame();
  bool TrackWithMotionModel();
  bool PredictStateIMU();

  bool Relocalization();

  void UpdateLocalMap();
  void UpdateLocalPoints();
  void UpdateLocalKeyFrames();

  bool TrackLocalMap();
  void SearchLocalPoints();

  bool NeedNewKeyFrame();
  void CreateNewKeyFrame();

  // Perform preintegration from last frame
  void PreintegrateIMU();

  // Reset IMU biases and compute frame velocity
  void ResetFrameIMU();

  bool mbMapUpdated;

  // Imu preintegration from last frame
  IMU::Preintegrated* imu_preint_kf_;

  // Queue of IMU measurements between frames
  std::list<IMU::Point> mlQueueImuData;

  // Vector of IMU measurements from previous to current frame (to be filled by
  // PreintegrateIMU)
  std::vector<IMU::Point> mvImuFromLastFrame;
  std::mutex mutex_imu_;

  // Imu calibration parameters
  IMU::Calib* imu_calib_;

  // Last Bias Estimation (at keyframe creation)
  IMU::Bias mLastBias;

  // In case of performing only localization, this flag is true when there are
  // no matches to points in the map. Still tracking will continue if there are
  // enough matches with temporal points. In that case we are doing visual
  // odometry. The system will try to do relocalization to recover "zero-drift"
  // localization to the map.
  bool mbVO;

  // Other Thread Pointers
  LocalMapping* mpLocalMapper;
  LoopClosing* mpLoopClosing;

  // ORB
  OrbExtractor *orb_extractor_left_, *orb_extractor_right_;
  OrbExtractor* mpIniORBextractor;

  // BoW
  ORBVocabulary* orb_voc_;
  KeyFrameDatabase* mpKeyFrameDB;

  // Initalization (only for monocular)
  bool mbReadyToInitializate;
  bool mbSetInit;

  // Local Map
  KeyFrame* mpReferenceKF;
  std::vector<KeyFrame*> mvpLocalKeyFrames;
  std::vector<MapPoint*> mvpLocalMapPoints;

  // System
  System* mpSystem;

  // Drawers
  Viewer* mpViewer;
  FrameDrawer* frame_drawer_;
  MapDrawer* mpMapDrawer;
  bool bStepByStep;

  // Atlas
  Atlas* atlas_;

  // Calibration matrix
  cv::Mat cv_K_;
  Eigen::Matrix3f eig_K_;
  cv::Mat dist_coef_;
  float bf_;
  float img_scale_;

  float mImuFreq;
  double mImuPer;
  bool mInsertKFsLost;

  // New KeyFrame rules (according to fps)
  int min_frames_;
  int max_frames_;

  int mnFirstImuFrameId;
  int mnFramesToResetIMU;

  // Threshold close/far points
  // Points seen as close by the stereo/kRgbd sensor are considered reliable
  // and inserted from just one frame. Far points requiere a match in two
  // keyframes.
  float th_depth_;

  // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are
  // scaled.
  float depth_map_factor_;

  // Current matches in frame
  int mnMatchesInliers;

  // Last Frame, KeyFrame and Relocalisation Info
  KeyFrame* mpLastKeyFrame;
  unsigned int mnLastKeyFrameId;
  unsigned int mnLastRelocFrameId;
  double mTimeStampLost;
  double time_recently_lost;

  unsigned int mnFirstFrameId;
  unsigned int mnInitialFrameId;
  unsigned int mnLastInitFrameId;

  bool mbCreatedMap;

  // Motion Model
  bool mbVelocity{false};
  Sophus::SE3f mVelocity;

  // Color order (true RGB, false BGR, ignored if grayscale)
  bool is_Rgb_;

  list<MapPoint*> mlpTemporalPoints;

  // int nMapChangeIndex;

  int num_dataset_;

  ofstream f_track_stats;

  ofstream f_track_times;
  double mTime_PreIntIMU;
  double mTime_PosePred;
  double mTime_LocalMapTrack;
  double mTime_NewKF_Dec;

  GeometricCamera *cam_, *cam2_;

  int initID, lastID;

  Sophus::SE3f so_Tlr_;

  void newParameterLoader(Settings* settings);

 public:
  cv::Mat img_right_;
};

}  // namespace ORB_SLAM_FUSION

#endif  // TRACKING_H
