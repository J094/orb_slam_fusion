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

#include <system.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <sstream>

#include "imu/imu_types.h"
#include "solver/g2o_solver/optimizer.h"

using namespace std;

// Breakpad stack analyse
#include "client/linux/handler/exception_handler.h"
static bool DumpCallback(const google_breakpad::MinidumpDescriptor &descriptor,
                         void *context, bool succeed) {
  cout << "Dump path: " << descriptor.path() << endl;
  return succeed;
}

void LoadImages(const string &strPathLeft, const string &strPathRight,
                const string &strPathTimes, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps,
             vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv) {
  // Breakpad stack analyse
  google_breakpad::MinidumpDescriptor descriptor("./");
  google_breakpad::ExceptionHandler eh(descriptor, NULL, DumpCallback, NULL,
                                       true, -1);

  if (argc < 5) {
    cerr << endl
         << "Usage: ./stereo_inertial_euroc [path_to_vocabulary] "
            "[path_to_settings] [path_to_sequence_folder] [path_to_times_file]"
         << endl;
    return 1;
  }

  bool bFileName = (((argc - 3) % 2) == 1);
  string file_name;
  if (bFileName) {
    file_name = string(argv[argc - 1]);
    cout << "file name: " << file_name << endl;
  }

  vector<string> vstrImageLeft;
  vector<string> vstrImageRight;
  vector<double> vTimestampsCam;
  vector<cv::Point3f> vAcc, vGyro;
  vector<double> vTimestampsImu;
  int nImages;
  int nImu;
  int first_imu = 0;

  int tot_images = 0;

  string pathSeq(argv[3]);
  string pathTimeStamps(argv[4]);

  string pathCam0 = pathSeq + "/mav0/cam0/data";
  string pathCam1 = pathSeq + "/mav0/cam1/data";
  string pathImu = pathSeq + "/mav0/imu0/data.csv";

  cout << "Loading Images...";
  LoadImages(pathCam0, pathCam1, pathTimeStamps, vstrImageLeft, vstrImageRight,
             vTimestampsCam);
  cout << "LOADED!" << endl;

  cout << "Loading IMU...";
  LoadIMU(pathImu, vTimestampsImu, vAcc, vGyro);
  cout << "LOADED!" << endl;

  nImages = vstrImageLeft.size();
  tot_images += nImages;
  nImu = vTimestampsImu.size();

  if ((nImages <= 0) || (nImu <= 0)) {
    cerr << "ERROR: Failed to load images or IMU" << endl;
    return 1;
  }

  // Find first imu to be considered, supposing imu measurements start first

  while (vTimestampsImu[first_imu] <= vTimestampsCam[0]) first_imu++;
  first_imu--;  // first imu measurement to be considered

  // Read rectification parameters
  cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "ERROR: Wrong path to settings" << endl;
    return -1;
  }

  // Vector for tracking time statistics
  vector<float> vTimesTrack;
  vTimesTrack.resize(tot_images);

  cout << endl << "-------" << endl;
  cout.precision(17);

  // Create SLAM system. It initializes all system threads and gets ready to
  // process frames.
  ORB_SLAM_FUSION::System SLAM(argv[1], argv[2], ORB_SLAM_FUSION::System::IMU_STEREO,
                         true);

  cv::Mat imLeft, imRight;
  vector<ORB_SLAM_FUSION::IMU::Point> vImuMeas;
  double t_rect = 0.f;
  double t_resize = 0.f;
  double t_track = 0.f;
  int num_rect = 0;
  int proccIm = 0;
  for (int ni = 0; ni < nImages; ni++, proccIm++) {
    // Read left and right images from file
    imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
    imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);

    if (imLeft.empty()) {
      cerr << endl
           << "Failed to load image at: " << string(vstrImageLeft[ni]) << endl;
      return 1;
    }

    if (imRight.empty()) {
      cerr << endl
           << "Failed to load image at: " << string(vstrImageRight[ni]) << endl;
      return 1;
    }

    double tframe = vTimestampsCam[ni];

    // Load imu measurements from previous frame
    vImuMeas.clear();

    if (ni > 0)
      // while(vTimestampsImu[first_imu]<=vTimestampsCam[ni])
      while (vTimestampsImu[first_imu] <= vTimestampsCam[ni]) {
        vImuMeas.push_back(ORB_SLAM_FUSION::IMU::Point(
            vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
            vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
            vTimestampsImu[first_imu]));
        first_imu++;
      }

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 =
        std::chrono::monotonic_clock::now();
#endif

    // Pass the images to the SLAM system
    SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas);

#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 =
        std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
    t_track =
        t_rect + t_resize +
        std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(
            t2 - t1)
            .count();
    SLAM.InsertTrackTime(t_track);
#endif

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1)
            .count();

    vTimesTrack[ni] = ttrack;

    // Wait to load the next frame
    double T = 0;
    if (ni < nImages - 1)
      T = vTimestampsCam[ni + 1] - tframe;
    else if (ni > 0)
      T = tframe - vTimestampsCam[ni - 1];

    if (ttrack < T) usleep((T - ttrack) * 1e6);  // 1e6
  }
  // Stop all threads
  SLAM.Shutdown();

  // Save camera trajectory
  if (bFileName) {
    const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
    const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
    SLAM.SaveTrajectoryTUM(f_file);
    SLAM.SaveKeyFrameTrajectoryTUM(kf_file);
  } else {
    SLAM.SaveTrajectoryTUM("f_euroc.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("kf_euroc.txt");
  }

  return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight,
                const string &strPathTimes, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimeStamps) {
  ifstream fTimes;
  fTimes.open(strPathTimes.c_str());
  vTimeStamps.reserve(5000);
  vstrImageLeft.reserve(5000);
  vstrImageRight.reserve(5000);
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      stringstream ss;
      ss << s;
      vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
      vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
      double t;
      ss >> t;
      vTimeStamps.push_back(t / 1e9);
    }
  }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps,
             vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro) {
  ifstream fImu;
  fImu.open(strImuPath.c_str());
  vTimeStamps.reserve(5000);
  vAcc.reserve(5000);
  vGyro.reserve(5000);

  while (!fImu.eof()) {
    string s;
    getline(fImu, s);
    if (s[0] == '#') continue;

    if (!s.empty()) {
      string item;
      size_t pos = 0;
      double data[7];
      int count = 0;
      while ((pos = s.find(',')) != string::npos) {
        item = s.substr(0, pos);
        data[count++] = stod(item);
        s.erase(0, pos + 1);
      }
      item = s.substr(0, pos);
      data[6] = stod(item);

      vTimeStamps.push_back(data[0] / 1e9);
      vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
      vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
    }
  }
}