#include "calibration/ExtrinsicCalibrator.h"
#include "utility.hpp"

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace ec;
using namespace std;
using namespace cv;


string g_imageFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/gooImages.txt";
string g_odomFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/goodOdom.txt";
string g_projectFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/reprojectedPoints.txt";
string g_outputFile = "./trajectories.txt";
bool g_debug = true;

double rad2degree(const double rad)
{
    return rad * 180 / M_PI;
}


int main(int argc, char *argv[])
{
    ec::ExtrinsicCalibrator ec;
    ec.setVerbose(g_debug);
    ec.readImageFromFile(g_imageFile);
    ec.readOdomFromFile(g_odomFile);
    ec.readCornersFromFile_Matlab(g_projectFile);
    ec.dataSync();

    ec.calculatePose();
//    Eigen::Matrix3d R_yx;
//    ec.estimatePitchRoll(R_yx);
//    double roll, pitch, yaw;
//    cvu::EigenMat2RPY(R_yx, roll, pitch, yaw);
//    printf("R_cam2odo: %f, %f, %f\n", rad2degree(roll), rad2degree(pitch), rad2degree(yaw));

    Eigen::Matrix4d Tbc;
    std::vector<double> scales;
    ec.estimate(Tbc, scales);
    double roll, pitch, yaw;
    cvu::EigenMat2RPY(Tbc, roll, pitch, yaw);
    printf("R_cam2odo[degree]: %f, %f, %f\n", rad2degree(roll), rad2degree(pitch), rad2degree(yaw));

//    cout << "Tbc: " << endl << Tbc << endl;
//    ec.writePose(g_outputFile);

//    if (g_debug) {
//        ec.showChessboardCorners();
//    }

//    ec.calculatePose();

    cout << "Done. " << endl;

    return 0;
}

