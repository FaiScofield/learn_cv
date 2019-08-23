#include "calibration/ExtrinsicCalibrator.h"
#include "utility.hpp"

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace ec;

string g_imageFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/gooImages.txt";
string g_odomFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/goodOdom.txt";
string g_projectFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/reprojectedPoints.txt";
string g_outputFile = "./trajectories.txt";
bool g_debug = true;

int main(int argc, char *argv[])
{
    ec::ExtrinsicCalibrator ec;
    ec.setVerbose(g_debug);
    ec.readImageFromFile(g_imageFile);
    ec.readOdomFromFile(g_odomFile);
    ec.readCornersFromFile(g_projectFile);
    ec.dataSync();

    ec.calculatePose();
    ec.writePose(g_outputFile);

//    if (g_debug) {
//        ec.showChessboardCorners();
//    }

//    ec.calculatePose();

    cout << "Done. " << endl;

    return 0;
}

