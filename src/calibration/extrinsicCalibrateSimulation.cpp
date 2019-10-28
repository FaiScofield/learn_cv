#include "calibration/ExtrinsicCalibrator.h"
#include "utility.hpp"

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace ec;
using namespace std;
using namespace cv;
using namespace Eigen;

string g_imageFile =
    "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/gooImages.txt";
string g_odomFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/goodOdom.txt";
string g_projectFile =
    "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/reprojectedPoints.txt";
string g_outputFile = "./trajectories.txt";
bool g_debug = true;

double rad2degree(const double rad)
{
    return rad * 180 / M_PI;
}

double degree2rad(const double deg)
{
    return deg * M_PI / 180;
}

cv::Mat vector2CvSE3(Eigen::Vector3d p)
{
    float c = cos(p[2]);
    float s = sin(p[2]);

    return (cv::Mat_<float>(4, 4) << c, -s, 0, p[0], s, c, 0, p[1], 0, 0, 1, 0, 0, 0, 0, 1);
}

void generateSimData(ec::ExtrinsicCalibrator* ec)
{
    //! true extrinsic
    AngleAxisd Rn(M_PI_2, Vector3d(0, 0, 1));  // yaw 90 degree
    Matrix3d R_truth = Rn.toRotationMatrix();
    Vector3d t_truth(-0.055, 0, 0);
    Mat R_cv, t_cv;
    Mat Tbc = Mat::eye(4, 4, CV_32F);
    eigen2cv(R_truth, R_cv);
    eigen2cv(t_truth, t_cv);
    R_cv.copyTo(Tbc.rowRange(0, 3).colRange(0, 3));
    t_cv.copyTo(Tbc.rowRange(0, 3).col(3));
    cout << "The true extrinsic of this simulation: " << endl;
    cout << "   R = " << Rn.matrix().eulerAngles(0, 1, 2).transpose() << endl;
    cout << "   t = " << t_truth.transpose() << endl << endl;

    //! parameters
    double v = 0.3 / 30;     // linear velocity: 1m/s, squize size: 0.3m
    double w = M_PI_2 / 30;  // angular velocity: pi/2 rad/s

    //! noise
    unsigned int N = 300;
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::normal_distribution<double> odomNoiseLinear(-0.1 * v, 0.1 * v);    // 1/5 noise, 2mm
    std::normal_distribution<double> odomNoiseAngular(-0.1 * w, 0.1 * w);   // 1/5 noise, 0.6 degree
    std::normal_distribution<double> cameraNoiseLinear(-0.001, 0.001);      // 2mm  noise
    std::normal_distribution<double> cameraNoiseAngular(-degree2rad(0.5),
                                                        degree2rad(0.5));   // 1 degree noise

    //! generate odom pose with noise. 走一个正方形
    int xFlag[10] = {1, 0, 0, 0, -1, 0, 0, 0, 1, 1};
    int yFlag[10] = {0, 0, 1, 0, 0, 0, -1, 0, 0, 0};
    int tFlag[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 0};
    double x, y, theta;
    x = y = theta = 0;
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < 30; ++i) {
            x += xFlag[j] * v + odomNoiseLinear(gen);
            y += yFlag[j] * v + odomNoiseLinear(gen);
            theta += tFlag[j] * w + odomNoiseAngular(gen);
            theta = cvu::normalizeAngleRad(theta);
            ec->addOdomPose(vector2CvSE3(Vector3d(x, y, theta)));
        }
    }

    //! generate camera pose with noise
    for (int i = 0; i < N; ++i) {
        Mat T_noise = Mat::eye(4, 4, CV_32F);
        Mat R_noise, t_noise;
        AngleAxisd R_n(cameraNoiseAngular(gen), Vector3d(1, 1, 1));
        eigen2cv(R_n.toRotationMatrix(), R_noise);
        t_noise = (Mat_<double>(3, 1) << cameraNoiseLinear(gen), cameraNoiseLinear(gen),
                   cameraNoiseLinear(gen));
        R_noise.copyTo(T_noise.rowRange(0, 3).colRange(0, 3));
        t_noise.copyTo(T_noise.rowRange(0, 3).col(3));
        Mat Tbw = ec->getOdomPose(i);
        Mat Tcw = T_noise * cvu::inv(Tbc) * Tbw;
        ec->addCameraPose(Tcw);
    }

    ec->setDataQuantity(N);
    ec->setTransforms();
}


int main(int argc, char* argv[])
{
    ec::ExtrinsicCalibrator ec;
    ec.setVerbose(g_debug);
    generateSimData(&ec);

    string outputFile = "./trajectories_sim.txt";
    ec.writePose(outputFile);


    if (ec.checkSystemReady()) {
        Eigen::Matrix4d Tbc;
        std::vector<double> scales;
        ec.estimate(Tbc, scales);
        double roll, pitch, yaw;
        cvu::EigenMat2RPY(Tbc, roll, pitch, yaw);
        printf("R_cam2odo[degree]: %f, %f, %f\n", rad2degree(roll), rad2degree(pitch),
               rad2degree(yaw));
    }

    cout << "Done. " << endl;

    return 0;
}
