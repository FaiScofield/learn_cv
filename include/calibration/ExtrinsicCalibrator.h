#ifndef EXTRINSICCALIBRATOR_H
#define EXTRINSICCALIBRATOR_H

#include "utility.hpp"

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <eigen3/Eigen/Core>
#include <g2o/types/sim3/sim3.h>
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/types/slam3d/vertex_se3.h>


namespace ec
{

class ExtrinsicCalibrator
{
public:
    ExtrinsicCalibrator();
    ~ExtrinsicCalibrator();

    bool lessThen(const ImageRaw& r1, const ImageRaw& r2) { return r1.timestamp < r2.timestamp; }
    void setVerbose(bool flag) { mbVerbose = flag; }
    void addCameraPose(cv::Mat Tcw) { mvTcw.emplace_back(Tcw); }
    void addOdomPose(cv::Mat Tbw) { mvTbw.emplace_back(Tbw); }
    cv::Mat getOdomPose(unsigned int idx) { return mvTbw[idx]; }

    void readCornersFromFile_Matlab(const std::string& cornerFile);
    void readImageFromFile(const std::string& imageFile);
    void readOdomFromFile(const std::string& odomFile);
    void dataSync();
    void setDataQuantity(unsigned int n) { N = n; }
    bool checkSystemReady();

    void calculatePose();
    void setTransforms();
    cv::Mat optimize(const cv::Mat& Tcw_, const std::vector<cv::Point2f> vFeatures_);
    bool solveQuadraticEquation(double a, double b, double c, double& x1, double& x2) const;
    bool estimatePitchRoll(Eigen::Matrix3d& R_yx);
    bool estimate(Eigen::Matrix4d& H_cam_odo, std::vector<double>& scales);


    void drawAxis(cv::Mat& image, const std::vector<cv::Point3f>& MPs, const cv::Mat& K,
                  const cv::Mat& D, const cv::Mat& R, const cv::Mat& tvec, const float& len);
    void writePose(const std::string& outputFile);

private:
    unsigned int N = 0;
    unsigned int nFeaturesPerFrame = 88;
    std::vector<OdomRaw> mvOdomRaws;
    std::vector<ImageRaw> mvImageRaws;
    std::vector<std::vector<cv::Point2f>> mvvCorners;
    std::vector<cv::Point3f> mvMapPoints;
    std::vector<long long int> mvTimeOdom;
    std::vector<long long int> mvTimeImage;
    std::vector<cv::Mat> mvImageMats;

    std::vector<cv::Mat> mvTbw;
    std::vector<cv::Mat> mvTcw;
    std::vector<cv::Mat> mvTcw_refined;

    std::vector<cv::Mat> mvTcjci;
    std::vector<cv::Mat> mvTbjbi;

    std::vector<cv::Mat> mvPoseCam;
    std::vector<cv::Mat> mvPoseOdo;
    std::vector<cv::Mat> mvPoseCam_refined;

    std::vector<cv::Mat> mvPosesCamera;
    std::vector<g2o::VertexSE3> mvVertexPoseCamera;
    std::vector<g2o::VertexSE2> mvVertexPoseOdom;

    cv::Mat K, D;
    cv::Size mBoardSize;
    float mSquareSize;
    //    Eigen::Matrix3f R;
    //    Eigen::Vector3f t;

    double fx, fy;
    double cx, cy;
    bool mbVerbose = false;
    bool mbSystemReady = false;
};

double normalizeAngle(const double angle)
{
    return angle + 2 * M_PI * floor((M_PI - angle) / (2 * M_PI));
}


}  // namespace ec

#endif  // EXTRINSICCALIBRATOR_H
