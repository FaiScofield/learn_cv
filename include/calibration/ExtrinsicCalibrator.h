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

    void setVerbose(bool flag) { mbVerbose = flag; }

    void readCornersFromFile(const std::string& cornerFile);
    void readImageFromFile(const std::string& imageFile);
    void readOdomFromFile(const std::string& odomFile);
    void dataSync();

    void calculatePose();
    int checkHomograpy(const cv::Mat& H, const std::vector<cv::Mat>& Rs, const std::vector<cv::Mat>& ts);
    cv::Mat computeH21(const std::vector<cv::Point2f>& vP1, const std::vector<cv::Point2f>& vP2);
    cv::Mat optimize(const cv::Mat& Tcw_, const std::vector<cv::Point2f> vFeatures_);

    bool lessThen(const ImageRaw& r1, const ImageRaw& r2) { return r1.timestamp < r2.timestamp; }

    void showChessboardCorners();
    void writePose(const std::string& outputFile);

private:
    unsigned int nTatalFrames;
    unsigned int nFeaturesPerFrame = 88;
    std::vector<OdomRaw> mvOdomRaws;
    std::vector<ImageRaw> mvImageRaws;
    std::vector<std::vector<cv::Point2f>> mvvCorners;
    std::vector<cv::Point3f> mvMapPoints;
    std::vector<long long int> mvTimeOdom;
    std::vector<long long int> mvTimeImage;
    std::vector<cv::Mat> mvImageMats;

    std::vector<cv::Mat> mvTwc;
    std::vector<cv::Mat> mvTwb;
    std::vector<cv::Mat> mvTwc_refined;
    std::vector<cv::Mat> mvPosesCamera;
    std::vector<g2o::VertexSE3> mvVertexPoseCamera;
    std::vector<g2o::VertexSE2> mvVertexPoseOdom;

    cv::Mat K, D;
    cv::Mat R, t;
    cv::Size mBoardSize;
    //    Eigen::Matrix3f R;
    //    Eigen::Vector3f t;

    double fx, fy;
    double cx, cy;
    bool mbVerbose = false;
};

double normalizeAngle(const double angle)
{
    return angle + 2 * M_PI * floor((M_PI - angle) / (2 * M_PI));
}


}  // namespace ec

#endif  // EXTRINSICCALIBRATOR_H
