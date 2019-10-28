/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef UTILITY_H
#define UTILITY_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

struct OdomRaw {
    long long int timestamp;
    double x, y, theta;
    double linearVelX, AngularVelZ;
    double deltaDistance, deltaTheta;

    OdomRaw()
    {
        timestamp = 0;
        x = y = theta = 0.0;
        linearVelX = AngularVelZ = 0.0;
        deltaDistance = deltaTheta = 0.0;
    }

    cv::Mat toCvSE3() const
    {
        float c = cos(theta);
        float s = sin(theta);

        return (cv::Mat_<float>(4, 4) << c, -s, 0, x, s, c, 0, y, 0, 0, 1, 0, 0, 0, 0, 1);
    }
};

struct ImageRaw {
    ImageRaw(const std::string& s, const long long int t) : fileName(s), timestamp(t) {}

    std::string fileName;
    long long int timestamp;
};

std::vector<std::string> readImageFromFolder(const std::string& folder)
{
    namespace bf = boost::filesystem;

    std::vector<std::string> files;

    bf::path folderPath(folder);
    if (!bf::exists(folderPath))
        return files;

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(folderPath); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;

        if (bf::is_regular_file(iter->status()))
            files.push_back(iter->path().string());
    }

    return files;
}

namespace cvu
{

using std::vector;
using std::string;
using cv::Mat;
using cv::Point2f;
using cv::Point3f;

//! 变换矩阵的逆
Mat inv(const Mat& T4x4)
{
    assert(T4x4.cols == 4 && T4x4.rows == 4);
    Mat RT = T4x4.rowRange(0, 3).colRange(0, 3).t();
    Mat t = -RT * T4x4.rowRange(0, 3).col(3);
    Mat T = Mat::eye(4, 4, CV_32FC1);
    RT.copyTo(T.rowRange(0, 3).colRange(0, 3));
    t.copyTo(T.rowRange(0, 3).col(3));
    return T;
}


//! 求向量的反对称矩阵^
Mat sk_sym(const Point3f _v)
{
    Mat mat(3, 3, CV_32FC1, cv::Scalar(0));
    mat.at<float>(0, 1) = -_v.z;
    mat.at<float>(0, 2) = _v.y;
    mat.at<float>(1, 0) = _v.z;
    mat.at<float>(1, 2) = -_v.x;
    mat.at<float>(2, 0) = -_v.y;
    mat.at<float>(2, 1) = _v.x;
    return mat;
}

/**
 * @brief triangulate 特征点三角化
 * @param pt1 - 参考帧KP
 * @param pt2 - 当前帧KP
 * @param P1 -
 * @param P2
 * @return
 */
Point3f triangulate(const Point2f& pt1, const Point2f& pt2, const Mat& P1, const Mat& P2)
{
    Mat A(4, 4, CV_32FC1);

    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);

    Mat u, w, vt, x3D;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

    return Point3f(x3D);
}

Point2f camprjc(const Mat& _K, const Point3f& _pt)
{
    Point3f uvw = cv::Matx33f(_K) * _pt;
    return Point2f(uvw.x / uvw.z, uvw.y / uvw.z);
}


bool checkParallax(const Point3f& o1, const Point3f& o2, const Point3f& pt3, int minDegree)
{
    float minCos[4] = {0.9998, 0.9994, 0.9986, 0.9976};
    Point3f p1 = pt3 - o1;
    Point3f p2 = pt3 - o2;
    float cosParallax = cv::norm(p1.dot(p2)) / (cv::norm(p1) * cv::norm(p2));
    return cosParallax < minCos[minDegree - 1];
}

Point3f se3map(const Mat& _Tcw, const Point3f& _pt)
{
    cv::Matx33f R(_Tcw.rowRange(0, 3).colRange(0, 3));
    Point3f t(_Tcw.rowRange(0, 3).col(3));
    return (R * _pt + t);
}

double normalizeAngleRad(double rad)
{
    if (rad < -M_PI)
        rad += 2 * M_PI;
    else if (rad > M_PI)
        rad -= 2 * M_PI;
}

// Gamma变换 gamma = 1.2 越小越亮
cv::Mat gamma(const cv::Mat& grayImg, float gamma)
{
    cv::Mat imgGamma, imgOut;
    grayImg.convertTo(imgGamma, CV_32F, 1.0 / 255, 0);
    cv::pow(imgGamma, gamma, imgOut);
    imgOut.convertTo(imgOut, CV_8U, 255, 0);

    return imgOut;
}

// Laplace边缘锐化 scale = 6~10  越小越强
cv::Mat sharpping(const cv::Mat& img, float scale)
{
    cv::Mat imgOut;

    cv::Mat kern = (cv::Mat_<float>(5, 5) << -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 40, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    kern = kern / scale;
    cv::filter2D(img, imgOut, img.depth(), kern);

    return imgOut;
}


Eigen::Vector3d MatRotation2Eular(const cv::Mat& R)
{
    assert(R.rows == 3 && R.cols == 3);
    Eigen::Matrix3d Rotation;
    cv::cv2eigen(R, Rotation);
    return Rotation.eulerAngles(2, 1, 0);
}

Eigen::Vector3d getAngleAxisFromCvMat(const cv::Mat& T)
{
    assert(T.rows == 4 && T.cols == 4);
    cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    Eigen::Vector3d angleAxis;
    cv::cv2eigen(rvec, angleAxis);
    return angleAxis;
}

Eigen::Vector3d getTranslationFromCvMat(const cv::Mat& T)
{
    assert(T.rows == 4 && T.cols == 4);
    Eigen::Vector3d t;
    Mat tt = T.rowRange(0, 3).col(3);
    cv::cv2eigen(tt, t);
    return t;
}

template <typename T> Eigen::Matrix<T, 4, 4> QuaternionMultMatLeft(const Eigen::Quaternion<T>& q)
{
    return (Eigen::Matrix<T, 4, 4>() << q.w(), -q.z(), q.y(), q.x(), q.z(), q.w(), -q.x(), q.y(),
            -q.y(), q.x(), q.w(), q.z(), -q.x(), -q.y(), -q.z(), q.w())
        .finished();
}

template <typename T> Eigen::Matrix<T, 4, 4> QuaternionMultMatRight(const Eigen::Quaternion<T>& q)
{
    return (Eigen::Matrix<T, 4, 4>() << q.w(), q.z(), -q.y(), q.x(), -q.z(), q.w(), q.x(), q.y(),
            q.y(), -q.x(), q.w(), q.z(), -q.x(), -q.y(), -q.z(), q.w())
        .finished();
}

template <typename T> void EigenMat2RPY(const Eigen::Matrix<T, 3, 3>& m, T& roll, T& pitch, T& yaw)
{
    roll = atan2(m(2, 1), m(2, 2));
    pitch = atan2(-m(2, 0), sqrt(m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2)));
    yaw = atan2(m(1, 0), m(0, 0));
}

template <typename T> void EigenMat2RPY(const Eigen::Matrix<T, 4, 4>& m, T& roll, T& pitch, T& yaw)
{
    roll = atan2(m(2, 1), m(2, 2));
    pitch = atan2(-m(2, 0), sqrt(m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2)));
    yaw = atan2(m(1, 0), m(0, 0));
}


void spaceToPlane(const Eigen::Vector3d& Pc, Point2f& p, const Eigen::Matrix3d& K)
{
    Eigen::Vector3d Pc_normalized = Pc / Pc[2];
    Eigen::Vector3d Puv = K * Pc_normalized;

    p.x = Puv[0];
    p.y = Puv[1];
}

}  // namespace cvu


namespace ug2o
{

using namespace Eigen;
using namespace g2o;

g2o::SE3Quat toSE3Quat(const cv::Mat& cvT)
{
    Eigen::Matrix<double, 3, 3> R;
    R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2), cvT.at<float>(1, 0),
        cvT.at<float>(1, 1), cvT.at<float>(1, 2), cvT.at<float>(2, 0), cvT.at<float>(2, 1),
        cvT.at<float>(2, 2);

    Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

    return g2o::SE3Quat(R, t);
}

cv::Mat toCvMat(const g2o::SE3Quat& SE3)
{
    Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
    cv::Mat cvMat(4, 4, CV_32F);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            cvMat.at<float>(i, j) = eigMat(i, j);

    return cvMat.clone();
}

Vector2d project2d(const Vector3d& v)
{
    Vector2d res;
    res(0) = v(0) / v(2);
    res(1) = v(1) / v(2);
    return res;
}


// class EdgeSE3ProjectXYZOnlyPose : public BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>
//{
// public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//    EdgeSE3ProjectXYZOnlyPose() {}

//    bool read(std::istream& is)
//    {
//        for (int i = 0; i < 2; i++) {
//            is >> _measurement[i];
//        }
//        for (int i = 0; i < 2; i++)
//            for (int j = i; j < 2; j++) {
//                is >> information()(i, j);
//                if (i != j)
//                    information()(j, i) = information()(i, j);
//            }
//        return true;
//    }

//    bool write(std::ostream& os) const
//    {
//        for (int i = 0; i < 2; i++) {
//            os << measurement()[i] << " ";
//        }

//        for (int i = 0; i < 2; i++)
//            for (int j = i; j < 2; j++) {
//                os << " " << information()(i, j);
//            }
//        return os.good();
//    }

//    void computeError()
//    {
//        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
//        Vector2d obs(_measurement);
//        _error = obs - cam_project(v1->estimate().map(Xw));
//    }

//    bool isDepthPositive()
//    {
//        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
//        return (v1->estimate().map(Xw))(2) > 0.0;
//    }

//    virtual void linearizeOplus()
//    {
//        VertexSE3Expmap* vi = static_cast<VertexSE3Expmap*>(_vertices[0]);
//        Vector3d xyz_trans = vi->estimate().map(Xw);

//        double x = xyz_trans[0];
//        double y = xyz_trans[1];
//        double invz = 1.0 / xyz_trans[2];
//        double invz_2 = invz * invz;

//        _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
//        _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
//        _jacobianOplusXi(0, 2) = y * invz * fx;
//        _jacobianOplusXi(0, 3) = -invz * fx;
//        _jacobianOplusXi(0, 4) = 0;
//        _jacobianOplusXi(0, 5) = x * invz_2 * fx;

//        _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
//        _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
//        _jacobianOplusXi(1, 2) = -x * invz * fy;
//        _jacobianOplusXi(1, 3) = 0;
//        _jacobianOplusXi(1, 4) = -invz * fy;
//        _jacobianOplusXi(1, 5) = y * invz_2 * fy;
//    }

//    Vector2d cam_project(const Vector3d& trans_xyz) const
//    {
//        Vector2d proj = project2d(trans_xyz);
//        Vector2d res;
//        res[0] = proj[0] * fx + cx;
//        res[1] = proj[1] * fy + cy;
//        return res;
//    }

//    Vector3d Xw;
//    double fx, fy, cx, cy;
//};


}  // namespace ug2o


#endif
