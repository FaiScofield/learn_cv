#include "calibration/ExtrinsicCalibrator.h"
#include "utility.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
//#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace ec
{


ExtrinsicCalibrator::ExtrinsicCalibrator()
{
    mBoardSize = cv::Size(11, 8);
    K = (cv::Mat_<double>(3, 3) << 415.3904681732305, 0., 317.2546831363181, 0., 415.6664203685940,
         241.0199225105337, 0., 0., 1.);
    D = (cv::Mat_<double>(5, 1) << 5.3963689585481e-02, -5.3999880307856e-02, 7.248873665656701e-04,
         7.696301272230405e-04, 0.0);

    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
}

ExtrinsicCalibrator::~ExtrinsicCalibrator()
{
}
void ExtrinsicCalibrator::readCornersFromFile(const std::string& cornerFile)
{
    if (nTatalFrames < 1) {
        std::cerr << "Please read image data first!" << std::endl;
        return;
    }

    std::ifstream ifs(cornerFile);
    if (!ifs.is_open()) {
        std::cerr << "File open error! : " << cornerFile << std::endl;
        return;
    }

    std::vector<cv::Point2f> tmp(nFeaturesPerFrame);
    mvvCorners.resize(nTatalFrames, tmp);

    int index = -1;
    std::string lineData;
    while (!ifs.eof()) {
        std::getline(ifs, lineData);
        if (lineData.empty())
            continue;

        // 确定图像索引index
        if (std::sscanf(lineData.c_str(), "val(:,:,%d) =", &index) == 1) {
            index -= 1;
            if (index < 0 || index >= nTatalFrames) {
                std::cerr << "Wrong index in file! : " << index + 1 << std::endl;
                continue;
            }
            std::getline(ifs, lineData);  // 去掉 "val(:,:,1) =" 下面的一行空行
        }

        // 读入该图像对应的88个角点
        std::string pointDate;
        for (int j = 0; j < nFeaturesPerFrame; ++j) {
            std::getline(ifs, pointDate);
            std::stringstream ss(pointDate);
            cv::Point2f p;
            ss >> p.x >> p.y;
            mvvCorners[index][j] = p;
        }
    }
    ifs.close();

    assert(index == nTatalFrames - 1);

    // generate fixed MapPoint
    const float squareSize = 0.03;  // unit:  m
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 11; j++)
            mvMapPoints.emplace_back(cv::Point3f(float(j * squareSize), float(i * squareSize), 0.));
    }
}

void ExtrinsicCalibrator::readImageFromFile(const std::string& imageFile)
{
    std::ifstream ifs(imageFile);
    if (!ifs.is_open()) {
        std::cerr << "File open error! : " << imageFile << std::endl;
        return;
    }

    std::string lineData;
    while (std::getline(ifs, lineData) && !lineData.empty()) {
        // format: */frameRaw12987978101.jpg
        auto i = lineData.find_last_of('w');
        auto j = lineData.find_last_of('.');
        long long int timestamp = atoll(lineData.substr(i + 1, j - i - 1).c_str());
        mvImageRaws.push_back(ImageRaw(lineData, timestamp));
        mvTimeImage.push_back(timestamp);
        cv::Mat img = cv::imread(lineData, CV_LOAD_IMAGE_GRAYSCALE);
        mvImageMats.push_back(img);
    }
    ifs.close();

    if (mvImageRaws.empty()) {
        std::cerr << "Not image data in the folder!" << std::endl;
        return;
    } else {
        std::cout << "Read " << mvImageRaws.size() << " files in the folder." << std::endl;
        nTatalFrames = mvImageRaws.size();
    }

    //! 注意不能直接对string排序
    //    std::sort(mvImageRaws.begin(), mvImageRaws.end(), lessThen);
}

void ExtrinsicCalibrator::readOdomFromFile(const std::string& odomFile)
{
    std::ifstream ifs(odomFile);
    if (!ifs.is_open()) {
        std::cerr << "File open error! : " << odomFile << std::endl;
        return;
    }

    std::string lineData;
    while (std::getline(ifs, lineData) && !lineData.empty()) {
        OdomRaw oraw;
        std::stringstream ss(lineData);// lineData = "a  aa asf asf"
        ss >> oraw.timestamp >> oraw.x >> oraw.y >> oraw.theta >> oraw.linearVelX >>
            oraw.AngularVelZ >> oraw.deltaDistance >> oraw.deltaTheta;
        mvOdomRaws.push_back(oraw);
        mvTimeOdom.push_back(oraw.timestamp);
    }

    if (mvOdomRaws.empty()) {
        std::cerr << "Not odom data in the folder!" << std::endl;
        return;
    } else {
        std::cout << "Read " << mvOdomRaws.size() << " odom data in the folder." << std::endl;
    }

    ifs.close();
}

void ExtrinsicCalibrator::dataSync()
{
    assert(nTatalFrames == mvImageRaws.size());
    assert(nTatalFrames == mvvCorners.size());
    assert(nTatalFrames < mvOdomRaws.size());
    //    assert(mvTimeImage.)  // sorted?

    std::cout << "Syncing image data and odom data with timestamps..." << std::endl;

    std::vector<OdomRaw> vOdomSynced(nTatalFrames);
    int skipFrames = 0;
    for (int i = 0; i < mvTimeImage.size(); ++i) {
        auto r = std::upper_bound(mvTimeOdom.begin(), mvTimeOdom.end(), mvTimeImage[i]) -
                 mvTimeOdom.begin();
        if (r > mvTimeOdom.size() - 1) {
            std::cout << "[Warning] 跳过此帧，因为找不到它的对应帧. " << mvTimeImage[i]
                      << std::endl;
            skipFrames++;
            continue;
        }
        if (r == 0) {
            vOdomSynced[r] = mvOdomRaws[0];
            continue;
        }
        //! TODO 插值方式需要改进，目前是直接线性插值，可能要改成Eular法
        auto l = r - 1;
        double alpha = (mvTimeImage[i] - mvOdomRaws[l].timestamp) /
                       (mvOdomRaws[r].timestamp - mvOdomRaws[l].timestamp);
        double x = mvOdomRaws[l].x + alpha * (mvOdomRaws[r].x - mvOdomRaws[l].x);
        double y = mvOdomRaws[l].y + alpha * (mvOdomRaws[r].y - mvOdomRaws[l].y);
        double theta = mvOdomRaws[l].theta + alpha * (mvOdomRaws[r].theta - mvOdomRaws[l].theta);
        OdomRaw odomTemp;
        odomTemp.timestamp = mvTimeImage[i];
        odomTemp.x = x;
        odomTemp.y = y;
        odomTemp.theta = normalizeAngle(theta);
        vOdomSynced[i] = odomTemp;
    }
    if (skipFrames > 0)
        printf("因为找不到对应帧共计跳过%d帧.\n", skipFrames);

    mvOdomRaws = vOdomSynced;
}

void ExtrinsicCalibrator::calculatePose()
{

//    mvPosesCamera.push_back(cv::Mat::eye(4, 4, CV_32F));
    for (int i = 0; i < nTatalFrames; ++i) {
        //! 通过Homograpy计算相机位姿. 参考OpenCV例子pose_from_homography.cpp
        //! [compute-image-points]
        std::vector<cv::Point2f> imagePoints;
        cv::undistortPoints(mvvCorners[i], imagePoints, K, D);

        //! [compute-object-points]
        std::vector<cv::Point2f> objectPointsPlanar;
        for (size_t i = 0; i < mvMapPoints.size(); i++) {
            objectPointsPlanar.push_back(cv::Point2f(mvMapPoints[i].x, mvMapPoints[i].y));
        }

        //! [estimate-homography]
        cv::Mat H = cv::findHomography(objectPointsPlanar, imagePoints);

        //! [pose-from-homography]
        // Normalization to ensure that ||c1|| = 1
        double norm =
            sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0) +
                 H.at<double>(2, 0) * H.at<double>(2, 0));
        H /= norm;
        cv::Mat c1 = H.col(0);
        cv::Mat c2 = H.col(1);
        cv::Mat c3 = c1.cross(c2);
        cv::Mat tvec = H.col(2);
        cv::Mat R(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            R.at<double>(i, 0) = c1.at<double>(i, 0);
            R.at<double>(i, 1) = c2.at<double>(i, 0);
            R.at<double>(i, 2) = c3.at<double>(i, 0);
        }

        //! [polar-decomposition-of-the-rotation-matrix]
        cv::Mat W, U, Vt;
        cv::SVDecomp(R, W, U, Vt);
        R = U * Vt;

        // 保存计算结果
        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
        R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
        tvec.copyTo(Tcw.rowRange(0, 3).col(3));

        static cv::Mat Tc0 = Tcw;
        cv::Mat Twc = Tc0 * cvu::inv(Tcw);
        mvTwc.push_back(Twc);  // 存入Twc，平移部分即为其Pose

        // 相机位姿优化
        cv::Mat Tcw_refined = optimize(Tcw, mvvCorners[i]);
        mvTwc_refined.push_back(Tcw_refined);


        //! 计算里程计位姿, w为首帧Odom的位姿，置为原点
        static cv::Mat Tb0 = mvOdomRaws[0].toCvSE3();
        cv::Mat Tbi = mvOdomRaws[i].toCvSE3();
        cv::Mat Twb = Tbi * cvu::inv(Tb0);
        mvTwb.push_back(Twb);

        // 位姿输出
        if (mbVerbose) {
            cv::Mat tc = Twc.rowRange(0, 3).col(3);
            cv::Mat tb = Twb.rowRange(0, 3).col(3);
            cv::Mat tc_r = cv::Mat(Tc0 * cvu::inv(Tcw_refined)).rowRange(0, 3).col(3);

            std::cout << "[DEBUG] Odom Pose t = " << tb.t() << std::endl;
            std::cout << "[DEBUG] Camera Pose t = " << tc.t() << std::endl;
            std::cout << "[DEBUG] Camera Pose Refined t = " << tc_r.t() << std::endl;

            // 显示角点图
            cv::Mat img_corners, img_pose;
            cv::cvtColor(mvImageMats[i], img_corners, cv::COLOR_GRAY2BGR);
            cv::cvtColor(mvImageMats[i], img_pose, cv::COLOR_GRAY2BGR);
            cv::drawChessboardCorners(img_corners, cv::Size(11, 8), mvvCorners[i], 1);
            cv::imshow("Chessboard corners detection", img_corners);

            // 显示坐标轴
            cv::Mat rvec;
            cv::Rodrigues(R, rvec);
            cv::aruco::drawAxis(img_pose, K, D, rvec, tvec, 5 * 0.03);
            cv::imshow("Pose from coplanar points", img_pose);

            cv::waitKey(30);
        }
    }
}

cv::Mat ExtrinsicCalibrator::computeH21(const std::vector<cv::Point2f>& vP1,
                                        const std::vector<cv::Point2f>& vP2)
{
    const int N = vP1.size();

    cv::Mat A(2 * N, 9, CV_32F);  // 2N*9

    for (int i = 0; i < N; i++) {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2 * i, 0) = 0.0;
        A.at<float>(2 * i, 1) = 0.0;
        A.at<float>(2 * i, 2) = 0.0;
        A.at<float>(2 * i, 3) = -u1;
        A.at<float>(2 * i, 4) = -v1;
        A.at<float>(2 * i, 5) = -1;
        A.at<float>(2 * i, 6) = v2 * u1;
        A.at<float>(2 * i, 7) = v2 * v1;
        A.at<float>(2 * i, 8) = v2;

        A.at<float>(2 * i + 1, 0) = u1;
        A.at<float>(2 * i + 1, 1) = v1;
        A.at<float>(2 * i + 1, 2) = 1;
        A.at<float>(2 * i + 1, 3) = 0.0;
        A.at<float>(2 * i + 1, 4) = 0.0;
        A.at<float>(2 * i + 1, 5) = 0.0;
        A.at<float>(2 * i + 1, 6) = -u2 * u1;
        A.at<float>(2 * i + 1, 7) = -u2 * v1;
        A.at<float>(2 * i + 1, 8) = -u2;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);  // v的最后一列
}

cv::Mat ExtrinsicCalibrator::optimize(const cv::Mat& Tcw_,
                                      const std::vector<cv::Point2f> vFeatures_)
{
    // 步骤1：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver =
        new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    g2o::CameraParameters* camera = new g2o::CameraParameters(
        K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // 步骤2：添加顶点：待优化当前帧的Tcw
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setId(0);
    vSE3->setEstimate(ug2o::toSE3Quat(Tcw_));
    vSE3->setFixed(false);
    vSE3->setMarginalized(false);
    optimizer.addVertex(vSE3);

    std::vector<g2o::EdgeProjectXYZ2UV*> vpEdgesMono;
    vpEdgesMono.reserve(nFeaturesPerFrame);
    const float deltaMono = sqrt(5.991);

    // 步骤3：添加一元边：相机投影模型
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    rk->setDelta(deltaMono);
    int id = 1;
    for (int i = 0; i < nFeaturesPerFrame; i++) {
        g2o::VertexSBAPointXYZ* vXYZ = new g2o::VertexSBAPointXYZ();
        Eigen::Vector3d est(mvMapPoints[i].x, mvMapPoints[i].y, mvMapPoints[i].z);
        vXYZ->setId(id);
        vXYZ->setEstimate(est);
        vXYZ->setFixed(true);
        vXYZ->setMarginalized(true);
        optimizer.addVertex(vXYZ);

        g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
        Eigen::Vector2d obs(vFeatures_[i].x, vFeatures_[i].y);
        e->setId(id);
        e->setVertex(0, vXYZ);
        e->setVertex(1, vSE3);
        e->setMeasurement(obs);
        e->setParameterId(0, 0);
        e->setInformation(Eigen::Matrix2d::Identity());
        //        e->setRobustKernel(rk);

        optimizer.addEdge(e);
        vpEdgesMono.push_back(e);
        id++;
    }

    // 步骤4：开始优化
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    // 步骤4：返回优化后的位姿
    g2o::SE3Quat SE3quat_recov = vSE3->estimate();
    cv::Mat Tcw_refined = ug2o::toCvMat(SE3quat_recov);

    return Tcw_refined;
}


void ExtrinsicCalibrator::showChessboardCorners()
{
    if (!mbVerbose)
        return;
    for (int i = 0; i < nTatalFrames; ++i) {
        cv::Mat imageOut;
        // 画角点
        cv::cvtColor(mvImageMats[i], imageOut, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(imageOut, mBoardSize, cv::Mat(mvvCorners[i]), 1);

        //        // 画坐标轴
        //        Eigen::Matrix3d Rcw;
        //        Eigen::Vector3d tcw;
        //        cv::cv2eigen(rotation, Rcw);
        //        cv::cv2eigen(cv_t, tcw);
        //        std::vector<Eigen::Vector3d> axis;
        //        axis.push_back(Rcw * Eigen::Vector3d(0, 0, 0) + tcw);
        //        axis.push_back(Rcw * Eigen::Vector3d(0.5, 0, 0) + tcw);
        //        axis.push_back(Rcw * Eigen::Vector3d(0, 0.5, 0) + tcw);
        //        axis.push_back(Rcw * Eigen::Vector3d(0, 0, 0.5) + tcw);
        //        std::vector<Eigen::Vector2d> imgpts(4);
        //        for (int i = 0; i < 4; ++i) {
        //            cam->spaceToPlane(axis[i], imgpts[i]);
        //        }
        //        cv::line(imageOut, cv::Point2f(imgpts[0](0), imgpts[0](1)),
        //                 cv::Point2f(imgpts[1](0), imgpts[1](1)), cv::Scalar(255, 0, 0), 2);  //
        //                 BGR
        //        cv::line(imageOut, cv::Point2f(imgpts[0](0), imgpts[0](1)),
        //                 cv::Point2f(imgpts[2](0), imgpts[2](1)), cv::Scalar(0, 255, 0), 2);
        //        cv::line(imageOut, cv::Point2f(imgpts[0](0), imgpts[0](1)),
        //                 cv::Point2f(imgpts[3](0), imgpts[3](1)), cv::Scalar(0, 0, 255), 2);

        //        cv::imshow("Current Image Corners", imageOut);
        //        cv::waitKey(50);
    }
    cv::destroyAllWindows();
}

void ExtrinsicCalibrator::writePose(const std::string& outputFile)
{
    std::ofstream ofs(outputFile);
    if (!ofs.is_open()) {
        std::cerr << "File open error! : " << outputFile << std::endl;
        return;
    }
    for (int i = 0; i < nTatalFrames; ++i) {
        cv::Mat Twci = mvTwc[i].rowRange(0, 3).col(3);
        cv::Mat Twbi = mvTwb[i].rowRange(0, 3).col(3);
        cv::Mat Twcr = mvTwc_refined[i].rowRange(0, 3).col(3);
        //        std::cout << "#" << i << " Twc: " << Twci.t() << " , Twb: " << Twbi.t() <<
        //        std::endl;
        ofs << Twbi.at<float>(0) << " " << Twbi.at<float>(1) << " " << Twbi.at<float>(2) << " "
            << Twci.at<float>(0) << " " << Twci.at<float>(1) << " " << Twci.at<float>(2) << " "
            << Twcr.at<float>(0) << " " << Twcr.at<float>(1) << " " << Twcr.at<float>(2)
            << std::endl;
    }
    ofs.close();
    std::cout << "Trajectories saved to " << outputFile << std::endl;
}

}  // namespace ec
