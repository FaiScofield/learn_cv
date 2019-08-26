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

using namespace cv;

ExtrinsicCalibrator::ExtrinsicCalibrator()
{
    mBoardSize = Size(8, 11);  // Size(width, height) 注意这里不能用11x8,要用8x11,根据Matlab而来.
    mSquareSize = 0.03; // [m]

    K = (Mat_<double>(3, 3) << 415.3904681732305, 0., 317.2546831363181, 0., 415.6664203685940,
         241.0199225105337, 0., 0., 1.);
    D = (Mat_<double>(5, 1) << 5.3963689585481e-02, -5.3999880307856e-02, 7.248873665656701e-04,
         7.696301272230405e-04, 0.0);

    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
}

ExtrinsicCalibrator::~ExtrinsicCalibrator()
{
}
void ExtrinsicCalibrator::readCornersFromFile_Matlab(const std::string& cornerFile)
{
    if (N < 1) {
        std::cerr << "Please read image data first!" << std::endl;
        return;
    }

    std::ifstream ifs(cornerFile);
    if (!ifs.is_open()) {
        std::cerr << "File open error! : " << cornerFile << std::endl;
        return;
    }

    std::vector<Point2f> tmp(nFeaturesPerFrame);
    mvvCorners.resize(N, tmp);

    int index = -1;
    std::string lineData;
    while (!ifs.eof()) {
        std::getline(ifs, lineData);
        if (lineData.empty())
            continue;

        // 确定图像索引index
        if (std::sscanf(lineData.c_str(), "val(:,:,%d) =", &index) == 1) {
            index -= 1;  // Matlab序号从1开始计,这里要减掉1
            if (index < 0 || index >= N) {
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
            Point2f p;
            ss >> p.x >> p.y;
            mvvCorners[index][j] = p;
        }
    }
    ifs.close();

    assert(index == N - 1);

    // generate fixed MapPoint
    for (int i = 0; i < mBoardSize.height; i++) {
        for (int j = 0; j < mBoardSize.width; j++)
            mvMapPoints.emplace_back(Point3f(float(j * mSquareSize), float(i * mSquareSize), 0.));
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
        Mat img = imread(lineData, CV_LOAD_IMAGE_GRAYSCALE);
        mvImageMats.push_back(img);
    }
    ifs.close();

    if (mvImageRaws.empty()) {
        std::cerr << "Not image data in the folder!" << std::endl;
        return;
    } else {
        std::cout << "Read " << mvImageRaws.size() << " files in the folder." << std::endl;
        N = mvImageRaws.size();
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
        std::stringstream ss(lineData);
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
    assert(N == mvImageRaws.size());
    assert(N == mvvCorners.size());
    assert(N <= mvOdomRaws.size());
    //    assert(mvTimeImage.)  // sorted?

    std::cout << "Syncing image data and odom data with timestamps..." << std::endl;

    std::vector<OdomRaw> vOdomSynced(N);
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
    for (int j = 0; j < N; ++j) {
        Mat rvec, tvec, R;
#ifdef withHomograpy
        //! 通过Homograpy计算相机位姿. 参考OpenCV例子pose_from_homography.cpp
        //! [compute-image-points]
        std::vector<Point2f> imagePoints;
        undistortPoints(mvvCorners[j], imagePoints, K, D);

        //! [compute-object-points]
        std::vector<Point2f> objectPointsPlanar;
        for (size_t i = 0; i < mvMapPoints.size(); i++) {
            objectPointsPlanar.push_back(Point2f(mvMapPoints[i].x, mvMapPoints[i].y));
        }

        //! [estimate-homography]
        Mat H = findHomography(objectPointsPlanar, imagePoints);

        //! [pose-from-homography]
        // Normalization to ensure that ||c1|| = 1
        double norm = std::sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) +
                                H.at<double>(1, 0) * H.at<double>(1, 0) +
                                H.at<double>(2, 0) * H.at<double>(2, 0));
        H /= norm;
        Mat c1 = H.col(0);
        Mat c2 = H.col(1);
        Mat c3 = c1.cross(c2);
        tvec = H.col(2);
        for (int i = 0; i < 3; i++) {
            R.at<double>(i, 0) = c1.at<double>(i, 0);
            R.at<double>(i, 1) = c2.at<double>(i, 0);
            R.at<double>(i, 2) = c3.at<double>(i, 0);
        }

        //! [polar-decomposition-of-the-rotation-matrix]
        Mat W, U, Vt;
        SVDecomp(R, W, U, Vt);
        R = U * Vt;
        Rodrigues(R, rvec);
#else
        solvePnP(mvMapPoints, mvvCorners[j], K, D, rvec, tvec);
        Rodrigues(rvec, R);
#endif
        // 保存计算结果
        Mat Tcw = Mat::eye(4, 4, CV_32F);
        R.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
        tvec.copyTo(Tcw.rowRange(0, 3).col(3));
        mvTcw.push_back(Tcw);

        // 相机位姿优化
        Mat Tcw_refined = optimize(Tcw, mvvCorners[j]);
        mvTcw_refined.push_back(Tcw_refined);

        //! 计算里程计位姿, w为首帧Odom的位姿，置为原点
        Mat Tbw = mvOdomRaws[j].toCvSE3();
        mvTbw.push_back(Tbw);

        // 可视化, 验证Tcw的准确性
        if (mbVerbose) {
            // 显示角点图与坐标轴, xyz对应rgb
            Mat img_corners, img_pose, img_joint;
            cvtColor(mvImageMats[j], img_corners, COLOR_GRAY2BGR);
            cvtColor(mvImageMats[j], img_pose, COLOR_GRAY2BGR);
            drawChessboardCorners(img_corners, mBoardSize, mvvCorners[j], 1);
            circle(img_corners, mvvCorners[j][0], 8, Scalar(0, 0, 255), 2);
            aruco::drawAxis(img_pose, K, D, rvec, tvec, 2 * mSquareSize);
            hconcat(img_corners, img_pose, img_joint);
            imshow("Chessboard corners and pose", img_joint);

            waitKey(10);
        }
    }
    assert(mvTcw.size() == N);
    assert(mvTbw.size() == N);

    // 计算Tcjci和Tbjbi,用于外参标定
    for (int j = 1; j < N; ++j) {
        int i = j - 1;
        Mat Tcjci = mvTcw[j] * cvu::inv(mvTcw[i]);
        Mat Tbjbi = mvTbw[j] * cvu::inv(mvTbw[i]);
        mvTcjci.push_back(Tcjci);
        mvTbjbi.push_back(Tbjbi);   // 已验证
    }
    assert(mvTcjci.size() == N - 1);
    assert(mvTbjbi.size() == N - 1);
}

Mat ExtrinsicCalibrator::optimize(const Mat& Tcw_, const std::vector<Point2f> vFeatures_)
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
    const float deltaMono = std::sqrt(5.991);

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
//        e->setRobustKernel(rk);   //! NOTE 设置了鲁棒核函数后有错误

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
    Mat Tcw_refined = ug2o::toCvMat(SE3quat_recov);

    return Tcw_refined;
}

bool ExtrinsicCalibrator::estimatePitchRoll(Eigen::Matrix3d& Rbc_yx)
{
    Eigen::MatrixXd M((N - 1) * 4, 4);
    M.setZero();

    size_t mark = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        const Eigen::Vector3d& rvec_cam = cvu::getAngleAxisFromCvMat(mvTcjci[i]);
        const Eigen::Vector3d& rvec_odo = cvu::getAngleAxisFromCvMat(mvTbjbi[i]);

        // Remove zero rotation.
        if (rvec_cam.norm() == 0 || rvec_odo.norm() == 0) {
            std::cout << "[Warning] A zero rotation occurred!" << std::endl;
            continue;
        }

        Eigen::Quaterniond q_cam;
        q_cam = Eigen::AngleAxisd(rvec_cam.norm(), rvec_cam.normalized());
        Eigen::Quaterniond q_odo;
        q_odo = Eigen::AngleAxisd(rvec_odo.norm(), rvec_odo.normalized());

        M.block<4, 4>(mark * 4, 0) = cvu::QuaternionMultMatLeft(q_odo) - cvu::QuaternionMultMatRight(q_cam);
        mark++;
    }


    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // 取出V_4X4的最后两列
    Eigen::Vector4d t1 = svd.matrixV().block<4, 1>(0, 2);
    Eigen::Vector4d t2 = svd.matrixV().block<4, 1>(0, 3);
    std::cout << "Sigular values of M: " << svd.singularValues().transpose() << std::endl;

    // solve constraint for q_yz: xy = -zw
    double x[2];
    if (!solveQuadraticEquation(t1(0) * t1(1) + t1(2) * t1(3),
                                t1(0) * t2(1) + t1(1) * t2(0) + t1(2) * t2(3) + t1(3) * t2(2),
                                t2(0) * t2(1) + t2(2) * t2(3), x[0], x[1])) {
        std::cout << "# ERROR: Quadratic equation cannot be solved due to negative determinant."
                  << std::endl;
        return false;
    }

    Eigen::Matrix3d R_yxs[2];
    double yaw[2];

    for (int i = 0; i < 2; ++i) {
        double t = x[i] * x[i] * t1.dot(t1) + 2 * x[i] * t1.dot(t2) + t2.dot(t2);

        // solve constraint ||q_yx|| = 1.
        double b = std::sqrt(1.0 / t);
        double a = x[i] * b;

        Eigen::Quaterniond q_yx;
        q_yx.coeffs() = a * t1 + b * t2;    // a,b为scale
        R_yxs[i] = q_yx.toRotationMatrix();

        double r, p;
        cvu::EigenMat2RPY(R_yxs[i], r, p, yaw[i]);
    }
    printf("the 2 Yaws: %f, %f\n", yaw[0], yaw[1]);
    if (fabs(yaw[0]) < fabs(yaw[1]))
        Rbc_yx = R_yxs[0];
    else
        Rbc_yx = R_yxs[1];

    return true;
}

// 解一元二次方程的两个根
bool ExtrinsicCalibrator::solveQuadraticEquation(double a, double b, double c, double& x1,
                                               double& x2) const
{
    if (fabs(a) < 1e-12) {
        x1 = x2 = -c / b;
        return true;
    }
    double delta2 = b * b - 4.0 * a * c;

    if (delta2 < 0.0) {
        return false;
    }

    double delta = std::sqrt(delta2);

    x1 = (-b + delta) / (2.0 * a);
    x2 = (-b - delta) / (2.0 * a);

    return true;
}

bool ExtrinsicCalibrator::estimate(Eigen::Matrix4d &H_cam_odo, std::vector<double> &scales)
{
    // Estimate Rbc_yx first
    Eigen::Matrix3d R_yx;
    estimatePitchRoll(R_yx);

    int segmentCount = 1;
    int motionCount = N - 1;
//    for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
//        motionCount += rvecs1.at(segmentId).size();
//    }

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(motionCount * 2, 2 + segmentCount * 2);
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(motionCount * 2, 1);

    int mark = 0;
    for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
        for (size_t motionId = 0; motionId < motionCount; ++motionId) {
            const Eigen::Vector3d& rvec1 = cvu::getAngleAxisFromCvMat(mvTbjbi[motionId]);
            const Eigen::Vector3d& tvec1 = cvu::getTranslationFromCvMat(mvTbjbi[motionId]);
            const Eigen::Vector3d& rvec2 = cvu::getAngleAxisFromCvMat(mvTcjci[motionId]);
            const Eigen::Vector3d& tvec2 = cvu::getTranslationFromCvMat(mvTcjci[motionId]);

//            const Eigen::Vector3d& rvec1 = rvecs1.at(segmentId).at(motionId);
//            const Eigen::Vector3d& tvec1 = tvecs1.at(segmentId).at(motionId);
//            const Eigen::Vector3d& rvec2 = rvecs2.at(segmentId).at(motionId);
//            const Eigen::Vector3d& tvec2 = tvecs2.at(segmentId).at(motionId);

            // Remove zero rotation.
            if (rvec1.norm() < 1e-10 || rvec2.norm() < 1e-10) {
                ++mark;
                continue;
            }

            Eigen::Quaterniond q1;
            q1 = Eigen::AngleAxisd(rvec1.norm(), rvec1.normalized());

            Eigen::Matrix2d J;
            J = q1.toRotationMatrix().block<2, 2>(0, 0) - Eigen::Matrix2d::Identity();

            // project tvec2 to plane with normal defined by 3rd row of R_yx
            Eigen::Vector3d n;
            n = R_yx.row(2);

            Eigen::Vector3d pi = R_yx * (tvec2 - tvec2.dot(n) * n);

            Eigen::Matrix2d K;
            K << pi(0), -pi(1), pi(1), pi(0);

            G.block<2, 2>(mark * 2, 0) = J;
            G.block<2, 2>(mark * 2, 2 + segmentId * 2) = K;

            w.block<2, 1>(mark * 2, 0) = tvec1.block<2, 1>(0, 0);

            ++mark;
        }
    }

    Eigen::MatrixXd m(2 + segmentCount * 2, 1);
    m = G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(w);

    Eigen::Vector2d t(-m(0), -m(1));

    std::vector<double> alpha_hypos;
    for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
        double alpha = atan2(m(2 + segmentId * 2 + 1), m(2 + segmentId * 2));
        double scale = m.block<2, 1>(2 + segmentId * 2, 0).norm();

        alpha_hypos.push_back(alpha);
        scales.push_back(scale);
    }

    double errorMin = std::numeric_limits<double>::max();
    double alpha_best = 0.0;

    for (size_t i = 0; i < alpha_hypos.size(); ++i) {
        double error = 0.0;
        double alpha = alpha_hypos.at(i);

        for (int segmentId = 0; segmentId < segmentCount; ++segmentId) {
            for (size_t motionId = 0; motionId < motionCount; ++motionId) {
                const Eigen::Vector3d& rvec1 = cvu::getAngleAxisFromCvMat(mvTbjbi[motionId]);
                const Eigen::Vector3d& tvec1 = cvu::getTranslationFromCvMat(mvTbjbi[motionId]);
                const Eigen::Vector3d& rvec2 = cvu::getAngleAxisFromCvMat(mvTcjci[motionId]);
                const Eigen::Vector3d& tvec2 = cvu::getTranslationFromCvMat(mvTcjci[motionId]);

//                const Eigen::Vector3d& rvec1 = rvecs1.at(segmentId).at(motionId);
//                const Eigen::Vector3d& tvec1 = tvecs1.at(segmentId).at(motionId);
//                const Eigen::Vector3d& rvec2 = rvecs2.at(segmentId).at(motionId);
//                const Eigen::Vector3d& tvec2 = tvecs2.at(segmentId).at(motionId);

                Eigen::Quaterniond q1;
                q1 = Eigen::AngleAxisd(rvec1.norm(), rvec1.normalized());

                Eigen::Matrix3d N;
                N = q1.toRotationMatrix() - Eigen::Matrix3d::Identity();

                Eigen::Matrix3d R = Eigen::AngleAxisd(alpha, Eigen::Vector3d::UnitZ()) * R_yx;

                // project tvec2 to plane with normal defined by 3rd row of R
                Eigen::Vector3d n;
                n = R.row(2);

                Eigen::Vector3d pc = tvec2 - tvec2.dot(n) * n;
                //                    Eigen::Vector3d pc = tvec2;

                Eigen::Vector3d A = R * pc;
                Eigen::Vector3d b = N * (Eigen::Vector3d() << t, 0.0).finished() + tvec1;

                error += (A * scales.at(segmentId) - b).norm();
            }
        }

        if (error < errorMin) {
            errorMin = error;
            alpha_best = alpha;
        }
    }

    H_cam_odo.setIdentity();
    H_cam_odo.block<3, 3>(0, 0) = Eigen::AngleAxisd(alpha_best, Eigen::Vector3d::UnitZ()) * R_yx;
    H_cam_odo.block<2, 1>(0, 3) = t;

    if (mbVerbose) {
        std::cout << "# INFO: Before refinement:" << std::endl;
        std::cout << "H_cam_odo = " << std::endl;
        std::cout << H_cam_odo << std::endl;
        std::cout << "scales = " << std::endl;
        for (size_t i = 0; i < scales.size(); ++i) {
            std::cout << scales.at(i) << " ";
        }
        std::cout << std::endl;
    }

//    refineEstimate(H_cam_odo, scales);

//    if (mbVerbose) {
//        std::cout << "# INFO: After refinement:" << std::endl;
//        std::cout << "H_cam_odo = " << std::endl;
//        std::cout << H_cam_odo << std::endl;
//        std::cout << "scales = " << std::endl;
//        for (size_t i = 0; i < scales.size(); ++i) {
//            std::cout << scales.at(i) << " ";
//        }
//        std::cout << std::endl;
//    }

    return true;
}



void ExtrinsicCalibrator::showChessboardCorners()
{
    if (!mbVerbose)
        return;
    for (int i = 0; i < N; ++i) {
        Mat imageOut;
        // 画角点
        cvtColor(mvImageMats[i], imageOut, COLOR_GRAY2BGR);
        drawChessboardCorners(imageOut, mBoardSize, Mat(mvvCorners[i]), 1);

//        // 画坐标轴
//        Eigen::Matrix3d Rcw;
//        Eigen::Vector3d tcw;
//        cv2eigen(rotation, Rcw);
//        cv2eigen(cv_t, tcw);
//        std::vector<Eigen::Vector3d> axis;
//        axis.push_back(Rcw * Eigen::Vector3d(0, 0, 0) + tcw);
//        axis.push_back(Rcw * Eigen::Vector3d(0.5, 0, 0) + tcw);
//        axis.push_back(Rcw * Eigen::Vector3d(0, 0.5, 0) + tcw);
//        axis.push_back(Rcw * Eigen::Vector3d(0, 0, 0.5) + tcw);
//        std::vector<Eigen::Vector2d> imgpts(4);
//        for (int i = 0; i < 4; ++i) {
//            cam->spaceToPlane(axis[i], imgpts[i]);
//        }
//        line(imageOut, Point2f(imgpts[0](0), imgpts[0](1)),
//                 Point2f(imgpts[1](0), imgpts[1](1)), Scalar(255, 0, 0), 2);  //
//                 BGR
//        line(imageOut, Point2f(imgpts[0](0), imgpts[0](1)),
//                 Point2f(imgpts[2](0), imgpts[2](1)), Scalar(0, 255, 0), 2);
//        line(imageOut, Point2f(imgpts[0](0), imgpts[0](1)),
//                 Point2f(imgpts[3](0), imgpts[3](1)), Scalar(0, 0, 255), 2);

//        imshow("Current Image Corners", imageOut);
//        waitKey(50);
    }
    destroyAllWindows();
}

void ExtrinsicCalibrator::writePose(const std::string& outputFile)
{
    Mat Tc0w = mvTcw[0];
    Mat Tb0w = mvTbw[0];
    Mat Tc0w_r = mvTcw_refined[0];
    for (int i = 0; i < N; ++i) {
        Mat Twci = Tc0w * cvu::inv(mvTcw[i]);   // Tc0ci = Twci, 以首帧坐标系为原点, 可视化用
        mvTwcPoseCam.push_back(Twci);           // 存入Twc，平移部分即为其Pose

        Mat Twbi = Tb0w * cvu::inv(mvTbw[i]);
        mvTwbPoseOdo.push_back(Twbi);

        Mat Twci_r = Tc0w_r * cvu::inv(mvTcw_refined[i]);
        mvTwc_refined.push_back(Twci_r);
    }

    std::ofstream ofs(outputFile);
    if (!ofs.is_open()) {
        std::cerr << "File open error! : " << outputFile << std::endl;
        return;
    }
    for (int i = 0; i < N; ++i) {
        Mat Twci = mvTwcPoseCam[i].rowRange(0, 3).col(3);
        Mat Twbi = mvTwbPoseOdo[i].rowRange(0, 3).col(3);
        Mat Twcr = mvTwc_refined[i].rowRange(0, 3).col(3);
        if (mbVerbose) {
            std::cout << "[DEBUG] Odom Pose t = " << Twbi.t() << std::endl;
            std::cout << "[DEBUG] Camera Pose t = " << Twci.t() << std::endl;
            std::cout << "[DEBUG] Camera Pose Refined t = " << Twcr.t() << std::endl;
        }

        ofs << Twbi.at<float>(0) << " " << Twbi.at<float>(1) << " " << Twbi.at<float>(2) << " "
            << Twci.at<float>(0) << " " << Twci.at<float>(1) << " " << Twci.at<float>(2) << " "
            << Twcr.at<float>(0) << " " << Twcr.at<float>(1) << " " << Twcr.at<float>(2)
            << std::endl;
    }
    ofs.close();
    std::cout << "Trajectories saved to " << outputFile << std::endl;
}

}  // namespace ec
