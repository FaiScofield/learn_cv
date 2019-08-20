#include "calibration/Chessboard.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
namespace bf = boost::filesystem;

string imageFolder = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_4/slamimg";
string outputFile = "result.xml";
Size g_boardSize(11, 8);    // 棋盘格内点数(width/cols, height/rows)
float g_squareSize = 30;    // 棋盘格宽度[mm]
const char ESC_KEY = 27;    // 退出按键ESC
bool g_half = false;

struct RK_IMAGE
{
    RK_IMAGE(const string& s, const long long int t)
        : fileName(s), timeStamp(t) {}

    string fileName;
    long long int timeStamp;
};


bool lessThen(const RK_IMAGE& r1, const RK_IMAGE& r2)
{
    return r1.timeStamp < r2.timeStamp;
}


void readImagesRK(const string& dataFolder, vector<string>& files)
{
    bf::path path(dataFolder);
    if (!bf::exists(path)) {
        cerr << "[Main] Data folder doesn't exist!" << endl;
        return;
    }

    vector<RK_IMAGE> allImages;
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /frameRaw12987978101.jpg
            string s = iter->path().string();
            auto i = s.find_last_of('w');
            auto j = s.find_last_of('.');
            auto t = atoll(s.substr(i+1, j-i-1).c_str());
            allImages.push_back(RK_IMAGE(s, t));
        }
    }

    if (allImages.empty()) {
        cerr << "[Main] Not image data in the folder!" << endl;
        return;
    } else
        cout << "[Main] Read " << allImages.size() << " files in the folder." << endl;

    //! 应该根据后面的时间戳数值来排序
    sort(allImages.begin(), allImages.end(), lessThen);

    files.clear();
    for (int i = 0; i < allImages.size(); ++i)
        files.push_back(allImages[i].fileName);
}

double computeReprojectionErrors(const vector<vector<Point3f>> &objectPoints,
                                 const vector<vector<Point2f>> &imagePoints,
                                 const vector<Mat> &rvecs, const vector<Mat> &tvecs,
                                 const Mat &cameraMatrix, const Mat &distCoeffs,
                                 vector<float> &perViewErrors)
{
    vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (size_t i = 0; i < objectPoints.size(); ++i) {
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

bool runCalibration(Size &imageSize, Mat &cameraMatrix, Mat &distCoeffs,
                    vector<vector<Point2f>> imagePoints, vector<Mat> &rvecs, vector<Mat> &tvecs,
                    vector<float> &reprojErrs, double &totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f>> objectPoints(1);
    for (int i = 0; i < g_boardSize.height; ++i)
        for (int j = 0; j < g_boardSize.width; ++j)
            objectPoints[0].push_back(Point3f(j * g_squareSize, i * g_squareSize, 0));

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    // Find intrinsic and extrinsic camera parameters
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
                                 rvecs, tvecs);

    cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix,
                                            distCoeffs, reprojErrs);

    return ok;
}

// Print camera parameters to the output file
static void saveCameraParams(Size &imageSize, Mat &cameraMatrix, Mat &distCoeffs,
                             const vector<Mat> &rvecs, const vector<Mat> &tvecs,
                             const vector<float> &reprojErrs,
                             const vector<vector<Point2f>> &imagePoints, double totalAvgErr)
{
    FileStorage fs(outputFile, FileStorage::WRITE);

    if (!rvecs.empty() || !reprojErrs.empty())
        fs << "nr_of_frames" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << g_boardSize.width;
    fs << "board_height" << g_boardSize.height;
    fs << "square_size" << g_squareSize;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (!reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if (!rvecs.empty() && !tvecs.empty()) {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, CV_MAKETYPE(rvecs[0].type(), 1));
        bool needReshapeR = rvecs[0].depth() != 1 ? true : false;
        bool needReshapeT = tvecs[0].depth() != 1 ? true : false;

        for (size_t i = 0; i < rvecs.size(); i++) {
            Mat r = bigmat(Range(int(i), int(i + 1)), Range(0, 3));
            Mat t = bigmat(Range(int(i), int(i + 1)), Range(3, 6));

            if (needReshapeR)
                rvecs[i].reshape(1, 1).copyTo(r);
            else {
                //*.t() is MatExpr (not Mat) so we can use assignment operator
                CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
                r = rvecs[i].t();
            }

            if (needReshapeT)
                tvecs[i].reshape(1, 1).copyTo(t);
            else {
                CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
                t = tvecs[i].t();
            }
        }

        fs << "extrinsic_parameters" << bigmat;
    }
}

int main(int argc, char *argv[])
{
    vector<string> fullImages;
    readImagesRK(imageFolder, fullImages);
    cout << "Read " << fullImages.size() << " files in the folder." << endl;

    Mat cameraMatrix, distCoeffs;        // 待求内参和畸变系数
    vector<vector<Point2f>> allCorners;  // 所有棋盘格角点

    //! 1.输入图像并检测棋盘格角点
    Mat imageOut;
    for (int i = 0; i < fullImages.size(); ++i) {
        Mat image = imread(fullImages[i], CV_LOAD_IMAGE_GRAYSCALE);
        if (g_half)
            resize(image, image, Size(image.cols/2, image.rows/2));
        cvtColor(image, imageOut, COLOR_GRAY2BGR);

        // CALIB_CB_FAST_CHECK saves a lot of time on images
        // that do not contain any chessboard corners
        vector<Point2f> corners;
        bool found = findChessboardCorners(image, g_boardSize, corners, /*CALIB_CB_FILTER_QUADS +*/
                                           CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
        learnCV::Chessboard cb(g_boardSize, image);
        cb.findCorners(false);
        cv::Mat sketch;
        cb.getSketch().copyTo(sketch);
        cv::imshow("Image", sketch);
        if (found) {
            cornerSubPix(image, corners, Size(5, 5), Size(-1, -1),
                         TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(imageOut, g_boardSize, Mat(corners), found);
            allCorners.push_back(corners);
            cout << "Find " << corners.size() << " corners in image " << i << endl;
        }
        imshow("Current Image Corners", imageOut);
        waitKey(10);
    }
    cout << "Detected " << allCorners.size() << " images with chessboard corners." << endl;

    //! 2.计算标定结果
    Size imageSize = imageOut.size();
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    bool ok = runCalibration(imageSize, cameraMatrix, distCoeffs, allCorners, rvecs, tvecs,
                             reprojErrs, totalAvgErr);
    cout << (ok ? "Calibration succeeded." : "Calibration failed!")
         << " avg reprojection error = " << totalAvgErr << endl;

    //! 3.根据标定结果校正畸变,并保存标定结果
    if (ok) {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(
            cameraMatrix, distCoeffs, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
            imageSize, CV_16SC2, map1, map2);

        for (size_t i = 0; i < fullImages.size(); i++) {
            view = imread(fullImages[i], IMREAD_COLOR);
            if (view.empty())
                continue;
            if (g_half)
                resize(view, view, Size(view.cols/2, view.rows/2));
            remap(view, rview, map1, map2, INTER_LINEAR);
//          undistort(temp, image, cameraMatrix, distCoeffs);
            imshow("Image Undistortion", rview);
            char c = (char)waitKey();
            if (c == ESC_KEY || c == 'q' || c == 'Q')
                break;
        }

        saveCameraParams(imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs,
                         allCorners, totalAvgErr);
        cerr << "Save calibration output to " << outputFile << endl;
    }

    return 0;
}
