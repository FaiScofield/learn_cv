#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;
namespace bf = boost::filesystem;

string imageFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/gooImages.txt";
string projectFile = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/reprojectedPoints.txt";
Size g_boardSize(11, 8);
bool g_half = false;


void readImages(const string& imageFile, vector<string>& fullImages)
{
    vector<string> res;

    ifstream ifs(imageFile);
    if (!ifs.is_open()) {
        cerr << "File open error! : " << imageFile << endl;
        return;
    }

    string lineData;
    while (getline(ifs, lineData) && !lineData.empty())
        res.push_back(lineData);
    ifs.close();

    fullImages = res;
}

void getAllCornersFromMatlabFile(const string& projectFile, const vector<string>& fullImages,
                                 vector<vector<Point2f>>& allCorners)
{
    ifstream ifs(projectFile);
    if (!ifs.is_open()) {
        cerr << "File open error! : " << imageFile << endl;
        return;
    }

    int n = fullImages.size();
    vector<Point2f> vp(88, Point2f());
    allCorners.resize(n, vp);

    int index = 0;
    string lineData;
    while (!ifs.eof()) {
        getline(ifs, lineData);
        if (lineData.empty())
            continue;

        // 确定图像索引index
        if (boost::starts_with(lineData, "val(:,:")) {
            auto i = lineData.find_last_of(',');
            auto j = lineData.find_last_of(')');
            index = stoi(lineData.substr(i+1, j-i-1)) - 1;
            if (index < 0 || index >= n) {
                cerr << "Wrong index in file! : " << index + 1 << endl;
                continue;
            }
            getline(ifs, lineData);    // 去掉 "val(:,:,1) =" 下面的一行空行
        }

        // 读入该图像对应的88个角点
        cout << "Reading " << index << " image corners." << endl;
        string pointDate;
        for (int j = 0; j < 88; ++j) {
            getline(ifs, pointDate);
            stringstream ss(pointDate);
            Point2f p;
            ss >> p.x >> p.y;
            allCorners[index][j] = p;
        }
    }
    ifs.close();

    assert(index == n - 1);
}

int main(int argc, char *argv[])
{
    vector<string> fullImages;
    readImages(imageFile, fullImages);
    cout << "Read " << fullImages.size() << " files in the file." << endl;

    Mat cameraMatrix, distCoeffs;        // 待求内参和畸变系数
    vector<vector<Point2f>> allCorners;  // 所有棋盘格角点

    getAllCornersFromMatlabFile(projectFile, fullImages, allCorners);

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
        corners = allCorners[i];
        drawChessboardCorners(imageOut, g_boardSize, Mat(corners), 1);
        imshow("Current Image Corners", imageOut);
        waitKey(33);
    }

//    //! 2.计算标定结果
//    Size imageSize = imageOut.size();
//    vector<Mat> rvecs, tvecs;
//    vector<float> reprojErrs;
//    double totalAvgErr = 0;
//    bool ok = runCalibration(imageSize, cameraMatrix, distCoeffs, allCorners, rvecs, tvecs,
//                             reprojErrs, totalAvgErr);
//    cout << (ok ? "Calibration succeeded." : "Calibration failed!")
//         << " avg reprojection error = " << totalAvgErr << endl;

//    //! 3.根据标定结果校正畸变,并保存标定结果
//    if (ok) {
//        Mat view, rview, map1, map2;
//        initUndistortRectifyMap(
//            cameraMatrix, distCoeffs, Mat(),
//            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
//            imageSize, CV_16SC2, map1, map2);

//        for (size_t i = 0; i < fullImages.size(); i++) {
//            view = imread(fullImages[i], IMREAD_COLOR);
//            if (view.empty())
//                continue;
//            if (g_half)
//                resize(view, view, Size(view.cols/2, view.rows/2));
//            remap(view, rview, map1, map2, INTER_LINEAR);
////          undistort(temp, image, cameraMatrix, distCoeffs);
//            imshow("Image Undistortion", rview);
//            char c = (char)waitKey();
//            if (c == ESC_KEY || c == 'q' || c == 'Q')
//                break;
//        }

//        saveCameraParams(imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs,
//                         allCorners, totalAvgErr);
//        cerr << "Save calibration output to " << outputFile << endl;
//    }

    return 0;
}
