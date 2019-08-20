#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
namespace bf = boost::filesystem;

string imageFolder = "/home/vance/dataset/rk/dibeaDataSet/ov7251_640";
bool g_half = true;


vector<string> readFolderFiles(const string &folder)
{
    vector<string> files;

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


int main(int argc, char *argv[])
{
    vector<string> fullImages = readFolderFiles(imageFolder);
    cout << "Read " << fullImages.size() << " files in the folder." << endl;


    Mat K_matlab = (cv::Mat_<double>(3, 3) << 2*219.9359613169054, 0., 2*161.5827136112504, 0.,
                    2*219.4159055585876, 2*117.7128673795551, 0., 0., 1.);

    Mat D_matlab = (cv::Mat_<double>(5, 1) << 0.064610443232716, -0.086814339668420,
                    -0.0009238134627751219/(-2), 0.0005452823230733891/(-3), 0.);

    Mat K_opencv = (cv::Mat_<double>(3, 3) << 437.05743494642360, 0., 320.53031768186491, 0.,
                    436.40672700188384, 236.57325092200287, 0., 0., 1.);

    Mat D_opencv = (cv::Mat_<double>(5, 1) << 5.038792e-02, -4.930388e-02, 3.611181e-04,
                    3.337900e-04, -3.367658e-02);

    Mat K_opencv2 = (cv::Mat_<double>(3, 3) << 216.20902398521909, 0., 158.63355745790227, 0.,
                     215.61524002182867, 117.64730638276232, 0., 0., 1.);

    Mat D_opencv2 = (cv::Mat_<double>(5, 1) << 4.956187e-02, -5.639085e-02, -1.795665e-04,
                     -1.138351e-04, -2.522290e-02);

    //    Mat Ks2;
    //    Kb.copyTo(Ks2);
    //    Ks2.at<double>(0, 0) *= 1. / 2;
    //    Ks2.at<double>(0, 2) *= 1. / 2;
    //    Ks2.at<double>(1, 1) *= 1. / 2;
    //    Ks2.at<double>(1, 2) *= 1. / 2;

    //    Mat Ds2 = Db.clone();
    //    Ds2.at<double>(0, 0) *= 1. / 4;   // k1
    //    Ds2.at<double>(1, 0) *= 1. / 16;  // k2
    //    Ds2.at<double>(2, 0) *= 1. / 2;   // p1
    //    Ds2.at<double>(3, 0) *= 1. / 2;   // p1

    Mat matlab, opencv1, opencv2;
    for (const auto &img : fullImages) {
        Mat image = imread(img, IMREAD_GRAYSCALE);
        undistort(image, matlab, K_matlab, D_matlab);
        imshow("Original Image (640x480)", image);
        imshow("Undistorted Image Matlab (640x480)", matlab);
        undistort(image, opencv1, K_opencv, D_opencv);
        imshow("Undistorted Image OpenCV (640x480)", opencv1);

        resize(image, image, Size(image.cols / 2, image.rows / 2));
        undistort(image, opencv2, K_opencv2, D_opencv2);
        imshow("Original Image (320x240)", image);
        imshow("Undistorted Image OpenCV (320x240)", opencv2);

        waitKey(0);
    }


    return 0;
}
