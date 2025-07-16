#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    Mat mats[32];

    mats[0] = Mat::zeros(10, 10, CV_8UC1);
    mats[1] = Mat::zeros(10, 10, CV_8UC2);
    mats[2] = Mat::zeros(10, 10, CV_8UC3);
    mats[3] = Mat::zeros(10, 10, CV_8UC4);

    mats[4] = Mat::zeros(10, 10, CV_8SC1);
    mats[5] = Mat::zeros(10, 10, CV_8SC2);
    mats[6] = Mat::zeros(10, 10, CV_8SC3);
    mats[7] = Mat::zeros(10, 10, CV_8SC4);

    mats[8] = Mat::zeros(10, 10, CV_16UC1);
    mats[9] = Mat::zeros(10, 10, CV_16UC2);
    mats[10] = Mat::zeros(10, 10, CV_16UC3);
    mats[11] = Mat::zeros(10, 10, CV_16UC4);

    mats[12] = Mat::zeros(10, 10, CV_16SC1);
    mats[13] = Mat::zeros(10, 10, CV_16SC2);
    mats[14] = Mat::zeros(10, 10, CV_16SC3);
    mats[15] = Mat::zeros(10, 10, CV_16SC4);

    mats[16] = Mat::zeros(10, 10, CV_32SC1);
    mats[17] = Mat::zeros(10, 10, CV_32SC2);
    mats[18] = Mat::zeros(10, 10, CV_32SC3);
    mats[19] = Mat::zeros(10, 10, CV_32SC4);

    mats[20] = Mat::zeros(10, 10, CV_32FC1);
    mats[21] = Mat::zeros(10, 10, CV_32FC2);
    mats[22] = Mat::zeros(10, 10, CV_32FC3);
    mats[23] = Mat::zeros(10, 10, CV_32FC4);

    mats[24] = Mat::zeros(10, 10, CV_64FC1);
    mats[25] = Mat::zeros(10, 10, CV_64FC2);
    mats[26] = Mat::zeros(10, 10, CV_64FC3);
    mats[27] = Mat::zeros(10, 10, CV_64FC4);

    mats[28] = Mat::zeros(10, 10, CV_16FC1);
    mats[29] = Mat::zeros(10, 10, CV_16FC2);
    mats[30] = Mat::zeros(10, 10, CV_16FC3);
    mats[31] = Mat::zeros(10, 10, CV_16FC4);

    cout << "   flags   | flags & 0xFFF | type | flags & 0x7 | depth | (flags >> 3) \t\t| channels ((flags >> 3) & 0x1FF)" << endl;
    for (int i = 0; i < 32; ++i) {
        int flags = mats[i].flags;
        cout << dec << flags << " | "
             << dec << (flags & 0xFFF) << hex << "\t(0x" << (flags & 0xFFF) << ")\t   | " << dec << mats[i].type() << "\t  | "
             << dec << (flags & 0x07) << hex << "\t(0x" << (flags & 0x07) << ")\t| " << dec << mats[i].depth() << "\t| "
             << dec << (flags >> 3) << hex << " (0x" << (flags >> 3) << ") | \t" << dec << mats[i].channels() << endl;
    }

    return 0;
}
