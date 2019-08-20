#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
namespace bf = boost::filesystem;

string imageFolder = "/home/vance/dataset/rk/dibeaDataSet/ov7251_640";

vector<string> readFolderFiles(const string& folder) {
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
    Mat imgMarker;
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::drawMarker(dictionary, 23, 200, imgMarker, 1);

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(7, 5, 50.f, 15.f, dictionary);

    Size imageSize;
    imageSize.width = 7 * (50. + 15.) - 15. + 2 * 15.;
    imageSize.height = 5 * (50. + 15.) - 15. + 2 * 15.;

    Mat boardImage;
    board->draw(imageSize, boardImage, 15., 15.);
    imshow("board", boardImage);


    imshow("Marker", imgMarker);
    waitKey(0);

    vector<string> fullImages = readFolderFiles(imageFolder);



    return 0;
}
