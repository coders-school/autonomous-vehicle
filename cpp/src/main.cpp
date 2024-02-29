#include <iostream>
#include <opencv2/highgui.hpp>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    std::string executablePath = std::filesystem::current_path();

    std::string imagePath = std::filesystem::path(executablePath).parent_path().parent_path().string()+"/img/test_image.jpg";
    std::cout << imagePath;


    cv::Mat image = cv::imread(imagePath);

    if(image.empty()) {
        std::cerr << "Error: Unable to load the image!" << std::endl;
        return 1;
    }

    cv::imshow("Image", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
