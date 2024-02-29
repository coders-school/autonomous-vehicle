#include "gtest/gtest.h"
#include <opencv2/highgui.hpp>
#include <filesystem>

// Demonstrate some basic assertions.

TEST(Test, NameTest) 
{
  ASSERT_TRUE(true);
}
TEST(TestOpenCV, IsAdded)
{
   std::string executablePath = std::filesystem::current_path();

    std::string imagePath = std::filesystem::path(executablePath).parent_path().string()+"/img/test_image.jpg";
    std::cout << imagePath;


    cv::Mat image = cv::imread(imagePath);
    ASSERT_FALSE(image.empty());
}


