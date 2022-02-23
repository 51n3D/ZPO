#include "EdgeDetection.h"
#include "zpo.h"

using namespace zpo;

cv::Mat EdgeDetection::filterImage(const cv::Mat& src, int type)
{
    cv::Mat horiz, vert;
    zpo::filter2D(src, horiz, this->kernel[type][HORIZ]);
    zpo::filter2D(src, vert, this->kernel[type][VERT]);

    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            auto pxl_horiz = horiz.at<float>(i, j);
            auto pxl_vert = vert.at<float>(i, j);
            dst.at<float>(i, j) = sqrt(pow(pxl_horiz, 2) + pow(pxl_vert, 2));
        }
    }

    // zpo::treshold(src, dst, 127);

    return dst;
}
