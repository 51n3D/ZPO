#include "ImageQuantization.h"

using namespace zpo;


void ImageQuantization::filter(const cv::Mat& src, cv::Mat& dst, int k, bool adaptive, bool gray = false)
{
    if (k < 1)
    {
        std::cerr << "At least 1 cluster has to be used." << std::endl;
        exit(-1);
    }

    cv::Mat labels;
    zpo::kmeans(src, k, labels, this->centroids, adaptive);

    dst = cv::Mat::zeros(src.size(), src.type());
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            dst.at<cv::Vec3f>(i, j) = (cv::Vec3f) centroids.at(labels.at<uchar>(i, j));
        }
    }

    if (gray)
    {
        zpo::gray(dst, dst);
    }
}
