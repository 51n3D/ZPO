#include "Emboss.h"
#include "zpo.h"

using namespace zpo;


cv::Mat Emboss::filterImage(const cv::Mat& src, int type)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    if (type == COMBINED)
    {
        cv::Mat bottom_l, bottom_r;
        zpo::filter2D(src, bottom_l, this->kernel[BOTTOM_LEFT]);
        zpo::filter2D(src, bottom_r, this->kernel[BOTTOM_RIGHT]);
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                auto max = bottom_r.at<float>(i, j);
                if (bottom_l.at<float>(i, j) > max) max = bottom_l.at<float>(i, j);

                dst.at<float>(i, j) = max;
            }
        }
    }
    else
    {
        zpo::filter2D(src, dst, this->kernel[type]);
    }

    return dst;
}
