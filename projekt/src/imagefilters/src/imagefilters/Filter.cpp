#include "Filter.h"
#include "zpo.h"

using namespace zpo;


void Filter::filter(const cv::Mat& src, cv::Mat& dst, int type, bool gray = false)
{
    if (src.channels() == zpo::COLORED)
    {
        std::vector<cv::Mat> channels(3);
        cv::split(src, channels);
        for (int i = 0; i < src.channels(); i++)
        {
            channels.at(i) = this->filterImage(channels[i], type);
        }
        cv::merge(channels, dst);
    }
    else if (src.channels() == GRAY)
    {
        dst = this->filterImage(src, type);
    }
    else
    {
        std::cerr << "Can not filter image.";
        exit(-1);
    }

    cv::normalize(dst, dst, 0, 256, cv::NORM_MINMAX);

    if (gray)
    {
        zpo::gray(dst, dst);
    }
}
