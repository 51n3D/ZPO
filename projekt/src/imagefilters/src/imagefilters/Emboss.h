#ifndef IMAGEFILTER_EMBOSS_H
#define IMAGEFILTER_EMBOSS_H

#include "Filter.h"

#include <map>


namespace zpo
{
    class Emboss : public Filter
    {
    private:
        // members
        float top_left[9] = { 1, 1, 0, 1, 0, -1, 0, -1, -1 };
        float top_right[9] = { 0, 1, 1, -1, 0, 1, -1, -1, 0 };
        float bottom_left[9] = { 0, -1, -1, 1, 0, -1, 1, 1, 0 };
        float bottom_right[9] = { -1, -1, 0, -1, 0, 1, 0, 1, 1 };

        std::map<int, cv::Mat> kernel = {
            { TOP_LEFT, cv::Mat(3, 3, CV_32F, this->top_left) },
            { TOP_RIGHT, cv::Mat(3, 3, CV_32F, this->top_right) },
            { BOTTOM_LEFT, cv::Mat(3, 3, CV_32F, this->bottom_left) },
            { BOTTOM_RIGHT, cv::Mat(3, 3, CV_32F, this->bottom_right) }
        };

        // methods
        cv::Mat filterImage(const cv::Mat& src, int type) override ;

    public:
        // members
        enum Kernel { COMBINED, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT };

        // constructor
        explicit Emboss() = default;

    };
}

#endif //IMAGEFILTER_EMBOSS_H
