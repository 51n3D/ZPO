#ifndef IMAGEFILTER_FILTER_H
#define IMAGEFILTER_FILTER_H

#include <opencv2/core/mat.hpp>


namespace zpo
{
    class Filter
    {
    private:
        // methods
        virtual cv::Mat filterImage(const cv::Mat& src, int type) = 0;

    public:
        // methods
        void filter(const cv::Mat& src, cv::Mat& dst, int type, bool gray);
    };
};

#endif //IMAGEFILTER_FILTER_H
