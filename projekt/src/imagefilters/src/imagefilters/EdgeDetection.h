#ifndef IMAGEFILTER_EDGEDETECTION_H
#define IMAGEFILTER_EDGEDETECTION_H

#include "Filter.h"

#include <map>


namespace zpo
{
    class EdgeDetection : public Filter
    {
    private:
        // members
        enum Directions { HORIZ, VERT };

        float sobel_horiz[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        float sobel_vert[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
        float prewitt_horiz[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        float prewitt_vert[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

        std::map<int, std::map<int, cv::Mat>> kernel = {
            { SOBEL, {
                 { HORIZ, cv::Mat(3, 3, CV_32F, this->sobel_horiz) },
                 { VERT, cv::Mat(3, 3, CV_32F, this->sobel_vert) }
             }},
            { PREWITT, {
                 { HORIZ, cv::Mat(3, 3, CV_32F, this->prewitt_horiz) },
                 { VERT, cv::Mat(3, 3, CV_32F, this->prewitt_vert) }
             }},
        };

        // methods
        cv::Mat filterImage(const cv::Mat& src, int type) override;

    public:
        // members
        enum Type { SOBEL, PREWITT };

        // constructor
        explicit EdgeDetection() = default;

    };
};

#endif //IMAGEFILTER_EDGEDETECTION_H
