#ifndef IMAGEFILTER_IMAGEQUANTIZATION_H
#define IMAGEFILTER_IMAGEQUANTIZATION_H

#include <opencv2/core/mat.hpp>
#include "zpo.h"


namespace zpo
{
    class ImageQuantization
    {
        private:
            // members
            std::vector<zpo::Pixel> centroids;


        public:
            // constructor
            explicit ImageQuantization() = default;

            // methods
            void filter(const cv::Mat& src, cv::Mat& dst, int k, bool adaptive, bool gray);
            inline void resetCentroids() { this->centroids = std::vector<zpo::Pixel>(); }
    };
}

#endif //IMAGEFILTER_IMAGEQUANTIZATION_H
