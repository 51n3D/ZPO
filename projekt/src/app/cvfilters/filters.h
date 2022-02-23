#ifndef IMAGEFILTERS_FILTERS_H
#define IMAGEFILTERS_FILTERS_H

#include <opencv2/core/mat.hpp>

namespace cvfilters
{
    /// emboss filter parameters ///////////////////////////////////////////////////////////////////////////////////////
    float top_left[9] = { 1, 1, 0, 1, 0, -1, 0, -1, -1 };
    float top_right[9] = { 0, 1, 1, -1, 0, 1, -1, -1, 0 };
    float bottom_left[9] = { 0, -1, -1, 1, 0, -1, 1, 1, 0 };
    float bottom_right[9] = { -1, -1, 0, -1, 0, 1, 0, 1, 1 };

    enum Kernel { COMBINED, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT };

    std::map<int, cv::Mat> emboss_kernel = {
            { TOP_LEFT, cv::Mat(3, 3, CV_32F, top_left) },
            { TOP_RIGHT, cv::Mat(3, 3, CV_32F, top_right) },
            { BOTTOM_LEFT, cv::Mat(3, 3, CV_32F, bottom_left) },
            { BOTTOM_RIGHT, cv::Mat(3, 3, CV_32F, bottom_right) }
    };
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// sobel/prewitt filter parameters ////////////////////////////////////////////////////////////////////////////////
    enum Directions { HORIZ, VERT };
    float prewitt_horiz[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    float prewitt_vert[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    std::map<int, cv::Mat> prewitt_kernel = {
         { HORIZ, cv::Mat(3, 3, CV_32F, prewitt_horiz) },
         { VERT, cv::Mat(3, 3, CV_32F, prewitt_vert) }
    };

    enum Type { SOBEL, PREWITT };
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void emboss(const cv::Mat& src, cv::Mat& dst, int type, bool gray)
    {
        std::vector<cv::Mat> channels(src.channels());
        cv::split(src, channels);
        for (int n = 0; n < src.channels(); n++)
        {
            if (type == COMBINED)
            {
                cv::Mat bottom_l, bottom_r;
                cv::filter2D(channels[n], bottom_l, CV_32F, emboss_kernel[TOP_LEFT]);
                cv::filter2D(channels[n], bottom_r, CV_32F, emboss_kernel[TOP_RIGHT]);
                channels[n] = cv::max(bottom_l, bottom_r);
            }
            else
            {
                cv::filter2D(channels[n], channels[n], CV_32F, emboss_kernel[type]);
            }
        }
        cv::merge(channels, dst);

        cv::normalize(dst, dst, 0, 256, cv::NORM_MINMAX);

        if (gray)
        {
            cv::cvtColor(dst, dst, CV_BGR2GRAY);
            cv::cvtColor(dst, dst, CV_GRAY2BGR);
        }
    }

    void edges(const cv::Mat& src, cv::Mat& dst, int type, bool gray)
    {
        std::vector<cv::Mat> channels(src.channels());
        cv::split(src, channels);
        for (int n = 0; n < src.channels(); n++)
        {
            if (type == SOBEL)
            {
                cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
                cv::Sobel(channels[n], grad_x, CV_32F, 1, 0);
                cv::Sobel(channels[n], grad_y, CV_32F, 0, 1);

                convertScaleAbs(grad_x, abs_grad_x);
                convertScaleAbs(grad_y, abs_grad_y);

                addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, channels[n]);
            }
            else
            {
                cv::Mat vert, horiz;
                cv::filter2D(channels[n], vert, CV_32F, prewitt_kernel[HORIZ]);
                cv::filter2D(channels[n], horiz, CV_32F, prewitt_kernel[VERT]);

                cv::pow(vert, 2, vert);
                cv::pow(horiz, 2, horiz);
                cv::sqrt(vert + horiz, channels[n]);
            }
        }
        cv::merge(channels, dst);

        cv::normalize(dst, dst, 0, 256, cv::NORM_MINMAX);

        if (gray)
        {
            cv::cvtColor(dst, dst, CV_BGR2GRAY);
            cv::cvtColor(dst, dst, CV_GRAY2BGR);
        }
    }

    void quantization(const cv::Mat& src, cv::Mat& dst, int k, cv::Mat& labels, bool gray)
    {
        cv::Mat samples(src.rows * src.cols, src.channels(), CV_32F);
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                for (int n = 0; n < src.channels(); n++) {
                    samples.at<float>(i + j * src.rows, n) = src.at<cv::Vec3f>(i, j)[n];
                }
            }
        }

        cv::Mat centroids;
        cv::kmeans(samples, k, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, centroids);

        dst = cv::Mat(src.size(), src.type());
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                int cluster_idx = labels.at<int>(i + j * src.rows, 0);
                for (int n = 0; n < src.channels(); n++)
                {
                    dst.at<cv::Vec3f>(i, j)[n] = centroids.at<float>(cluster_idx, n);
                }
            }
        }

        if (gray)
        {
            cv::cvtColor(dst, dst, CV_BGR2GRAY);
            cv::cvtColor(dst, dst, CV_GRAY2BGR);
        }
    }
}

#endif //IMAGEFILTERS_FILTERS_H
