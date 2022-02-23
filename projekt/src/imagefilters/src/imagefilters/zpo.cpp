#include "zpo.h"


void zpo::treshold(const cv::Mat& src, cv::Mat& dst, float tresh)
{
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            dst.at<float>(i, j) = src.at<float>(i, j) > tresh ? 255 : 0;
        }
    }
}

void zpo::gray(const cv::Mat& src, cv::Mat& dst)
{
    cv::cvtColor(src, dst, CV_BGR2GRAY);
    cv::cvtColor(dst, dst, CV_GRAY2BGR);
}

void zpo::filter2D(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel)
{
    dst = cv::Mat::zeros(src.size(), src.type());

    for (int i = kernel.rows / 2; i < src.rows - kernel.rows / 2; i++)
    {
        for (int j = kernel.cols / 2; j < src.cols - kernel.cols / 2; j++)
        {
            // apply flipped kernel
            for (int k = kernel.rows - 1; k >= 0; k--)
            {
                for (int l = kernel.cols - 1; l >= 0; l--)
                {
                    int x = i + 1 - k;
                    int y = j + 1 - l;

                    // border replicant
                    if (x < 0) x = 0;
                    else if (x >= src.rows) x = src.rows - 1;
                    if (y < 0) y = 0;
                    else if (y >= src.cols) y = src.cols - 1;

                    // ignore out of border
                    if (x >= 0 && x < src.rows && y >= 0 && y < src.cols)
                    {
                        dst.at<float>(i, j) += src.at<float>(x, y) * kernel.at<float>(k, l);
                    }
                }
            }
        }
    }
}

void zpo::kmeans(const cv::Mat& src, int k, cv::Mat& labels, std::vector<Pixel>& centroids, bool adaptive)
{
    labels = cv::Mat::zeros(src.size(), CV_8UC1);
    auto centroids_empty = centroids.empty();
    if (centroids_empty)
    {
        for (int i = 0; i < k; i++)
        {
            auto init_v = (i + 1) * 255 / (k + 1);
            centroids.emplace_back(Pixel(init_v, init_v, init_v, i));
        }
    }

    zpo::assignClusters(src, labels, centroids);

    if (adaptive || centroids_empty)
    {
        std::vector<Pixel> prev_centroids(k);
        while (!std::equal(centroids.begin(), centroids.end(), prev_centroids.begin()))
        {
            prev_centroids = centroids;
            zpo::computeCentroids(src, labels, centroids);
            zpo::assignClusters(src, labels, centroids);
        }
    }
}

void zpo::assignClusters(const cv::Mat& src, cv::Mat& labels, const std::vector<Pixel>& centroids)
{
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            auto min_dist = centroids[0].distance(src.at<cv::Vec3f>(i, j));
            for (auto it = centroids.begin() + 1; it != centroids.end(); it++)
            {
                auto dist = it->distance(src.at<cv::Vec3f>(i, j));
                if (dist < min_dist)
                {
                    min_dist = dist;
                    labels.at<uchar>(i, j) = (uchar) it->cluster;
                }
            }
        }
    }
}

void zpo::computeCentroids(const cv::Mat& src, const cv::Mat& labels, std::vector<Pixel>& centroids)
{
    for (auto it = centroids.begin(); it != centroids.end(); it++)
    {
        Pixel mean(it->cluster);
        auto n = 0;
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                if (labels.at<uchar>(i, j) == (uint) it->cluster)
                {
                    mean += Pixel(src.at<cv::Vec3f>(i, j));
                    n++;
                }
            }
        }

        centroids.at(it->cluster) = mean / n;
    }
}
