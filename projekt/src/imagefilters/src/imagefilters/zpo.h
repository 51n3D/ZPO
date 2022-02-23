#ifndef IMAGEFILTER_ZPO_H
#define IMAGEFILTER_ZPO_H

#include <opencv2/opencv.hpp>


namespace zpo
{
    enum ImageType { GRAY=1, COLORED=3 };

    struct Pixel
    {
        public:
            float x, y, z;
            int cluster;

            Pixel()
                    : x(.0), y(.0), z(0.0), cluster(-1) {}

            Pixel(int cluster)
                    : x(.0), y(.0), z(0.0), cluster(cluster) {}

            Pixel(float x, float y, float z)
                    : x(x), y(y), z(z), cluster(-1) {}

            Pixel(float x, float y, float z, int cluster)
                    : x(x), y(y), z(z), cluster(cluster) {}

            Pixel(cv::Vec3f pos)
                    : x(pos[0]), y(pos[1]), z(pos[2]), cluster(-1) {}

            Pixel(cv::Vec3f pos, int cluster)
                    : x(pos[0]), y(pos[1]), z(pos[2]), cluster(cluster) {}

            [[nodiscard]] inline float distance(const Pixel& p) const
            {
                return sqrt(pow(p.x - x, 2) + pow(p.y - y, 2) + pow(p.z - z, 2));
            }

            Pixel& operator+=(const Pixel& pxl)
            {
                this->x += pxl.x; this->y += pxl.y; this->z += pxl.z;
                return *this;
            }

            Pixel operator/ (float n) const
            {
                return Pixel(this->x / n, this->y / n, this->z / n, this->cluster);
            }

            bool operator== (const Pixel& pxl)
            {
                return this->x == pxl.x && this->y == pxl.y && this->z == pxl.z;
            }

            explicit operator cv::Vec3f() const { return cv::Vec3f(this->x, this->y, this->z); }
    };

    // final methods
    void gray(const cv::Mat& src, cv::Mat& dst);
    void treshold(const cv::Mat& src, cv::Mat& dst, float tresh);
    void filter2D(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel);
    void kmeans(const cv::Mat& src, int k, cv::Mat& labels, std::vector<Pixel>& centroids, bool adaptive);

    // helper methods
    void assignClusters(const cv::Mat& src, cv::Mat& labels, const std::vector<Pixel>& centroids);
    void computeCentroids(const cv::Mat& src, const cv::Mat& labels, std::vector<Pixel>& centroids);
}

#endif //IMAGEFILTER_ZPO_H
