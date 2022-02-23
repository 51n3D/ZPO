#ifndef IMAGEFILTER_WINDOW_H
#define IMAGEFILTER_WINDOW_H

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QMainWindow>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <imagefilters/EdgeDetection.h>
#include <imagefilters/ImageQuantization.h>
#include <imagefilters/Emboss.h>

#define MEASURING

namespace Ui { class Window; }

class Window : public QMainWindow
{
    Q_OBJECT

    public:
        explicit Window(QWidget *parent = nullptr);
        ~Window();

        void connectWidgets();

    signals:
        void initWidgets();
        void updateFilters();
        void useFilter(int index);

    private slots:
        void setupWidgets();
        void update_window();
        void changeFilter(int index);
        void updateLibrary();
        void updateFrameType();
        void updateEmboss();
        void updateEdgehighlighting();
        void updateImageQuantization();

    private:
        Ui::Window *ui;

        cv::VideoCapture cap;
        cv::Mat frame;
        bool gray_frame;
        bool using_zpo;
        QTimer *timer;

        zpo::EdgeDetection edges;
        zpo::Emboss emboss;
        zpo::ImageQuantization quantization;

        enum Filters { EMBOSS, HIGHLIGHTED_EDGES, IMAGE_QUANTIZATION };
        enum Settings { TYPE, CLUSTERS, ADAPTIVE, GRAY };

        void (Window::*currentFilter)(const cv::Mat& src, cv::Mat& dst);

        std::unordered_map<int, void (Window::*)(const cv::Mat& src, cv::Mat& dst)> filter = {
            { EMBOSS, &Window::useEmboss },
            { HIGHLIGHTED_EDGES, &Window::useHighlitedEdges },
            { IMAGE_QUANTIZATION, &Window::useImageQuantization }
        };

        std::unordered_map<int, std::unordered_map<int, int>> settings = {
            { EMBOSS, {
                { TYPE, zpo::Emboss::COMBINED }
            }},
            { HIGHLIGHTED_EDGES, {
                { TYPE, zpo::EdgeDetection::SOBEL }
            }},
            { IMAGE_QUANTIZATION, {
                { CLUSTERS, 16 },
                { ADAPTIVE, 0 }
            }}
        };

        cv::Mat best_labels;

        void useEmboss(const cv::Mat& src, cv::Mat& dst);
        void useHighlitedEdges(const cv::Mat& src, cv::Mat& dst);
        void useImageQuantization(const cv::Mat& src, cv::Mat& dst);
};

#endif //IMAGEFILTER_WINDOW_H
