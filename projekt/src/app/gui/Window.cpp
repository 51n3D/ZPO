#include "Window.h"
#include "ui_window.h"

#include "cvfilters/filters.h"

#include <iostream>

#include <QImage>
#include <QSize>
#include <QScreen>
#include <QApplication>
#include <QTabWidget>
#include <chrono>


Window::Window(QWidget *parent) : QMainWindow(parent), ui(new Ui::Window), timer(new QTimer(this))
{
    this->currentFilter = this->filter[EMBOSS];
    this->ui->setupUi(this);
    this->setFixedSize(this->ui->video->size());
    this->cap.open(0);

    if (!this->cap.isOpened())
    {
        std::cerr << "camera is not open" << std::endl;
        exit(-1);
    }

    this->connectWidgets();

    emit initWidgets();
    emit updateFilters();
    emit useFilter(EMBOSS);

    timer->start(20);
}

void Window::connectWidgets()
{
    QObject::connect(this, SIGNAL(initWidgets()), this, SLOT(setupWidgets()));
    QObject::connect(this, SIGNAL(updateFilters()), this, SLOT(updateLibrary()));
    QObject::connect(this, SIGNAL(updateFilters()), this, SLOT(updateFrameType()));
    QObject::connect(this, SIGNAL(updateFilters()), this, SLOT(updateEmboss()));
    QObject::connect(this, SIGNAL(updateFilters()), this, SLOT(updateEdgehighlighting()));
    QObject::connect(this, SIGNAL(updateFilters()), this, SLOT(updateImageQuantization()));
    QObject::connect(this, SIGNAL(useFilter(int)), this, SLOT(changeFilter(int)));
    QObject::connect(this->ui->zpo, SIGNAL(clicked(bool)), this, SLOT(updateLibrary()));
    QObject::connect(this->ui->cv, SIGNAL(clicked(bool)), this, SLOT(updateLibrary()));
    QObject::connect(this->ui->rgb, SIGNAL(clicked(bool)), this, SLOT(updateFrameType()));
    QObject::connect(this->ui->gray, SIGNAL(clicked(bool)), this, SLOT(updateFrameType()));
    QObject::connect(this->ui->filters, SIGNAL(tabBarClicked(int)), this, SLOT(changeFilter(int)));
    QObject::connect(this->ui->sobel, SIGNAL(clicked(bool)), this, SLOT(updateEdgehighlighting()));
    QObject::connect(this->ui->prewitt, SIGNAL(clicked(bool)), this, SLOT(updateEdgehighlighting()));
    QObject::connect(this->ui->type, SIGNAL(currentIndexChanged(int)), this, SLOT(updateEmboss()));
    QObject::connect(this->ui->clusters, SIGNAL(valueChanged(int)), this, SLOT(updateImageQuantization()));
    QObject::connect(this->ui->adaptive, SIGNAL(stateChanged(int)), this, SLOT(updateImageQuantization()));
    QObject::connect(this->timer, SIGNAL(timeout()), this, SLOT(update_window()));
}

void Window::useEmboss(const cv::Mat& src, cv::Mat& dst)
{
    if (this->using_zpo)
    {
#ifdef MEASURING
        // cas pred algoritmom
    auto start = std::chrono::steady_clock::now();
#endif
        this->emboss.filter(src, dst, this->settings[EMBOSS][TYPE], this->gray_frame);
#ifdef MEASURING
        // cas po algoritme
    auto end = std::chrono::steady_clock::now();
    // casovy rozdiel
    auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ulozenie do suboru
    std::ofstream file("zpo_emboss_measuring.txt", std::ios_base::app | std::ios_base::out);
    file << time_elapsed_ms << std::endl;
#endif
    }
    else
    {
#ifdef MEASURING
        // cas pred algoritmom
    auto start = std::chrono::steady_clock::now();
#endif
        cvfilters::emboss(src, dst, this->settings[EMBOSS][TYPE], this->gray_frame);
#ifdef MEASURING
        // cas po algoritme
    auto end = std::chrono::steady_clock::now();
    // casovy rozdiel
    auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ulozenie do suboru
    std::ofstream file("cv_emboss_measuring.txt", std::ios_base::app | std::ios_base::out);
    file << time_elapsed_ms << std::endl;
#endif
    }
}

void Window::useHighlitedEdges(const cv::Mat& src, cv::Mat& dst)
{
    if (this->using_zpo)
    {
#ifdef MEASURING
        // cas pred algoritmom
    auto start = std::chrono::steady_clock::now();
#endif
        this->edges.filter(src, dst, this->settings[HIGHLIGHTED_EDGES][TYPE], this->gray_frame);
#ifdef MEASURING
        // cas po algoritme
    auto end = std::chrono::steady_clock::now();
    // casovy rozdiel
    auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ulozenie do suboru
    if (this->settings[HIGHLIGHTED_EDGES][TYPE] == zpo::EdgeDetection::SOBEL)
    {
        std::ofstream file("zpo_edges_sobel_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
    else
    {
        std::ofstream file("zpo_edges_prewitt_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
#endif
    }
    else
    {
#ifdef MEASURING
        // cas pred algoritmom
    auto start = std::chrono::steady_clock::now();
#endif
        cvfilters::edges(src, dst, this->settings[HIGHLIGHTED_EDGES][TYPE], this->gray_frame);
#ifdef MEASURING
        // cas po algoritme
    auto end = std::chrono::steady_clock::now();
    // casovy rozdiel
    auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ulozenie do suboru
    if (this->settings[HIGHLIGHTED_EDGES][TYPE] == zpo::EdgeDetection::SOBEL)
    {
        std::ofstream file("cv_edges_sobel_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
    else
    {
        std::ofstream file("cv_edges_prewitt_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
#endif
    }
}

void Window::useImageQuantization(const cv::Mat& src, cv::Mat& dst)
{
    if (this->using_zpo)
    {
#ifdef MEASURING
        // cas pred algoritmom
    auto start = std::chrono::steady_clock::now();
#endif
        this->quantization.filter(
                src, dst,
                this->settings[IMAGE_QUANTIZATION][CLUSTERS],
                this->settings[IMAGE_QUANTIZATION][ADAPTIVE],
                this->gray_frame
        );
#ifdef MEASURING
        // cas po algoritme
    auto end = std::chrono::steady_clock::now();
    // casovy rozdiel
    auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ulozenie do suboru
    if (this->settings[IMAGE_QUANTIZATION][ADAPTIVE])
    {
        std::ofstream file("zpo_kmeans_adaptive_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
    else
    {
        std::ofstream file("zpo_kmeans_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
#endif
    }
    else
    {
#ifdef MEASURING
        // cas pred algoritmom
    auto start = std::chrono::steady_clock::now();
#endif
        cv::Mat labels;
        if (this->settings[IMAGE_QUANTIZATION][ADAPTIVE])
        {
            labels = this->best_labels;
        }
        cvfilters::quantization(
                src, dst,
                this->settings[IMAGE_QUANTIZATION][CLUSTERS],
                labels,
                this->gray_frame
        );
        this->best_labels = labels;
#ifdef MEASURING
        // cas po algoritme
    auto end = std::chrono::steady_clock::now();
    // casovy rozdiel
    auto time_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // ulozenie do suboru
    if (this->settings[IMAGE_QUANTIZATION][ADAPTIVE])
    {
        std::ofstream file("cv_kmeans_adaptive_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
    else
    {
        std::ofstream file("cv_kmeans_measuring.txt", std::ios_base::app | std::ios_base::out);
        file << time_elapsed_ms << std::endl;
    }
#endif
    }
}

Window::~Window()
{
    this->cap.release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// SLOTS /////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Window::changeFilter(int index)
{
    this->currentFilter = this->filter[index];
}

void Window::update_window()
{
    this->cap >> this->frame;

    cv::Mat dst;
    this->frame.convertTo(this->frame, CV_32FC3);
    (this->*currentFilter)(this->frame, dst);
    dst.convertTo(dst, CV_8UC3);

    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    auto qt_image = QImage((const unsigned char*) (dst.data), dst.cols, dst.rows, QImage::Format_RGB888);
    auto a = QPixmap::fromImage(qt_image);
    this->ui->video->setPixmap(a.scaled(this->ui->video->size()));
}

void Window::setupWidgets()
{
    this->ui->zpo->setChecked(true);
    this->ui->rgb->setChecked(true);
    this->ui->sobel->setChecked(true);
    this->ui->clusters->setMinimum(2);
    this->ui->clusters->setMaximum(64);
    this->ui->clusters->setValue(4);
}

void Window::updateLibrary()
{
    if (this->ui->zpo->isChecked())
    {
        this->using_zpo = true;
    }
    else if (this->ui->cv->isChecked())
    {
        this->using_zpo = false;
    }
}

void Window::updateFrameType()
{
    if (this->ui->rgb->isChecked())
    {
        this->gray_frame = false;
    }
    else if (this->ui->gray->isChecked())
    {
        this->gray_frame = true;
    }
}

void Window::updateEmboss()
{
    this->settings[EMBOSS][TYPE] = this->ui->type->currentIndex();
}

void Window::updateEdgehighlighting()
{
    if (this->ui->sobel->isChecked())
    {
        this->settings[HIGHLIGHTED_EDGES][TYPE] = zpo::EdgeDetection::SOBEL;
    }
    else if (this->ui->prewitt->isChecked())
    {
        this->settings[HIGHLIGHTED_EDGES][TYPE] = zpo::EdgeDetection::PREWITT;
    }
}

void Window::updateImageQuantization()
{
    this->quantization.resetCentroids();
    this->settings[IMAGE_QUANTIZATION][CLUSTERS] = this->ui->clusters->value();
    this->settings[IMAGE_QUANTIZATION][ADAPTIVE] = this->ui->adaptive->isChecked();
}
