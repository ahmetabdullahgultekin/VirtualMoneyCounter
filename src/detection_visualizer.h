// File: detection_visualizer.h
#ifndef VIRTUALMONEYCOUNTER_DETECTION_VISUALIZER_H
#define VIRTUALMONEYCOUNTER_DETECTION_VISUALIZER_H

#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "coin_classifier.h"
#include "coin_detection.h"
#include "coin_tracker.h"

using namespace std;
using namespace cv;

// Draws the detected coins onto the frame, annotates each coin with its value,
// and overlay frame statistics (coin count, average total, frame number).
// Returns true if the drawing is successful.
bool draw(cv::Mat &frame, const std::vector<cv::Vec3f> &circles, int currentframe);

// Calculates the total monetary value of the detected coins.
// The first time it runs, it sets gMmPerPixel using the largest circle
// diameter.
// It returns the average value of the coins.
double calculateTotal(const vector<cv::Vec3f> &circles);

#endif // VIRTUALMONEYCOUNTER_DETECTION_VISUALIZER_H