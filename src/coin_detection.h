// File: coin_detection.h
#ifndef VIRTUALMONEYCOUNTER_COIN_DETECTION_H
#define VIRTUALMONEYCOUNTER_COIN_DETECTION_H

#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/**
 * @brief Hough Transform parameters
 * @details
 * These variables are used to configure the Hough Transform parameters for circle detection.
 * They are defined globally to be accessible in the setup, preprocess, detectCoins, and draw functions.
 * The DP variable is the inverse ratio of the accumulator resolution to the image resolution.
 * The MIN_DIST variable is the minimum distance between detected centers.
 * The PARAM1 and PARAM2 variables are the first and second method parameters for HoughCircles.
 * The MIN_RADIUS and MAX_RADIUS variables define the minimum and maximum radius of circles to be detected.
 * These parameters can be adjusted to improve the detection results based on the specific video being processed.
 */

// Global variables for Hough Transform parameters
extern int dp; // Scaled by 10 (e.g., 1.2 -> 12)
extern int param1;
extern int param2;
extern int minRadius;
extern int maxRadius;
extern int minDist; // Minimum distance between detected centers

/** @brief Detect coins function
 * @details
 * This function detects coins in the preprocessed image using the Hough Transform method.
 * It uses the cv::HoughCircles function to detect circles in the image.
 * The detected circles are stored in the circles vector.
 *
 * @param preproc - The preprocessed input image.
 * @return std::vector<cv::Vec3f> - A vector of detected circles (coins).
 */
vector<Vec3f> detectCoins(const Mat &preproc);

/**
 * @brief Global variables
 * @var cap cv::VideoCapture - video capture handle
 * @var quitKeyVar int - quitKeyVar pressed (global so helpers can exit)
 * @var DP double - inverse ratio of the accumulator resolution to the image resolution
 * @var MIN_DIST int - minimum distance between detected centers
 * @var PARAM1 int - first method parameter for HoughCircles
 * @var PARAM2 int - second method parameter for HoughCircles
 * @var MIN_RADIUS int - minimum radius of circles to be detected
 * @var MAX_RADIUS int - maximum radius of circles to be detected
 * @var VIDEO_FILE_PATH std::string - path to the video file
 *
 * @details
 * These variables are used to configure the Hough Transform parameters for circle detection.
 * They are defined globally to be accessible in the setup, preprocess, detectCoins, and draw functions.
 * The DP variable is the inverse ratio of the accumulator resolution to the image resolution.
 * The MIN_DIST variable is the minimum distance between detected centers.
 * The PARAM1 and PARAM2 variables are the first and second method parameters for HoughCircles.
 * The MIN_RADIUS and MAX_RADIUS variables define the minimum and maximum radius of circles to be detected.
 * These parameters can be adjusted to improve the detection results based on the specific video being processed.
 * The default values are set to reasonable values for detecting coins in a video.
 * These values can be modified based on the specific requirements of the application.
 * The VIDEO_FILE_PATH variable is the path to the video file to be processed.
 * It should be in the same directory as the source code.
 * The video file should be in a supported format (e.g., .mp4, .avi).
 */


#endif // VIRTUALMONEYCOUNTER_COIN_DETECTION_H