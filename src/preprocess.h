#ifndef VIRTUALMONEYCOUNTER_PREPROCESS_H
#define VIRTUALMONEYCOUNTER_PREPROCESS_H

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

// Global variables for Gaussian blur
extern int blurSize; // Size of the Gaussian kernel (must be odd)
extern int blurSigma; // Standard deviation for Gaussian blur

// Global variables for Thresholding
extern int thresholdValue; // Threshold value for binary thresholding
extern int thresholdMaxValue; // Maximum value for binary thresholding
extern int thresholdBlockSize; // Block size for adaptive thresholding (must be odd)
extern int thresholdC; // This constant subtracted from the mean in adaptive thresholding

/**
 * @brief Preprocess function
 * @details
 * This function preprocesses the input image by converting it to grayscale and applying Gaussian blur.
 * The grayscale conversion is done using the cv::cvtColor function with the COLOR_BGR2GRAY flag.
 * The Gaussian blur is applied using the cv::GaussianBlur function with a kernel size of 9x9 and a standard deviation of 2.
 * The preprocessed image is stored in the dst parameter.
 *
 * @param src - The input image to be preprocessed.
 * @param dst - The preprocessed output image.
 *
 * @return int - Returns 0 if the preprocessing was successful.
 */
bool preprocess(const Mat &src, Mat &dst);

#endif //VIRTUALMONEYCOUNTER_PREPROCESS_H
