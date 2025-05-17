#include "preprocess.h"

// Global variables for Gaussian blur
int blurSize = 55; // Size of the Gaussian kernel (must be odd)
int blurSigma = 2; // Standard deviation for Gaussian blur

// Global variables for Thresholding
int thresholdValue = 111; // Threshold value for binary thresholding
int thresholdMaxValue = 255; // Maximum value for binary thresholding
int thresholdBlockSize = 11; // Block size for adaptive thresholding (must be odd)
int thresholdC = 2; // This constant subtracted from the mean in adaptive thresholding

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
/*bool preprocess(const Mat &src, Mat &dst) {
    try {
        // Convert to grayscale
        cvtColor(src, dst, COLOR_BGR2GRAY);
        // Apply Gaussian blur
        GaussianBlur(dst, dst, Size(55, 55), 3,3);
        // Apply Binary thresholding
        threshold(dst, dst, 120, 255, THRESH_BINARY);
        // Apply Canny edge detection
        *//*Canny(dst, dst, 25, 75);*//*
    }
    catch (const exception &e) {
        return false;
    }
    return true;
}*/
bool preprocess(const Mat &src, Mat &dst) {
    try {
        cvtColor(src, dst, COLOR_BGR2GRAY);
        GaussianBlur(dst, dst, Size(blurSize, blurSize), blurSigma, blurSigma);
        // use Otsu to auto‐tune the threshold:
        threshold(dst, dst, thresholdValue, thresholdMaxValue, THRESH_BINARY_INV | THRESH_OTSU);
        // optional morphology to fill holes / remove speckle:
        /*Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(dst, dst, MORPH_CLOSE, kernel);
        morphologyEx(dst, dst, MORPH_OPEN, kernel);*/
    } catch (const exception &e) {
        return false;
    }
    return true;
}