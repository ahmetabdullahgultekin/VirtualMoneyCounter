// File: setup.h
#ifndef VIRTUALMONEYCOUNTER_SETUP_H
#define VIRTUALMONEYCOUNTER_SETUP_H

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <filesystem>
#include <atomic>

#include "coin_classifier.h"
#include "coin_detection.h"
#include "detection_visualizer.h"
#include "preprocess.h"
#include "coin_tracker.h"
#include "info_windows.h"

/**
 * @brief Include the vc.h header file
 */
extern "C" {
#include "vc.h"
}

/**
 * @brief Namespaces for convenience
 * using namespace std;
 * using namespace cv;
 */
using namespace std;
using namespace cv;

/**
 * @brief Global variables
 * @details
 * These variables are used to store the video capture object, frame delay, quit key,
 * and other parameters for coin detection.
 */
/**
* @brief Video capture handle
* @details
* This variable is used to capture video frames from the specified video file.
* It is defined globally to be accessible in the setup, preprocess, detectCoins, and draw functions.
* The video capture handle is initialized in the setup function and released in the main function.
*/
extern VideoCapture cap;
extern const int frameDelay; // milliseconds
extern int quitKeyVar;
extern const char quitKey;

/**
 * @brief Video file path
 * @details
 * This variable stores the path to the video file to be processed.
 * The video file should be in the same directory as the source code.
 * The video file should be in a supported format (e.g., .mp4, .avi).
 */
const string VIDEO_FILE_DIR = "videos/";
const string VIDEO_FILE_NAME = "video1.mp4";
const string VIDEO_FILE_PATH = VIDEO_FILE_DIR + VIDEO_FILE_NAME;

void vc_timer();

/**
 * @brief Setup function
 * @details
 * This function initializes the video capture and creates a window for displaying the video.
 * It sets the video file path and opens the video file for processing.
 * The function returns true if the setup was successful, otherwise false.
 *
 * @param videoFile - The path to the video file to be processed.
 * @return bool - Returns true if the setup was successful, otherwise false.
 */
bool setup();

/**
 * @brief Main function
 * @details
 * This function runs the main process of the program.
 * It captures video frames, preprocesses them, detects coins, and draws the results.
 * The function returns an integer indicating the result of the process.
 *
 * @return int - Returns 0 if the process was successful, otherwise an error code.
 */
int runProcess();

#endif //VIRTUALMONEYCOUNTER_SETUP_H