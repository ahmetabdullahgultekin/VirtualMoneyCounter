// File: coin_classifier.h
#ifndef VIRTUALMONEYCOUNTER_COIN_CLASSIFIER_H
#define VIRTUALMONEYCOUNTER_COIN_CLASSIFIER_H

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace chrono;

/**
 * @brief Coin specifications
 * @details
 * This struct defines the specifications of euro coins.
 * It includes the diameter and value of each coin.
 * The EURO_COINS array contains the specifications of all euro coins.
 * The N_EURO_COINS variable stores the number of euro coins in the array.
 */
struct CoinSpec {
    double diameter;
    double value;
};
const CoinSpec EURO_COINS[] = {
        {16.25, 0.01},
        {18.75, 0.02},
        {21.25, 0.05},
        {19.75, 0.10},
        {22.25, 0.20},
        {24.25, 0.50},
        {23.25, 1.00},
        {25.75, 2.00}
};
const int N_EURO_COINS = sizeof(EURO_COINS) / sizeof(EURO_COINS[0]);
extern bool gScaleReady;
extern double gMmPerPixel;
static int seenLargestR = 0;

// Global variables to track cumulative totals
// Array to count coins of each denomination
extern int coinsCount[N_EURO_COINS];
extern double totalValueSum;
extern int totalCoins;

// Detected coin structure
struct DetectedCircle {
    Vec3f circle; // Circle parameters (x, y, radius)
};

// Classifies an euro coin based on its measured diameter in millimeters.
// Returns the coin value in euros; returns -1.0 if no match is found.
double classifyEuroCoin(double dMM);

// Calculates the average monetary value per coin by processing the detected circles.
// Returns the average value computed.
double calculateTotal(const std::vector<cv::Vec3f> &circles);

// Checks if the coin looks fake based on its chroma in BGR format.
// Returns true if the coin is considered fake.
bool looksFakeByChroma(const cv::Mat &rgb, const cv::Vec3f &c);

// Checks if the coin looks fake based on its Lab colour representation.
// Returns true if the coin is considered fake.
bool looksFakeByLab(const cv::Mat &rgb, const cv::Vec3f &c);

#endif // VIRTUALMONEYCOUNTER_COIN_CLASSIFIER_H