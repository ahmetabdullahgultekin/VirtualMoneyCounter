// File: coin_classifier.h
#ifndef VIRTUALMONEYCOUNTER_COIN_CLASSIFIER_H
#define VIRTUALMONEYCOUNTER_COIN_CLASSIFIER_H

#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "coin_tracker.h"
#include "coin_detection.h"

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

// Classifies the euro coin based on its measured diameter in millimeters.
// Returns the coin value in euros; returns -1.0 if no match is found.
double classifyEuroCoin(double dMM);

// Computes the average color of a circular region in the image.
// Takes the image, center point, and radius as input.
// Returns the average color in BGR format.
Scalar getAverageColor(const Mat &image, Point2f center, float radius);

// Checks if the detected color is within the expected range for euro coins.
bool isCoinColor(const Scalar &color);

#endif // VIRTUALMONEYCOUNTER_COIN_CLASSIFIER_H