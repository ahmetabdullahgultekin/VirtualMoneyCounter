#include "coin_classifier.h"

bool gScaleReady = false;
double gMmPerPixel = 0.0;

/**
 * @brief Classify a euro coin based on its measured diameter.
 *
 * This function determines the value of a euro coin by comparing its
 * measured diameter to the predefined diameters of euro coins. It finds
 * the closest match dynamically by calculating the absolute difference
 * between the measured diameter and each coin's diameter.
 *
 * @param dMM - The measured diameter of the coin in millimeters.
 * @return double - The value of the coin in euros; returns -1.0 if no match is found.
 */
double classifyEuroCoin(double dMM) {
    double closestValue = -1.0;
    double minDifference = numeric_limits<double>::max();

    for (const auto &coin: EURO_COINS) {
        double difference = fabs(dMM - coin.diameter);
        if (difference < minDifference) {
            minDifference = difference;
            closestValue = coin.value;
        }
    }

    return closestValue;
}

Scalar getAverageColor(const Mat &image, Point2f center, float radius) {
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    circle(mask, center, (int) radius, 255, -1);  // filled circle as a mask

    Scalar meanColor = mean(image, mask);
    return meanColor;  // returns Scalar(B, G, R)
}

bool isCoinColor(const Scalar &color) {
    double r = color[0];
    double g = color[1];
    double b = color[2];

    return (r >= 29 && r <= 92) &&
           (g >= 44 && g <= 116) &&
           (b >= 59 && b <= 125);
}
