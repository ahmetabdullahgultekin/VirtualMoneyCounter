#include "coin_detection.h"

// Global variables for Hough Transform parameters
int dp = 12; // Scaled by 10 (e.g., 1.2 -> 12)
int param1 = 60;
int param2 = 60;
int minRadius = 55;
int maxRadius = 110;
int minDist = minRadius * 2; // Minimum distance between detected centers
/**
 * @brief Detect coins function
 * @details
 * This function detects coins in the preprocessed image using the Hough Transform method.
 * It uses the cv::HoughCircles function to detect circles in the image.
 * The detected circles are stored in the circles vector.
 *
 * @param preproc - The preprocessed input image.
 * @return std::vector<cv::Vec3f> - A vector of detected circles (coins).
 */
/*vector<Vec3f> detectCoins(const Mat &preproc) {
    try {
        vector<Vec3f> circles;
        double dpValue = dp / 10.0; // Convert scaled DP back to a double
        HoughCircles(preproc, circles, HOUGH_GRADIENT,
                     dpValue, minDist, param1, param2,
                     minRadius, maxRadius);
        return circles;
    } catch (const exception &e) {
        return {};
    }
}*/
vector<Vec3f> detectCoins(const Mat &preproc) {
    vector<Vec3f> circles;
    // 1) Find all external contours
    vector<vector<Point>> contours;
    findContours(preproc, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto &c: contours) {
        double area = contourArea(c);
        if (area < 100.0)
            continue;               // skip tiny noise blobs

        double peri = arcLength(c, true);
        double circ = 4.0 * CV_PI * area / (peri * peri);

        if (circ < 0.75)
            continue;               // require at least ~75% circular

        // 2) Fit minimum enclosing circle
        Point2f center;
        float radius;
        minEnclosingCircle(c, center, radius);

        // 3) Optionally enforce a radius window too
        if (radius < (float) minRadius || radius > (float) maxRadius)
            continue;

        circles.emplace_back(center.x, center.y, radius);
    }
    return circles;
}