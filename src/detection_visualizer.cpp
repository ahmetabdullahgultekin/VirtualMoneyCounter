#include "detection_visualizer.h"

/**
 * @brief Draw function
 * @details
 * This function draws the detected circles (coins) on the original image.
 * It uses the cv::circle function to draw the circles and their centers.
 * The number of detected circles and the current frame number are displayed on the image.
 *
 * @param frame - The original image to draw on.
 * @param circles - A vector of detected circles (coins).
 * @param currentframe - The current frame number.
 *
 * @return bool - Returns true if the drawing was successful, otherwise false.
 */
bool draw(Mat &frame, const vector<Vec3f> &circles, int currentframe) {
    try {
        for (const auto &c: circles) {
            Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);
            circle(frame, center, radius, Scalar(0, 255, 0), 2);
            circle(frame, center, 2, Scalar(0, 0, 255), 3);

            // Classify the coin and add text
            double diameterMM = 2.0 * radius * gMmPerPixel;
            double coinValue = classifyEuroCoin(diameterMM);

            if (coinValue > 0.0) {
                totalValueSum += coinValue;
                ostringstream oss;
                oss.setf(ios::fixed);
                oss.precision(2);
                oss << "€" << coinValue;

                putText(frame, oss.str(), Point(center.x - 20, center.y - radius - 10),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
                putText(frame, oss.str(), Point(center.x - 20, center.y - radius - 10),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
            }
        }

        double total = calculateTotal(circles); /*totalValueSum / coinsCount;*/
        ostringstream oss;
        oss.setf(ios::fixed);
        oss.precision(2);
        oss << "COINS: " << circles.size()
            << "  |  TOTAL AVG: €" << total
            << "  |  FRAME: " << currentframe;

        string txt = oss.str();
        putText(frame, txt, Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(0, 0, 0), 2);
        putText(frame, txt, Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(255, 255, 255), 1);

        imshow("Coin Detection", frame);
    }
    catch (const exception &e) {
        return false;
    }
    return true;
}
