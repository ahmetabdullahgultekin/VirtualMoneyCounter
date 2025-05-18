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
        // Draw the detected objectInfo
        for (const auto &[id, obj]: objectInfo) {
            // Do not draw if the object is not a coin
            //if (!isCoinColor(c)) continue;
            // Draw the circle outline
            /*Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);*/
            // Draw the green circle outline if the object is a coin,
            // Draw the red circle outline if the object is not a coin
            // Do not draw if the object is disappeared
            if (obj.isDisappeared) continue;
            if (!obj.isCoin) {
                circle(frame, obj.center, obj.radius, Scalar(0, 0, 255), 2);
                circle(frame, obj.center, 2, Scalar(255, 0, 0), 3);
            } else {
                // Draw the green circle outline if the object is a coin
                circle(frame, obj.center, obj.radius, Scalar(0, 255, 0), 2);
                circle(frame, obj.center, 2, Scalar(0, 0, 255), 3);
            }
            /*circle(frame, obj.center, obj.radius, Scalar(0, 255, 0), 2);
            circle(frame, obj.center, 2, Scalar(0, 0, 255), 3);*/

            // Classify the coin and add text
            double diameterMM = 2.0 * obj.radius * gMmPerPixel;
            double coinValue = obj.value; //classifyEuroCoin(diameterMM);

            if (coinValue > 0.0) {
                //totalValueSum = calculateTotal(circles);
                ostringstream oss;
                oss.setf(ios::fixed);
                oss.precision(2);
                oss << "€" << coinValue;

                putText(frame, oss.str(), Point((int) obj.center.x - 20, (int) (obj.center.y - obj.radius - 10)),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
                putText(frame, oss.str(), Point((int) obj.center.x - 20, (int) (obj.center.y - obj.radius - 10)),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
            }
        }

        double total = calculateTotal(circles);
        ostringstream oss;
        oss.setf(ios::fixed);
        oss.precision(2);
        oss << "COINS: " << objectInfo.size()
            << "  |  TOTAL: €" << total
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

/**
 * @brief Calculate the total monetary value of the detected coins.
 *
 * The first time it runs, it sets gMmPerPixel using the largest circle
 * (assumed to be a €2 coin: 25.75 mm).  From then on it just applies
 * that scale to every circle and looks up a euro denomination.
 */
double calculateTotal(const vector<Vec3f> &circles) {
    if (circles.empty()) return 0.0;

    /* --- one‑off scale calibration ------------------------------------ */
    if (!gScaleReady) {
        int maxR = 0;
        for (auto &c: circles) maxR = max(maxR, cvRound(c[2]));
        gMmPerPixel = 25.75 / (2.0 * maxR);   // assume biggest coin = €2 (25.75 mm Ø)
        /*seenLargestR = max(seenLargestR, maxR);
        gMmPerPixel   = 25.75 / (2.0 * seenLargestR);   // update whenever we see larger*/
        gScaleReady = true;
    }

    // return a sum of all coins
    double totalValue = 0.0;
    for (auto &c: objectInfo) {
        if (c.second.isCoin)
            totalValue += c.second.value;
    }

    return totalValue;
}
