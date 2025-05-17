#include "coin_classifier.h"

bool gScaleReady = false;
double gMmPerPixel = 0.0;

// Global variables to track cumulative totals
// Array to count coins of each denomination
int coinsCount[N_EURO_COINS] = {0};
double totalValueSum = 0.0;
int totalCoins = 0;

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
    // Printing the diameter for debugging
    cout << "Diameter: " << dMM << endl;

    double closestValue = -1.0;
    double minDifference = numeric_limits<double>::max();

    for (const auto &coin: EURO_COINS) {
        double difference = fabs(dMM - coin.diameter);
        if (difference < minDifference) {
            minDifference = difference;
            closestValue = coin.value;
        }
    }

    // Printing the closest diameter and value for debugging
    if (closestValue != -1.0) {
        cout << "Closest Diameter: " << (dMM - minDifference) << " Value: " << closestValue << endl;
    }

    return closestValue;
}

/**
 * @brief Check if a detected coin is the same as a previously tracked coin.
 * @param newCoin - The newly detected coin (circle).
 * @return bool - True if the coin is already tracked, false otherwise.
 */
/*bool isSameCoin(const TrackedCoin &oldC, const cv::Vec3f &newC)
{
    const double posThr = 10.0;  // centre-to-centre px
    const double radThr = 5.0;   // radius px

    double dx = newC[0] - oldC.circle[0];
    double dy = newC[1] - oldC.circle[1];
    double dist = std::sqrt(dx * dx + dy * dy);

    return (dist < posThr) && (std::abs(newC[2] - oldC.circle[2]) < radThr);
}*/


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

    /* --- accumulate value -------------------------------------------- */
    double sum = 0.0;
    for (auto &c: circles) {
        double dMM = 2.0 * cvRound(c[2]) * gMmPerPixel; // diameter in mm
        double val = classifyEuroCoin(dMM);
        /*if (val > 0.0) sum += val;*/
        switch (int(val * 100)) { // convert to cents
            case 1:
                coinsCount[0]++;
                break; // 0.01
            case 2:
                coinsCount[1]++;
                break; // 0.02
            case 5:
                coinsCount[2]++;
                break; // 0.05
            case 10:
                coinsCount[3]++;
                break; // 0.10
            case 20:
                coinsCount[4]++;
                break; // 0.20
            case 50:
                coinsCount[5]++;
                break; // 0.50
            case 100:
                coinsCount[6]++;
                break; // 1.00
            case 200:
                coinsCount[7]++;
                break; // 2.00
        }
    }
    totalValueSum = coinsCount[0] * 0.01 / 100 +
                    coinsCount[1] * 0.02 / 100 +
                    coinsCount[2] * 0.05 / 100 +
                    coinsCount[3] * 0.10 / 100 +
                    coinsCount[4] * 0.20 / 100 +
                    coinsCount[5] * 0.50 / 100 +
                    coinsCount[6] * 1.00 / 100 +
                    coinsCount[7] * 2.00 / 100;

    totalCoins = coinsCount[0] + coinsCount[1] + coinsCount[2] +
                 coinsCount[3] + coinsCount[4] +
                 coinsCount[5] + coinsCount[6] +
                 coinsCount[7];
    return totalValueSum / totalCoins; // average value
}

/**
 * @brief Extract per-coin image patches from a frame.
 *
 * @param frame         BGR input frame (any depth, any channel count ≥3)
 * @param circles       Detected circles: {x, y, r} in pixel units
 * @param circularMask  true  – return tight circular patches on black background
 *                      false – return simple square crops that enclose the coin
 *
 * @return std::vector<cv::Mat>  Deep-copied patches, same type as @p frame.
 *                               Caller owns the data; safe after @p frame is reused.
 *
 * Usage:
 *     auto patches = extractCoinPatches(frame, circles);          // default = masked
 *     for (size_t i = 0; i < patches.size(); ++i)
 *         cv::imshow("coin " + std::to_string(i), patches[i]);
 */
/*std::vector<cv::Mat> extractCoinPatches(const cv::Mat&                      frame,
                                        const std::vector<cv::Vec3f>&       circles,
                                        bool                                circularMask = true)
{
    using namespace cv;

    std::vector<Mat> patches;
    patches.reserve(circles.size());          // avoid realloc each push_back

    Mat mask(frame.rows, frame.cols, CV_8UC1);   // reused every iteration

    for (const auto& c : circles)
    {
        int r   = cvRound(c[2]);
        int x0  = std::max(0,           cvRound(c[0] - r));
        int y0  = std::max(0,           cvRound(c[1] - r));
        int x1  = std::min(frame.cols,  cvRound(c[0] + r));
        int y1  = std::min(frame.rows,  cvRound(c[1] + r));

        if (x1 <= x0 || y1 <= y0) continue;      // degenerate ROI → skip

        Rect roi(x0, y0, x1 - x0, y1 - y0);

        if (circularMask)
        {
            mask.setTo(0);
            circle(mask,
                   Point(cvRound(c[0]), cvRound(c[1])),
                   r, 255, -1);                  // draw 255 inside coin

            Mat patch;
            frame(roi).copyTo(patch, mask(roi)); // copy only masked pixels
            patches.emplace_back(patch.clone()); // deep copy = safe after reuse
        }
        else
        {
            patches.emplace_back(frame(roi).clone());
        }
    }
    return patches;
}*/

/* ------------------------------------------------------------------
 *  Return true iff the coin patch looks like a real euro alloy
 *  and give back the average BGR in @avg (optional).
 *  The patch must come from extractCoinPatches(... circularMask=true)
 *  so the background is pure black (0,0,0).
 * ------------------------------------------------------------------ */
/*
bool isRealCoinColour(const cv::Mat& patch,
                      cv::Vec3d*     avgOut = nullptr)
{
    CV_Assert(patch.channels() >= 3);

    */
/* 1. Build mask: coin pixels are non-black *//*

    Mat gray; cvtColor(patch, gray, COLOR_BGR2GRAY);
    Mat mask = gray > 0;                      // 8-bit 0/255

    if (countNonZero(mask) < 50) return false;  // almost empty

    */
/* 2. Mean BGR of coin *//*

    Scalar m = mean(patch, mask);
    Vec3d  avg(m[0], m[1], m[2]);             // B, G, R order

    if (avgOut) *avgOut = avg;

    */
/* 3. Distance to each alloy reference colour *//*

    static const Vec3d copper ( 60,  80, 145);
    static const Vec3d brass  ( 85, 140, 155);
    static const Vec3d nickel (180, 180, 180);

    auto dist = [](const Vec3d& a, const Vec3d& b){
        return norm(a - b);                   // Euclidean in RGB
    };

    double dMin = std::min({ dist(avg,copper),
                             dist(avg,brass),
                             dist(avg,nickel) });

    return dMin < 25.0;                       // threshold – tune 20-30
}
*/

bool looksFakeByChroma(const cv::Mat &rgb, const cv::Vec3f &c) {
    /* build a circular mask so we measure only the coin pixels */
    cv::Mat mask(rgb.rows, rgb.cols, CV_8UC1, cv::Scalar(0));
    cv::circle(mask,
               cv::Point(cvRound(c[0]), cvRound(c[1])),
               cvRound(c[2] * 0.9),             // 90 % radius to skip the rim
               cv::Scalar(255), -1);

    /* convert ROI to HSV – S (channel 1) encodes “colourfulness” */
    cv::Mat hsv;
    cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);

    cv::Scalar meanHSV = cv::mean(hsv, mask);
    double meanSat = meanHSV[1];         // 0‥255

    return meanSat < 20;                      // tweak: 10-25 works in practice
}

bool looksFakeByLab(const cv::Mat &rgb, const cv::Vec3f &c) {
    cv::Mat mask(rgb.rows, rgb.cols, CV_8UC1, cv::Scalar(0));
    cv::circle(mask, cv::Point(cvRound(c[0]), cvRound(c[1])),
               cvRound(c[2] * 0.9), cv::Scalar(255), -1);

    cv::Mat lab;
    cv::cvtColor(rgb, lab, cv::COLOR_BGR2Lab);

    cv::Scalar meanLab = cv::mean(lab, mask);
    double a = meanLab[1] - 128.0;            // centre Lab to 0 ± 127
    double b = meanLab[2] - 128.0;
    double C = std::sqrt(a * a + b * b);          // chroma

    return C < 8.0;                           // <8 ≈ neutral grey
}