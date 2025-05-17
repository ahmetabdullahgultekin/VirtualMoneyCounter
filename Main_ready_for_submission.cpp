
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

extern "C" {
#include "vc.h"
}
using namespace std;
using namespace cv;

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

double gMmPerPixel = 0.0;
int coinsCount[N_EURO_COINS] = {0};
double totalValueSum = 0.0;
int totalCoins = 0;

int main() {
    string videoFilePath;
    cout << "Enter video filename (e.g., videos/video1.mp4): ";
    cin >> videoFilePath;
    VideoCapture cap(videoFilePath);
    if (!cap.isOpened()) {
        cerr << "Cannot open video file!" << endl;
        return -1;
    }

    Mat frame, gray, blurred, thresh;
    int frame_num = 0;
    while (cap.read(frame)) {
        frame_num++;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(15, 15), 2, 2);
        adaptiveThreshold(blurred, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);

        vector<Vec3f> circles;
        HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 50, 120, 30, 30, 70);

        for (size_t i = 0; i < circles.size(); i++) {
            float x = circles[i][0], y = circles[i][1], r = circles[i][2];
            double area = M_PI * r * r;
            double perimeter = 2 * M_PI * r;
            double circularity = 4 * M_PI * area / (perimeter * perimeter);

            Rect bbox(cvRound(x - r), cvRound(y - r), cvRound(2 * r), cvRound(2 * r));
            rectangle(frame, bbox, Scalar(0, 255, 0), 2);
            circle(frame, Point(cvRound(x), cvRound(y)), 3, Scalar(255, 0, 0), -1);

            double estimateDiameterPx = 2 * r;
            if (gMmPerPixel == 0.0) gMmPerPixel = EURO_COINS[7].diameter / estimateDiameterPx;
            double estimateDiameterMm = estimateDiameterPx * gMmPerPixel;
            double minDiff = 1e9;
            int bestIdx = -1;
            for (int c = 0; c < N_EURO_COINS; ++c) {
                double diff = abs(EURO_COINS[c].diameter - estimateDiameterMm);
                if (diff < minDiff) {
                    minDiff = diff;
                    bestIdx = c;
                }
            }
            double typeThresh = 1.2;
            if (bestIdx >= 0 && minDiff < typeThresh) {
                coinsCount[bestIdx]++;
                totalCoins++;
                totalValueSum += EURO_COINS[bestIdx].value;
                putText(frame, to_string(EURO_COINS[bestIdx].value) + "€", Point(x + r, y), FONT_HERSHEY_SIMPLEX, 0.6,
                        Scalar(0, 255, 255), 2);
            } else {
                putText(frame, "Unknown", Point(x + r, y), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
            putText(frame, "A=" + to_string(int(area)), Point(x - r, y - r - 10), FONT_HERSHEY_SIMPLEX, 0.5,
                    Scalar(0, 0, 255), 1);
            putText(frame, "P=" + to_string(int(perimeter)), Point(x - r, y - r - 25), FONT_HERSHEY_SIMPLEX, 0.5,
                    Scalar(255, 0, 0), 1);

            cout << "Frame " << frame_num << ", Coin " << i + 1 << ": Area=" << area << ", Perimeter=" << perimeter
                 << ", Center=(" << x << "," << y << "), Circularity=" << circularity << ", Diameter="
                 << estimateDiameterMm << "mm, Type: "
                 << (bestIdx >= 0 && minDiff < typeThresh ? to_string(EURO_COINS[bestIdx].value) + "€" : "Unknown")
                 << endl;
        }
        imshow("Detection", frame);
        if (waitKey(1) == 27) break;
    }

    cout << "\nRESULT SUMMARY:\n";
    for (int i = 0; i < N_EURO_COINS; ++i) {
        cout << fixed << setprecision(2) << EURO_COINS[i].value << "€: " << coinsCount[i] << " coins\n";
    }
    cout << "TOTAL COINS: " << totalCoins << endl;
    cout << "TOTAL VALUE: " << totalValueSum << "€" << endl;

    cap.release();
    destroyAllWindows();
    return 0;
}
