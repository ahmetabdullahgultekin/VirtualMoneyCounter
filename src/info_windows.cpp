#include "info_windows.h"

atomic<bool> keepRunning = true;

void showSummary(int realCount, int fakeCount, double realSum, double fakeSum) {
    int width = 400, height = 250;
    cv::Mat summaryImg(height, width, CV_8UC3, cv::Scalar(30, 30, 30));

    // Draw the summary information
    // 2 Digit Precision
    std::ostringstream realSumStream;
    realSumStream.setf(std::ios::fixed);
    realSumStream.precision(2);
    realSumStream << realSum;

    std::ostringstream fakeSumStream;
    fakeSumStream.setf(std::ios::fixed);
    fakeSumStream.precision(2);
    fakeSumStream << fakeSum;

    std::ostringstream realCountStream;
    realCountStream << realCount;

    std::ostringstream fakeCountStream;
    fakeCountStream << fakeCount;

    std::string title = "Detection Summary";
    std::string realCountStr = "Real coins: " + realCountStream.str();
    std::string fakeCountStr = "Fake coins: " + fakeCountStream.str();
    std::string realSumStr = "Real total: " + realSumStream.str() + " EUR";
    std::string fakeSumStr = "Fake total: " + fakeSumStream.str() + " EUR";

    cv::putText(summaryImg, title, {40, 40}, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
    cv::putText(summaryImg, realCountStr, {40, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(summaryImg, fakeCountStr, {40, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::putText(summaryImg, realSumStr, {40, 170}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(summaryImg, fakeSumStr, {40, 210}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    cv::imshow("Summary", summaryImg);
    waitKey(0);
}

// Show enter your choice for a video or camera info window
/*void onOffInputInfoWindow() {
        cv::Mat infoImg(200, 1100, CV_8UC3, cv::Scalar(30, 30, 30));
        std::string title = "Close this window and continue via the command line for input";
        std::string videoOption = "0. Camera";
        std::string cameraOption = "1. Video File";

        cv::putText(infoImg, title, {40, 40}, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,255), 2);
        cv::putText(infoImg, videoOption, {40, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
        cv::putText(infoImg, cameraOption, {40, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);

        cv::imshow("Input Method", infoImg);
        while (keepRunning) {
            if (cv::waitKey(100) == 27) {
                keepRunning = false;
                destroyAllWindows();
                break;
            }
        }
}*/
