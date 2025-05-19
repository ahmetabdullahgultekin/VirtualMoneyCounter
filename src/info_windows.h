#ifndef VIRTUALMONEYCOUNTER_INFO_WINDOWS_H
#define VIRTUALMONEYCOUNTER_INFO_WINDOWS_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

extern atomic<bool> keepRunning;

void showSummary(int realCount, int fakeCount, double realSum, double fakeSum);

void onOffInputInfoWindow();

#endif //VIRTUALMONEYCOUNTER_INFO_WINDOWS_H
