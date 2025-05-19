#ifndef VIRTUALMONEYCOUNTER_COIN_TRACKER_H
#define VIRTUALMONEYCOUNTER_COIN_TRACKER_H

#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>

#include "coin_classifier.h"
#include "info_windows.h"

using namespace std;
using namespace cv;
using namespace chrono;

struct TrackedObject {
    int id;
    Point2f center;
    float radius;
    double circularity;
    double area;
    int firstSeen;
    int lastSeen;
    int disappearedFrame = 0;
    bool isDisappeared = false; // true if the object is not seen for a while
    Scalar avgColor;  // BGR format
    bool isCoin = false; // true if the object is a coin
    double diameterMM = 0.0; // diameter in mm
    double value = 0.0; // value in euros
};

extern map<int, TrackedObject> objectInfo;
extern int frameIndex;
extern int nextObjectId;
extern map<int, Point2f> trackedObjects; // id -> last-known position

// Coin Counters
extern int realCoinCount;
extern int fakeCoinCount;
extern double totalRealCoinValue;
extern double totalFakeCoinValue;

float distance(Point2f a, Point2f b);

map<int, Point2f> updateTracks(const vector<Point2f> &detectedCenters,
                               const vector<float> &radii,
                               const Mat &frame,
                               float maxDistance);

void prepareSummary();

#endif //VIRTUALMONEYCOUNTER_COIN_TRACKER_H
