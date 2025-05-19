#include "coin_tracker.h"

int frameIndex = 0;
int nextObjectId = 0;
map<int, TrackedObject> objectInfo;
map<int, Point2f> trackedObjects; // id -> last-known position

// Coin Counters
int realCoinCount = 0;
int fakeCoinCount = 0;
double totalRealCoinValue = 0;
double totalFakeCoinValue = 0;

float distance(Point2f a, Point2f b) {
    return sqrt((float) (a.x - b.x) * (a.x - b.x) +
                (float) (a.y - b.y) * (a.y - b.y));
}

map<int, Point2f> updateTracks(const vector<Point2f> &detectedCenters,
                               const vector<float> &radii,
                               const Mat &frame,
                               float maxDistance) {
    map<int, Point2f> updatedPositions;
    vector<bool> matched(detectedCenters.size(), false);

    for (auto &[id, prevCenter]: trackedObjects) {
        float minDistance = maxDistance;
        int bestIndex = -1;

        // Debuger
        cout << "ID: " << id << ", Previous Center: (" << prevCenter.x << ", " << prevCenter.y << ")" << endl;

        // Check if the object is still in the frame
        for (int i = 0; i < detectedCenters.size(); ++i) {
            if (matched[i]) continue;
            float dist = distance(prevCenter, detectedCenters[i]);
            // Debuger
            cout << "ID: " << id << ", Distance: " << dist << endl;
            if (dist < minDistance) {
                minDistance = dist;
                bestIndex = i;
                // Debuger
                cout << "ID: " << id << ", Best Index: " << bestIndex
                     << ", Min Distance: " << minDistance
                     << ", Radius: " << radii[i] << endl;
            }
        }

        // If a match is found, update the position
        if (bestIndex != -1) {
            Point2f newCenter = detectedCenters[bestIndex];
            updatedPositions[id] = newCenter;
            matched[bestIndex] = true;

            // Update the tracked object
            objectInfo[id].center = newCenter;
            objectInfo[id].lastSeen = frameIndex;
            // The Object has not disappeared yet
            objectInfo[id].isDisappeared = false;
            objectInfo[id].disappearedFrame = 0; // Reset the disappeared frame count
            // Update the average color by calculating the average color with the previous average color
            //objectInfo[id].avgColor = 0.5 * (objectInfo[id].avgColor + 2.0 * getAverageColor(frame, newCenter, radii[bestIndex])); // average color in BGR
            // Update diameter by calculating the average diameter with the previous diameter
            objectInfo[id].diameterMM =
                    0.5 * (objectInfo[id].diameterMM + 2.0 * radii[bestIndex] * gMmPerPixel); // diameter in mm
            objectInfo[id].value = classifyEuroCoin(objectInfo[id].diameterMM); // Get the value of the coin
            // Update the radius by calculating the average radius with the previous radius
            //objectInfo[id].radius = 0.5 * (objectInfo[id].radius + 2.0 * radii[bestIndex]); // radius in pixels
            objectInfo[id].isCoin = isCoinColor(objectInfo[id].avgColor);
        }
            // If no match is found, mark the object as disappeared
        else {
            objectInfo[id].isDisappeared = true;
            objectInfo[id].disappearedFrame++;
            if (objectInfo[id].disappearedFrame > 5) { // If the object has disappeared for more than 30 frames
                objectInfo[id].isDisappeared = true;
                objectInfo[id].disappearedFrame = 0; // Reset the disappeared frame count
            } else {
                //objectInfo[id].isDisappeared = false; // The object has not disappeared yet
                //return updatedPositions;
            }
        }
    }

    // Handle unmatched detected centers
    // Add new objects
    for (int i = 0; i < detectedCenters.size(); ++i) {
        if (!matched[i]) {
            int id = nextObjectId++;
            Point2f center = detectedCenters[i];
            float radius = radii[i];
            Scalar color = getAverageColor(frame, center, radius);

            updatedPositions[id] = center;
            trackedObjects[id] = center;

            TrackedObject obj;
            obj.id = id;
            obj.center = center;
            obj.radius = radius;
            obj.area = radius * radius * CV_PI;
            obj.circularity = 4.0 * CV_PI * obj.area / (radius * radius);
            obj.firstSeen = frameIndex;
            obj.lastSeen = frameIndex;
            obj.isDisappeared = false; // The object has just been seen
            obj.avgColor = color;  // <<<< STORE COLOR
            obj.isCoin = isCoinColor(color); // Check if the object is a coin
            obj.diameterMM = 2.0 * radius * gMmPerPixel; // diameter in mm
            obj.value = classifyEuroCoin(obj.diameterMM); // Get the value of the coin

            objectInfo[id] = obj;
        }
    }

    // Debuger
    for (const auto &[id, obj]: objectInfo) {
        cout << "ID: " << id
             << ", Center: (" << obj.center.x << ", " << obj.center.y << ")"
             << ", First Seen: " << obj.firstSeen
             << ", Last Seen: " << obj.lastSeen
             << ", Radius: " << obj.radius
             << ", Area: " << obj.area
             << ", Circularity: " << obj.circularity
             << ", Disappeared: " << (obj.isDisappeared ? "Yes" : "No")
             << ", AvgColor (BGR): " << obj.avgColor
             << ", Is Coin: " << (obj.isCoin ? "Yes" : "No")
             << ", Diameter (mm): " << obj.diameterMM
             << ", Value (€): " << obj.value
             << endl;
    }

    return updatedPositions;
}

// Prepare summary
void prepareSummary() {
    // Initialize the summary variables
    realCoinCount = 0;
    fakeCoinCount = 0;
    totalRealCoinValue = 0;
    totalFakeCoinValue = 0;

    // Iterate through the tracked objects and count real and fake coins
    for (const auto &[id, obj]: objectInfo) {
        if (obj.isCoin) {
            realCoinCount++;
            totalRealCoinValue += obj.value;
        } else {
            fakeCoinCount++;
            totalFakeCoinValue += obj.value;
        }
    }

    // Show the summary
    showSummary(realCoinCount, fakeCoinCount, totalRealCoinValue, totalFakeCoinValue);
}
