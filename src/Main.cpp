/**
 * @file Main.cpp
 * @brief Coin detection using Hough Transform
 * @author Ahmet Abdullah GULTEKIN
 * @date 2025-04-17
 * @details
 * This program demonstrates how to detect coins in a video using the Hough Transform method.
 * It captures video frames, processes them to detect circles (coins), and displays the results.
 * The program uses OpenCV for image processing and video capture.
 * It also includes a timer to measure the elapsed time during processing.
 * The program can be run with a video file.
 * The video file should be in the same directory as the source code.
 */

#include <iostream>
#include "setup.h"

/**
 * @brief Namespaces for convenience
 * using namespace std;
 * using namespace cv;
 */
using namespace std;
using namespace cv;

/**
 * @brief Main function
 * @details
 * This is the main function of the program.
 * It calls the start to runProcess the coin detection process.
 * The program returns 0 if it executed successfully.
 *
 * @return int - Returns 0 if the program executed successfully.
 */
int main() {
    int result = runProcess();
    switch (result) {
        case 1:
            cerr << "Error: Unable to open video file." << endl;
            break;
        case 2:
            cerr << "Error: Preprocessing failed." << endl;
            break;
        case 3:
            cerr << "Error: Drawing failed." << endl;
            break;
        case 4:
            cerr << "Error: Coin detection failed." << endl;
            break;
        default:
            cout << "Coin detection completed successfully." << endl;
    }

    return 0;
}
