#include "setup.h"

/**
* @brief Video capture handle
* @details
* This variable is used to capture video frames from the specified video file.
* It is defined globally to be accessible in the setup, preprocess, detectCoins, and draw functions.
* The video capture handle is initialized in the setup function and released in the main function.
*/
VideoCapture cap(0);
const int frameDelay = 1; // milliseconds
int quitKeyVar = 0;
const char quitKey = 'q';

void vc_timer() {
    static bool running = false;
    static chrono::steady_clock::time_point previousTime = chrono::steady_clock::now();

    if (!running) {
        running = true;
    } else {
        chrono::steady_clock::time_point currentTime = chrono::steady_clock::now();
        chrono::steady_clock::duration elapsedTime = currentTime - previousTime;

        // Tempo em segundos.
        auto time_span = chrono::duration_cast<chrono::duration<double>>(elapsedTime);
        double nseconds = time_span.count();

        cout << "Tempo decorrido: " << nseconds << "segundos" << endl;
        cout << "Pressione qualquer tecla para continuar...\n";
        cin.get();
    }
}

/**
 * @brief Setup function
 * @details
 * This function initializes the video capture and creates a window for displaying the results.
 * It opens the video file specified by the user and checks if it was opened successfully.
 * If the video file is opened successfully, it creates a window named "Coin Detection" for displaying the results.
 * It also starts the timer to measure the elapsed time during processing.
 * The function returns true if the setup was successful, otherwise it returns false.
 *
 * @param videoFile - The path to the video file to be processed.
 * @return bool - Returns true if the setup was successful, otherwise false.
 *
 */
bool setup(const string &videoFile) {
    if (!cap.open(videoFile)) {
        return false;
    }
    namedWindow("Coin Detection", WINDOW_AUTOSIZE);
    vc_timer();                        // runProcess timing
    return true;
}

/*
void onTrackbarChange(int, void *) {
    // Callback function for trackbar changes (can be left empty)
}

void setupTrackbars() {
    namedWindow("Parameters", WINDOW_AUTOSIZE);
/*    createTrackbar("DP (x10)", "Parameters", &dp, 50, onTrackbarChange);
    createTrackbar("Min Dist", "Parameters", &minDist, 200, onTrackbarChange);
    createTrackbar("Param1", "Parameters", &param1, 200, onTrackbarChange);
    createTrackbar("Param2", "Parameters", &param2, 200, onTrackbarChange);
    createTrackbar("Min Radius", "Parameters", &minRadius, 200, onTrackbarChange);
    createTrackbar("Max Radius", "Parameters", &maxRadius, 200, onTrackbarChange);/

    // Trackbars for Thresholding parameters
    createTrackbar("Threshold Value", "Parameters", &thresholdValue, 255, onTrackbarChange);
    createTrackbar("Max Value", "Parameters", &thresholdMaxValue, 255, onTrackbarChange);
    createTrackbar("Block Size", "Parameters", &thresholdBlockSize, 50, onTrackbarChange);
    createTrackbar("Threshold C", "Parameters", &thresholdC, 50, onTrackbarChange);
}
*/

/**
 * @brief Main function
 * @details
 * This is the main function of the program.
 * It calls the setup function to initialize the video capture and create a window.
 * It then enters a loop to read frames from the video, preprocess them, detect coins, and draw the results.
 * The loop continues until the user presses the 'q' quitKeyVar or there are no more frames to read.
 * After processing all frames, it stops the timer and releases the video capture.
 *
 * @return int - Returns 0 if the program executed successfully.
 */
int runProcess() {
    try {
        const string videoFile = VIDEO_FILE_PATH;
        if (!setup(videoFile)) return 1;

        /*setupTrackbars();*/

        Mat frame, preproc;
        int currentFrame = 0;
        bool paused = false; // Pause state

        while (true) {
            if (!paused) {
                if (!cap.read(frame) || frame.empty()) break;
                currentFrame++;

                if (!preprocess(frame, preproc)) return 2;

                imshow("Preprocessed Image", preproc);

                auto circles = detectCoins(preproc);
                //auto coinPatches = extractCoinPatches(frame, circles  *//*or trackedCoins[i].circle*//* );
                if (!draw(frame, circles, currentFrame)) return 3;
            }

            int key = waitKey(frameDelay);
            if (key == quitKey) break; // Quit if 'q' is pressed
            if (key == 'p') paused = !paused; // Toggle pause state
        }

        vc_timer();
        destroyWindow("Coin Detection");
        cap.release();
        destroyAllWindows();
    }
    catch (const exception &e) {
        return 4;
    }

    return 0;
}