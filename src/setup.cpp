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

std::string promptVideoFile(const std::string &dir) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    for (const auto &entry: fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
                files.push_back(entry.path().string());
            }
        }
    }
    if (files.empty()) {
        std::cout << "No video files found in " << dir << std::endl;
        return "";
    }
    std::cout << "\n========================================\n";
    std::cout << "SELECT A VIDEO FILE\n";
    std::cout << "========================================\n";
    for (size_t i = 0; i < files.size(); ++i) {
        std::cout << i << ": " << files[i] << std::endl;
    }
    size_t idx = 0;
    std::cout << "\nENTER AN INDEX (0-" << files.size() - 1 << "): ";
    std::cin >> idx;
    if (idx >= files.size()) {
        std::cout << "Invalid selection.\n";
        return "";
    }
    return files[idx];
}

bool promptVideoOrCamera(string *pString) {
    std::cout << "\n========================================\n";
    std::cout << "SELECT AN INPUT METHOD\n";
    std::cout << "========================================\n";
    std::cout << "0: CAMERA\n";
    std::cout << "1: VIDEO FILE\n";
    std::cout << "========================================\n";
    std::cout << "ENTER YOUR CHOICE (0 or 1): ";
    int choice = 0;
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    if (choice == 0) {
        return false; // Camera
    } else if (choice == 1) {
        std::string dir = VIDEO_FILE_DIR;
        std::string videoFile = promptVideoFile(dir);
        if (videoFile.empty()) {
            return false;
        }
        *pString = videoFile;
        return true; // Video file
    } else {
        std::cout << "Invalid choice. Please enter 0 or 1.\n";
        return false;
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
bool setup() {
    string videoFile = VIDEO_FILE_PATH;
    bool choice = promptVideoOrCamera(&videoFile);

    if (choice) {
        cap.open(videoFile); // Open the video file
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video file " << videoFile << std::endl;
            return false;
        }
    } else {
        cap.open(0); // Open the camera
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open camera." << std::endl;
            return false;
        }
    }
    namedWindow("Coin Detection", WINDOW_AUTOSIZE);
    return true;
}

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
        //const string videoFile = VIDEO_FILE_PATH;
        if (!setup()) return 1;

        Mat frame, preproc;
        bool paused = false; // Pause state
        double totalFrames = cap.get(CAP_PROP_FRAME_COUNT);

        while (true/*totalFrames == 0 || frameIndex < totalFrames*/) {
            if (!paused) {
                if (!cap.read(frame) || frame.empty()) break;

                if (!preprocess(frame, preproc)) return 2;

                imshow("Preprocessed Image", preproc);

                auto circles = detectCoins(preproc);

                vector<Point2f> centers;
                vector<float> radii;

                for (const auto &c: circles) {
                    centers.emplace_back(c[0], c[1]);
                    radii.emplace_back(c[2]);
                }

                trackedObjects = updateTracks(centers, radii, frame, 100.0f);

                if (!draw(frame, circles, frameIndex)) return 3;

                frameIndex++;
            }

            int key = waitKey(frameDelay);
            if (key == quitKey) break; // Quit if 'q' is pressed
            if (key == 'p') paused = !paused; // Toggle pause state
        }

        destroyWindow("Coin Detection");
        destroyAllWindows();
        cap.release();

    }
    catch (const exception &e) {
        return 4;
    }

    return 0;
}