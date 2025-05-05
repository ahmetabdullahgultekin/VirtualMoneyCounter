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
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief Include the vc.h header file
 */
extern "C" {
#include "vc.h"
}

/**
 * @brief Namespaces for convenience
 * using namespace std;
 * using namespace cv;
 */
using namespace std;
using namespace cv;

/**
 * @brief Global variables
 * @var cap cv::VideoCapture - video capture handle
 * @var quitKeyVar int - quitKeyVar pressed (global so helpers can exit)
 * @var DP double - inverse ratio of the accumulator resolution to the image resolution
 * @var MIN_DIST int - minimum distance between detected centers
 * @var PARAM1 int - first method parameter for HoughCircles
 * @var PARAM2 int - second method parameter for HoughCircles
 * @var MIN_RADIUS int - minimum radius of circles to be detected
 * @var MAX_RADIUS int - maximum radius of circles to be detected
 * @var VIDEO_FILE_PATH std::string - path to the video file
 *
 * @details
 * These variables are used to configure the Hough Transform parameters for circle detection.
 * They are defined globally to be accessible in the setup, preprocess, detectCoins, and draw functions.
 * The DP variable is the inverse ratio of the accumulator resolution to the image resolution.
 * The MIN_DIST variable is the minimum distance between detected centers.
 * The PARAM1 and PARAM2 variables are the first and second method parameters for HoughCircles.
 * The MIN_RADIUS and MAX_RADIUS variables define the minimum and maximum radius of circles to be detected.
 * These parameters can be adjusted to improve the detection results based on the specific video being processed.
 * The default values are set to reasonable values for detecting coins in a video.
 * These values can be modified based on the specific requirements of the application.
 * The VIDEO_FILE_PATH variable is the path to the video file to be processed.
 * It should be in the same directory as the source code.
 * The video file should be in a supported format (e.g., .mp4, .avi).
 */

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
// Global vector to store unique coins
vector<Vec3f> trackedCoins;

/**
 * @brief Hough Transform parameters
 * @details
 * These variables are used to configure the Hough Transform parameters for circle detection.
 * They are defined globally to be accessible in the setup, preprocess, detectCoins, and draw functions.
 * The DP variable is the inverse ratio of the accumulator resolution to the image resolution.
 * The MIN_DIST variable is the minimum distance between detected centers.
 * The PARAM1 and PARAM2 variables are the first and second method parameters for HoughCircles.
 * The MIN_RADIUS and MAX_RADIUS variables define the minimum and maximum radius of circles to be detected.
 * These parameters can be adjusted to improve the detection results based on the specific video being processed.
 */
/*const double DP = 1.2;
const int PARAM1 = 60;
const int PARAM2 = 60;
const int MIN_RADIUS = 55;
const int MAX_RADIUS = 110;
const int MIN_DIST = MIN_RADIUS * 2;*/

// Global variables for Hough Transform parameters
int dp = 12; // Scaled by 10 (e.g., 1.2 -> 12)
int param1 = 60;
int param2 = 60;
int minRadius = 55;
int maxRadius = 110;
int minDist = minRadius * 2; // Minimum distance between detected centers

// Global variables for Gaussian blur
int blurSize = 9; // Size of the Gaussian kernel (must be odd)
int blurSigma = 2; // Standard deviation for Gaussian blur

// Global variables for Thresholding
int thresholdValue = 150; // Threshold value for binary thresholding
int thresholdMaxValue = 255; // Maximum value for binary thresholding
int thresholdBlockSize = 11; // Block size for adaptive thresholding (must be odd)
int thresholdC = 2; // Constant subtracted from the mean in adaptive thresholding

/**
 * @brief Video file path
 * @details
 * This variable stores the path to the video file to be processed.
 * The video file should be in the same directory as the source code.
 * The video file should be in a supported format (e.g., .mp4, .avi).
 */
const string VIDEO_FILE_DIR = "videos/";
const string VIDEO_FILE_NAME = "video2.mp4";
const string VIDEO_FILE_PATH = VIDEO_FILE_DIR + VIDEO_FILE_NAME;

/**
 * @brief Coin specifications
 * @details
 * This struct defines the specifications of euro coins.
 * It includes the diameter and value of each coin.
 * The EURO_COINS array contains the specifications of all euro coins.
 * The N_EURO_COINS variable stores the number of euro coins in the array.
 */
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
bool gScaleReady = false;
double gMmPerPixel = 0.0;

/**
 * @brief Function prototypes
 * @details
 * These function prototypes declare the functions used in the program.
 * They are defined in the implementation section.
 * The setup function initializes the video capture and creates a window for displaying the results.
 * The preprocess function converts the input image to grayscale and applies Gaussian blur.
 * The detectCoins function uses the Hough Transform to detect circles in the preprocessed image.
 * The draw function draws the detected circles on the original image and displays the results.
 * These functions are called in the main loop of the program to process each frame of the video.
 * The setup function is called once at the beginning to initialize the video capture and create the window.
 * The preprocess function is called for each frame to convert it to grayscale and apply Gaussian blur.
 * The detectCoins function is called to detect circles in the preprocessed image.
 * The draw function is called to draw the detected circles on the original image and display the results.
 * The vc_timer function is called to measure the elapsed time during processing.
 * The vc_timer function is called at the beginning and end of the main loop to measure the elapsed time.
 * The elapsed time is displayed in the console.
 * The elapsed time is measured in seconds.
 *
 */
bool setup(const string &videoFile);

bool preprocess(const Mat &src, Mat &dst);

vector<Vec3f> detectCoins(const Mat &preproc);

bool draw(Mat &frame, const vector<Vec3f> &circles, int currentframe);

double classifyEuroCoin(double dMM);

double calculateTotal(const vector<Vec3f> &circles);

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

int professors_function() {
    // V�deo
    char videofile[20] = "video1.mp4";
    VideoCapture capture;
    struct {
        int width, height;
        int ntotalframes;
        int fps;
        int nframe;
    } video{};
    // Outros
    string str;
    int key = 0;

    /* Leitura de v�deo de um ficheiro */
    /* NOTA IMPORTANTE:
    O ficheiro video.avi dever� estar localizado no mesmo direct�rio que o ficheiro de c�digo fonte.
    */
    capture.open(videofile);

    /* Em alternativa, abrir captura de v�deo pela Webcam #0 */
    //capture.open(0, CAP_DSHOW); // Pode-se utilizar apenas capture.open(0);

    /* Verifica se foi poss�vel abrir o ficheiro de v�deo */
    if (!capture.isOpened()) {
        cerr << "Erro ao abrir o ficheiro de v�deo!\n";
        return 1;
    }

    /* N�mero total de frames no v�deo */
    video.ntotalframes = (int) capture.get(CAP_PROP_FRAME_COUNT);
    /* Frame rate do v�deo */
    video.fps = (int) capture.get(CAP_PROP_FPS);
    /* Resolu��o do v�deo */
    video.width = (int) capture.get(CAP_PROP_FRAME_WIDTH);
    video.height = (int) capture.get(CAP_PROP_FRAME_HEIGHT);

    /* Cria uma janela para exibir o v�deo */
    namedWindow("VC - VIDEO", WINDOW_AUTOSIZE);

    /* Inicia o timer */
    vc_timer();

    Mat frame;
    while (key != 'q') {
        /* Leitura de uma frame do v�deo */
        capture.read(frame);

        /* Verifica se conseguiu ler a frame */
        if (frame.empty()) break;

        /* N�mero da frame a processar */
        video.nframe = (int) capture.get(CAP_PROP_POS_FRAMES);

        // Fa�a o seu c�digo aqui...
        /*
        // Cria uma nova imagem IVC
        IVC *image = vc_image_new(video.width, video.height, 3, 255);
        // Copia dados de imagem da estrutura cv::Mat para uma estrutura IVC
        memcpy(image->data, frame.data, video.width * video.height * 3);
        // Executa uma fun��o da nossa biblioteca vc
        vc_rgb_get_green(image);
        // Copia dados de imagem da estrutura IVC para uma estrutura cv::Mat
        memcpy(frame.data, image->data, video.width * video.height * 3);
        // Liberta a mem�ria da imagem IVC que havia sido criada
        vc_image_free(image);
        */
        // +++++++++++++++++++++++++

        /* Exemplo de inser��o texto na frame */
        str = string("RESOLUCAO: ").append(to_string(video.width)).append("x").append(
                to_string(video.height));
        putText(frame, str, Point(20, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 2);
        putText(frame, str, Point(20, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1);
        str = string("TOTAL DE FRAMES: ").append(to_string(video.ntotalframes));
        putText(frame, str, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 2);
        putText(frame, str, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1);
        str = string("FRAME RATE: ").append(to_string(video.fps));
        putText(frame, str, Point(20, 75), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 2);
        putText(frame, str, Point(20, 75), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1);
        str = string("N. DA FRAME: ").append(to_string(video.nframe));
        putText(frame, str, Point(20, 100), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0), 2);
        putText(frame, str, Point(20, 100), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 1);

        /* Exibe a frame */
        imshow("VC - VIDEO", frame);

        /* Sai da aplica��o, se o utilizador premir a tecla 'q' */
        key = waitKey(1);
    }

    /* Para o timer e exibe o tempo decorrido */
    vc_timer();

    /* Fecha a janela */
    destroyWindow("VC - VIDEO");

    /* Fecha o ficheiro de v�deo */
    capture.release();

    return 0;
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

void onTrackbarChange(int, void*) {
    // Callback function for trackbar changes (can be left empty)
}

void setupTrackbars() {
    namedWindow("Parameters", WINDOW_AUTOSIZE);
/*    createTrackbar("DP (x10)", "Parameters", &dp, 50, onTrackbarChange);
    createTrackbar("Min Dist", "Parameters", &minDist, 200, onTrackbarChange);
    createTrackbar("Param1", "Parameters", &param1, 200, onTrackbarChange);
    createTrackbar("Param2", "Parameters", &param2, 200, onTrackbarChange);
    createTrackbar("Min Radius", "Parameters", &minRadius, 200, onTrackbarChange);
    createTrackbar("Max Radius", "Parameters", &maxRadius, 200, onTrackbarChange);*/

    // Trackbars for Thresholding parameters
    createTrackbar("Threshold Value", "Parameters", &thresholdValue, 255, onTrackbarChange);
    createTrackbar("Max Value", "Parameters", &thresholdMaxValue, 255, onTrackbarChange);
    createTrackbar("Block Size", "Parameters", &thresholdBlockSize, 50, onTrackbarChange);
    createTrackbar("Threshold C", "Parameters", &thresholdC, 50, onTrackbarChange);
}

/**
 * @brief Preprocess function
 * @details
 * This function preprocesses the input image by converting it to grayscale and applying Gaussian blur.
 * The grayscale conversion is done using the cv::cvtColor function with the COLOR_BGR2GRAY flag.
 * The Gaussian blur is applied using the cv::GaussianBlur function with a kernel size of 9x9 and a standard deviation of 2.
 * The preprocessed image is stored in the dst parameter.
 *
 * @param src - The input image to be preprocessed.
 * @param dst - The preprocessed output image.
 *
 * @return int - Returns 0 if the preprocessing was successful.
 */
/*bool preprocess(const Mat &src, Mat &dst) {
    try {
        // Convert to grayscale
        cvtColor(src, dst, COLOR_BGR2GRAY);
        // Apply Gaussian blur
        GaussianBlur(dst, dst, Size(55, 55), 3,3);
        // Apply Binary thresholding
        threshold(dst, dst, 120, 255, THRESH_BINARY);
        // Apply Canny edge detection
        *//*Canny(dst, dst, 25, 75);*//*
    }
    catch (const exception &e) {
        return false;
    }
    return true;
}*/
bool preprocess(const Mat &src, Mat &dst) {
    cvtColor(src, dst, COLOR_BGR2GRAY);
    GaussianBlur(dst, dst, Size(blurSize, blurSize), blurSigma, blurSigma);
    // use Otsu to auto‐tune the threshold:
    threshold(dst, dst, thresholdValue, thresholdMaxValue,THRESH_BINARY_INV | THRESH_OTSU);
    // optional morphology to fill holes / remove speckle:
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(dst, dst, MORPH_CLOSE, kernel);
    morphologyEx(dst, dst, MORPH_OPEN,  kernel);
    return true;
}


/**
 * @brief Detect coins function
 * @details
 * This function detects coins in the preprocessed image using the Hough Transform method.
 * It uses the cv::HoughCircles function to detect circles in the image.
 * The detected circles are stored in the circles vector.
 *
 * @param preproc - The preprocessed input image.
 * @return std::vector<cv::Vec3f> - A vector of detected circles (coins).
 */
/*vector<Vec3f> detectCoins(const Mat &preproc) {
    try {
        vector<Vec3f> circles;
        double dpValue = dp / 10.0; // Convert scaled DP back to a double
        HoughCircles(preproc, circles, HOUGH_GRADIENT,
                     dpValue, minDist, param1, param2,
                     minRadius, maxRadius);
        return circles;
    } catch (const exception &e) {
        return {};
    }
}*/
vector<Vec3f> detectCoins(const Mat &preproc) {
    vector<Vec3f> circles;
    // 1) Find all external contours
    vector<vector<Point>> contours;
    findContours(preproc, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto &c : contours) {
        double area = contourArea(c);
        if (area < 100.0)
            continue;               // skip tiny noise blobs

        double peri = arcLength(c, true);
        double circ = 4.0*CV_PI*area/(peri*peri);
        if (circ < 0.75)
            continue;               // require at least ~75% circular

        // 2) Fit minimum enclosing circle
        Point2f center;
        float   radius;
        minEnclosingCircle(c, center, radius);

        // 3) Optionally enforce a radius window too
        if (radius < minRadius || radius > maxRadius)
            continue;

        circles.emplace_back(center.x, center.y, radius);
    }
    return circles;
}

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

    for (const auto &coin : EURO_COINS) {
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
bool isSameCoin(const Vec3f &newCoin) {
    const double positionThreshold = 10.0; // Max distance between centers
    const double radiusThreshold = 5.0;   // Max difference in radius

    for (const auto &trackedCoin : trackedCoins) {
        double dx = newCoin[0] - trackedCoin[0];
        double dy = newCoin[1] - trackedCoin[1];
        double distance = std::sqrt(dx * dx + dy * dy);
        double radiusDiff = std::abs(newCoin[2] - trackedCoin[2]);

        if (distance < positionThreshold && radiusDiff < radiusThreshold) {
            return true; // Coin is the same
        }
    }
    return false;
}

/**
 * @brief Update the list of tracked coins with new detections.
 * @param detectedCoins - The coins detected in the current frame.
 */
void updateTrackedCoins(const std::vector<Vec3f> &detectedCoins) {
    for (const auto &coin : detectedCoins) {
        if (!isSameCoin(coin)) {
            trackedCoins.push_back(coin); // Add new coin to the list
        }
    }
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
        gScaleReady = true;
    }

    /* --- accumulate value -------------------------------------------- */
    double sum = 0.0;
    for (auto &c: circles) {
        double dMM = 2.0 * c[2] * gMmPerPixel;
        double val = classifyEuroCoin(dMM);
        if (val > 0.0) sum += val;
    }
    return sum;
}


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
        for (const auto &c: circles) {
            Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);
            circle(frame, center, radius, Scalar(0, 255, 0), 2);
            circle(frame, center, 2, Scalar(0, 0, 255), 3);

            // Classify the coin and add text
            double diameterMM = 2.0 * radius * gMmPerPixel;
            double coinValue = classifyEuroCoin(diameterMM);

            if (coinValue > 0.0) {
                ostringstream oss;
                oss.setf(ios::fixed);
                oss.precision(2);
                oss << "€" << coinValue;

                putText(frame, oss.str(), Point(center.x - 20, center.y - radius - 10),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
                putText(frame, oss.str(), Point(center.x - 20, center.y - radius - 10),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
            }
        }

        double total = calculateTotal(circles);
        ostringstream oss;
        oss.setf(ios::fixed);
        oss.precision(2);
        oss << "COINS: " << circles.size()
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

        setupTrackbars();

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
