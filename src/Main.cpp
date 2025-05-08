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
int blurSize = 55; // Size of the Gaussian kernel (must be odd)
int blurSigma = 2; // Standard deviation for Gaussian blur

// Global variables for Thresholding
int thresholdValue = 111; // Threshold value for binary thresholding
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
static int seenLargestR = 0;

// Global variables to track cumulative totals
// Array to count coins of each denomination
int coinsCount[N_EURO_COINS] = {0};
double totalValueSum = 0.0;
int totalCoins = 0;

// Detected coin structure
struct DetectedCircle {
    Vec3f circle; // Circle parameters (x, y, radius)

};

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

bool looksFakeByChroma(const cv::Mat &rgb, const cv::Vec3f &c)
{
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
    double meanSat      = meanHSV[1];         // 0‥255

    return meanSat < 20;                      // tweak: 10-25 works in practice
}

bool looksFakeByLab(const cv::Mat &rgb, const cv::Vec3f &c)
{
    cv::Mat mask(rgb.rows, rgb.cols, CV_8UC1, cv::Scalar(0));
    cv::circle(mask, cv::Point(cvRound(c[0]), cvRound(c[1])),
               cvRound(c[2] * 0.9), cv::Scalar(255), -1);

    cv::Mat lab;
    cv::cvtColor(rgb, lab, cv::COLOR_BGR2Lab);

    cv::Scalar meanLab = cv::mean(lab, mask);
    double a = meanLab[1] - 128.0;            // centre Lab to 0 ± 127
    double b = meanLab[2] - 128.0;
    double C = std::sqrt(a*a + b*b);          // chroma

    return C < 8.0;                           // <8 ≈ neutral grey
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
    try {
        cvtColor(src, dst, COLOR_BGR2GRAY);
        GaussianBlur(dst, dst, Size(blurSize, blurSize), blurSigma, blurSigma);
        // use Otsu to auto‐tune the threshold:
        threshold(dst, dst, thresholdValue, thresholdMaxValue, THRESH_BINARY_INV | THRESH_OTSU);
        // optional morphology to fill holes / remove speckle:
        /*Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(dst, dst, MORPH_CLOSE, kernel);
        morphologyEx(dst, dst, MORPH_OPEN, kernel);*/
    } catch (const exception &e) {
        return false;
    }
    return true;
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
        cout << "Circularity: " << circ << endl;
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
bool isSameCoin(const TrackedCoin &oldC, const cv::Vec3f &newC)
{
    const double posThr = 10.0;  // centre-to-centre px
    const double radThr = 5.0;   // radius px

    double dx = newC[0] - oldC.circle[0];
    double dy = newC[1] - oldC.circle[1];
    double dist = std::sqrt(dx * dx + dy * dy);

    return (dist < posThr) && (std::abs(newC[2] - oldC.circle[2]) < radThr);
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
            case 1: coinsCount[0]++; break; // 0.01
            case 2: coinsCount[1]++; break; // 0.02
            case 5: coinsCount[2]++; break; // 0.05
            case 10: coinsCount[3]++; break; // 0.10
            case 20: coinsCount[4]++; break; // 0.20
            case 50: coinsCount[5]++; break; // 0.50
            case 100: coinsCount[6]++; break; // 1.00
            case 200: coinsCount[7]++; break; // 2.00
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
                totalValueSum += coinValue;
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

        double total = calculateTotal(circles); /*totalValueSum / coinsCount;*/
        ostringstream oss;
        oss.setf(ios::fixed);
        oss.precision(2);
        oss << "COINS: " << circles.size()
            << "  |  TOTAL AVG: €" << total
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
