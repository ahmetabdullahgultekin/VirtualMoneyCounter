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
 * @brief Global variables
 * @var cap cv::VideoCapture - video capture handle
 * @var quitKey int - quitKey pressed (global so helpers can exit)
 * @var DP double - Hough parameters
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
cv::VideoCapture cap;
int quitKey = 0;
const double DP = 1.2;
const int MIN_DIST = 50;
const int PARAM1 = 60;
const int PARAM2 = 30;
const int MIN_RADIUS = 20;
const int MAX_RADIUS = 120;
const std::string VIDEO_FILE_DIR = "videos/";
const std::string VIDEO_FILE_NAME = "video1.mp4";
const std::string VIDEO_FILE_PATH = VIDEO_FILE_DIR + VIDEO_FILE_NAME;
/* ---------- COIN‑SIZE DATA (EURO) ------------------------------------ */
struct CoinSpec {
    double diameter;
    double value;
};
const CoinSpec EURO_COINS[] = {                // diameter in mm
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

/* will be filled lazily the first time we see the biggest coin */
bool gScaleReady = false;
double gMmPerPixel = 0.0;   // mm = gMmPerPixel * pixel


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
bool setup(const std::string &videoFile);

bool preprocess(const cv::Mat &src, cv::Mat &dst);

std::vector<cv::Vec3f> detectCoins(const cv::Mat &preproc);

bool draw(cv::Mat &frame, const std::vector<cv::Vec3f> &circles, int currentframe);

double classifyEuroCoin(double dMM);

double calculateTotal(const std::vector<cv::Vec3f> &circles);

void vc_timer() {
    static bool running = false;
    static std::chrono::steady_clock::time_point previousTime = std::chrono::steady_clock::now();

    if (!running) {
        running = true;
    } else {
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration elapsedTime = currentTime - previousTime;

        // Tempo em segundos.
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(elapsedTime);
        double nseconds = time_span.count();

        std::cout << "Tempo decorrido: " << nseconds << "segundos" << std::endl;
        std::cout << "Pressione qualquer tecla para continuar...\n";
        std::cin.get();
    }
}

int professors_function() {
    // V�deo
    char videofile[20] = "video1.mp4";
    cv::VideoCapture capture;
    struct {
        int width, height;
        int ntotalframes;
        int fps;
        int nframe;
    } video{};
    // Outros
    std::string str;
    int key = 0;

    /* Leitura de v�deo de um ficheiro */
    /* NOTA IMPORTANTE:
    O ficheiro video.avi dever� estar localizado no mesmo direct�rio que o ficheiro de c�digo fonte.
    */
    capture.open(videofile);

    /* Em alternativa, abrir captura de v�deo pela Webcam #0 */
    //capture.open(0, cv::CAP_DSHOW); // Pode-se utilizar apenas capture.open(0);

    /* Verifica se foi poss�vel abrir o ficheiro de v�deo */
    if (!capture.isOpened()) {
        std::cerr << "Erro ao abrir o ficheiro de v�deo!\n";
        return 1;
    }

    /* N�mero total de frames no v�deo */
    video.ntotalframes = (int) capture.get(cv::CAP_PROP_FRAME_COUNT);
    /* Frame rate do v�deo */
    video.fps = (int) capture.get(cv::CAP_PROP_FPS);
    /* Resolu��o do v�deo */
    video.width = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
    video.height = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    /* Cria uma janela para exibir o v�deo */
    cv::namedWindow("VC - VIDEO", cv::WINDOW_AUTOSIZE);

    /* Inicia o timer */
    vc_timer();

    cv::Mat frame;
    while (key != 'q') {
        /* Leitura de uma frame do v�deo */
        capture.read(frame);

        /* Verifica se conseguiu ler a frame */
        if (frame.empty()) break;

        /* N�mero da frame a processar */
        video.nframe = (int) capture.get(cv::CAP_PROP_POS_FRAMES);

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
        str = std::string("RESOLUCAO: ").append(std::to_string(video.width)).append("x").append(
                std::to_string(video.height));
        cv::putText(frame, str, cv::Point(20, 25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, str, cv::Point(20, 25), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
        str = std::string("TOTAL DE FRAMES: ").append(std::to_string(video.ntotalframes));
        cv::putText(frame, str, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, str, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
        str = std::string("FRAME RATE: ").append(std::to_string(video.fps));
        cv::putText(frame, str, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, str, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
        str = std::string("N. DA FRAME: ").append(std::to_string(video.nframe));
        cv::putText(frame, str, cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, str, cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);

        /* Exibe a frame */
        cv::imshow("VC - VIDEO", frame);

        /* Sai da aplica��o, se o utilizador premir a tecla 'q' */
        key = cv::waitKey(1);
    }

    /* Para o timer e exibe o tempo decorrido */
    vc_timer();

    /* Fecha a janela */
    cv::destroyWindow("VC - VIDEO");

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
bool setup(const std::string &videoFile) {
    if (!cap.open(videoFile)) {
        return false;
    }
    cv::namedWindow("Coin Detection", cv::WINDOW_AUTOSIZE);
    vc_timer();                        // runProcess timing
    return true;
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
bool preprocess(const cv::Mat &src, cv::Mat &dst) {
    try {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(dst, dst, cv::Size(9, 9), 2);
    }
    catch (const std::exception &e) {
        return false;
    }
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
std::vector<cv::Vec3f> detectCoins(const cv::Mat &preproc) {
    try {
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(preproc, circles, cv::HOUGH_GRADIENT,
                         DP, MIN_DIST, PARAM1, PARAM2,
                         MIN_RADIUS, MAX_RADIUS);
        return circles;
    }
    catch (const std::exception &e) {
        return {};
    }
}

/**
 * @brief Classify a euro coin by its measured diameter.
 * @return coin value in €;  -1.0  if no match
 */
double classifyEuroCoin(double dMM) {
    const double TOL = 0.6;                 // ± 0.6 mm tolerance band
    for (auto i: EURO_COINS)
        if (std::fabs(dMM - i.diameter) <= TOL)
            return i.value;
    return -1.0;
}

/**
 * @brief Calculate the total monetary value of the detected coins.
 *
 * The first time it runs, it sets gMmPerPixel using the largest circle
 * (assumed to be a €2 coin: 25.75 mm).  From then on it just applies
 * that scale to every circle and looks up a euro denomination.
 */
double calculateTotal(const std::vector<cv::Vec3f> &circles) {
    if (circles.empty()) return 0.0;

    /* --- one‑off scale calibration ------------------------------------ */
    if (!gScaleReady) {
        int maxR = 0;
        for (auto &c: circles) maxR = std::max(maxR, cvRound(c[2]));
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
bool draw(cv::Mat &frame, const std::vector<cv::Vec3f> &circles, int currentframe) {
    try {
        for (const auto &c: circles) {
            cv::Point center(cvRound(c[0]), cvRound(c[1]));
            int radius = cvRound(c[2]);
            cv::circle(frame, center, radius, cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, center, 2, cv::Scalar(0, 0, 255), 3);
        }

        double total = calculateTotal(circles);
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << "COINS: " << circles.size()
            << "  |  TOTAL: €" << total
            << "  |  FRAME: " << currentframe;

        std::string txt = oss.str();
        cv::putText(frame, txt, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, txt, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(255, 255, 255), 1);

        cv::putText(frame, txt, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, txt, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(255, 255, 255), 1);

        cv::imshow("Coin Detection", frame);
    }
    catch (const std::exception &e) {
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
 * The loop continues until the user presses the 'q' quitKey or there are no more frames to read.
 * After processing all frames, it stops the timer and releases the video capture.
 *
 * @return int - Returns 0 if the program executed successfully.
 */
int runProcess() {
    try {
        const std::string videoFile = VIDEO_FILE_PATH;
        if (!setup(videoFile)) return 1;

        cv::Mat frame, preproc;
        int currentFrame = 0;

        while (quitKey != 'q') {
            if (!cap.read(frame) || frame.empty()) break;
            currentFrame++;

            if (!preprocess(frame, preproc)) return 2;
            auto circles = detectCoins(preproc);
            if (!draw(frame, circles, currentFrame)) return 3;

            quitKey = cv::waitKey(1);
        }

        vc_timer();                                   // stop timing
        cv::destroyWindow("Coin Detection");
        cap.release();
        // Close window
        cv::destroyAllWindows();
    }
    catch (const std::exception &e) {
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
            std::cerr << "Error: Unable to open video file." << std::endl;
            break;
        case 2:
            std::cerr << "Error: Preprocessing failed." << std::endl;
            break;
        case 3:
            std::cerr << "Error: Drawing failed." << std::endl;
            break;
        case 4:
            std::cerr << "Error: Coin detection failed." << std::endl;
            break;
        default:
            std::cout << "Coin detection completed successfully." << std::endl;
    }

    return 0;
}
