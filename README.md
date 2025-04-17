```markdown
# Virtual Money Counter

## Overview
The **Virtual Money Counter** is a program that detects coins in a video using the Hough Transform method. It processes video frames, identifies circular objects (coins), and displays the results. The program is built using C++ and OpenCV for image processing and video capture.

## Features
- Detects coins in video files.
- Displays the number of detected coins and their positions.
- Supports video preprocessing with grayscale conversion and Gaussian blur.
- Includes a timer to measure processing time.

## Requirements
- **C++ Compiler** (e.g., GCC, MSVC)
- **OpenCV** (version 4.11 or compatible)
- **CMake** (for building the project)
- **CLion IDE** (optional, for development)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmetabdullahgultekin/VirtualMoneyCounter.git
   cd VirtualMoneyCounter
   ```

2. Install OpenCV and ensure it is properly configured.

3. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## Usage
1. Place the video file (`video1.mp4`) in the same directory as the executable or update the file path in the code.

2. Run the program:
   ```bash
   ./VirtualMoneyCounter
   ```

3. Press `q` to exit the program.

## File Structure
- `main.cpp`: Main program logic for coin detection.
- `vc.c` and `vc.h`: Custom image processing library.
- `CMakeLists.txt`: Build configuration file.

## Troubleshooting
- **Video file not found**: Ensure the video file is in the correct directory or update the file path in the code.
- **OpenCV errors**: Verify that OpenCV is installed and linked correctly.

## Author
Ahmet Abdullah Gultekin
```