# Image Retrieval System Based on Multi-source Information Fusion

## Overview

This project is an image retrieval system based on multi-source information fusion technology, implemented using MATLAB. It supports two retrieval methods: uploading images and hand-drawn sketches. The system extracts HOG (Histogram of Oriented Gradients) features from images and uses cosine similarity algorithm for image matching, achieving efficient and accurate image-based image retrieval.

## Features

### Dual-mode Retrieval
- **Image Upload**: Upload local images as query samples
- **Sketch Drawing**: Built-in drawing panel for hand-drawn sketch retrieval
- **Drawing Tools**: Support for pen thickness adjustment and eraser functionality

### Intelligent Image Processing
- **Automatic Preprocessing**: Grayscale conversion, size standardization, normalization
- **Sketch Processing**: Automatic binarization, cropping, and filling of hand-drawn sketches
- **Feature Extraction**: HOG feature extraction for accurate representation

### Intuitive User Interface
- **Partitioned Layout**: Query image area, drawing panel area, and results display area
- **Real-time Progress Bar**: Shows retrieval process
- **Similarity Percentage**: Intuitive display of matching degree

### Efficient Retrieval Algorithm
- **HOG Feature Extraction**: Captures local shape information of images
- **Cosine Similarity Calculation**: Ensures matching accuracy
- **Similarity Ranking**: Returns top 10 most similar results

## System Requirements

- MATLAB (R2018a or later)
- Image Processing Toolbox
- Computer Vision Toolbox

## Installation

1. Clone or download this repository to your local machine
2. Ensure MATLAB is installed with the required toolboxes
3. Navigate to the project directory in MATLAB
4. Run `main.m` to start the application

## Usage

### Method 1: Upload Image
1. Click the "上传图片" (Upload Image) button
2. Select an image file from your local directory
3. Click the "开始检索" (Start Retrieval) button to search for similar images

### Method 2: Draw Sketch
1. Use the drawing panel to sketch your query
2. Adjust pen thickness using the slider if needed
3. Use the eraser tool to correct any mistakes
4. Click the "使用绘画检索" (Use Drawing for Retrieval) button
5. Click the "开始检索" (Start Retrieval) button to search for similar images

## Project Structure

```

├── main.m                 # Main entry point
├── src/                  # Source code
│   ├── features/         # Feature extraction module
│   │   └── extractFeatures.m
│   ├── gui/              # User interface module
│   │   └── imageRetrievalGUI.m
│   ├── matching/         # Image matching module
│   │   └── matchImages.m
│   └── preprocess/       # Image preprocessing module
│       └── preprocessImage.m
├── data/                 # Image dataset
│   ├── 01_01.bmp - 01_20.bmp
│   ├── 02_01.bmp - 02_20.bmp
│   └── ... more image files
└── README.md             # Project documentation
```

## Technical Implementation

- **Development Language**: MATLAB
- **Core Algorithms**: HOG feature extraction, cosine similarity matching
- **Interface Design**: MATLAB GUI
- **Image Processing**: Preprocessing, binarization, feature extraction
- **Data Storage**: Local image dataset

## Application Scenarios

- **Similar Image Retrieval**: Upload an image to quickly find similar images in the dataset
- **Sketch Retrieval**: Retrieve images with similar shapes or structures through hand-drawn sketches
- **Image Database Management**: Provide fast retrieval functionality for large image libraries
- **Academic Research**: Serve as a teaching and research tool for image processing and pattern recognition

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational purposes only.

## Acknowledgments

- This project was developed as part of a multi-source information fusion course
- Thanks to the MATLAB community for their valuable resources and support
