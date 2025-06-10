# Face Detection App with Haar Cascade Classifier

This is a Streamlit-based face detection application that uses the **Haar Cascade classifier** to detect faces in uploaded images. The app is built using modern Python practices and follows PEP8 standards.

---

## Table of Contents

  - [About the Project](#-about-the-project)
    - [What is the Haar Cascade Classifier?](#-what-is-the-haar-cascade-classifier)
      - [How It Works:](#-how-it-works)
  - [Features](#-features)
  - [Getting Started](#-getting-started)
    - [Prerequisites](#-prerequisites)
    - [Folder Structure](#-folder-structure)
  - [Installation](#-installation)
  - [Running Script](#-running-script)
  - [Expectations When Running This App](#-expectations-when-running-this-app)
  - [Demo](#-demo)
  - [Acknowledgments](#-acknowledgments)
  - [License](#-license)
  - [Notes](#-notes)

---

## About the Project

This project is a simple yet powerful face detection tool that allows users to upload images and detect faces using the **Haar Cascade classifier**. It's built with Python, OpenCV, and Streamlit.

### What is the Haar Cascade Classifier?

The **Haar Cascade** is a machine learning object detection algorithm developed by Paul Viola and Michael Jones. It's widely used for detecting objects in images, such as:

- Faces
- Eyes
- Smile
- Vehicles

#### How It Works:
1. **Haar Features**: The algorithm uses simple rectangular features to detect patterns in images.
2. **Cascade of Classifiers**: A series of classifiers is trained, each one filtering out non-matching regions.
3. + **Real-Time Detection**: The algorithm is fast and efficient, making it ideal for real-time applications.

The `haarcascade_frontalface_default.xml` file used in this app is a pre-trained model for detecting frontal faces.

---

## Features

- Upload images in `.jpg`, `.jpeg`, or `.png` format
- Face detection using the Haar Cascade classifier
- Display of detected faces with bounding boxes
- Full-width image display in Streamlit

---

## Getting Started

### Prerequisites
Make sure you have the following installed:

```bash
pip install streamlit opencv-python numpy
```

### Folder Structure

Ensure the following structure exists in your project root:

```
project/
│
├── cascade-model/
│   └── haarcascade_frontalface_default.xml
├── app.py        # This file (you're reading now)
└── README.md      # This file
```

---

## Installation

1. Clone or download the project files.
2. Place the `haarcascade_frontalface_default.xml` file in the `cascade-model/` directory.
3. Run the app using Streamlit.

---

## Running Script

Run the following command in your terminal:

```bash
streamlit run app.py
```

---

## Expectations When Running This App

- The user should upload images in `.jpg`, `.jpeg`, or `.png` format.
- If the file is not valid, an error message will be displayed.
- The app will display detected faces with blue rectangles around them.

---

## Demo

Here's what the app does:

1. User uploads a face image.
2. The app loads the image and detects faces using the Haar Cascade classifier.
3. Bounding boxes are drawn around detected faces.
4. The processed image is displayed in the browser.

---

## Acknowledgments

- [OpenCV](https://opencv.org) — For the Haar Cascade algorithm
- [Streamlit](https://streamlit.io) — For building the web app

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Notes

- The `haarcascade_frontalface_default.xml` file is required for face detection. You can download it from [OpenCV's GitHub](https://github.com/opencv/opencv/tree/master/data/hcascades).
- This app is designed to be lightweight and easy to deploy using Streamlit.

---

