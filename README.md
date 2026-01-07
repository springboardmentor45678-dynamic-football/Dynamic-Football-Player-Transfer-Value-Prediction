# CircuitGuard: PCB Defect Detection

## Overview

**CircuitGuard** is an AI-powered system that automatically detects and classifies defects in **Printed Circuit Boards (PCBs)** using deep learning and computer vision. It replaces manual visual inspection with a fast, accurate, and scalable solution suitable for modern electronics manufacturing.

## Problem Statement

Manual PCB inspection is time-consuming, inconsistent, and prone to human error. Even small defects can cause device failure and financial loss.

## Solution

CircuitGuard uses a **YOLO-based object detection model** served via a **FastAPI backend** and an interactive **Streamlit frontend** to perform real-time PCB defect detection.

## Key Features

* Automated PCB defect detection
* YOLO-based deep learning model
* FastAPI inference API
* Streamlit web interface
* Real-time visualization with bounding boxes

## Tech Stack

* **ML/DL:** YOLO, PyTorch, OpenCV
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Language:** Python

## System Flow

1. Upload PCB image via web UI
2. Image sent to FastAPI backend
3. YOLO model performs defect detection
4. Results returned and visualized

## Use Cases

* Electronics manufacturing quality control
* Automated Optical Inspection (AOI)
* Smart factory and Industry 4.0 systems

## Outcome

Delivered an end-to-end AI application demonstrating:

* Applied computer vision
* Production-style API design
* Clean ML-to-UI integration

## Future Scope

* Edge deployment
* Batch inspection
* Defect analytics dashboard
