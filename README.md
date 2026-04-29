---
title: RetailLens AI
emoji: 🛒
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8001
pinned: false
---

# 🛒 RetailLens-AI
**Advanced Retail Product Detection & Inventory Analysis Dashboard**

RetailLens-AI is a full-stack, production-ready machine learning pipeline deployed on Hugging Face Spaces. It specializes in high-speed, SKU-level product detection and visual similarity grouping for retail shelf imagery.

## 🚀 Live Demo
The application is live and hosted on Hugging Face Spaces:
**[View RetailLens-AI on Hugging Face](https://huggingface.co/spaces/Rkx-01/RetailLens-AI)**

## ✨ Core Features
- **Object Detection (YOLOv8n)**: High-speed, CPU-optimized bounding box detection tailored for densely packed retail shelves.
- **Visual Similarity Grouping (MobileNetV2)**: AI-driven feature extraction that visually clusters identical products together, assigning unique Group IDs without requiring barcode scans.
- **Robust Production Server**: Powered by a multi-threaded Gunicorn WSGI server, guaranteeing zero timeouts and instant health-check responses under heavy inference loads.
- **Hardware Optimized**: Aggressively configured PyTorch OpenMP threading to prevent Docker container thread explosions and memory thrashing.
- **Premium User Interface**: A responsive, glassmorphism-styled web dashboard with dynamic sample galleries, real-time metrics, and instant visual downloads.

## 🏗️ Architecture Stack
- **Backend API**: Python 3.10, Flask, Gunicorn
- **Computer Vision**: OpenCV (Headless), NumPy
- **Deep Learning**: PyTorch, Torchvision, Ultralytics YOLOv8
- **Clustering**: Scikit-Learn (DBSCAN)
- **Deployment**: Docker, Hugging Face Spaces

## 💻 Run Locally

### Prerequisites
- Python 3.9+
- Git

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rkx-01/AI-Retail-Product-Detection.git
   cd AI-Retail-Product-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   # Run via Gunicorn for production parity (Recommended)
   gunicorn -b 0.0.0.0:8001 -w 1 --threads 4 --timeout 120 app:app
   
   # OR run the standard Flask dev server
   python app.py
   ```

4. **Access the Dashboard:**
   Open your browser and navigate to `http://127.0.0.1:8001`

## 📡 API Endpoints

### `POST /process`
Upload an image directly from your local machine to run the detection pipeline.
- **Body:** `multipart/form-data` containing an `image` file.
- **Response:** JSON containing bounding boxes, group IDs, confidence scores, and a URL to download the visualized image.

### `POST /process-url`
Analyze an image hosted on a remote server (e.g., GitHub raw content).
- **Body:** JSON `{"url": "https://..."}`
- **Response:** JSON containing the analysis results.

## 🛡️ Production Stability Notes
This repository implements strict resource management constraints required for cloud-hosted Docker containers:
- **Lazy Loading**: PyTorch models are loaded lazily to satisfy instantaneous container boot health checks.
- **Non-Root Execution**: Implements a dedicated `user` with UID 1000 for strict Hugging Face security compliance.

## 🔮 Future Implementation
The architecture has been designed to support rapid scaling and the integration of more advanced machine learning models in future iterations:
- **RF-DETR Integration**: Upgrading the bounding box detection backend from YOLOv8 to RT-DETR (Real-Time DEtection TRansformer) for superior recall in heavily occluded retail shelf environments.
- **SKU-110K Fine-Tuning**: Training the detection model specifically on the SKU-110K dataset to specialize in dense object detection scenarios rather than generalist categories.
- **Brand & Logo Recognition**: Integrating OCR (Optical Character Recognition) and zero-shot logo classification to automatically identify specific brands and variants within the grouped products.
- **Database Integration**: Connecting the grouping pipeline to a PostgreSQL or Pinecone vector database to track product stock and anomalies across multiple shelves and time periods.
