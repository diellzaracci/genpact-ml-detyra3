# Designing & Implementing an SOD System

## Overview

This project implements a Salient Object Detection (SOD) system using deep learning and convolutional neural networks (CNNs). The goal of SOD is to detect and segment the most visually important object or region within an image.

The project was developed using PyTorch in Google Colab and trained on the DUTS dataset. The system includes preprocessing, augmentation, model training, evaluation, experimentation, visualization, and an interactive Gradio demo.

---

## Features

* CNN encoder–decoder architecture implemented from scratch
* DUTS dataset preprocessing and augmentation
* BCE + IoU combined loss function
* IoU, Precision, Recall, F1-score, and MAE evaluation metrics
* Early stopping and checkpoint saving
* Experimental analysis with different configurations
* Gradio web demo for image upload and prediction

---

## Dataset

The project uses the DUTS dataset:

* DUTS-TR: 10,553 training images
* DUTS-TE: 5,019 testing images

Dataset structure:

```text
DUTS-TR/
├── DUTS-TR-Image
├── DUTS-TR-Mask

DUTS-TE/
├── DUTS-TE-Image
├── DUTS-TE-Mask
```

---

## Model Architecture

The implemented model follows a CNN encoder–decoder architecture.

### Encoder

* Convolutional layers
* Batch normalization
* ReLU activation
* Max pooling

### Decoder

* Transposed convolution layers
* Upsampling
* Sigmoid output layer

The final output is a single-channel saliency mask.

---

## Training Configuration

| Parameter     | Value     |
| ------------- | --------- |
| Image Size    | 224 × 224 |
| Optimizer     | Adam      |
| Learning Rate | 1e-3      |
| Epochs        | 25        |
| Loss Function | BCE + IoU |

---

## Results

| Model        |    IoU | Precision | Recall | F1-score |    MAE |
| ------------ | -----: | --------: | -----: | -------: | -----: |
| Baseline     | 0.5615 |    0.7124 | 0.7290 |   0.7159 | 0.2001 |
| Experiment 1 | 0.5609 |    0.7118 | 0.7288 |   0.7154 | 0.2001 |
| Experiment 2 | 0.1926 |    0.7308 | 0.2083 |   0.3157 | 0.2778 |

---

## Demo

A Gradio-based demo application was created to allow users to upload custom images and generate:

* Predicted saliency masks
* Overlay visualizations
* Inference times

Run the demo:

```python
!python app.py
```

---

## Project Structure

```text
project/
├── app.py
├── data_loader.py
├── evaluate.py
├── train.py
├── sod_model.py
├── sistemi_SOD.ipynb
├── checkpoints/
├── outputs/
└── README.md
```

---

## Installation

Install dependencies:

```python
!pip install torch torchvision opencv-python gradio matplotlib
```

---

## How to Run

### 1. Upload Dataset

Upload the DUTS dataset zip file to Google Colab.

### 2. Extract Dataset

```python
import zipfile

with zipfile.ZipFile("duts_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("duts_dataset")
```

### 3. Train Model

```python
!python train.py
```

### 4. Evaluate Model

```python
!python evaluate.py
```

### 5. Run Demo

```python
!python app.py
```

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Gradio
* Google Colab

---

## Author

Diellza Raçi
