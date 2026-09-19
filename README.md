# Skin Cancer Detection Web Application

A deep learning-based web application for binary classification of skin lesion images as **benign** or **malignant**.

The project uses transfer learning with MobileNetV2 for image classification and serves the trained model through a Flask web application. The model was trained using skin lesion images from the International Skin Imaging Collaboration (ISIC) dataset and compared against alternative deep learning configurations during experimentation.

## Project Overview

The objective of this project was to explore the use of convolutional neural networks and transfer learning for automated skin lesion classification.

The project involved:

* preparing benign and malignant skin lesion images for training and testing
* comparing ResNet and MobileNetV2-based approaches
* fine-tuning MobileNetV2
* evaluating the resulting models
* exporting the selected trained model
* integrating the model with a Flask web application 

The final application allows a user to upload a skin lesion image and returns a prediction indicating whether the lesion is classified as benign or malignant.

## Model Development

The dataset is divided into two classes:

```text
Benign
Malignant
```

Images are converted to RGB format and organised into separate training and testing sets.

The notebook contains:

```text
Training images: 2,637
Testing images:   660
Total images:     3,297
```

Images are processed at:

```text
224 × 224 × 3
```

which matches the input dimensions used by the deployed model.

## Transfer Learning

The project explores transfer learning rather than training a convolutional neural network entirely from scratch.

Multiple model configurations were evaluated, including:

* ResNet-50
* MobileNetV2 with a linear output configuration
* MobileNetV2 with a sigmoid-based configuration
* Fine-tuned MobileNetV2

The experiments showed substantially stronger classification performance with MobileNetV2 than with the ResNet configuration used.

The best MobileNetV2 configurations achieved approximately:

```text
Test Accuracy: 83.94%
```

The model comparison recorded in the notebook includes:

```text
ResNet                         ~68.33%
MobileNetV2 with linear        ~82.88%
MobileNetV2 with sigmoid       ~83.94%
Fine-tuned MobileNetV2         ~83.94%
```

## Why MobileNetV2?

MobileNetV2 was particularly useful for this project because it provides a comparatively lightweight convolutional architecture while retaining strong image-classification performance.

This was important not only from an accuracy perspective, but also because a smaller trained model is more practical for deployment in resource-constrained environments such as lightweight web or mobile applications.

The trained model used in the broader study was approximately **9.2 MB**, which motivated further consideration of model size and deployability alongside predictive performance.

## Web Application

The trained model is integrated into a Flask application.

The application follows this inference flow:

1. The user uploads an image of a skin lesion.
2. The image is saved temporarily by the Flask application.
3. The image is opened using Pillow.
4. It is resized to `224 × 224`.
5. The image is reshaped into the format expected by the neural network:

```python
(1, 224, 224, 3)
```

6. The trained Keras model generates a prediction.
7. The predicted class is mapped to:

```text
0 → Benign
1 → Malignant
```

8. The prediction is displayed in the web interface.

## Project Structure

```text
skin-cancer-detection-webapp/
│
├── model/
│   └── model.h5
│
├── static/
│
├── templates/
│
├── SkinCancer_Classification_MobileNetV2.ipynb
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

### `SkinCancer_Classification_MobileNetV2.ipynb`

Contains the deep-learning experimentation and model-development workflow, including:

* loading and preparing image data
* creation of benign/malignant labels
* train/test preparation
* transfer-learning experiments
* ResNet-50 experimentation
* MobileNetV2 experimentation
* model training
* fine-tuning
* evaluation
* comparison of model performance
* serialisation of the trained model

### `app.py`

Contains the Flask application used to serve the trained model.

It:

* loads the saved Keras model
* receives image uploads
* resizes images to `224 × 224`
* prepares the input tensor
* performs inference
* converts model output into a benign/malignant label
* renders the result in the browser


## Installation

Clone the repository:

```bash
git clone https://github.com/Ritwika101/skin-cancer-detection-webapp.git
cd skin-cancer-detection-webapp
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Set the Flask application:

### Windows

```bash
set FLASK_APP=app.py
```

### macOS/Linux

```bash
export FLASK_APP=app.py
```

Start the Flask server:

```bash
flask run
```

Then open:

```text
http://127.0.0.1:5000/
```

Upload a skin lesion image to generate a classification.


## Research Extension

This classification project later formed part of a broader investigation into automated skin cancer detection.

The work was extended beyond image classification to:

* object detection using multiple YOLOv5 variants
* lesion localisation using bounding boxes
* image segmentation using U-Net
* experimentation with MobileNetV2 and ResNet-50 backbones for U-Net

This broader work was later developed into the research paper:

**“Automation of Skin Cancer Detection with Image Processing Using Efficient and Lightweight CNN Models”**

published at the **IEEE International Conference on Pervasive Computing and Social Networking (ICPCSN), 2023**.

## Author

**Ritwika Pal**

