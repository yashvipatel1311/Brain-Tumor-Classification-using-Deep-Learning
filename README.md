# Brain Tumor Classification Using Deep Learning

## Overview
This project focuses on classifying brain tumors from MRI scans using Deep Learning.
A Convolutional Neural Network (CNN) is trained to identify different types of brain tumors
and provide accurate predictions through an interactive web application.

The system classifies MRI images into four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

## Motivation
Brain tumor diagnosis through manual MRI analysis can be time-consuming and prone to errors.
Deep learning models can automatically learn patterns from MRI images and assist doctors
in making faster and more accurate decisions.

## Dataset
- Brain Tumor MRI Dataset (Kaggle)
- Four classes: Glioma, Meningioma, Pituitary, No Tumor
- Images split into training and testing sets

## Methodology
- Image resizing to 150×150 pixels
- Normalization of pixel values
- Data augmentation (rotation, zoom, shifts, flips)
- CNN model with multiple convolutional and pooling layers
- Dropout used to reduce overfitting
- Softmax output layer for multi-class classification

## Model Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 50
- Batch Size: 32

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results
- Test Accuracy: ~96%
- High precision, recall, and F1-score across all classes
- Model shows reliable performance in distinguishing tumor types

## Web Application
- Built using Streamlit
- Users can upload MRI images
- Real-time tumor classification output
- Simple and user-friendly interface

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV
- Streamlit
- Matplotlib, Seaborn

## Limitations
- Dataset imbalance among tumor classes
- Limited dataset size
- Model trained from scratch without transfer learning

## Future Scope
- Use larger and more diverse datasets
- Apply transfer learning (VGG, ResNet, EfficientNet)
- Improve explainability using Grad-CAM
- Deploy as a real-time clinical support system
