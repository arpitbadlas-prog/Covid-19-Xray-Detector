# 🩺 COVID-19 Chest X-ray Detection using Deep Learning

An end-to-end Deep Learning project that detects **COVID-19, Normal, and Viral Pneumonia** cases from chest X-ray images using Convolutional Neural Networks (CNN) and Transfer Learning.

The project compares multiple deep learning architectures, evaluates their performance, and identifies the best model for medical image classification.

---

# 📌 Project Overview

Early detection of COVID-19 is crucial for timely treatment and reducing disease transmission. Chest X-ray imaging combined with deep learning provides an efficient approach for assisting medical diagnosis.

The primary objectives of this project are to:

✅ Classify chest X-ray images into multiple disease categories

✅ Compare different CNN architectures

✅ Improve model performance using Transfer Learning

✅ Reduce overfitting through data augmentation and regularization

---

# 🧠 Deep Learning Problem Type

This is a **Multi-Class Image Classification** problem.

### Target Classes

```text
COVID-19
Normal
Viral Pneumonia
```

---

# 📂 Dataset Information

The dataset contains labeled chest X-ray images belonging to different respiratory conditions.

### Image Categories

* COVID-19
* Normal
* Viral Pneumonia

Each image is resized and preprocessed before being used for model training.

---

# 🔍 Data Preprocessing

Several preprocessing techniques were applied before training:

* Image resizing
* Image normalization
* Label Encoding
* One-Hot Encoding
* Train, Validation and Test Split
* Data Augmentation
* Batch Generation

### Data Augmentation Techniques

* Random Rotation
* Width Shift
* Height Shift
* Zoom
* Shear Transformation
* Horizontal Flip

These techniques improve model generalization and reduce overfitting.

---

# ⚙️ Models Developed

Multiple deep learning models were trained and compared.

### Model 1

* Custom CNN

### Model 2

* VGG16 Transfer Learning

### Model 3

* VGG16 + Data Augmentation

### Additional Experiments

* CNN with Class Weights
* Tuned CNN Architecture
* Hyperparameter Optimization using Keras Tuner

---

# 📊 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Validation Accuracy
* Loss
* ROC-AUC Score
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

---

# 📈 Model Comparison

Different deep learning models were compared based on:

* Test Accuracy
* Validation Performance
* ROC-AUC Score
* Generalization Ability
* Overfitting Analysis

The best-performing model was selected after comparing multiple CNN architectures and transfer learning approaches.

---

# 🚀 Features

✨ Multi-class chest X-ray classification

✨ Transfer Learning using VGG16

✨ Custom CNN architecture

✨ Data augmentation pipeline

✨ Hyperparameter tuning

✨ ROC Curve analysis

✨ Confusion Matrix visualization

✨ Performance comparison of multiple models

---

# 📊 Visualizations Included

### 📌 Class Distribution

Displays the number of images in each class.

### 📌 Sample X-ray Images

Visualizes examples from different disease categories.

### 📌 Augmented Images

Shows the effect of data augmentation techniques.

### 📌 Training & Validation Accuracy

Tracks model learning performance across epochs.

### 📌 Training & Validation Loss

Evaluates convergence during training.

### 📌 ROC Curves

Compares ROC-AUC performance for all classes.

### 📌 Confusion Matrix

Visualizes prediction performance across disease categories.

### 📌 Model Comparison

Compares all trained deep learning models based on evaluation metrics.

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# 📚 Deep Learning Concepts Covered

* Convolutional Neural Networks (CNN)
* Transfer Learning
* VGG16
* Data Augmentation
* Batch Normalization
* Dropout Regularization
* Early Stopping
* Learning Rate Scheduling
* Hyperparameter Tuning
* Medical Image Classification

---

# 🔮 Future Improvements

Potential enhancements include:

* Fine-tuning deeper pre-trained models (ResNet50, EfficientNet, DenseNet)
* Explainable AI using Grad-CAM
* Real-time chest X-ray prediction system
* Deployment using Streamlit or Flask
* Larger medical imaging datasets
* Integration into clinical decision-support systems

---

# 🎯 Conclusion

This project demonstrates how Deep Learning techniques can accurately classify chest X-ray images into **COVID-19, Normal, and Viral Pneumonia** categories.

The project combines:

* Image Preprocessing
* Data Augmentation
* CNN Development
* Transfer Learning
* Hyperparameter Optimization
* Model Evaluation
* Performance Comparison

into a complete end-to-end medical image classification solution.
