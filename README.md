# ECG-Anomaly-Detection-using-Autoencoders
Anomaly detection is critical in various domains, especially in healthcare, where early detection of irregularities in ECG signals can significantly impact patient care. The project "ECG Time Series Anomaly Detection using CNN Autoencoder" focuses on using Convolutional Neural Networks (CNNs) and Autoencoders to develop a model that can identify abnormal heartbeats in real-time, potentially improving cardiac health monitoring and patient outcomes.
## Introduction
**1. Relevance of the Project**

- Anomaly Detection in ECG Signals: Detecting anomalies in ECG signals is essential for diagnosing conditions like arrhythmias and myocardial infarction. This project aims to improve the accuracy and efficiency of ECG anomaly detection using deep learning techniques, enabling timely medical intervention.
- Real-time Monitoring: The model's capability to detect anomalies in real-time is crucial for continuous monitoring of patients' cardiac health, allowing immediate response to irregularities.
- Transfer Learning and Feature Clustering: The approach leverages CNNs and Autoencoders validated through feature clustering and transfer learning, enhancing the model's performance and broadening its application across different datasets.

**2. Problem Statement and Objective**
- The primary goal is to accurately detect anomalies in ECG signals, differentiating between normal and abnormal data in real-time. This capability is crucial for improving cardiac health monitoring and patient care.

**3. Scope of the Project**
- Technical Application: The project involves developing and implementing a deep learning model for ECG anomaly detection, including data preprocessing, model building, and evaluation.
- Healthcare Impact: Beyond technical development, the project aims to contribute to cardiac health monitoring by providing a tool for early detection and treatment of heart conditions.
- Research and Development: The project explores new methodologies in ECG anomaly detection, contributing to advancements in healthcare technology and patient care.

## Process
**1. Data Collection**
- The project uses the PTB Diagnostic ECG Database from Physionet, available on Kaggle. The dataset includes normal and abnormal ECG signals, with binary labels indicating the presence of anomalies.

**2. Data Preprocessing**
- Necessary libraries such as NumPy, pandas, and TensorFlow were imported. The dataset was found to be sufficiently clean, requiring minimal preprocessing.

**3. Exploratory Data Analysis (EDA)**
- Comparing Samples: Visual comparisons of random samples from normal and anomalous datasets helped identify potential patterns or differences in their characteristics.
- Smoothed Mean and Confidence Intervals: Plots of the smoothed mean and confidence intervals for both classes provided insights into the distribution and variability of the ECG signals.

![download (1)](https://github.com/user-attachments/assets/9b745e5f-729e-4635-a5bb-030449899210)
![download](https://github.com/user-attachments/assets/0467b98e-cae5-49de-bf85-7f84c700bb56)


**4. Model Building**
- CNN Autoencoder Structure: The model comprises an encoder that compresses ECG data and a decoder that reconstructs it. The encoder captures essential features through convolutional layers, while the decoder reconstructs the original signals.
- Training and Optimization: The model was trained on normal ECG signals, with a focus on minimizing reconstruction error. The training process included early stopping to prevent overfitting, using a batch size of 128.
- Threshold Setting: A threshold was established based on the mean training loss plus one standard deviation to distinguish between normal and anomalous data.

![Autoencoder](https://github.com/user-attachments/assets/c28ba701-168f-47a0-8d40-4f3e2167c9ea)
![download (3)](https://github.com/user-attachments/assets/44b2adde-fedb-4956-8095-f2fe3e10ef37)
![download (2)](https://github.com/user-attachments/assets/bfabcec5-d941-4de0-8712-46fc565015ac)
![download (4)](https://github.com/user-attachments/assets/f7e83770-6edb-4868-87dd-b44a5645ba6c)

**5. Model Evaluation**
- Evaluation Metrics: The model's performance was assessed using accuracy, precision, recall, and F1-score. Visualization tools like confusion matrices and classification reports provided additional insights.
- Function Implementation: Custom functions, such as evaluate_model, prepare_labels, and plot_confusion_matrix, were developed to streamline model evaluation and metric calculation.
- Insights from Evaluation: The evaluation process revealed the model's strengths and areas for improvement, guiding further refinements to enhance anomaly detection accuracy.

![Screenshot 2024-04-23 174920](https://github.com/user-attachments/assets/cece130d-8d1a-4c27-888a-cfb1264ce1d1)


## Conclusion
The project successfully implemented a CNN Autoencoder for detecting anomalies in ECG signals, achieving an accuracy of 80.32%. Despite challenges like data imbalance, the model demonstrated potential in identifying abnormal ECG patterns. These results suggest that the model could significantly improve patient outcomes and healthcare efficiency by enabling early detection of cardiac conditions. Future work will focus on addressing data imbalance and refining the model for even higher accuracy, further contributing to the field of cardiac health monitoring and patient care.
