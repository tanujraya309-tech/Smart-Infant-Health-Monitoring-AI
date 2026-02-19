# Smart Infant Health Monitoring System using AI

## Project Description
In this project, we built a machine learning model to automatically classify different types of baby cries. The goal was to understand what a baby might be expressing — such as hunger, pain, burping, fear (scared), or non-cry sounds — based purely on audio signals.
To achieve this, we used a Random Forest classifier, which is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It works well with complex, high-dimensional data, making it suitable for audio-based classification tasks.

## Features
- Baby cry classification
- Audio signal processing
- Real-time prediction 
- Alert system (future scope)

## Feature Extraction

Since raw audio signals cannot be directly fed into a machine learning model, we first processed the audio files using Librosa, a popular Python library for music and audio analysis.
To capture meaningful patterns from the audio, we extracted several acoustic features:
MFCCs (Mel-Frequency Cepstral Coefficients) – 13 coefficients to represent the timbral characteristics of the sound
Delta features – First-order derivatives of MFCCs (captures change over time)
Delta-Delta features – Second-order derivatives (captures acceleration of change)
Chroma features – Represents energy distribution across pitch classes
Zero Crossing Rate (ZCR) – Measures how frequently the signal changes sign
Spectral Centroid – Indicates where the “center of mass” of the spectrum lies (brightness of sound)

##Model Training
After extracting and organizing the features into a structured dataset, we trained a Random Forest classifier to predict the cry category. The model learns patterns in the acoustic features and associates them with specific cry types.
 Outcome
The final system can classify baby cries into categories like:
Hunger
Pain
Burping
Scared
Non-cry
##WITH 
Model Accuracy:69%
macro avg:59%
weighted avg:67%

This project demonstrates how audio signal processing and machine learning can work together to build intelligent systems that could potentially assist parents or caregivers in understanding a baby’s needs more effectively.
C## Status
Project under development,using various model for the best training purpose and also for the best results
🚧
