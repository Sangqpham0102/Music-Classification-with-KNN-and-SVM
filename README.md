# Vietnamese Traditional Music Classification

- Course project
- This project focuses on classifying various genres of Vietnamese traditional music using two classification algorithms: K-Nearest Neighbors (KNN) and Support Vector Machine (SVM).
- Based on audio features such as Spectral Centroid, Rolloff, Flux, Zero Crossing, Low Energy, and MFCC.
- Completed on 10/06/2024.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Implementation Process](#implementation-process)
- [Program Interface](#program-interface)
- [References](#references)

## Introduction
- This project aims to build a machine learning model to classify genres of Vietnamese traditional music, including Cai Luong, Ca Tru, Chau Van, Cheo, and Xam.
- We utilize Python libraries for audio processing and machine learning, such as librosa for feature extraction and TensorFlow/Keras for model construction and training.

## Setup
- To set up this project, you need to install Python 3.7.10 and the following necessary libraries:
  ```bash
  numpy
  librosa
  tensorflow 
  matplotlib
  pydub
  sklearn
  seaborn
- Clone the repository:
  ```bash
  https://github.com/Sangqpham0102/Machine-learning-project.git
  cd Machine-learning-project

## Implementation Process
- Prepare the data
- Extract features
- Train the model
- Evaluate
- Predict genre
Details are provided in the file: [Audi_training](https://github.com/Sangqpham0102/Music-Classification-with-KNN-and-SVM/blob/master/Audi_Training.ipynb)

- Run the Flask application [app.py](https://github.com/Sangqpham0102/Music-Classification-with-KNN-and-SVM/blob/48674b6d78a589024313513d7f1d16a6748e715c/app.py), upload new audio, and receive predictions.

## Program Interface
![image](https://github.com/Sangqpham0102/Machine-learning-project/assets/119334855/8cc0a27a-b3ca-4112-b1ef-5a960ee8a3c9)

## References
[1] LTPhat/ Phân loại Việt-Truyền thống-Âm nhạc-Phân loại, https://github.com/LTPhat/Vietnamese-Traditional-Music-Classification

[2] Librosa Library, https://librosa.org/doc/latest/index.html

[3] TensorFlow, https://www.tensorflow.org/
