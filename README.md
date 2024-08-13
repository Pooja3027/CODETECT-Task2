**Name:** POOJASHREE N

**Company:** CODTECH IT SOLUTIONS

**ID:** CT08DS4780

**Domain:** ARTIFICIAL INTELLIGENCE

**Duration:** July To August 2024

## Overview of the Project:

### Project: CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

**Objective:**
The goal of this project is to develop a Convolutional Neural Network (CNN) model using TensorFlow and Keras to classify images from the CIFAR-10 dataset into one of ten predefined classes.

**Dataset:**
- **CIFAR-10**: A dataset consisting of 60,000 32x32 color images across 10 different classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It is divided into 50,000 training images and 10,000 testing images.

**Key Steps Involved:**

1. **Data Preprocessing:**
   - The CIFAR-10 dataset is loaded and divided into training and testing datasets.
   - Pixel values of images are normalized to the range [0, 1] to facilitate better training of the neural network.

2. **Model Architecture:**
   - A Sequential CNN model is constructed using Keras, consisting of the following layers:
     - **Convolutional Layers**: Three Conv2D layers are used with increasing filters (32, 64, 64) and a kernel size of (3,3). Each convolutional layer is followed by a ReLU activation function.
     - **Pooling Layers**: Two MaxPooling2D layers with a pool size of (2, 2) are used after the first two convolutional layers to reduce the spatial dimensions.
     - **Flatten Layer**: The output from the last convolutional layer is flattened into a 1D array.
     - **Dense Layers**: Two fully connected Dense layers are used. The first has 64 neurons with a ReLU activation, and the final layer has 10 neurons representing the class scores.

3. **Model Compilation and Training:**
   - The model is compiled with the Adam optimizer, Sparse Categorical Crossentropy loss function (with logits), and accuracy as the evaluation metric.
   - The model is trained for 10 epochs on the training data, with validation performed on the test data.

4. **Model Evaluation:**
   - After training, the model is evaluated on the test dataset, achieving an accuracy score, which is printed.
   - A plot is generated to visualize the accuracy and validation accuracy across epochs.

5. **Classification and Prediction:**
   - The model is used to predict labels for the test images.
   - The predicted class labels are compared against the true labels, and a classification report is generated to provide detailed metrics such as precision, recall, and F1-score for each class.
   - Additionally, the predicted label for a specific test image (e.g., the 3rd image) is displayed to showcase the model's capability.

**Outcome:**
The project demonstrates the effectiveness of CNNs in image classification tasks. The model successfully classifies images from the CIFAR-10 dataset with a reasonably good accuracy and provides insights into the model's performance through detailed metrics and visualization.


### Result:

![Screenshot 2024-08-14 002301](https://github.com/user-attachments/assets/4b673371-c194-45f5-809e-428e98d1b01e)

### classification_report with Test Accuracy

![Screenshot 2024-08-14 002319](https://github.com/user-attachments/assets/acf78228-159c-4bef-b038-1747e4733ca8)



