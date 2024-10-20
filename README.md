# MNIST Handwritten Digit Recognition App

## Overview

This app is a simple yet powerful demonstration of using a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0-9) from the **MNIST dataset**. The app allows users to draw digits on a canvas and then uses a deep learning model hosted on a Flask back-end to classify the drawn digit with a high level of accuracy.

## Features

- **User Interface (Front-End)**: A simple drawing interface built with **React.js** allows users to draw a digit (0-9).
- **Back-End**: A **Flask API** takes the user's drawing, processes the image, and runs it through a pre-trained **CNN** model for classification.
- **Machine Learning**: The app leverages a **Convolutional Neural Network** (CNN) trained on the MNIST dataset using **PyTorch** to classify the input digits.
- **Real-Time Inference**: The Flask API provides real-time inference by returning the predicted digit to the user almost instantly.

## How It Works

1. The user draws a digit on a canvas using the front-end interface.
2. The drawing is sent as an image to the Flask API hosted on the back end.
3. The image is preprocessed (converted to grayscale, resized to 28x28 pixels).
4. The preprocessed image is passed to the CNN model for classification.
5. The predicted digit is returned to the front-end and displayed to the user.

## Technology Stack

- **Front-End**: React.js, JavaScript, HTML, CSS
- **Back-End**: Flask, Python
- **Machine Learning**: PyTorch, CNN model
- **Cloud & Containerization**: Docker, AWS (Lambda, API Gateway, S3, CloudFront)
  
## Lessons Learned

### 1. **Building a Full Stack ML Application**
   Developing this app allowed me to gain hands-on experience in integrating machine learning with web development. I learned how to serve a machine learning model using a Flask API and connect it with a React-based front-end.

### 2. **Model Deployment and Cloud Integration**
   Deploying the Flask app using **AWS services** (Lambda, CloudFront, and API Gateway) and containerizing the application with **Docker** provided valuable insights into how machine learning models can be operationalized and deployed at scale.

### 3. **Working with Flask and REST APIs**
   I gained experience in creating RESTful APIs using Flask, including handling image data, preprocessing it for the model, and returning real-time results.

### 4. **Frontend-Backend Integration**
   The project taught me how to integrate a React.js front-end with a Flask back-end, ensuring smooth communication between the client and server for real-time inference.

### 5. **Security Best Practices**
   While working on this project, I implemented security best practices such as **input validation** and **securing API endpoints**, which are essential for safeguarding machine learning APIs deployed in production.

## How to Run This Project

### Prerequisites

- **Python 3.x**
- **Flask**
- **PyTorch**
- **React.js**
