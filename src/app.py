# app.py 
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import awsgi


app = Flask(__name__)
CORS(app)

# Define the CNN model (same as in mnist.py)
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Output: 32 x 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # Output: 32 x 14 x 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # Output: 64 x 7 x 7

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# Instantiate the model
model = CNNClassifier()

# Load the trained model weights
try:
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

model.eval()  # Set model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = data.get('pixels', None)  # Safely get 'pixels' from JSON

    if pixels is None or not isinstance(pixels, list) or len(pixels) != 784:
        return jsonify({'error': 'Invalid input data. Expecting a list of 784 pixel values.'}), 400

    # Convert list to numpy array and reshape to [1, 28, 28]
    try:
        pixels = np.array(pixels, dtype=np.float32).reshape(1, 28, 28)
    except ValueError:
        return jsonify({'error': 'Pixel data could not be reshaped to 28x28.'}), 400

    # Normalize the pixels (same as during training)
    pixels = (pixels - 0.5) / 0.5  # Normalize to [-1, 1]

    # Convert to torch tensor and add batch and channel dimensions
    input_tensor = torch.from_numpy(pixels).unsqueeze(0)  # Shape: [1, 1, 28, 28]

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})

# Lambda handler
def lambda_handler(event, context):
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': ''
        }

    if event['httpMethod'] == 'POST' and event['path'] == '/predict':
        response = awsgi.response(app, event, context)
        response['headers']['Access-Control-Allow-Origin'] = '*'
        return response
    else:
        return {
            'statusCode': 404,
            'body': 'Not Found'
        }



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
