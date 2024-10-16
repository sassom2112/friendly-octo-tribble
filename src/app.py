# app.py 
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

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

# Load the trained model weights with weights_only=True
model = CNNClassifier()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()  # Set model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = data['pixels']  # Should be a list of 784 pixel values

    # Convert list to numpy array and reshape to [1, 28, 28]
    pixels = np.array(pixels, dtype=np.float32).reshape(1, 28, 28)

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

if __name__ == '__main__':
    app.run(debug=True)
