from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import io
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import your model architecture
class SimpleClassifier3Layer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(SimpleClassifier3Layer, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# Load the trained model
input_size = 28 * 28
hidden_size1 = 64
hidden_size2 = 64
hidden_size3 = 64
num_classes = 10
model = SimpleClassifier3Layer(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
model.load_state_dict(torch.load("model.pth"))  # Load your trained model
model.eval()  # Set the model to evaluation mode

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the POST request
    file = request.files['image'].read()
    img = Image.open(io.BytesIO(file)).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 like MNIST
    img = np.array(img) / 255.0  # Normalize the image
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Predict with the model
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return jsonify({'prediction': int(predicted.item())})

if __name__ == '__main__':
    app.run(debug=True)
