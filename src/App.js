// src/App.js
import React, { useState } from 'react';
import Canvas from './static/canvas'; // Import your Canvas component
import './DigitRecognizer.css'; // Import the custom stylesheet

function App() {
  const [prediction, setPrediction] = useState(null); // Set up state for prediction

  return (
    <div className="app-container"> {/* Applies overall page styling */}
      <h1 className="app-title">MNIST Digit Recognition</h1> {/* Added Title */}
      <Canvas setPrediction={setPrediction} /> {/* Pass setPrediction as a prop */}
      {prediction !== null && <h2 className="prediction">Prediction: {prediction}</h2>} {/* Display the prediction */}
    </div>
  );
}

export default App;
