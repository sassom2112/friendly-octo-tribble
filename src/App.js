// src/App.js
import React, { useState } from 'react';
import Canvas from './static/canvas';
import ConfidenceBars from './static/ConfidenceBars';
import ActivationGrid from './static/ActivationGrid';
import './DigitRecognizer.css';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="app-container">
      <h1 className="app-title">MNIST Digit Recognition</h1>
      <div className="viz-layout">
        <div className="center-panel">
          <Canvas onResult={setResult} />
          {result !== null && (
            <h2 className="prediction">Prediction: {result.prediction}</h2>
          )}
        </div>
        <ConfidenceBars
          confidences={result?.confidences}
          prediction={result?.prediction}
        />
      </div>
      {result && (
        <div className="maps-row">
          <ActivationGrid
            maps={result.conv1}
            nativeSize={14}
            title="Conv Layer 1 — 32 filters"
            cols={8}
          />
          <ActivationGrid
            maps={result.conv2}
            nativeSize={7}
            title="Conv Layer 2 — 64 filters"
            cols={8}
          />
        </div>
      )}
    </div>
  );
}

export default App;
