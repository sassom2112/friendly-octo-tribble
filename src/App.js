// src/App.js
import React, { useState } from 'react';
import Canvas from './static/canvas';
import ConfidenceBars from './static/ConfidenceBars';
import ActivationGrid from './static/ActivationGrid';
import './DigitRecognizer.css';

function App() {
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('try');

  return (
    <div className="app-container">
      <h1 className="app-title">MNIST Digit Recognition</h1>

      <div className="tab-bar">
        <button
          className={`tab-btn${activeTab === 'try' ? ' active' : ''}`}
          onClick={() => setActiveTab('try')}
        >
          Try It Out
        </button>
        <button
          className={`tab-btn${activeTab === 'github' ? ' active' : ''}`}
          onClick={() => setActiveTab('github')}
        >
          GitHub
        </button>
      </div>

      {activeTab === 'try' && (
        <>
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
          <div className="maps-row">
            <ActivationGrid
              maps={result?.conv1}
              nativeSize={14}
              title="Conv Layer 1 — 32 filters"
              cols={8}
              count={32}
            />
            <ActivationGrid
              maps={result?.conv2}
              nativeSize={7}
              title="Conv Layer 2 — 64 filters"
              cols={8}
              count={64}
            />
          </div>
        </>
      )}

      {activeTab === 'github' && (
        <div className="github-panel">
          <p className="github-desc">
            Source code for the CNN model, training pipeline, and this React app.
          </p>
          <a
            href="https://github.com/sassom2112/friendly-octo-tribble"
            target="_blank"
            rel="noopener noreferrer"
            className="action-button github-link"
          >
            View on GitHub →
          </a>
        </div>
      )}
    </div>
  );
}

export default App;
