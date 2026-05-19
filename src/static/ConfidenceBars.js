// src/static/ConfidenceBars.js
import React from 'react';

function ConfidenceBars({ confidences, prediction }) {
  return (
    <div className="confidence-panel">
      <h3 className="section-title">Confidence</h3>
      {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((digit) => {
        const pct = confidences ? confidences[digit] * 100 : 0;
        const isPredicted = prediction === digit;
        return (
          <div key={digit} className={`bar-row${isPredicted ? ' bar-row--predicted' : ''}`}>
            <span className="digit-label">{digit}</span>
            <div className="bar-track">
              <div
                className={`bar-fill${isPredicted ? ' bar-fill--predicted' : ''}`}
                style={{ width: `${pct.toFixed(1)}%` }}
              />
            </div>
            <span className="bar-pct">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default ConfidenceBars;
