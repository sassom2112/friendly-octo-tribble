// src/static/ActivationGrid.js
import React, { useEffect, useRef } from 'react';

// black → red → yellow → white
function hotColormap(val) {
  return [
    Math.min(255, Math.round(val * 3 * 255)),
    Math.min(255, Math.max(0, Math.round((val * 3 - 1) * 255))),
    Math.min(255, Math.max(0, Math.round((val * 3 - 2) * 255))),
  ];
}

function FeatureMap({ data, nativeSize }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(nativeSize, nativeSize);

    for (let y = 0; y < nativeSize; y++) {
      for (let x = 0; x < nativeSize; x++) {
        const [r, g, b] = hotColormap(data[y][x]);
        const idx = (y * nativeSize + x) * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [data, nativeSize]);

  return (
    <canvas
      ref={canvasRef}
      width={nativeSize}
      height={nativeSize}
      className="feature-map"
    />
  );
}

function ActivationGrid({ maps, nativeSize, title, cols = 8, count }) {
  const total = maps ? maps.length : count;

  return (
    <div className="activation-grid">
      <h3 className="section-title">{title}</h3>
      <div className="maps-grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {maps
          ? maps.map((map, i) => <FeatureMap key={i} data={map} nativeSize={nativeSize} />)
          : Array.from({ length: total }, (_, i) => (
              <div key={i} className="feature-map feature-map--placeholder" />
            ))}
      </div>
    </div>
  );
}

export default ActivationGrid;
