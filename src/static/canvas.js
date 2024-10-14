import React, { useRef, useState } from 'react';

function Canvas() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineWidth = 10;
    ctx.strokeStyle = 'black';
    ctx.beginPath();
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.closePath();
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset the prediction state after clearing the canvas
    setPrediction(null);
  };

  const submitDrawing = () => {
    const canvas = canvasRef.current;
    canvas.toBlob((blob) => {
      const formData = new FormData();
      formData.append('image', blob);

      fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => {
          setPrediction(data.prediction); // Update the prediction state
        })
        .catch((err) => {
          console.error('Error:', err);
        });
    }, 'image/png');
  };

  return (
    <div>
      <h1>Draw a Digit</h1>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: '1px solid black' }}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
      />
      <br />
      <button onClick={clearCanvas}>Clear</button>
      <button onClick={submitDrawing}>Submit</button>
      {prediction !== null && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default Canvas;
