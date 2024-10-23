// src/static/canvas.js
import React, { useRef, useState, useEffect } from 'react';
import '../../src/DigitRecognizer.css'; // Ensure the CSS is imported here if needed
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'; // If using icons
import { faEraser, faPaperPlane } from '@fortawesome/free-solid-svg-icons'; // Icon imports

function Canvas({ setPrediction }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const GRID_SIZE = 28; // 28x28 grid size (for data processing)

  // Initialize the canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true }); // Added willReadFrequently
    ctx.fillStyle = 'black'; // Set background to black
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.lineCap = 'round';
    ctx.lineWidth = 20; // Increase line width for thicker digits
    ctx.strokeStyle = 'white'; // Set drawing color to white
    ctx.beginPath();
    ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.closePath();
    setIsDrawing(false);
  };

  const captureGridPixels = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const canvasWidth = canvas.width;
    const cellSize = canvasWidth / GRID_SIZE;
    const pixels = [];

    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        const imageData = ctx.getImageData(
          j * cellSize,
          i * cellSize,
          cellSize,
          cellSize
        );
        let total = 0;
        for (let k = 0; k < imageData.data.length; k += 4) {
          // Sum the grayscale values
          const r = imageData.data[k];
          const g = imageData.data[k + 1];
          const b = imageData.data[k + 2];
          const alpha = imageData.data[k + 3];
          const grayscale = (r + g + b) / 3;
          total += (grayscale / 255) * (alpha / 255);
        }
        const average = total / (imageData.data.length / 4);
        const pixelValue = average; // No inversion needed
        pixels.push(pixelValue);
      }
    }

    return pixels; // Return pixels for immediate use
  };

  const centerImage = (pixels) => {
    // Convert the 1D pixels array back to a 2D array
    let image = [];
    for (let i = 0; i < GRID_SIZE; i++) {
      image.push(pixels.slice(i * GRID_SIZE, (i + 1) * GRID_SIZE));
    }

    // Find the bounding box of the digit
    let rows = image.length;
    let cols = image[0].length;
    let xmin = cols,
      xmax = -1,
      ymin = rows,
      ymax = -1;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (image[y][x] > 0.1) {
          // Threshold to ignore noise
          if (x < xmin) xmin = x;
          if (x > xmax) xmax = x;
          if (y < ymin) ymin = y;
          if (y > ymax) ymax = y;
        }
      }
    }

    // Handle empty drawing
    if (xmin > xmax || ymin > ymax) {
      return new Array(rows * cols).fill(0);
    }

    // Crop the image to the bounding box
    let cropped = [];
    for (let y = ymin; y <= ymax; y++) {
      cropped.push(image[y].slice(xmin, xmax + 1));
    }

    // Create a new GRID_SIZE x GRID_SIZE image and center the cropped digit
    let centered = new Array(GRID_SIZE)
      .fill(0)
      .map(() => new Array(GRID_SIZE).fill(0));
    let offsetX = Math.floor((GRID_SIZE - (xmax - xmin + 1)) / 2);
    let offsetY = Math.floor((GRID_SIZE - (ymax - ymin + 1)) / 2);

    for (let y = 0; y < cropped.length; y++) {
      for (let x = 0; x < cropped[0].length; x++) {
        centered[y + offsetY][x + offsetX] = cropped[y][x];
      }
    }

    // Flatten the centered image back to a 1D array
    return centered.flat();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black'; // Clear with black background
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null); // Clear prediction
  };

  const submitDrawing = () => {
    // Capture the latest pixels
    let flattenedGrid = captureGridPixels();

    if (!flattenedGrid) {
      console.error('flattenedGrid is undefined after captureGridPixels.');
      return;
    }

    // Center the image
    flattenedGrid = centerImage(flattenedGrid);

    if (!flattenedGrid) {
      console.error('flattenedGrid is undefined after centerImage.');
      return;
    }

    // Validate that we have 784 pixel values
    if (flattenedGrid.length !== 28 * 28) {
      console.error(
        'Invalid grid data length after centering:',
        flattenedGrid.length
      );
      return;
    }

    // Send the flattened grid to the backend
    fetch('https://ctjpw1q0q3.execute-api.us-west-1.amazonaws.com/prod/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json', // Make sure to send JSON
      },
      body: JSON.stringify({ pixels: flattenedGrid }), // Send pixel data as JSON
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error('Network response was not ok');
        }
        return res.json();
      })
      .then((data) => {
        setPrediction(data.prediction); // Set the prediction from the backend
      })
      .catch((err) => {
        console.error('Error:', err);
      });
  };

  return (
    <div className="action-card"> {/* Applies card styling */}
      <div className="canvas-container"> {/* New container for overlay */}
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="digit-canvas" // Applies canvas-specific styling
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing} // Handle mouse leaving the canvas
        />
        <div className="overlay"> {/* Overlay Text */}
          Draw a number between 0 and 9
        </div>
      </div>
      <div className="button-group"> {/* Container for buttons */}
        <button onClick={clearCanvas} className="action-button" aria-label="Clear Canvas">
          <FontAwesomeIcon icon={faEraser} /> Clear
        </button>
        <button onClick={submitDrawing} className="action-button" aria-label="Submit Drawing">
          <FontAwesomeIcon icon={faPaperPlane} /> Submit
        </button>
      </div>
      {/* Removed the prediction display from here */}
    </div>
  );
}

export default Canvas;
