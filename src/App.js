import React, { useState } from 'react';
import Canvas from './static/canvas';  // Import your Canvas component

function App() {
  const [prediction, setPrediction] = useState(null);  // Set up state for prediction

  return (
    <div className="App">
      <Canvas setPrediction={setPrediction} />  {/* Pass setPrediction as a prop */}
      {prediction !== null && <h2>Prediction: {prediction}</h2>}  {/* Display the prediction */}
    </div>
  );
}

export default App;
