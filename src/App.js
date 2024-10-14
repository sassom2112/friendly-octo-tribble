import React from 'react';
import './App.css';
import Canvas from './static/canvas';  // Import your Canvas component

function App() {
  return (
    <div className="App">
      <h1>Welcome to the MNIST Digit Recognition App</h1>
      <Canvas />  {/* Render your canvas drawing component here */}
    </div>
  );
}

export default App;
