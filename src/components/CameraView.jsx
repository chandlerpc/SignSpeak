import React, { useRef, useEffect, useState } from 'react';
import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import './CameraView.css';

const CameraView = ({ onLetterRecognized }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [fps, setFps] = useState(0);
  const lastFrameTime = useRef(Date.now());
  const frameCount = useRef(0);
  const lastPredictionTime = useRef(0);

  useEffect(() => {
    let mounted = true;
    let camera = null;

    const initialize = async () => {
      try {
        const hands = new Hands({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
          },
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5,
        });

        hands.onResults(onResults);

        if (videoRef.current && mounted) {
          camera = new Camera(videoRef.current, {
            onFrame: async () => {
              if (mounted) {
                await hands.send({ image: videoRef.current });
              }
            },
            width: 640,
            height: 480,
          });
          camera.start();
        }
      } catch (error) {
        console.error('Error initializing MediaPipe:', error);
      }
    };

    initialize();

    return () => {
      mounted = false;
      if (camera) {
        camera.stop();
      }
    };
  }, []);

  const onResults = async (results) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = results.image.width;
    canvas.height = results.image.height;

    // Draw camera feed
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      setIsDetecting(true);
      const landmarks = results.multiHandLandmarks[0];

      // Draw hand landmarks
      drawHandLandmarks(ctx, landmarks, canvas.width, canvas.height);

      // Get prediction from Flask API
      await classifyHand(results.image, landmarks, canvas.width, canvas.height);
    } else {
      setIsDetecting(false);
    }

    ctx.restore();
    updateFPS();
  };

  const drawHandLandmarks = (ctx, landmarks, width, height) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17],
    ];

    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    connections.forEach(([start, end]) => {
      ctx.beginPath();
      ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height);
      ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height);
      ctx.stroke();
    });

    ctx.fillStyle = '#ff0000';
    landmarks.forEach((landmark) => {
      ctx.beginPath();
      ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const classifyHand = async (image, landmarks, width, height) => {
    try {
      // Throttle predictions to once every 2 seconds
      const now = Date.now();
      if (now - lastPredictionTime.current < 2000) {
        return; // Skip this prediction
      }

      console.log('🖐️ Hand detected, sending to model server...');

      // Extract bounding box around hand with generous padding
      const xs = landmarks.map((l) => l.x * width);
      const ys = landmarks.map((l) => l.y * height);

      // Calculate hand size and use percentage-based padding
      const handWidth = Math.max(...xs) - Math.min(...xs);
      const handHeight = Math.max(...ys) - Math.min(...ys);
      const handSize = Math.max(handWidth, handHeight);

      // Use 60% padding (training images show hand taking ~70% of frame)
      const padding = handSize * 0.6;

      let minX = Math.min(...xs) - padding;
      let minY = Math.min(...ys) - padding;
      let maxX = Math.max(...xs) + padding;
      let maxY = Math.max(...ys) + padding;

      // Make it square (training images are square)
      const boxWidth = maxX - minX;
      const boxHeight = maxY - minY;
      const size = Math.max(boxWidth, boxHeight);

      // Center the hand in the square
      const centerX = (minX + maxX) / 2;
      const centerY = (minY + maxY) / 2;
      minX = centerX - size / 2;
      minY = centerY - size / 2;
      maxX = centerX + size / 2;
      maxY = centerY + size / 2;

      // Clamp to image bounds
      minX = Math.max(0, minX);
      minY = Math.max(0, minY);
      maxX = Math.min(width, maxX);
      maxY = Math.min(height, maxY);

      // Create square canvas for hand region
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = size;
      tempCanvas.height = size;
      const tempCtx = tempCanvas.getContext('2d');

      // Fill with light background (training images have light backgrounds)
      tempCtx.fillStyle = '#E0E0E0';
      tempCtx.fillRect(0, 0, size, size);

      // Draw hand region centered
      const cropWidth = maxX - minX;
      const cropHeight = maxY - minY;
      const offsetX = (size - cropWidth) / 2;
      const offsetY = (size - cropHeight) / 2;

      tempCtx.drawImage(
        image,
        minX, minY, cropWidth, cropHeight,
        offsetX, offsetY, cropWidth, cropHeight
      );

      // Resize to 128x128 for the model
      const resizedCanvas = document.createElement('canvas');
      resizedCanvas.width = 128;
      resizedCanvas.height = 128;
      const resizedCtx = resizedCanvas.getContext('2d');
      resizedCtx.drawImage(tempCanvas, 0, 0, 128, 128);

      // Convert canvas to image data array [0-1 normalized]
      const imageData = resizedCtx.getImageData(0, 0, 128, 128);
      const pixels = imageData.data;
      const normalized = [];

      // Convert RGBA to RGB and normalize to [0, 1]
      for (let i = 0; i < pixels.length; i += 4) {
        normalized.push(pixels[i] / 255.0);     // R
        normalized.push(pixels[i + 1] / 255.0); // G
        normalized.push(pixels[i + 2] / 255.0); // B
      }

      // Reshape to [128, 128, 3]
      const imageArray = [];
      for (let y = 0; y < 128; y++) {
        const row = [];
        for (let x = 0; x < 128; x++) {
          const idx = (y * 128 + x) * 3;
          row.push([normalized[idx], normalized[idx + 1], normalized[idx + 2]]);
        }
        imageArray.push(row);
      }

      // Send to Flask API
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageArray })
      });

      const result = await response.json();

      if (result.error) {
        console.error('Prediction error:', result.error);
        return;
      }

      const predictedLetter = result.prediction;
      const confidence = result.confidence;

      console.log(`Prediction: ${predictedLetter} (confidence: ${(confidence * 100).toFixed(2)}%)`);

      if (confidence > 0.5) {
        lastPredictionTime.current = Date.now();
        onLetterRecognized(predictedLetter, confidence);
      }
    } catch (error) {
      console.error('Classification error:', error);
    }
  };

  const updateFPS = () => {
    frameCount.current++;
    const now = Date.now();
    const elapsed = now - lastFrameTime.current;

    if (elapsed >= 1000) {
      setFps(frameCount.current);
      frameCount.current = 0;
      lastFrameTime.current = now;
    }
  };

  return (
    <div className="camera-view">
      <div className="camera-header">
        <h2>Camera Feed</h2>
      </div>

      <div className="camera-container">
        <video ref={videoRef} style={{ display: 'none' }} />
        <canvas ref={canvasRef} className="camera-canvas" />
      </div>

      <div className="camera-status">
        <div className={`status-indicator ${isDetecting ? 'detected' : ''}`}>
          <span className="status-dot"></span>
          {isDetecting ? 'Hand Detected' : 'No Hand Detected'}
        </div>
        <div className="fps-counter">
          FPS: {fps}
        </div>
      </div>
    </div>
  );
};

export default CameraView;
