import React, { useState, useEffect, useRef } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import './DataCollector.css'

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
const SAMPLES_PER_LETTER = 500  // Increased from 50 to 500 for better accuracy

const DataCollector = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [currentLetter, setCurrentLetter] = useState('A')
  const [capturedImages, setCapturedImages] = useState([])
  const handsRef = useRef(null)
  const currentLandmarksRef = useRef(null)
  const [loadingError, setLoadingError] = useState(null)

  useEffect(() => {
    // Clear old localStorage data to prevent quota errors
    localStorage.removeItem('asl_captured_images')

    initializeCamera()
  }, [])

  useEffect(() => {
    // Keyboard shortcut: Space bar to capture
    const handleKeyPress = (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'BUTTON') {
        e.preventDefault()
        captureHandImage()
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [currentLetter, capturedImages])

  // Auto-download every 100 images to prevent data loss
  useEffect(() => {
    if (capturedImages.length > 0 && capturedImages.length % 100 === 0) {
      console.log(`Auto-downloading at ${capturedImages.length} images...`)
      downloadData()
    }
  }, [capturedImages.length])

  const initializeCamera = async () => {
    try {
      setLoadingError(null)

      const hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        }
      })

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
      })

      hands.onResults(onHandsResults)
      handsRef.current = hands

      if (videoRef.current) {
        const camera = new Camera(videoRef.current, {
          onFrame: async () => {
            await hands.send({ image: videoRef.current })
          },
          width: 640,
          height: 480
        })
        camera.start()
      }
    } catch (error) {
      console.error('Error initializing camera:', error)
      setLoadingError('Failed to load MediaPipe. Please refresh the page.')
    }
  }

  const onHandsResults = (results) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    canvas.width = results.image.width
    canvas.height = results.image.height

    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0]
      drawHandLandmarks(ctx, landmarks, canvas.width, canvas.height)

      // Store current landmarks and canvas for manual capture
      currentLandmarksRef.current = {
        landmarks,
        width: canvas.width,
        height: canvas.height,
        ctx
      }
    } else {
      currentLandmarksRef.current = null
    }

    ctx.restore()
  }

  const drawHandLandmarks = (ctx, landmarks, width, height) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17]
    ]

    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2
    connections.forEach(([start, end]) => {
      ctx.beginPath()
      ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height)
      ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height)
      ctx.stroke()
    })

    ctx.fillStyle = '#ff0000'
    landmarks.forEach(landmark => {
      ctx.beginPath()
      ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI)
      ctx.fill()
    })
  }

  const captureHandImage = () => {
    if (!currentLandmarksRef.current) {
      alert('No hand detected! Please show your hand to the camera.')
      return
    }

    const { landmarks, width, height, ctx } = currentLandmarksRef.current

    // Calculate bounding box
    const xs = landmarks.map(l => l.x * width)
    const ys = landmarks.map(l => l.y * height)
    const minX = Math.max(0, Math.min(...xs) - 30)
    const minY = Math.max(0, Math.min(...ys) - 30)
    const maxX = Math.min(width, Math.max(...xs) + 30)
    const maxY = Math.min(height, Math.max(...ys) + 30)

    // Extract ROI
    const roi = ctx.getImageData(minX, minY, maxX - minX, maxY - minY)

    // Create temporary canvas for resizing
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = maxX - minX
    tempCanvas.height = maxY - minY
    const tempCtx = tempCanvas.getContext('2d')
    tempCtx.putImageData(roi, 0, 0)

    // Create final canvas at 160x160
    const finalCanvas = document.createElement('canvas')
    finalCanvas.width = 160
    finalCanvas.height = 160
    const finalCtx = finalCanvas.getContext('2d')
    finalCtx.drawImage(tempCanvas, 0, 0, 160, 160)

    // Convert to base64
    const imageData = finalCanvas.toDataURL('image/jpeg', 0.95)

    setCapturedImages(prev => [...prev, {
      letter: currentLetter,
      image: imageData,
      timestamp: Date.now()
    }])

    console.log(`Captured image for letter ${currentLetter}`)
  }

  const nextLetter = () => {
    const currentIndex = LETTERS.indexOf(currentLetter)
    if (currentIndex < LETTERS.length - 1) {
      setCurrentLetter(LETTERS[currentIndex + 1])
    }
  }

  const previousLetter = () => {
    const currentIndex = LETTERS.indexOf(currentLetter)
    if (currentIndex > 0) {
      setCurrentLetter(LETTERS[currentIndex - 1])
    }
  }

  const downloadData = async () => {
    // Group images by letter
    const groupedData = {}
    capturedImages.forEach(item => {
      if (!groupedData[item.letter]) {
        groupedData[item.letter] = []
      }
      groupedData[item.letter].push(item.image)
    })

    // Download as JSON
    const dataStr = JSON.stringify(groupedData, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `asl_training_data_${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)

    alert(`Downloaded ${capturedImages.length} images!\n\nNext: Upload this JSON file to ml/training_data/ and run the training script.`)
  }

  const clearData = () => {
    if (confirm('Clear all captured data? Make sure you downloaded your data first!')) {
      setCapturedImages([])
      console.log('Cleared all captured images')
    }
  }

  const letterStats = LETTERS.map(letter => ({
    letter,
    count: capturedImages.filter(img => img.letter === letter).length
  }))

  return (
    <div className="data-collector">
      <div className="collector-header">
        <h2>📸 Training Data Collector</h2>
        <p>Capture {SAMPLES_PER_LETTER} samples of each letter</p>
        {capturedImages.length > 0 && (
          <p style={{ color: '#f57c00', fontWeight: 'bold' }}>
            ⚠️ {capturedImages.length} images in memory - Download regularly to avoid losing data!
          </p>
        )}
      </div>

      {loadingError && (
        <div style={{
          background: '#fff3cd',
          border: '2px solid #ffc107',
          padding: '15px',
          borderRadius: '8px',
          marginBottom: '20px',
          textAlign: 'center'
        }}>
          <strong>⚠️ {loadingError}</strong>
          <br />
          <button onClick={initializeCamera} style={{ marginTop: '10px', padding: '8px 16px' }}>
            Retry Loading
          </button>
        </div>
      )}

      <div className="collector-content">
        <div className="camera-section">
          <video ref={videoRef} style={{ display: 'none' }} />
          <canvas ref={canvasRef} className="collector-canvas" />
        </div>

        <div className="control-panel">
          <div className="letter-selector">
            <button onClick={previousLetter} disabled={LETTERS.indexOf(currentLetter) === 0}>
              ← Prev
            </button>
            <div className="current-letter-display">
              <h1>{currentLetter}</h1>
              <p>{capturedImages.filter(img => img.letter === currentLetter).length} / {SAMPLES_PER_LETTER}</p>
            </div>
            <button onClick={nextLetter} disabled={LETTERS.indexOf(currentLetter) === LETTERS.length - 1}>
              Next →
            </button>
          </div>

          <div className="capture-controls">
            <button className="btn-capture" onClick={captureHandImage}>
              📷 Capture (Space)
            </button>
          </div>

          <div className="data-controls">
            <button onClick={downloadData} disabled={capturedImages.length === 0}>
              💾 Download Data ({capturedImages.length} images)
            </button>
            <button onClick={clearData} disabled={capturedImages.length === 0}>
              🗑️ Clear All
            </button>
          </div>

          <div className="progress-grid">
            <h3>Progress</h3>
            <div className="letter-progress">
              {letterStats.map(({ letter, count }) => (
                <div
                  key={letter}
                  className={`letter-badge ${letter === currentLetter ? 'active' : ''} ${count >= SAMPLES_PER_LETTER ? 'complete' : ''}`}
                >
                  <span>{letter}</span>
                  <small>{count}</small>
                </div>
              ))}
            </div>
          </div>

          <div className="instructions">
            <h3>Instructions</h3>
            <ol>
              <li>Position your hand to form letter "{currentLetter}"</li>
              <li>Click "Capture" or press SPACE to capture</li>
              <li>Capture {SAMPLES_PER_LETTER} samples with slight variations</li>
              <li>Use ← Prev / Next → to navigate letters</li>
              <li>⚠️ Auto-downloads every 100 images</li>
              <li>Download when done and retrain the model</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DataCollector
