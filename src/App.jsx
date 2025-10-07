import React, { useState, lazy, Suspense } from 'react'
import './App.css'
import OutputDisplay from './components/OutputDisplay'

// Lazy load heavy components
const CameraView = lazy(() => import('./components/CameraView'))
const PracticeMode = lazy(() => import('./components/PracticeMode'))
const DataCollector = lazy(() => import('./components/DataCollector'))
const Analytics = lazy(() => import('./components/Analytics'))
const SavedPhrases = lazy(() => import('./components/SavedPhrases'))

function App() {
  const [mode, setMode] = useState('translate') // 'translate', 'practice', or 'collect'
  const [recognizedLetters, setRecognizedLetters] = useState([])
  const [currentWord, setCurrentWord] = useState('')
  const [confidence, setConfidence] = useState(0)

  const handleLetterRecognized = (letter, conf) => {
    setRecognizedLetters(prev => [...prev, { letter, confidence: conf, timestamp: Date.now() }])
    setCurrentWord(prev => prev + letter)
    setConfidence(conf)
  }

  const clearWord = () => {
    setCurrentWord('')
  }

  const addSpace = () => {
    setCurrentWord(prev => prev + ' ')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>SignSpeak</h1>
        <p>Real-Time ASL Finger-Spelling Translator</p>
      </header>

      <nav className="mode-selector">
        <button
          className={mode === 'translate' ? 'active' : ''}
          onClick={() => setMode('translate')}
        >
          Translation Mode
        </button>
        <button
          className={mode === 'practice' ? 'active' : ''}
          onClick={() => setMode('practice')}
        >
          Practice Mode
        </button>
        <button
          className={mode === 'collect' ? 'active' : ''}
          onClick={() => setMode('collect')}
        >
          Data Collection
        </button>
      </nav>

      <main className="app-main">
        {mode === 'translate' ? (
          <div className="translate-mode">
            <div className="camera-section">
              <Suspense fallback={<div className="loading">Loading Camera...</div>}>
                <CameraView onLetterRecognized={handleLetterRecognized} />
              </Suspense>
            </div>
            <div className="output-section">
              <OutputDisplay
                currentWord={currentWord}
                recognizedLetters={recognizedLetters}
                confidence={confidence}
                onClear={clearWord}
                onSpace={addSpace}
              />
            </div>
          </div>
        ) : mode === 'practice' ? (
          <Suspense fallback={<div className="loading">Loading Practice Mode...</div>}>
            <PracticeMode />
          </Suspense>
        ) : (
          <Suspense fallback={<div className="loading">Loading Data Collector...</div>}>
            <DataCollector />
          </Suspense>
        )}

        {mode !== 'collect' && (
          <div className="side-panel">
            <Suspense fallback={<div className="loading">Loading...</div>}>
              <Analytics recognizedLetters={recognizedLetters} />
              <SavedPhrases />
            </Suspense>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
