import React, { useEffect, useState } from 'react'
import { savePhrase } from '../utils/database'
import './OutputDisplay.css'

const OutputDisplay = ({ currentWord, recognizedLetters, confidence, onClear, onSpace }) => {
  const [wordSuggestions, setWordSuggestions] = useState([])
  const [showSaveModal, setShowSaveModal] = useState(false)

  useEffect(() => {
    // Generate word suggestions based on current word
    if (currentWord.length >= 2) {
      generateSuggestions(currentWord)
    } else {
      setWordSuggestions([])
    }
  }, [currentWord])

  const generateSuggestions = async (partial) => {
    // Common English words for suggestions (in production, use a proper dictionary API)
    const commonWords = [
      'hello', 'world', 'help', 'please', 'thank', 'you', 'good', 'morning',
      'afternoon', 'evening', 'night', 'how', 'are', 'what', 'when', 'where',
      'who', 'why', 'yes', 'no', 'maybe', 'today', 'tomorrow', 'yesterday'
    ]

    const suggestions = commonWords
      .filter(word => word.toLowerCase().startsWith(partial.toLowerCase()))
      .slice(0, 5)

    setWordSuggestions(suggestions)
  }

  const handleSuggestionClick = (suggestion) => {
    const words = currentWord.split(' ')
    words[words.length - 1] = suggestion
    onClear()
    words.forEach((word, i) => {
      Array.from(word).forEach(letter => {
        // Simulate letter recognition for the suggestion
      })
      if (i < words.length - 1) onSpace()
    })
  }

  const handleSave = async () => {
    if (currentWord.trim()) {
      await savePhrase(currentWord.trim())
      setShowSaveModal(false)
      alert('Phrase saved successfully!')
    }
  }

  const getRecentLetters = () => {
    return recognizedLetters.slice(-10).reverse()
  }

  return (
    <div className="output-display">
      <div className="output-header">
        <h2>Translation Output</h2>
        <div className="confidence-meter">
          <span>Confidence:</span>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{
                width: `${confidence * 100}%`,
                backgroundColor: confidence > 0.8 ? '#10b981' : confidence > 0.6 ? '#f59e0b' : '#ef4444'
              }}
            ></div>
          </div>
          <span className="confidence-value">{(confidence * 100).toFixed(0)}%</span>
        </div>
      </div>

      <div className="current-text">
        <p>{currentWord || 'Start signing to see text...'}</p>
      </div>

      {wordSuggestions.length > 0 && (
        <div className="word-suggestions">
          <span className="suggestions-label">Suggestions:</span>
          <div className="suggestions-list">
            {wordSuggestions.map((word, i) => (
              <button
                key={i}
                className="suggestion-chip"
                onClick={() => handleSuggestionClick(word)}
              >
                {word}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="letter-history">
        <h3>Recent Letters</h3>
        <div className="letter-grid">
          {getRecentLetters().map((item, i) => (
            <div key={i} className="letter-item">
              <span className="letter">{item.letter}</span>
              <span className="letter-confidence">{(item.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="output-controls">
        <button className="control-btn space-btn" onClick={onSpace}>
          Add Space
        </button>
        <button className="control-btn clear-btn" onClick={onClear}>
          Clear
        </button>
        <button className="control-btn save-btn" onClick={() => setShowSaveModal(true)}>
          Save Phrase
        </button>
      </div>

      {showSaveModal && (
        <div className="modal-overlay" onClick={() => setShowSaveModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Save Phrase</h3>
            <p className="phrase-preview">{currentWord}</p>
            <div className="modal-actions">
              <button className="modal-btn cancel" onClick={() => setShowSaveModal(false)}>
                Cancel
              </button>
              <button className="modal-btn save" onClick={handleSave}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default OutputDisplay
