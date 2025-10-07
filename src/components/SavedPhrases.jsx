import React, { useEffect, useState } from 'react'
import { getSavedPhrases, deletePhrase, exportPhrases } from '../utils/database'
import './SavedPhrases.css'

const SavedPhrases = () => {
  const [phrases, setPhrases] = useState([])

  useEffect(() => {
    loadPhrases()
  }, [])

  const loadPhrases = async () => {
    const savedPhrases = await getSavedPhrases()
    setPhrases(savedPhrases)
  }

  const handleDelete = async (id) => {
    await deletePhrase(id)
    loadPhrases()
  }

  const handleExport = async () => {
    const data = await exportPhrases()
    const blob = new Blob([data], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `signspeak-phrases-${Date.now()}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text)
    alert('Phrase copied to clipboard!')
  }

  return (
    <div className="saved-phrases">
      <div className="phrases-header">
        <h3>Saved Phrases</h3>
        {phrases.length > 0 && (
          <button className="export-btn" onClick={handleExport}>
            Export All
          </button>
        )}
      </div>

      <div className="phrases-list">
        {phrases.length === 0 ? (
          <p className="empty-message">No saved phrases yet</p>
        ) : (
          phrases.map((phrase) => (
            <div key={phrase.id} className="phrase-item">
              <div className="phrase-text">{phrase.text}</div>
              <div className="phrase-date">
                {new Date(phrase.timestamp).toLocaleDateString()}
              </div>
              <div className="phrase-actions">
                <button
                  className="phrase-action-btn copy"
                  onClick={() => handleCopy(phrase.text)}
                  title="Copy"
                >
                  📋
                </button>
                <button
                  className="phrase-action-btn delete"
                  onClick={() => handleDelete(phrase.id)}
                  title="Delete"
                >
                  🗑️
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default SavedPhrases
