import { openDB } from 'idb'

const DB_NAME = 'SignSpeakDB'
const DB_VERSION = 1
const PHRASES_STORE = 'phrases'
const ANALYTICS_STORE = 'analytics'

// Initialize IndexedDB
export const initDB = async () => {
  const db = await openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      // Create phrases store
      if (!db.objectStoreNames.contains(PHRASES_STORE)) {
        const phrasesStore = db.createObjectStore(PHRASES_STORE, {
          keyPath: 'id',
          autoIncrement: true
        })
        phrasesStore.createIndex('timestamp', 'timestamp')
      }

      // Create analytics store
      if (!db.objectStoreNames.contains(ANALYTICS_STORE)) {
        const analyticsStore = db.createObjectStore(ANALYTICS_STORE, {
          keyPath: 'id',
          autoIncrement: true
        })
        analyticsStore.createIndex('date', 'date')
      }
    }
  })
  return db
}

// Phrase operations
export const savePhrase = async (text) => {
  const db = await initDB()
  const tx = db.transaction(PHRASES_STORE, 'readwrite')
  const store = tx.objectStore(PHRASES_STORE)

  await store.add({
    text,
    timestamp: Date.now()
  })

  await tx.done
}

export const getSavedPhrases = async () => {
  const db = await initDB()
  const tx = db.transaction(PHRASES_STORE, 'readonly')
  const store = tx.objectStore(PHRASES_STORE)
  const index = store.index('timestamp')

  const phrases = await index.getAll()
  return phrases.reverse() // Most recent first
}

export const deletePhrase = async (id) => {
  const db = await initDB()
  const tx = db.transaction(PHRASES_STORE, 'readwrite')
  const store = tx.objectStore(PHRASES_STORE)

  await store.delete(id)
  await tx.done
}

export const exportPhrases = async () => {
  const phrases = await getSavedPhrases()
  return phrases
    .map(p => `[${new Date(p.timestamp).toLocaleString()}] ${p.text}`)
    .join('\n')
}

// Analytics operations
export const saveAnalytics = async (data) => {
  const db = await initDB()
  const tx = db.transaction(ANALYTICS_STORE, 'readwrite')
  const store = tx.objectStore(ANALYTICS_STORE)

  const today = new Date().toDateString()

  await store.add({
    date: today,
    timestamp: Date.now(),
    ...data
  })

  await tx.done
}

export const getAnalyticsByDate = async (startDate, endDate) => {
  const db = await initDB()
  const tx = db.transaction(ANALYTICS_STORE, 'readonly')
  const store = tx.objectStore(ANALYTICS_STORE)

  const allAnalytics = await store.getAll()

  return allAnalytics.filter(item => {
    const itemDate = new Date(item.timestamp)
    return itemDate >= startDate && itemDate <= endDate
  })
}

export const clearAllData = async () => {
  const db = await initDB()

  const phrasesTx = db.transaction(PHRASES_STORE, 'readwrite')
  await phrasesTx.objectStore(PHRASES_STORE).clear()
  await phrasesTx.done

  const analyticsTx = db.transaction(ANALYTICS_STORE, 'readwrite')
  await analyticsTx.objectStore(ANALYTICS_STORE).clear()
  await analyticsTx.done
}
