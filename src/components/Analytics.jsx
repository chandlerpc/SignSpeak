import React, { useEffect, useState } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import './Analytics.css'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

const Analytics = ({ recognizedLetters }) => {
  const [analyticsData, setAnalyticsData] = useState({
    avgConfidence: 0,
    totalLetters: 0,
    accuracyTrend: []
  })

  useEffect(() => {
    if (recognizedLetters.length > 0) {
      calculateAnalytics()
    }
  }, [recognizedLetters])

  const calculateAnalytics = () => {
    const total = recognizedLetters.length
    const avgConf = recognizedLetters.reduce((sum, item) => sum + item.confidence, 0) / total

    // Group by time windows (every 10 letters)
    const trend = []
    for (let i = 0; i < recognizedLetters.length; i += 10) {
      const window = recognizedLetters.slice(i, i + 10)
      const windowAvg = window.reduce((sum, item) => sum + item.confidence, 0) / window.length
      trend.push(windowAvg)
    }

    setAnalyticsData({
      avgConfidence: avgConf,
      totalLetters: total,
      accuracyTrend: trend
    })
  }

  const chartData = {
    labels: analyticsData.accuracyTrend.map((_, i) => `${i * 10}-${(i + 1) * 10}`),
    datasets: [
      {
        label: 'Confidence Trend',
        data: analyticsData.accuracyTrend,
        borderColor: '#667eea',
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        tension: 0.4
      }
    ]
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Recognition Confidence Over Time',
        color: '#667eea',
        font: {
          size: 14,
          weight: 'bold'
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          callback: (value) => `${(value * 100).toFixed(0)}%`
        }
      }
    }
  }

  const getLetterFrequency = () => {
    const frequency = {}
    recognizedLetters.forEach(item => {
      frequency[item.letter] = (frequency[item.letter] || 0) + 1
    })

    return Object.entries(frequency)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
  }

  const topLetters = getLetterFrequency()

  return (
    <div className="analytics">
      <h3>Performance Analytics</h3>

      <div className="analytics-summary">
        <div className="summary-card">
          <span className="summary-label">Total Letters</span>
          <span className="summary-value">{analyticsData.totalLetters}</span>
        </div>
        <div className="summary-card">
          <span className="summary-label">Avg Confidence</span>
          <span className="summary-value">
            {(analyticsData.avgConfidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {analyticsData.accuracyTrend.length > 0 && (
        <div className="chart-container">
          <Line data={chartData} options={chartOptions} />
        </div>
      )}

      {topLetters.length > 0 && (
        <div className="top-letters">
          <h4>Most Frequent Letters</h4>
          <div className="letters-list">
            {topLetters.map(([letter, count], i) => (
              <div key={i} className="letter-freq-item">
                <span className="freq-letter">{letter}</span>
                <div className="freq-bar-container">
                  <div
                    className="freq-bar"
                    style={{
                      width: `${(count / topLetters[0][1]) * 100}%`
                    }}
                  ></div>
                </div>
                <span className="freq-count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Analytics
