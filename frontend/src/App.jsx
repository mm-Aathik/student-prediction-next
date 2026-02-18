import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [zscore, setZscore] = useState('')
  const [stream, setStream] = useState('')
  const [district, setDistrict] = useState('')
  const [streams, setStreams] = useState([])
  const [districts, setDistricts] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [optionsLoading, setOptionsLoading] = useState(true)
  const [explanation, setExplanation] = useState(null)
  const [explaining, setExplaining] = useState(false)

  // Load options on mount
  useEffect(() => {
    axios.get(`${API_URL}/api/options`)
      .then(res => {
        setStreams(res.data.streams || [])
        setDistricts(res.data.districts || [])
      })
      .catch(err => {
        setError('Failed to load options: ' + err.message)
      })
      .finally(() => {
        setOptionsLoading(false)
      })
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setPrediction(null)
    setExplanation(null)
    setLoading(true)

    try {
      const response = await axios.post(`${API_URL}/api/predict`, {
        zscore: parseFloat(zscore),
        stream: stream,
        district: district
      }, {
        timeout: 10000 // 10 seconds
      })
      setPrediction(response.data)
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.')
      } else if (err.response?.data?.error) {
        setError(err.response.data.error)
      } else {
        setError('Error: ' + (err.message || 'Unknown error'))
      }
    } finally {
      setLoading(false)
    }
  }

  const handleExplain = async () => {
    setExplaining(true)
    try {
      const response = await axios.post(`${API_URL}/api/explain`, {
        zscore: parseFloat(zscore),
        stream: stream,
        district: district
      })
      setExplanation(response.data)
    } catch (err) {
      setError('Failed to get explanation: ' + (err.response?.data?.error || err.message))
    } finally {
      setExplaining(false)
    }
  }

  return (
    <div className="container">
      <div className="card">
        <h1>Student Course & University Prediction</h1>
        <p>Enter your details to get a prediction</p>

        {optionsLoading && <p className="loading">Loading options...</p>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Z-Score</label>
            <input
              type="number"
              step="any"
              value={zscore}
              onChange={(e) => setZscore(e.target.value)}
              placeholder="Enter Z-score"
              required
              disabled={loading || optionsLoading}
            />
          </div>

          <div className="form-group">
            <label>Stream</label>
            <select
              value={stream}
              onChange={(e) => setStream(e.target.value)}
              required
              disabled={loading || optionsLoading}
            >
              <option value="">Select Stream</option>
              {streams.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>District</label>
            <select
              value={district}
              onChange={(e) => setDistrict(e.target.value)}
              required
              disabled={loading || optionsLoading}
            >
              <option value="">Select District</option>
              {districts.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <button type="submit" disabled={loading || optionsLoading} className="submit-btn">
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </form>

        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {prediction && (
          <div className="prediction">
            <h2>Prediction Result</h2>
            <div className="prediction-main">
              <p className="prediction-text">{prediction.prediction}</p>
              <p className="confidence">
                Confidence: {(prediction.confidence * 100).toFixed(1)}%
              </p>
            </div>
            
            {prediction.top_3 && prediction.top_3.length > 1 && (
              <div className="top-predictions">
                <h3>Top 3 Recommendations:</h3>
                <ul>
                  {prediction.top_3.map((item, index) => (
                    <li key={index} className={index === 0 ? 'top-choice' : ''}>
                      <span className="rank">#{index + 1}</span>
                      <span className="course">{item.course}</span>
                      <span className="conf">{(item.confidence * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <button onClick={handleExplain} disabled={explaining} className="explain-btn">
              {explaining ? 'Analyzing...' : '🔍 Explain Why'}
            </button>

            {explanation && (
              <div className="explanation">
                <h3>Why this prediction? (SHAP Analysis)</h3>

                {explanation.summary && (
                  <div className="explain-summary">
                    <p>{explanation.summary}</p>
                  </div>
                )}

                {explanation.simple_explanations?.map((item, index) => (
                  <div key={index} className="simple-explain-card">
                    <span className="simple-icon">{item.icon}</span>
                    <p className="simple-text">{item.text}</p>
                  </div>
                ))}

                <h4 className="technical-heading">Technical Details</h4>
                <p className="explanation-subtitle">Feature contributions to: <strong>{explanation.predicted_course}</strong></p>
                <div className="shap-bars">
                  {explanation.contributions?.map((item, index) => {
                    const maxVal = Math.max(...explanation.contributions.map(c => Math.abs(c.shap_value)))
                    const barWidth = maxVal > 0 ? (Math.abs(item.shap_value) / maxVal) * 100 : 0
                    return (
                      <div key={index} className="shap-row">
                        <div className="shap-label">
                          <span className="shap-feature">{item.feature}</span>
                          <span className="shap-input">= {item.value}</span>
                        </div>
                        <div className="shap-bar-container">
                          <div
                            className={`shap-bar ${item.impact}`}
                            style={{ width: `${barWidth}%` }}
                          />
                          <span className={`shap-val ${item.impact}`}>
                            {item.shap_value > 0 ? '+' : ''}{item.shap_value.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
                <p className="explanation-note">
                  <strong>Green (positive)</strong> = pushes toward this course. <strong>Red (negative)</strong> = pushes away from it.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App

