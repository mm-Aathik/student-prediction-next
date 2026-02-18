import { useState, useEffect } from 'react'
import axios from 'axios'
import { Bar, Doughnut } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend } from 'chart.js'
import './App.css'

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend)

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
  const [modelInfo, setModelInfo] = useState(null)
  const [showModelInfo, setShowModelInfo] = useState(false)
  const [modelInfoLoading, setModelInfoLoading] = useState(false)

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

  const handleModelInfo = async () => {
    if (modelInfo) {
      setShowModelInfo(!showModelInfo)
      return
    }
    setModelInfoLoading(true)
    try {
      const response = await axios.get(`${API_URL}/api/model-info`)
      setModelInfo(response.data)
      setShowModelInfo(true)
    } catch (err) {
      setError('Failed to load model info: ' + err.message)
    } finally {
      setModelInfoLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Student Course & University Prediction</h1>
      <p className="subtitle">Enter your details to get a prediction</p>

      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="three-columns">
        {/* Column 1: Form */}
        <div className="panel panel-form">
          <h2 className="panel-title">Input Details</h2>

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
        </div>

        {/* Column 2: Prediction Result */}
        <div className="panel panel-prediction">
          <h2 className="panel-title">Prediction Result</h2>
          {!prediction ? (
            <div className="panel-empty">Submit the form to see predictions</div>
          ) : (
            <>
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
                {explaining ? 'Analyzing...' : '🔍 Explain Why →'}
              </button>
            </>
          )}
        </div>

        {/* Column 3: SHAP Explanation */}
        <div className="panel panel-explanation">
          <h2 className="panel-title">Why this prediction?</h2>
          {!explanation ? (
            <div className="panel-empty">Click "Explain Why" to see SHAP analysis</div>
          ) : (
            <>
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
                <strong>Green</strong> = pushes toward this course. <strong>Red</strong> = pushes away.
              </p>
            </>
          )}
        </div>
      </div>

      {/* Model Performance Section */}
      <div className="model-info-section">
        <button onClick={handleModelInfo} disabled={modelInfoLoading} className="model-info-toggle">
          {modelInfoLoading ? 'Loading...' : showModelInfo ? '📉 Hide Model Performance' : '📊 Show Model Performance & Dataset Info'}
        </button>

        {showModelInfo && modelInfo && (
          <div className="model-info-content">
            {/* Row 1: Key Metrics */}
            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-value">{(modelInfo.accuracy * 100).toFixed(1)}%</span>
                <span className="metric-label">Accuracy</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{(modelInfo.top3_accuracy * 100).toFixed(1)}%</span>
                <span className="metric-label">Top-3 Accuracy</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{(modelInfo.top5_accuracy * 100).toFixed(1)}%</span>
                <span className="metric-label">Top-5 Accuracy</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{modelInfo.f1_weighted?.toFixed(3)}</span>
                <span className="metric-label">F1 (Weighted)</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{modelInfo.f1_macro?.toFixed(3)}</span>
                <span className="metric-label">F1 (Macro)</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{modelInfo.num_classes}</span>
                <span className="metric-label">Total Classes</span>
              </div>
            </div>

            {/* Row 2: Charts */}
            <div className="charts-row">
              {/* Accuracy Comparison Bar Chart */}
              <div className="chart-panel">
                <h3 className="info-panel-title">📊 Accuracy Comparison</h3>
                <div className="chart-container">
                  <Bar
                    data={{
                      labels: ['Top-1', 'Top-3', 'Top-5'],
                      datasets: [{
                        label: 'Accuracy %',
                        data: [
                          (modelInfo.accuracy * 100).toFixed(1),
                          (modelInfo.top3_accuracy * 100).toFixed(1),
                          (modelInfo.top5_accuracy * 100).toFixed(1)
                        ],
                        backgroundColor: ['#667eea', '#764ba2', '#f59e0b'],
                        borderRadius: 6,
                        barThickness: 48
                      }]
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ctx.raw + '%' } } },
                      scales: { y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } } }
                    }}
                  />
                </div>
              </div>

              {/* Feature Importance Donut */}
              <div className="chart-panel">
                <h3 className="info-panel-title">🎯 Feature Importance</h3>
                <div className="chart-container">
                  <Doughnut
                    data={{
                      labels: modelInfo.feature_importances?.map(f => f.feature) || [],
                      datasets: [{
                        data: modelInfo.feature_importances?.map(f => (f.importance * 100).toFixed(1)) || [],
                        backgroundColor: ['#667eea', '#22c55e', '#f59e0b'],
                        borderWidth: 2,
                        borderColor: '#fff'
                      }]
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { position: 'bottom', labels: { padding: 16, font: { size: 12 } } },
                        tooltip: { callbacks: { label: (ctx) => ctx.label + ': ' + ctx.raw + '%' } }
                      }
                    }}
                  />
                </div>
              </div>

              {/* Per-Class F1 Bar Chart */}
              <div className="chart-panel">
                <h3 className="info-panel-title">📈 F1 Score by Course (Top 10)</h3>
                <div className="chart-container">
                  <Bar
                    data={{
                      labels: modelInfo.per_class_top15?.slice(0, 10).map(c => c.class.length > 20 ? c.class.slice(0, 18) + '...' : c.class) || [],
                      datasets: [{
                        label: 'F1 Score',
                        data: modelInfo.per_class_top15?.slice(0, 10).map(c => c.f1.toFixed(3)) || [],
                        backgroundColor: modelInfo.per_class_top15?.slice(0, 10).map(c =>
                          c.f1 >= 0.5 ? '#22c55e' : c.f1 >= 0.2 ? '#f59e0b' : '#ef4444'
                        ) || [],
                        borderRadius: 4,
                        barThickness: 18
                      }]
                    }}
                    options={{
                      indexAxis: 'y',
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: false } },
                      scales: { x: { beginAtZero: true, max: 1, ticks: { callback: v => v.toFixed(1) } } }
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Row 3: Three info panels */}
            <div className="info-three-columns">
              {/* Dataset Info */}
              <div className="info-panel">
                <h3 className="info-panel-title">📁 Dataset Info</h3>
                <table className="info-table">
                  <tbody>
                    <tr><td>Total Rows</td><td>{modelInfo.dataset_info?.total_rows?.toLocaleString()}</td></tr>
                    <tr><td>After Cleaning</td><td>{modelInfo.dataset_info?.rows_after_cleaning?.toLocaleString()}</td></tr>
                    <tr><td>Features</td><td>{modelInfo.dataset_info?.num_features}</td></tr>
                    <tr><td>Streams</td><td>{modelInfo.dataset_info?.num_streams}</td></tr>
                    <tr><td>Districts</td><td>{modelInfo.dataset_info?.num_districts}</td></tr>
                    <tr><td>Courses</td><td>{modelInfo.dataset_info?.num_courses}</td></tr>
                    <tr><td>Z-Score Range</td><td>{modelInfo.dataset_info?.zscore_min} — {modelInfo.dataset_info?.zscore_max}</td></tr>
                    <tr><td>Z-Score Mean</td><td>{modelInfo.dataset_info?.zscore_mean}</td></tr>
                  </tbody>
                </table>
                <h4 className="preprocess-heading">Preprocessing</h4>
                <ul className="preprocess-list">
                  {modelInfo.dataset_info?.preprocessing?.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ul>
              </div>

              {/* Hyperparameters & Training */}
              <div className="info-panel">
                <h3 className="info-panel-title">⚙️ Hyperparameters</h3>
                <table className="info-table">
                  <tbody>
                    <tr><td>Algorithm</td><td>Random Forest</td></tr>
                    {modelInfo.hyperparameters && Object.entries(modelInfo.hyperparameters).map(([key, val]) => (
                      <tr key={key}><td>{key}</td><td>{String(val)}</td></tr>
                    ))}
                    <tr><td>Train / Test Split</td><td>{modelInfo.train_size?.toLocaleString()} / {modelInfo.test_size?.toLocaleString()}</td></tr>
                  </tbody>
                </table>

                <h4 className="preprocess-heading">Feature Importances</h4>
                <div className="fi-bars">
                  {modelInfo.feature_importances?.map((f, i) => (
                    <div key={i} className="fi-row">
                      <span className="fi-name">{f.feature}</span>
                      <div className="fi-bar-bg">
                        <div className="fi-bar-fill" style={{ width: `${(f.importance * 100)}%` }} />
                      </div>
                      <span className="fi-val">{(f.importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Per-Class Performance */}
              <div className="info-panel">
                <h3 className="info-panel-title">📋 Per-Class Metrics (Top 15)</h3>
                <div className="class-table-wrapper">
                  <table className="class-table">
                    <thead>
                      <tr>
                        <th>Course</th>
                        <th>Prec.</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>N</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelInfo.per_class_top15?.map((c, i) => (
                        <tr key={i}>
                          <td className="class-name" title={c.class}>{c.class.length > 40 ? c.class.slice(0, 37) + '...' : c.class}</td>
                          <td>{c.precision.toFixed(2)}</td>
                          <td>{c.recall.toFixed(2)}</td>
                          <td>{c.f1.toFixed(2)}</td>
                          <td>{c.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

