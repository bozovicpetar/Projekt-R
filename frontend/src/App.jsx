import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import { TrendingUp, TrendingDown, Activity, Target, BarChart3, Zap, Search, AlertCircle } from 'lucide-react'
import axios from 'axios'
import './App.css'

export default function App() {
  const [ticker, setTicker] = useState('')
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const API_URL = 'http://localhost:8000'

  const handleAnalyze = async () => {
    if (!ticker.trim()) {
      setError('Molimo unesite ticker simbol')
      return
    }

    setLoading(true)
    setError(null)
    setData(null)

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        ticker: ticker.toUpperCase()
      })
      setData(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Greška pri dohvaćanju podataka')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !loading) {
      handleAnalyze()
    }
  }

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      const color = data.is_prediction ? '#fb923c' : '#60a5fa'
      return (
        <div className="tooltip">
          <p className="tooltip-date">{data.date}</p>
          <p className="tooltip-price" style={{ color }}>
            ${data.close.toFixed(2)}
          </p>
          {data.is_prediction && (
            <p className="prediction-label">📊 Predikcija</p>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <div className="app-container">
      {/* Animated Background */}
      <div className="background-container">
        <div className="bg-blob" style={{ animationDelay: '0s', top: '-200px', right: '-200px', width: '400px', height: '400px' }}></div>
        <div className="bg-blob" style={{ animationDelay: '1s', bottom: '-200px', left: '-200px', width: '400px', height: '400px' }}></div>
        <div className="bg-blob" style={{ animationDelay: '2s', top: '50%', left: '50%', width: '300px', height: '300px' }}></div>
      </div>

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <div className="logo-box">
              <Zap className="logo-icon" />
            </div>
            <div>
              <h1 className="title">Projekt R</h1>
              <p className="subtitle">Predikcija Dionica</p>
            </div>
          </div>
          <div className="version">v1.0</div>
        </div>
      </header>

      <div className="main-container">
        {/* Input Section */}
        <div className="input-section">
          <div style={{ marginBottom: '1.5rem' }}>
            <h2 className="section-title">Analiziraj Dionicu</h2>
            <p className="section-subtitle">Unesite ticker simbol (npr. AAPL, MSFT, TSLA, GOOGL)</p>
          </div>
          
          <div className="input-container">
            <div className="input-wrapper">
              <Search className="search-icon" />
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                onKeyPress={handleKeyPress}
                placeholder="npr. AAPL"
                className="input"
                disabled={loading}
              />
            </div>
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="button"
            >
              {loading ? (
                <>
                  <span className="spin-icon">⚙️</span>
                  Analiziram...
                </>
              ) : (
                <>
                  <Search style={{ width: '20px', height: '20px' }} />
                  Analiziraj
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="error-box">
              <AlertCircle className="error-icon" />
              <p className="error-text">{error}</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        {data && (
          <>
            {/* Cards Grid */}
            <div className="cards-grid">
              {/* Current Price Card */}
              <div className="card">
                <div className="card-header">
                  <div className="card-icon">
                    <Activity className="card-icon-img" />
                  </div>
                  <h3 className="card-title">Trenutna Cijena</h3>
                </div>
                <p className="card-price">${data.current_price.toFixed(2)}</p>
                <p className="card-ticker">{data.ticker}</p>
              </div>

              {/* Prediction Card */}
              <div className="card" style={{
                borderColor: data.is_positive ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'
              }}>
                <div className="card-header">
                  <div className="card-icon" style={{
                    backgroundColor: data.is_positive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'
                  }}>
                    {data.is_positive ? (
                      <TrendingUp className="card-icon-img" style={{ color: '#86efac' }} />
                    ) : (
                      <TrendingDown className="card-icon-img" style={{ color: '#fca5a5' }} />
                    )}
                  </div>
                  <h3 className="card-title">Predikcija Sutra</h3>
                </div>
                <p className="card-price" style={{
                  color: data.is_positive ? '#4ade80' : '#f87171'
                }}>
                  ${data.predicted_price_tomorrow.toFixed(2)}
                </p>
                <div className="change-container">
                  <span className="change-badge" style={{
                    backgroundColor: data.is_positive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                    color: data.is_positive ? '#86efac' : '#fca5a5'
                  }}>
                    {data.is_positive ? '📈' : '📉'} {data.is_positive ? '+' : ''}{data.change_percent.toFixed(2)}%
                  </span>
                </div>
              </div>

              {/* Model Reliability Card */}
              <div className="card">
                <div className="card-header">
                  <div className="card-icon">
                    <Target className="card-icon-img" />
                  </div>
                  <h3 className="card-title">Kvalitet Modela</h3>
                </div>
                <div className="metrics-container">
                  <div>
                    <p className="metric-label">R² Score</p>
                    <p className="metric-value">{(data.model_metrics.r2_score * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="metric-label">MAPE (Greška)</p>
                    <p className="metric-value">{data.model_metrics.mape}%</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Chart Section */}
            <div className="chart-section">
              <div className="chart-header">
                <div className="card-icon">
                  <BarChart3 className="card-icon-img" />
                </div>
                <div>
                  <h3 className="chart-title">Grafikon - Zadnjih 30 Dana</h3>
                  <p className="chart-subtitle">{data.explanation}</p>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={420}>
                <LineChart data={data.chart_data}>
                  <defs>
                    <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF', fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value)
                      return `${date.getMonth() + 1}/${date.getDate()}`
                    }}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF', fontSize: 12 }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend 
                    wrapperStyle={{ paddingTop: '20px' }}
                    iconType="line"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#3B82F6" 
                    strokeWidth={3}
                    dot={(props) => {
                      const { cx, cy, payload } = props
                      if (payload.is_prediction) {
                        return (
                          <circle cx={cx} cy={cy} r={7} fill="#F97316" stroke="#fff" strokeWidth={2} />
                        )
                      }
                      return <circle cx={cx} cy={cy} r={4} fill="#3B82F6" />
                    }}
                    name="Cijena dionice"
                  />
                  <ReferenceLine 
                    x={data.chart_data[data.chart_data.length - 2]?.date} 
                    stroke="#6B7280" 
                    strokeDasharray="5 5"
                    label={{ value: 'Danas', position: 'top', fill: '#9CA3AF', fontSize: 12 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* Empty State */}
        {!data && !loading && !error && (
          <div className="empty-state">
            <div className="empty-icon">
              <Activity className="empty-icon-img" />
            </div>
            <p className="empty-title">Unesite ticker dionice</p>
            <p className="empty-subtitle">za početak analize tržišta</p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p className="loading-text">Analiziram podatke...</p>
            <p className="loading-subtext">Molimo čekajte</p>
          </div>
        )}
      </div>
    </div>
  )
}