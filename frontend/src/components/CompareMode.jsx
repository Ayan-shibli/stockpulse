import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search, ArrowRight, Loader2, TrendingUp, TrendingDown, Minus,
  Target, Shield, Zap, Activity, BarChart2, Scale, ArrowLeft
} from 'lucide-react'

const API_BASE = ''

const cardAnim = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.4, 0, 0.2, 1] } },
}

function fmt(val, type = 'price') {
  if (val == null) return '—'
  if (type === 'price') return `$${Number(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  if (type === 'pct') return `${val > 0 ? '+' : ''}${Number(val).toFixed(2)}%`
  if (type === 'cap') {
    const n = Number(val)
    if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
    if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
    return `$${n.toLocaleString()}`
  }
  return String(val)
}

function getSentColor(s) {
  if (s === 'bullish') return '#8052ff'
  if (s === 'bearish') return '#ffb829'
  return '#9a9a9a'
}

function CompareCard({ data, label, winner }) {
  if (!data) return null
  const sentColor = getSentColor(data.sentiment)
  const isWinner = winner === label

  return (
    <div className={`compare-card ${isWinner ? 'compare-winner' : ''}`}>
      {isWinner && (
        <div className="compare-winner-badge">
          <Zap size={10} /> STRONGER
        </div>
      )}

      {/* Ticker Header */}
      <div className="compare-card-header">
        <div className="compare-card-avatar">{(data.ticker || '??').slice(0, 2)}</div>
        <div>
          <div className="compare-card-ticker">{data.ticker}</div>
          <div className="compare-card-name">{data.company_name || data.ticker}</div>
        </div>
        <span className={`tag ${data.sentiment}`} style={{ marginLeft: 'auto' }}>
          {data.sentiment === 'bullish' ? <TrendingUp size={12} /> : data.sentiment === 'bearish' ? <TrendingDown size={12} /> : <Minus size={12} />}
          {data.sentiment}
        </span>
      </div>

      {/* Price */}
      <div className="compare-card-price">
        <div className="compare-card-price-value">{fmt(data.current_price)}</div>
        {data.change_percent_today != null && (
          <span className={`compare-card-change ${data.change_percent_today >= 0 ? 'up' : 'down'}`}>
            {fmt(data.change_percent_today, 'pct')}
          </span>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="compare-metrics-grid">
        <div className="compare-metric">
          <span className="compare-metric-label">Sentiment Score</span>
          <span className="compare-metric-value" style={{ color: sentColor }}>
            {data.sentiment_score != null ? (data.sentiment_score > 0 ? '+' : '') + data.sentiment_score.toFixed(3) : '—'}
          </span>
        </div>
        <div className="compare-metric">
          <span className="compare-metric-label">Market Cap</span>
          <span className="compare-metric-value">{fmt(data.market_cap, 'cap')}</span>
        </div>
        <div className="compare-metric">
          <span className="compare-metric-label">P/E Ratio</span>
          <span className="compare-metric-value">{data.pe_ratio != null ? data.pe_ratio.toFixed(2) : '—'}</span>
        </div>
        <div className="compare-metric">
          <span className="compare-metric-label">Analyst Target</span>
          <span className="compare-metric-value" style={{ color: 'var(--color-plum-voltage)' }}>{fmt(data.analyst_target)}</span>
        </div>
        <div className="compare-metric">
          <span className="compare-metric-label">Trend Signal</span>
          <span className="compare-metric-value" style={{
            color: data.trend_signal?.includes('up') ? 'var(--color-lichen)' : data.trend_signal?.includes('down') ? 'var(--color-amber-spark)' : 'var(--color-smoke)',
          }}>
            {data.trend_signal?.replace(/_/g, ' ') || '—'}
          </span>
        </div>
        <div className="compare-metric">
          <span className="compare-metric-label">Beta</span>
          <span className="compare-metric-value">{data.beta != null ? data.beta.toFixed(2) : '—'}</span>
        </div>
        <div className="compare-metric">
          <span className="compare-metric-label">Confidence</span>
          <span className="compare-metric-value">{data.confidence_score}/10</span>
        </div>
        <div className="compare-metric">
          <span className="compare-metric-label">Revenue Growth</span>
          <span className="compare-metric-value" style={{
            color: data.revenue_growth > 0 ? 'var(--color-lichen)' : data.revenue_growth < 0 ? 'var(--color-amber-spark)' : undefined,
          }}>
            {data.revenue_growth != null ? `${(data.revenue_growth * 100).toFixed(1)}%` : '—'}
          </span>
        </div>
      </div>

      {/* Key Events */}
      {data.key_events?.length > 0 && (
        <div className="compare-events">
          <div className="compare-events-label">Key Events</div>
          {data.key_events.slice(0, 3).map((ev, i) => (
            <div key={i} className="compare-event-item">
              <div className="compare-event-dot" style={{ background: sentColor }} />
              <span>{ev.length > 90 ? ev.slice(0, 90) + '…' : ev}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function CompareMode({ onBack, searchByName, trending }) {
  const [tickerA, setTickerA] = useState('')
  const [tickerB, setTickerB] = useState('')
  const [loading, setLoading] = useState(false)
  const [resultA, setResultA] = useState(null)
  const [resultB, setResultB] = useState(null)
  const [error, setError] = useState(null)

  // Suggestion state for both inputs
  const [suggestionsA, setSuggestionsA] = useState([])
  const [suggestionsB, setSuggestionsB] = useState([])
  const [showSugA, setShowSugA] = useState(false)
  const [showSugB, setShowSugB] = useState(false)
  const debounceA = useRef(null)
  const debounceB = useRef(null)

  const handleInputA = useCallback((val) => {
    setTickerA(val)
    clearTimeout(debounceA.current)
    if (!val.trim()) { setSuggestionsA([]); setShowSugA(false); return }
    const localMatches = trending.filter(t => t.ticker.includes(val.toUpperCase()) || t.name.toLowerCase().includes(val.toLowerCase())).slice(0, 5).map(t => ({ ticker: t.ticker, name: t.name }))
    setSuggestionsA(localMatches)
    setShowSugA(true)
    debounceA.current = setTimeout(async () => {
      const results = await searchByName(val)
      if (results.length > 0) setSuggestionsA(results.slice(0, 5).map(r => ({ ticker: r.ticker, name: r.name })))
    }, 400)
  }, [trending, searchByName])

  const handleInputB = useCallback((val) => {
    setTickerB(val)
    clearTimeout(debounceB.current)
    if (!val.trim()) { setSuggestionsB([]); setShowSugB(false); return }
    const localMatches = trending.filter(t => t.ticker.includes(val.toUpperCase()) || t.name.toLowerCase().includes(val.toLowerCase())).slice(0, 5).map(t => ({ ticker: t.ticker, name: t.name }))
    setSuggestionsB(localMatches)
    setShowSugB(true)
    debounceB.current = setTimeout(async () => {
      const results = await searchByName(val)
      if (results.length > 0) setSuggestionsB(results.slice(0, 5).map(r => ({ ticker: r.ticker, name: r.name })))
    }, 400)
  }, [trending, searchByName])

  const pickA = (t) => { setTickerA(t); setShowSugA(false); setSuggestionsA([]) }
  const pickB = (t) => { setTickerB(t); setShowSugB(false); setSuggestionsB([]) }

  const runCompare = async () => {
    if (!tickerA.trim() || !tickerB.trim()) return
    setLoading(true); setError(null); setResultA(null); setResultB(null)
    try {
      const res = await fetch(`${API_BASE}/api/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker_a: tickerA.trim(), ticker_b: tickerB.trim() }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
      setResultA(data.ticker_a)
      setResultB(data.ticker_b)
    } catch (e) {
      setError(e.message || 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  // Determine which stock is "stronger" based on sentiment score
  const winner = resultA && resultB
    ? (resultA.sentiment_score || 0) > (resultB.sentiment_score || 0) ? 'A' : (resultB.sentiment_score || 0) > (resultA.sentiment_score || 0) ? 'B' : null
    : null

  return (
    <div className="compare-container">
      {/* Header */}
      <div className="compare-header">
        <div>
          <h2 className="compare-title">
            <Scale size={20} style={{ color: 'var(--color-plum-voltage)' }} />
            Compare Stocks
          </h2>
          <p className="compare-subtitle">Side-by-side sentiment & signal analysis</p>
        </div>
        <button className="btn-secondary" style={{ padding: '8px 16px', fontSize: 12 }} onClick={onBack}>
          <ArrowLeft size={14} /> Home
        </button>
      </div>

      {/* Search Inputs */}
      <motion.div className="compare-search-row" {...cardAnim}>
        <div className="compare-search-col">
          <div className="compare-search-label">STOCK A</div>
          <div className="compare-search-bar">
            <Search size={14} />
            <input
              type="text"
              placeholder="Enter ticker (e.g. AAPL)"
              value={tickerA}
              onChange={e => handleInputA(e.target.value)}
              onFocus={() => { if (suggestionsA.length > 0) setShowSugA(true) }}
              onBlur={() => setTimeout(() => setShowSugA(false), 200)}
              onKeyDown={e => { if (e.key === 'Enter') runCompare() }}
            />
          </div>
          {showSugA && suggestionsA.length > 0 && (
            <div className="compare-suggestions">
              {suggestionsA.map((s, i) => (
                <div key={i} className="compare-suggestion-item" onMouseDown={() => pickA(s.ticker)}>
                  <span style={{ fontWeight: 600 }}>{s.ticker}</span>
                  <span style={{ fontSize: 11, color: 'var(--color-smoke)' }}>{s.name}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="compare-vs">VS</div>

        <div className="compare-search-col">
          <div className="compare-search-label">STOCK B</div>
          <div className="compare-search-bar">
            <Search size={14} />
            <input
              type="text"
              placeholder="Enter ticker (e.g. MSFT)"
              value={tickerB}
              onChange={e => handleInputB(e.target.value)}
              onFocus={() => { if (suggestionsB.length > 0) setShowSugB(true) }}
              onBlur={() => setTimeout(() => setShowSugB(false), 200)}
              onKeyDown={e => { if (e.key === 'Enter') runCompare() }}
            />
          </div>
          {showSugB && suggestionsB.length > 0 && (
            <div className="compare-suggestions">
              {suggestionsB.map((s, i) => (
                <div key={i} className="compare-suggestion-item" onMouseDown={() => pickB(s.ticker)}>
                  <span style={{ fontWeight: 600 }}>{s.ticker}</span>
                  <span style={{ fontSize: 11, color: 'var(--color-smoke)' }}>{s.name}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <button
          className="btn-primary compare-run-btn"
          onClick={runCompare}
          disabled={loading || !tickerA.trim() || !tickerB.trim()}
        >
          {loading ? (
            <><Loader2 size={14} className="spin-icon" /> Analyzing…</>
          ) : (
            <><Activity size={14} /> Compare</>
          )}
        </button>
      </motion.div>

      {/* Error */}
      {error && (
        <motion.div className="compare-error" {...cardAnim}>
          ⚠️ {error}
        </motion.div>
      )}

      {/* Loading */}
      {loading && (
        <div className="compare-loading">
          <Loader2 size={28} className="spin-icon" />
          <p>Running AI research on both stocks…</p>
          <p style={{ fontSize: 12, color: 'var(--color-smoke)' }}>This may take 15-30 seconds</p>
        </div>
      )}

      {/* Results */}
      {resultA && resultB && !loading && (
        <motion.div className="compare-results" {...cardAnim}>
          {/* Verdict banner */}
          <div className="compare-verdict">
            <div className="compare-verdict-icon">
              <Scale size={16} />
            </div>
            {winner === 'A' ? (
              <span><strong>{resultA.ticker}</strong> has stronger sentiment ({(resultA.sentiment_score || 0).toFixed(3)}) vs {resultB.ticker} ({(resultB.sentiment_score || 0).toFixed(3)})</span>
            ) : winner === 'B' ? (
              <span><strong>{resultB.ticker}</strong> has stronger sentiment ({(resultB.sentiment_score || 0).toFixed(3)}) vs {resultA.ticker} ({(resultA.sentiment_score || 0).toFixed(3)})</span>
            ) : (
              <span>Both stocks have similar sentiment profiles</span>
            )}
          </div>

          {/* Side-by-side cards */}
          <div className="compare-cards-row">
            <CompareCard data={resultA} label="A" winner={winner} />
            <CompareCard data={resultB} label="B" winner={winner} />
          </div>
        </motion.div>
      )}
    </div>
  )
}
