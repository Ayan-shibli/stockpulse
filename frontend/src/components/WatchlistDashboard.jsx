import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Plus, X, Search, TrendingUp, TrendingDown, Minus, RefreshCw,
  Star, Trash2, ArrowRight, BarChart2, Loader2
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || ''

const cardAnim = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.4, 0, 0.2, 1] } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.2 } },
}

function fmt(val, type = 'price') {
  if (val == null) return '—'
  if (type === 'price') return `$${Number(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  if (type === 'cap') {
    const n = Number(val)
    if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
    if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
    if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
    return `$${n.toLocaleString()}`
  }
  return String(val)
}

export default function WatchlistDashboard({ onSearch, onBack, searchByName, trending }) {
  const [watchlist, setWatchlist] = useState(() => {
    try { return JSON.parse(localStorage.getItem('watchlist') || '[]') } catch { return [] }
  })
  const [stockData, setStockData] = useState([])
  const [loading, setLoading] = useState(false)
  const [addMode, setAddMode] = useState(false)
  const [addQuery, setAddQuery] = useState('')
  const [addSuggestions, setAddSuggestions] = useState([])
  const [searching, setSearching] = useState(false)
  const debounceRef = useRef(null)

  // Save to localStorage whenever watchlist changes
  useEffect(() => {
    localStorage.setItem('watchlist', JSON.stringify(watchlist))
  }, [watchlist])

  // Fetch batch data for the watchlist
  const fetchData = useCallback(async () => {
    if (watchlist.length === 0) {
      setStockData([])
      return
    }
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/watchlist/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers: watchlist }),
      })
      const data = await res.json()
      setStockData(data.stocks || [])
    } catch {
      setStockData([])
    } finally {
      setLoading(false)
    }
  }, [watchlist])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 60_000)
    return () => clearInterval(interval)
  }, [fetchData])

  const addTicker = (ticker) => {
    const t = ticker.trim().toUpperCase()
    if (t && !watchlist.includes(t)) {
      setWatchlist(prev => [...prev, t])
    }
    setAddMode(false)
    setAddQuery('')
    setAddSuggestions([])
  }

  const removeTicker = (ticker) => {
    setWatchlist(prev => prev.filter(t => t !== ticker))
  }

  const handleAddInput = (value) => {
    setAddQuery(value)
    clearTimeout(debounceRef.current)
    if (!value.trim()) {
      setAddSuggestions([])
      return
    }
    // Local trending matches
    const localMatches = trending
      .filter(t =>
        t.ticker.includes(value.toUpperCase()) ||
        t.name.toLowerCase().includes(value.toLowerCase())
      )
      .slice(0, 5)
      .map(t => ({ ticker: t.ticker, name: t.name }))
    setAddSuggestions(localMatches)

    debounceRef.current = setTimeout(async () => {
      setSearching(true)
      const results = await searchByName(value)
      setSearching(false)
      if (results.length > 0) {
        setAddSuggestions(results.slice(0, 6).map(r => ({ ticker: r.ticker, name: r.name })))
      }
    }, 400)
  }

  // Summary stats
  const totalValue = stockData.reduce((s, d) => s + (d.price || 0), 0)
  const avgChange  = stockData.length > 0
    ? stockData.reduce((s, d) => s + (d.change || 0), 0) / stockData.length
    : 0
  const upCount   = stockData.filter(d => d.up).length
  const downCount = stockData.filter(d => !d.up).length

  return (
    <div className="watchlist-container">
      {/* Header */}
      <div className="watchlist-header">
        <div>
          <h2 className="watchlist-title">
            <Star size={20} style={{ color: 'var(--color-amber-spark)' }} />
            Portfolio Watchlist
          </h2>
          <p className="watchlist-subtitle">{watchlist.length} stocks tracked</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button className="watchlist-refresh-btn" onClick={fetchData} disabled={loading} title="Refresh">
            <RefreshCw size={14} className={loading ? 'spin-icon' : ''} />
          </button>
          <button className="btn-primary" style={{ padding: '8px 16px', fontSize: 12 }}
            onClick={() => setAddMode(true)}>
            <Plus size={14} /> Add Stock
          </button>
          <button className="btn-secondary" style={{ padding: '8px 16px', fontSize: 12 }}
            onClick={onBack}>
            ← Home
          </button>
        </div>
      </div>

      {/* Add Ticker Flyout */}
      <AnimatePresence>
        {addMode && (
          <motion.div className="watchlist-add-panel" {...cardAnim}>
            <div className="watchlist-add-bar">
              <Search size={14} />
              <input
                type="text"
                placeholder="Search ticker or company name…"
                value={addQuery}
                onChange={e => handleAddInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && addQuery.trim()) addTicker(addQuery) }}
                autoFocus
              />
              <button onClick={() => { setAddMode(false); setAddQuery(''); setAddSuggestions([]) }}>
                <X size={14} />
              </button>
            </div>
            {searching && (
              <div style={{ padding: '8px 16px', fontSize: 11, color: 'var(--color-smoke)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <Loader2 size={12} className="spin-icon" /> Searching…
              </div>
            )}
            {addSuggestions.length > 0 && (
              <div className="watchlist-add-suggestions">
                {addSuggestions.map((s, i) => (
                  <button key={i} className="watchlist-add-suggestion-item"
                    onClick={() => addTicker(s.ticker)}>
                    <span style={{ fontWeight: 600, fontSize: 12 }}>{s.ticker}</span>
                    <span style={{ fontSize: 11, color: 'var(--color-smoke)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {s.name}
                    </span>
                    <Plus size={12} style={{ color: 'var(--color-lichen)', flexShrink: 0 }} />
                  </button>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Summary Strip */}
      {stockData.length > 0 && (
        <motion.div className="watchlist-summary-strip" {...cardAnim}>
          <div className="watchlist-summary-item">
            <div className="watchlist-summary-value">{stockData.length}</div>
            <div className="watchlist-summary-label">Stocks</div>
          </div>
          <div className="watchlist-summary-divider" />
          <div className="watchlist-summary-item">
            <div className="watchlist-summary-value" style={{ color: upCount >= downCount ? 'var(--color-lichen)' : 'var(--color-amber-spark)' }}>
              {avgChange >= 0 ? '+' : ''}{avgChange.toFixed(2)}%
            </div>
            <div className="watchlist-summary-label">Avg Change</div>
          </div>
          <div className="watchlist-summary-divider" />
          <div className="watchlist-summary-item">
            <div className="watchlist-summary-value" style={{ color: 'var(--color-lichen)' }}>
              {upCount} <TrendingUp size={12} />
            </div>
            <div className="watchlist-summary-label">Gainers</div>
          </div>
          <div className="watchlist-summary-divider" />
          <div className="watchlist-summary-item">
            <div className="watchlist-summary-value" style={{ color: 'var(--color-amber-spark)' }}>
              {downCount} <TrendingDown size={12} />
            </div>
            <div className="watchlist-summary-label">Losers</div>
          </div>
        </motion.div>
      )}

      {/* Stock Cards Grid */}
      {watchlist.length === 0 ? (
        <motion.div className="watchlist-empty" {...cardAnim}>
          <Star size={32} style={{ color: 'var(--color-smoke)', opacity: 0.4 }} />
          <p>Your watchlist is empty</p>
          <p style={{ fontSize: 12, color: 'var(--color-smoke)' }}>
            Add stocks to track them here. Your watchlist is saved in your browser.
          </p>
          <button className="btn-primary" style={{ marginTop: 12, padding: '8px 20px', fontSize: 12 }}
            onClick={() => setAddMode(true)}>
            <Plus size={14} /> Add Your First Stock
          </button>
        </motion.div>
      ) : loading && stockData.length === 0 ? (
        <div className="watchlist-loading">
          <Loader2 size={24} className="spin-icon" />
          <p>Loading watchlist data…</p>
        </div>
      ) : (
        <div className="watchlist-grid">
          <AnimatePresence>
            {watchlist.map((ticker) => {
              const data = stockData.find(d => d.ticker === ticker) || {}
              const hasData = !!data.price
              return (
                <motion.div
                  key={ticker}
                  className="watchlist-card"
                  layout
                  {...cardAnim}
                >
                  <div className="watchlist-card-header">
                    <div className="watchlist-card-ticker">{ticker}</div>
                    <button className="watchlist-card-remove" onClick={() => removeTicker(ticker)} title="Remove">
                      <Trash2 size={12} />
                    </button>
                  </div>

                  {hasData ? (
                    <>
                      <div className="watchlist-card-name">{data.name || ticker}</div>
                      <div className="watchlist-card-price">{fmt(data.price)}</div>
                      <div className={`watchlist-card-change ${data.up ? 'up' : 'down'}`}>
                        {data.up ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                        {data.change >= 0 ? '+' : ''}{data.change?.toFixed(2)}%
                      </div>
                      <div className="watchlist-card-meta">
                        {data.sector && <span>{data.sector}</span>}
                        {data.market_cap && <span>{fmt(data.market_cap, 'cap')}</span>}
                      </div>
                    </>
                  ) : (
                    <div style={{ padding: '12px 0', color: 'var(--color-smoke)', fontSize: 11 }}>
                      {loading ? 'Loading…' : 'No data available'}
                    </div>
                  )}

                  <button className="watchlist-card-research" onClick={() => onSearch(ticker)}>
                    <BarChart2 size={12} /> Research <ArrowRight size={10} />
                  </button>
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
