import { useRef, useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, ArrowRight, Loader2, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import ParticleCosmos from './ParticleCosmos'

const fadeUp = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.4, 0, 0.2, 1] } },
}
const stagger = { animate: { transition: { staggerChildren: 0.08 } } }

// ── Live ticker strip ─────────────────────────────────────────────────────────
// Fallback data shown instantly while the live fetch loads
const TICKER_FALLBACK = [
  { symbol: 'AAPL',  price: '—', change: '—', up: true  },
  { symbol: 'NVDA',  price: '—', change: '—', up: true  },
  { symbol: 'TSLA',  price: '—', change: '—', up: false },
  { symbol: 'MSFT',  price: '—', change: '—', up: true  },
  { symbol: 'AMZN',  price: '—', change: '—', up: true  },
  { symbol: 'GOOGL', price: '—', change: '—', up: false },
  { symbol: 'META',  price: '—', change: '—', up: true  },
  { symbol: 'NFLX',  price: '—', change: '—', up: true  },
  { symbol: 'SPY',   price: '—', change: '—', up: true  },
  { symbol: 'QQQ',   price: '—', change: '—', up: true  },
  { symbol: 'BRK-B', price: '—', change: '—', up: false },
  { symbol: 'JPM',   price: '—', change: '—', up: true  },
  { symbol: 'AMD',   price: '—', change: '—', up: true  },
  { symbol: 'COIN',  price: '—', change: '—', up: false },
  { symbol: 'PLTR',  price: '—', change: '—', up: true  },
]

function TickerStrip() {
  const [tickers, setTickers] = useState(TICKER_FALLBACK)

  useEffect(() => {
    // Fetch live prices on mount, refresh every 60 seconds
    const load = () => {
      fetch(`${import.meta.env.VITE_API_URL || ''}/api/prices`)
        .then(r => r.json())
        .then(data => {
          if (data.prices && data.prices.length > 0) setTickers(data.prices)
        })
        .catch(() => {}) // silently keep fallback on error
    }
    load()
    const interval = setInterval(load, 60_000)
    return () => clearInterval(interval)
  }, [])

  const duplicated = [...tickers, ...tickers]
  return (
    <div className="ticker-strip-bar">
      <div className="ticker-strip-track">
        {duplicated.map((t, i) => (
          <div key={i} className="ticker-strip-item">
            <span className="ticker-strip-symbol">{t.symbol}</span>
            <span className="ticker-strip-price">
              {t.price === '—' ? '—' : `$${t.price}`}
            </span>
            <span className={`ticker-strip-change ${t.up ? 'up' : 'down'}`}>
              {t.price !== '—' && (t.up ? <TrendingUp size={10} /> : <TrendingDown size={10} />)}
              {t.change}{t.change !== '—' ? '%' : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function HeroSection({
  query, setQuery, onSearch, searchByName, loading, loadingStep, trending, history
}) {
  const [suggestions, setSuggestions]       = useState([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [searching, setSearching]           = useState(false)
  const [focused, setFocused]               = useState(false)
  const debounceRef = useRef(null)
  const inputRef    = useRef(null)

  const handleInput = useCallback((value) => {
    setQuery(value)
    clearTimeout(debounceRef.current)
    if (!value.trim()) {
      setSuggestions([])
      setShowSuggestions(false)
      return
    }
    const trendingMatches = trending
      .filter(t =>
        t.ticker.startsWith(value.toUpperCase()) ||
        t.name.toLowerCase().startsWith(value.toLowerCase()) ||
        t.ticker.includes(value.toUpperCase()) ||
        t.name.toLowerCase().includes(value.toLowerCase())
      )
      .map(t => ({ ticker: t.ticker, name: t.name, exchange: t.sector, type: 'Equity' }))
    setSuggestions(trendingMatches)
    setShowSuggestions(true)
    debounceRef.current = setTimeout(async () => {
      setSearching(true)
      const results = await searchByName(value)
      setSearching(false)
      if (results.length > 0) setSuggestions(results)
    }, 350)
  }, [trending, searchByName, setQuery])

  const handleFocus = useCallback(() => {
    setFocused(true)
    if (!query.trim()) {
      const all = trending.slice(0, 8).map(t => ({
        ticker: t.ticker, name: t.name, exchange: t.sector, type: 'Equity'
      }))
      setSuggestions(all)
      setShowSuggestions(true)
    } else {
      setShowSuggestions(true)
    }
  }, [query, trending])

  const handleKey = (e) => {
    if (e.key === 'Enter') { setShowSuggestions(false); onSearch() }
    if (e.key === 'Escape') setShowSuggestions(false)
  }

  const pick = (ticker) => {
    setQuery(ticker)
    setSuggestions([])
    setShowSuggestions(false)
    onSearch(ticker)
  }

  return (
    <div style={{ width: '100%' }}>

      {/* ── Ticker Strip ──────────────────────────────────────────────────── */}
      <TickerStrip />

      {/* ── Section 1: Hero Split ─────────────────────────────────────────── */}
      <section className="hero">

        {/* Radial glow behind globe */}
        <div className="hero-globe-glow" />

        <div className="container">
          <div className="hero-layout">

            {/* Left Content Block */}
            <div className="hero-text-block">
              <motion.div variants={stagger} initial="initial" animate="animate">

                {/* Eyebrow Flag */}
                <motion.div variants={fadeUp}>
                  <span className="hero-eyebrow accented">
                    AI-POWERED STOCK RESEARCH
                  </span>
                </motion.div>

                {/* Display Headline */}
                <motion.h1 variants={fadeUp} className="hero-title">
                  Real-time market<br />
                  intelligence,<br />
                  on demand.
                </motion.h1>

                {/* Body Paragraph */}
                <motion.p variants={fadeUp} className="hero-subtitle">
                  Search any stock ticker to get an instant AI research report — live prices, news sentiment, technical signals, earnings data, and a 7-day AI price forecast.
                </motion.p>

                {/* Search Bar — Glassmorphism */}
                <motion.div variants={fadeUp} className="search-wrapper">
                  <div className={`search-bar glass-search${focused ? ' focused' : ''}`}>
                    <div className="search-icon">
                      {loading ? (
                        <Loader2 size={16} className="spin-icon" style={{ color: 'var(--color-plum-voltage)' }} />
                      ) : (
                        <Search size={16} />
                      )}
                    </div>
                    <input
                      ref={inputRef}
                      className="search-input"
                      type="text"
                      placeholder='Type a company name or ticker — e.g. "Apple", "NVDA", "HBL"…'
                      value={query}
                      onChange={e => handleInput(e.target.value)}
                      onKeyDown={handleKey}
                      onFocus={handleFocus}
                      onBlur={() => { setFocused(false); setTimeout(() => setShowSuggestions(false), 220) }}
                      disabled={loading}
                      autoComplete="off"
                      spellCheck={false}
                    />
                    <button
                      className="search-btn"
                      onClick={() => { setShowSuggestions(false); onSearch() }}
                      disabled={loading || !query.trim()}
                    >
                      {loading ? (
                        <><Loader2 size={13} className="spin-icon" /> Analyzing</>
                      ) : (
                        <><ArrowRight size={13} /> Research</>
                      )}
                    </button>
                  </div>

                  {/* Suggestions */}
                  <AnimatePresence>
                    {showSuggestions && suggestions.length > 0 && (
                      <motion.div
                        className="search-suggestions"
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.12 }}
                      >
                        <div className="suggestion-header">
                          {searching ? 'Searching…' : 'Suggested — click any to research'}
                        </div>
                        {searching && (
                          <div className="suggestion-searching">
                            <Loader2 size={12} className="spin-icon" />
                            Loading results…
                          </div>
                        )}
                        {suggestions.slice(0, 8).map((t, i) => {
                          const isPK = t.ticker?.includes('.KA')
                          const flag = isPK ? '🇵🇰' : '🇺🇸'
                          const cleanName = t.name || t.ticker
                          return (
                            <div
                              key={i}
                              className="suggestion-item"
                              onMouseDown={() => pick(t.ticker)}
                            >
                              <span style={{ fontSize: 15 }}>{flag}</span>
                              <div style={{ flex: 1, minWidth: 0 }}>
                                <div className="suggestion-name" style={{ fontWeight: 500, fontSize: 13 }}>
                                  {cleanName}
                                </div>
                                <div style={{ fontSize: 11, color: 'var(--color-smoke)', marginTop: 1 }}>
                                  {t.ticker} · {t.exchange || 'Equity'}
                                </div>
                              </div>
                              <span className="suggestion-sector" style={{ fontSize: 11 }}>
                                {t.exchange || ''}
                              </span>
                            </div>
                          )
                        })}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>

                {/* Loading Status */}
                {loading && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="loading-step"
                  >
                    <Loader2 size={12} className="spin-icon" />
                    {loadingStep}
                  </motion.p>
                )}

                {/* Trending Chips */}
                {!loading && (
                  <motion.div variants={fadeUp} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

                    {/* US Stocks */}
                    <div className="trending-strip">
                      <span className="trending-label">🇺🇸 US</span>
                      {trending
                        .filter(t => !t.ticker.includes('.KA'))
                        .slice(0, 6)
                        .map(t => (
                          <button
                            key={t.ticker}
                            className="trending-chip"
                            onClick={() => pick(t.ticker)}
                            title={t.ticker}
                          >
                            <span style={{ fontWeight: 600, fontSize: 11 }}>{t.ticker}</span>
                            <span style={{
                              fontSize: 10,
                              color: 'var(--color-smoke)',
                              maxWidth: 80,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}>
                              {t.name.split(' ')[0]}
                            </span>
                          </button>
                        ))}
                    </div>

                    {/* Pakistan Stocks */}
                    <div className="trending-strip">
                      <span className="trending-label">🇵🇰 PSX</span>
                      {trending
                        .filter(t => t.ticker.includes('.KA'))
                        .slice(0, 6)
                        .map(t => (
                          <button
                            key={t.ticker}
                            className="trending-chip"
                            onClick={() => pick(t.ticker)}
                            title={t.name}
                            style={{ borderColor: 'rgba(0,200,100,0.2)' }}
                          >
                            <span style={{ fontWeight: 600, fontSize: 11 }}>
                              {t.ticker.replace('.KA', '')}
                            </span>
                            <span style={{
                              fontSize: 10,
                              color: 'var(--color-smoke)',
                              maxWidth: 80,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}>
                              {t.name.split(' ')[0]}
                            </span>
                          </button>
                        ))}
                    </div>

                  </motion.div>
                )}

                {/* Recent Items */}
                {!loading && history.length > 0 && (
                  <motion.div variants={fadeUp} className="trending-strip" style={{ marginTop: 12 }}>
                    <span className="trending-label">Recent</span>
                    {history.map(t => (
                      <button key={t} className="history-chip" onClick={() => pick(t)}>
                        <div className="history-chip-dot" />
                        {t}
                      </button>
                    ))}
                  </motion.div>
                )}

                {/* Stats Strip */}
                <motion.div variants={fadeUp} className="stats-strip">
                  {[
                    { value: '10K+', label: 'Tickers Covered' },
                    { value: '5',    label: 'Research Agents'  },
                    { value: 'Live', label: 'Market Data'      },
                  ].map(s => (
                    <div key={s.label} className="stat-item">
                      <div className="stat-value">{s.value}</div>
                      <div className="stat-label">{s.label}</div>
                    </div>
                  ))}
                </motion.div>

              </motion.div>
            </div>

            {/* Right: Globe — full height, slightly overflowing */}
            <div className="hero-visual">
              <ParticleCosmos state="globe" />
            </div>

          </div>
        </div>
      </section>

      {/* Section Spacer */}
      <div className="section-spacer" />

      {/* ── Section 2: Centered Info Block ── */}
      <section className="section-centered">
        <div style={{ position: 'absolute', inset: 0, zIndex: 0, overflow: 'hidden' }}>
          <ParticleCosmos state="dandelion" speedMultiplier={0.3} />
        </div>
        <div
          className="container"
          style={{
            position: 'relative', zIndex: 2,
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', gap: 24,
          }}
        >
          <span className="hero-eyebrow accented" style={{ marginBottom: 0 }}>
            HOW IT WORKS
          </span>
          <h2 style={{
            fontFamily: 'var(--font-acronym)',
            fontSize: 'var(--text-heading-sm)',
            fontWeight: 'var(--font-weight-extralight)',
            color: 'var(--color-bone)',
            letterSpacing: '-0.01em',
            textAlign: 'center',
            lineHeight: 1.3,
          }}>
            Multi-agent research, in seconds.
          </h2>
          <p
            className="centered-body-text"
            style={{
              background: 'rgba(0,0,0,0.55)',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
              borderRadius: 16,
              padding: '20px 32px',
              border: '1px solid rgba(255,255,255,0.07)',
              maxWidth: '56ch',
            }}
          >
            Enter any stock ticker and our AI agents fan out in parallel — fetching live prices, scanning news sentiment, pulling technical indicators, analyzing earnings history, and forecasting the next 7 trading days.
          </p>
        </div>
      </section>

      <div className="section-spacer" />
    </div>
  )
}