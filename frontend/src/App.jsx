import { useState, useEffect, useCallback, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import HeroSection from './components/HeroSection'
import ResultsDashboard from './components/ResultsDashboard'
import ErrorBoundary from './components/ErrorBoundary'
import AboutModal from './components/AboutModal'
import WatchlistDashboard from './components/WatchlistDashboard'
import CompareMode from './components/CompareMode'
import AgeGate from './components/AgeGate'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -16 },
  transition: { duration: 0.4, ease: [0.4, 0, 0.2, 1] },
}

export default function App() {
  const [ageVerified, setAgeVerified] = useState(() => {
    try { return localStorage.getItem('age_verified') === 'true' }
    catch { return false }
  })
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [trending, setTrending] = useState([])
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('search_history') || '[]') }
    catch { return [] }
  })
  const [loadingStep, setLoadingStep] = useState('')
  const [showAbout, setShowAbout] = useState(false)
  // View: 'home' | 'results' | 'watchlist' | 'compare'
  const [view, setView] = useState('home')
  const wsRef = useRef(null)
  const wsTimeoutRef = useRef(null)
  const gotResultRef = useRef(false)

  const handleAgeVerified = useCallback(() => {
    setAgeVerified(true)
    try { localStorage.setItem('age_verified', 'true') } catch {}
  }, [])

  // Force dark mode attribute to align with Dala's cosmic design guidelines
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'dark')
  }, [])

  // Fetch trending tickers
  useEffect(() => {
    axios.get(`${API_BASE}/api/trending`)
      .then(r => setTrending(r.data.tickers || []))
      .catch(() => setTrending([]))
  }, [])

  const addToHistory = useCallback((ticker) => {
    setHistory(prev => {
      const next = [ticker, ...prev.filter(t => t !== ticker)].slice(0, 10)
      localStorage.setItem('search_history', JSON.stringify(next))
      return next
    })
  }, [])

  // ── WebSocket-based research ─────────────────────────────────────────────
  const handleSearch = useCallback(async (ticker) => {
    const t = (ticker || query).trim().toUpperCase()
    if (!t) return
    setQuery(t)
    setLoading(true)
    setResult(null)
    setError(null)
    setLoadingStep('Connecting to research agent…')
    setView('home') // stay in hero view while loading

    // Reset result flag used by onclose guard
    gotResultRef.current = false

    // Cancel any pending fallback timeout
    if (wsTimeoutRef.current) {
      clearTimeout(wsTimeoutRef.current)
      wsTimeoutRef.current = null
    }

    // Close any existing WebSocket
    if (wsRef.current) {
      try { wsRef.current.close() } catch {}
      wsRef.current = null
    }

    // Determine WebSocket URL.
    // In development: use current host so Vite's proxy forwards /ws to the backend.
    // In production: use VITE_API_URL (Render backend URL) directly, not Vercel host.
    let wsUrl
    if (API_BASE && API_BASE !== '') {
      // Production — API_BASE is https://xxx.onrender.com, swap scheme to wss://
      wsUrl = API_BASE.replace(/^https/, 'wss').replace(/^http/, 'ws') + '/ws/research'
    } else {
      // Development — proxy via Vite
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      wsUrl = `${wsProtocol}//${window.location.host}/ws/research`
    }

    const doHTTPFallback = (reason) => {
      console.warn(`WS fallback (${reason}), switching to HTTP`)
      if (wsRef.current) {
        try { wsRef.current.close() } catch {}
        wsRef.current = null
      }
      handleSearchHTTP(t)
    }

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      // Safety net: if the WS connects but we get no messages in 8s,
      // fall back to HTTP (handles silent proxy hijacking by Vite HMR).
      wsTimeoutRef.current = setTimeout(() => {
        if (!gotResultRef.current) {
          doHTTPFallback('timeout')
        }
      }, 8000)

      ws.onopen = () => {
        ws.send(JSON.stringify({ ticker: t }))
        setLoadingStep('Agent started — resolving ticker…')
        // Reset timeout on successful open — agent may take a while
        clearTimeout(wsTimeoutRef.current)
        // Give a generous 120s once the agent is actually running
        wsTimeoutRef.current = setTimeout(() => {
          if (!gotResultRef.current) doHTTPFallback('agent-timeout')
        }, 120000)
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'step') {
            setLoadingStep(msg.data)
          } else if (msg.type === 'result') {
            gotResultRef.current = true
            clearTimeout(wsTimeoutRef.current)
            addToHistory(msg.data.ticker || t)
            setResult(msg.data)
            setLoading(false)
            setLoadingStep('')
            ws.close()
          } else if (msg.type === 'error') {
            gotResultRef.current = true
            clearTimeout(wsTimeoutRef.current)
            setError(msg.data || 'Research failed.')
            setLoading(false)
            setLoadingStep('')
            ws.close()
          }
        } catch {
          // Ignore parse errors
        }
      }

      ws.onerror = () => {
        clearTimeout(wsTimeoutRef.current)
        if (!gotResultRef.current) doHTTPFallback('onerror')
      }

      ws.onclose = () => {
        clearTimeout(wsTimeoutRef.current)
        // If the socket closed without us ever getting a result, fall back
        if (!gotResultRef.current) {
          doHTTPFallback('onclose-no-result')
        }
      }
    } catch {
      // WebSocket construction failed — fallback to HTTP
      clearTimeout(wsTimeoutRef.current)
      handleSearchHTTP(t)
    }
  }, [query, addToHistory])

  // HTTP fallback for research
  const handleSearchHTTP = useCallback(async (t) => {
    const steps = [
      'Resolving ticker symbol…',
      'Scanning news & sentiment feeds…',
      'Fetching live market prices…',
      'Gathering technical indicators…',
      'Pulling earnings history…',
      'Synthesizing AI research report…',
      'Running quality reflection & scoring…',
    ]
    let stepIdx = 0
    setLoadingStep(steps[0])
    const stepTimer = setInterval(() => {
      stepIdx = Math.min(stepIdx + 1, steps.length - 1)
      setLoadingStep(steps[stepIdx])
    }, 2200)

    try {
      const { data } = await axios.post(`${API_BASE}/api/research`, { ticker: t, limit: 8 })
      clearInterval(stepTimer)
      addToHistory(data.ticker || t)
      setResult(data)
    } catch (e) {
      clearInterval(stepTimer)
      const isNetworkError = !e.response
      setError(
        isNetworkError
          ? 'Cannot connect to the backend. Please ensure the FastAPI server is running on port 8001.'
          : (e.response?.data?.detail || e.message || 'Research failed. Please try again.')
      )
    } finally {
      setLoading(false)
      setLoadingStep('')
    }
  }, [addToHistory])

  const searchByName = useCallback(async (q) => {
    if (!q || q.length < 2) return []
    try {
      const { data } = await axios.get(`${API_BASE}/api/search`, { params: { q } })
      return data.results || []
    } catch {
      return []
    }
  }, [])

  const goHome = useCallback(() => {
    setResult(null)
    setError(null)
    setQuery('')
    setView('home')
  }, [])

  const hasResult = !loading && (result || error)

  // Show age gate if not verified
  if (!ageVerified) {
    return <AgeGate onVerified={handleAgeVerified} />
  }

  return (
    <div className="app-wrapper">
      {/* ── Fixed Top Navigation ── */}
      <nav className="nav">
        <div className="nav-inner">
          <a className="nav-logo" href="/" onClick={(e) => { e.preventDefault(); goHome() }}>
            <div className="nav-logo-icon">
              {/* Chart line icon */}
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            </div>
            <span>AstraQuant</span>
          </a>

          {/* Navigation Links */}
          <div className="nav-links">
            <button className="nav-link" onClick={() => setShowAbout(true)}>ABOUT</button>
            <button className={`nav-link ${view === 'watchlist' ? 'active' : ''}`}
              onClick={() => setView(view === 'watchlist' ? 'home' : 'watchlist')}>
              WATCHLIST
            </button>
            <button className={`nav-link ${view === 'compare' ? 'active' : ''}`}
              onClick={() => setView(view === 'compare' ? 'home' : 'compare')}>
              COMPARE
            </button>
          </div>

          <div className="nav-actions">
            <span className="nav-badge">V3.0.0</span>
          </div>
        </div>
      </nav>

      {/* About Modal */}
      <AboutModal open={showAbout} onClose={() => setShowAbout(false)} />

      {/* Main Views */}
      <AnimatePresence mode="wait">
        {view === 'watchlist' ? (
          <motion.div key="watchlist" {...fadeUp} style={{ paddingTop: 64 }}>
            <div className="container">
              <WatchlistDashboard
                onSearch={(t) => { setView('home'); handleSearch(t) }}
                onBack={goHome}
                searchByName={searchByName}
                trending={trending}
              />
            </div>
          </motion.div>
        ) : view === 'compare' ? (
          <motion.div key="compare" {...fadeUp} style={{ paddingTop: 64 }}>
            <div className="container">
              <CompareMode
                onBack={goHome}
                searchByName={searchByName}
                trending={trending}
              />
            </div>
          </motion.div>
        ) : !hasResult ? (
          <motion.div key="hero" {...fadeUp}>
            <HeroSection
              query={query}
              setQuery={setQuery}
              onSearch={handleSearch}
              searchByName={searchByName}
              loading={loading}
              loadingStep={loadingStep}
              trending={trending}
              history={history}
            />
          </motion.div>
        ) : (
          <motion.div key="results" {...fadeUp} style={{ paddingTop: 64 }}>
            <div className="container results-section">
              <ErrorBoundary onReset={goHome}>
                <ResultsDashboard
                  result={result}
                  error={error}
                  ticker={query}
                  onNewSearch={goHome}
                  onSearch={handleSearch}
                  history={history}
                  trending={trending}
                  loading={loading}
                  loadingStep={loadingStep}
                  query={query}
                  setQuery={setQuery}
                />
              </ErrorBoundary>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Branded Footer */}
      <footer className="footer">
        <p>
          AstraQuant — AI Stock Research Agent.
          {' '}© 2026 <a href="#" onClick={(e) => { e.preventDefault(); goHome() }}>AstraQuant</a>
        </p>
        <p className="footer-credit">
          Made by{' '}
          <a href="https://ayan-shibli-portfolio-g2dci0iwp-ayanhero1859-6169s-projects.vercel.app/" target="_blank" rel="noopener noreferrer">
            Ayan Shibli
          </a>
        </p>
      </footer>
    </div>
  )
}
