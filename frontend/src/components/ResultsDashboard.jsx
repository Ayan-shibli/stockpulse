import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadialBarChart, RadialBar, PolarAngleAxis, Cell
} from 'recharts'
import {
  TrendingUp, TrendingDown, Minus, ExternalLink, RotateCcw,
  Search, Newspaper, Zap, Activity, Target, BookOpen,
  DollarSign, BarChart2, ChevronDown, ChevronUp,
  Cpu, CheckCircle, ArrowUpRight, ArrowDownRight, Shield
} from 'lucide-react'
import ParticleCosmos from './ParticleCosmos'
import StockChart from './StockChart'

const stagger = {
  animate: { transition: { staggerChildren: 0.07 } }
}
const cardAnim = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.4, 0, 0.2, 1] } },
}

function getSentimentColor(sentiment) {
  if (sentiment === 'bullish') return '#8052ff' // Plum Voltage
  if (sentiment === 'bearish') return '#ffb829' // Amber Spark
  return '#9a9a9a' // Smoke
}

function getSentimentIcon(sentiment) {
  if (sentiment === 'bullish') return <TrendingUp size={16} />
  if (sentiment === 'bearish') return <TrendingDown size={16} />
  return <Minus size={16} />
}

function extractDomain(url) {
  try {
    const u = new URL(url)
    return u.hostname.replace('www.', '')
  } catch {
    return url
  }
}

function fmt(val, type = 'number') {
  if (val === null || val === undefined) return '—'
  if (type === 'price') return `$${Number(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  if (type === 'pct') return `${val > 0 ? '+' : ''}${Number(val).toFixed(2)}%`
  if (type === 'cap') {
    const n = Number(val)
    if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
    if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
    if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`
    return `$${n.toLocaleString()}`
  }
  if (type === 'ratio') return Number(val).toFixed(2)
  if (type === 'pctRaw') return `${(Number(val) * 100).toFixed(1)}%`
  return String(val)
}

function ScoreBar({ score }) {
  const pct = Math.round(((score + 1) / 2) * 100)
  return (
    <div className="score-bar-wrapper">
      <div className="score-bar-track">
        <motion.div
          className="score-bar-fill"
          style={{ width: `${pct}%` }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: [0.4, 0, 0.2, 1] }}
        />
      </div>
      <div className="score-bar-labels">
        <span>Bearish −1.0</span>
        <span>Neutral 0</span>
        <span>+1.0 Bullish</span>
      </div>
    </div>
  )
}

function SentimentGauge({ score, sentiment }) {
  const color = getSentimentColor(sentiment)
  const normalised = Math.round(((score + 1) / 2) * 100)
  const data = [{ value: normalised, fill: color }]
  return (
    <ResponsiveContainer width="100%" height={180}>
      <RadialBarChart
        innerRadius="75%"
        outerRadius="100%"
        data={data}
        startAngle={180}
        endAngle={0}
      >
        <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
        <RadialBar dataKey="value" cornerRadius={12} background={{ fill: 'rgba(255, 255, 255, 0.05)' }}>
          {data.map((_, i) => <Cell key={i} fill={color} />)}
        </RadialBar>
        <text
          x="50%"
          y="72%"
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontSize={28}
          fontWeight={600}
          fontFamily="var(--font-acronym)"
        >
          {score > 0 ? '+' : ''}{score.toFixed(2)}
        </text>
        <text
          x="50%"
          y="86%"
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#9a9a9a"
          fontSize={10}
          fontWeight={600}
          letterSpacing="0.05em"
        >
          AI SCORE
        </text>
      </RadialBarChart>
    </ResponsiveContainer>
  )
}

function SentimentBreakdownChart({ score }) {
  const abs = Math.abs(score)
  const data = [
    { name: 'Bullish', value: score > 0 ? Math.round(abs * 100) : 0, fill: '#8052ff' },
    { name: 'Neutral', value: Math.round((1 - abs) * 60),              fill: '#9a9a9a' },
    { name: 'Bearish', value: score < 0 ? Math.round(abs * 100) : 0, fill: '#ffb829' },
  ]
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
        <CartesianGrid strokeDasharray="1 4" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: '#9a9a9a', fontSize: 11, fontWeight: 600, letterSpacing: '0.05em' }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#9a9a9a', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            background: '#000000',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: 'var(--radius-cards)',
            color: '#ffffff',
            fontSize: 13,
            fontFamily: 'var(--font-acronym)'
          }}
          cursor={{ fill: 'rgba(255,255,255,0.02)' }}
        />
        <Bar dataKey="value" radius={[12, 12, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.fill} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function DataRow({ label, value, highlight }) {
  return (
    <div className="data-row">
      <span className="data-row-label">{label}</span>
      <span className="data-row-value" style={highlight ? { color: highlight } : {}}>{value}</span>
    </div>
  )
}

function AgentStepsPanel({ steps, reflection }) {
  const [open, setOpen] = useState(false)
  if (!steps || steps.length === 0) return null
  const score = reflection?.score
  const passed = reflection?.passed
  const critique = reflection?.critique

  return (
    <motion.div variants={cardAnim} className="agent-steps-card">
      <button className="agent-steps-toggle" onClick={() => setOpen(o => !o)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="agent-steps-icon">
            <Cpu size={14} />
          </div>
          <span className="agent-steps-title">Agent Reasoning Steps</span>
          {score && (
            <span className={`agent-score-badge ${passed ? 'passed' : 'retry'}`}>
              {passed ? <CheckCircle size={11} /> : <Activity size={11} />}
              Quality Score: {score}/10
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: 'var(--color-smoke)' }}>{steps.length} steps</span>
          {open ? <ChevronUp size={16} style={{ color: 'var(--color-smoke)' }} /> : <ChevronDown size={16} style={{ color: 'var(--color-smoke)' }} />}
        </div>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            style={{ overflow: 'hidden' }}
          >
            <div className="agent-steps-body">
              {steps.map((step, i) => (
                <div key={i} className="agent-step-item">
                  <div className="agent-step-dot" />
                  <span className="agent-step-text">{step}</span>
                </div>
              ))}
              {critique && (
                <div className="agent-critique">
                  <Shield size={12} style={{ color: 'var(--color-plum-voltage)', flexShrink: 0 }} />
                  <span><strong>Reflection Critique:</strong> {critique}</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function ResultsDashboard({
  result, error, ticker, onNewSearch, onSearch,
  history, trending, loading, loadingStep, query, setQuery
}) {
  if (loading) {
    return (
      <div className="loading-container">
        {/* Dynamic swirling canvas background on loading */}
        <ParticleCosmos state="vortex" speedMultiplier={1.5} />
        
        <div style={{ position: 'relative', zIndex: 2 }}>
          <motion.div
            className="loading-spinner"
            animate={{ rotate: 360 }}
            transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
          />
          <p className="loading-text">Researching <strong>{ticker}</strong></p>
          <p className="loading-step">
            <Activity size={12} className="spin-icon" />
            {loadingStep}
          </p>
          
          {/* Flat Hairline Skeleton Preview */}
          <div style={{ width: '100%', minWidth: 320, maxWidth: 800, marginTop: 48, display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="skeleton" style={{ height: 80, borderRadius: 'var(--radius-cards)' }} />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
              {[...Array(4)].map((_, i) => <div key={i} className="skeleton" style={{ height: 100, borderRadius: 'var(--radius-cards)' }} />)}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div className="skeleton" style={{ height: 200, borderRadius: 'var(--radius-cards)' }} />
              <div className="skeleton" style={{ height: 200, borderRadius: 'var(--radius-cards)' }} />
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <motion.div {...cardAnim}>
        <div className="error-card">
          <div className="error-icon">⚠️</div>
          <div className="error-title">Synthesis Failed</div>
          <div className="error-msg">{error}</div>
          <button className="btn-primary" onClick={onNewSearch}>
            <RotateCcw size={14} /> Ask Again
          </button>
        </div>
      </motion.div>
    )
  }

  if (!result) return null

  const {
    sentiment, sentiment_score, summary, key_events = [], sources = [],
    current_price, change_percent_today, market_cap, pe_ratio,
    analyst_target, analyst_recommendation, week_52_low, week_52_high,
    trend_signal, beta, revenue_growth, profit_margin, next_earnings_date,
    confidence_score, company_name, sector, tools_used = [],
    _steps = [], _reflection = {}
  } = result

  const sentColor = getSentimentColor(sentiment)
  const changePositive = change_percent_today > 0
  const changeNegative = change_percent_today < 0

  return (
    <motion.div variants={stagger} initial="initial" animate="animate" className="results-container">
      {/* Background sparse drifting particles for results dashboard */}
      <ParticleCosmos state="drift" speedMultiplier={0.2} />

      {/* Top Header */}
      <motion.div variants={cardAnim} className="result-header">
        <div className="result-ticker-block">
          <div className="ticker-avatar">{ticker.slice(0, 2).toUpperCase()}</div>
          <div>
            <div className="ticker-title">{ticker.toUpperCase()}</div>
            {company_name && <div className="ticker-company">{company_name}</div>}
            <div className="ticker-subtitle">
              {sector && <span className="data-badge">{sector}</span>}
              {tools_used.length > 0 && (
                <span className="data-badge data-badge-tools">
                  <Cpu size={10} /> {tools_used.length} tools indexed
                </span>
              )}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <span className={`tag ${sentiment}`}>
            {getSentimentIcon(sentiment)} {sentiment}
          </span>
          {confidence_score && (
            <span className="confidence-badge">
              <Shield size={11} /> {confidence_score}/10 Confidence
            </span>
          )}
          <button
            className="btn-secondary"
            style={{ padding: '10px 20px', position: 'relative', zIndex: 10, cursor: 'pointer' }}
            onClick={(e) => { e.stopPropagation(); onNewSearch(); }}
          >
            <Search size={14} /> New Query
          </button>
        </div>
      </motion.div>

      {/* Quick Picks */}
      {(history.length > 0 || trending.length > 0) && (
        <motion.div variants={cardAnim} style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <span className="trending-label">Quick Links:</span>
            {[...new Set([...history, ...trending.slice(0, 5).map(t => t.ticker)])].slice(0, 8).map(t => (
              <button key={t} className="trending-chip" onClick={() => onSearch(t)}>
                {t}
              </button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Price Hero Section */}
      {current_price && (
        <motion.div variants={cardAnim} className="price-hero-card">
          <div className="price-hero-left">
            <div className="price-hero-label">Current Value</div>
            <div className="price-hero-value">{fmt(current_price, 'price')}</div>
            {change_percent_today !== null && change_percent_today !== undefined && (
              <div className={`price-change ${changePositive ? 'up' : changeNegative ? 'down' : ''}`}>
                {changePositive ? <ArrowUpRight size={14} /> : changeNegative ? <ArrowDownRight size={14} /> : <Minus size={14} />}
                {fmt(change_percent_today, 'pct')} today
              </div>
            )}
          </div>
          <div className="price-hero-divider" />
          <div className="price-hero-grid">
            <div className="price-stat">
              <div className="price-stat-label">Market Value</div>
              <div className="price-stat-value">{fmt(market_cap, 'cap')}</div>
            </div>
            <div className="price-stat">
              <div className="price-stat-label">P/E Multiple</div>
              <div className="price-stat-value">{fmt(pe_ratio, 'ratio')}</div>
            </div>
            <div className="price-stat">
              <div className="price-stat-label">Target Level</div>
              <div className="price-stat-value" style={{ color: 'var(--color-plum-voltage)' }}>{fmt(analyst_target, 'price')}</div>
            </div>
            <div className="price-stat">
              <div className="price-stat-label">Consensus</div>
              <div className="price-stat-value" style={{ textTransform: 'capitalize' }}>
                {analyst_recommendation || '—'}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* AI Price Prediction Chart */}
      {current_price && (
        <StockChart ticker={result.ticker || ticker} sources={sources} />
      )}


      <motion.div variants={cardAnim} className="metrics-grid">
        {/* Sentiment Block */}
        <div className="metric-card" style={{ borderColor: sentiment === 'bullish' ? 'rgba(128,82,255,0.3)' : sentiment === 'bearish' ? 'rgba(255,184,41,0.3)' : 'rgba(255,255,255,0.08)' }}>
          <div className="metric-icon" style={{ color: sentColor }}>
            {getSentimentIcon(sentiment)}
          </div>
          <div className="metric-label">Sentiment Pulse</div>
          <div className="metric-value" style={{ color: sentColor, textTransform: 'capitalize' }}>{sentiment}</div>
          <ScoreBar score={sentiment_score} />
        </div>

        {/* AI Score */}
        <div className="metric-card">
          <div className="metric-icon" style={{ color: 'var(--color-plum-voltage)' }}>
            <Target size={18} />
          </div>
          <div className="metric-label">AI Consensus</div>
          <div className="metric-value" style={{ color: 'var(--color-plum-voltage)', fontFamily: 'var(--font-acronym)' }}>
            {sentiment_score > 0 ? '+' : ''}{sentiment_score.toFixed(3)}
          </div>
          <div className="metric-sub">Range: −1.000 to +1.000</div>
        </div>

        {/* Core Insights */}
        <div className="metric-card">
          <div className="metric-icon" style={{ color: 'var(--color-bone)' }}>
            <Zap size={18} />
          </div>
          <div className="metric-label">Events Located</div>
          <div className="metric-value">{key_events.length}</div>
          <div className="metric-sub">Critical data points</div>
        </div>

        {/* Channels */}
        <div className="metric-card">
          <div className="metric-icon" style={{ color: 'var(--color-bone)' }}>
            <Newspaper size={18} />
          </div>
          <div className="metric-label">Sources Indexed</div>
          <div className="metric-value">{sources.length}</div>
          <div className="metric-sub">Validated channels</div>
        </div>
      </motion.div>

      {/* Charts Grid */}
      <motion.div variants={cardAnim} className="charts-row">
        <div className="chart-card">
          <div className="chart-title">
            <Activity size={12} style={{ color: 'var(--color-plum-voltage)' }} />
            Sentiment Orbit Gauge
          </div>
          <SentimentGauge score={sentiment_score} sentiment={sentiment} />
        </div>
        <div className="chart-card">
          <div className="chart-title">
            <BarChart2 size={12} style={{ color: 'var(--color-smoke)' }} />
            Distribution Breakdown
          </div>
          <SentimentBreakdownChart score={sentiment_score} />
        </div>
      </motion.div>

      {/* Technicals + Financials */}
      {(week_52_low || beta || revenue_growth !== undefined || trend_signal) && (
        <motion.div variants={cardAnim} className="fundamentals-row">
          {/* Technical Indicators */}
          {(week_52_low || week_52_high || trend_signal || beta) && (
            <div className="fund-card">
              <div className="fund-card-title">
                <BarChart2 size={14} style={{ color: 'var(--color-plum-voltage)' }} />
                Technical Signals
              </div>
              {week_52_low && week_52_high && (
                <div className="range-bar-section">
                  <div className="range-bar-labels">
                    <span>52W Low: {fmt(week_52_low, 'price')}</span>
                    <span>52W High: {fmt(week_52_high, 'price')}</span>
                  </div>
                  <div className="range-bar-track">
                    {current_price && (
                      <motion.div
                        className="range-bar-thumb"
                        style={{
                          left: `${Math.min(100, Math.max(0, ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100))}%`
                        }}
                        initial={{ left: '0%' }}
                        animate={{
                          left: `${Math.min(100, Math.max(0, ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100))}%`
                        }}
                        transition={{ duration: 1, ease: [0.4, 0, 0.2, 1] }}
                      />
                    )}
                  </div>
                </div>
              )}
              <DataRow
                label="Trend Pulse"
                value={trend_signal?.replace(/_/g, ' ') || '—'}
                highlight={trend_signal?.includes('up') ? 'var(--color-plum-voltage)' : trend_signal?.includes('down') ? 'var(--color-amber-spark)' : undefined}
              />
              <DataRow label="System Beta" value={fmt(beta, 'ratio')} />
            </div>
          )}

          {/* Financial Metrics */}
          {(revenue_growth !== undefined || profit_margin !== undefined || next_earnings_date) && (
            <div className="fund-card">
              <div className="fund-card-title">
                <DollarSign size={14} stroke="var(--color-lichen)" />
                Workspace Earnings Metrics
              </div>
              <DataRow
                label="Revenue Expansion"
                value={fmt(revenue_growth, 'pctRaw')}
                highlight={revenue_growth > 0 ? 'var(--color-plum-voltage)' : revenue_growth < 0 ? 'var(--color-amber-spark)' : undefined}
              />
              <DataRow
                label="Margin Ratio"
                value={fmt(profit_margin, 'pctRaw')}
                highlight={profit_margin > 0.15 ? 'var(--color-plum-voltage)' : profit_margin < 0 ? 'var(--color-amber-spark)' : undefined}
              />
              <DataRow
                label="Next Ledger Event"
                value={
                  next_earnings_date
                    ? Array.isArray(next_earnings_date)
                      ? new Date(next_earnings_date[0] * 1000).toLocaleDateString()
                      : typeof next_earnings_date === 'number'
                        ? new Date(next_earnings_date * 1000).toLocaleDateString()
                        : String(next_earnings_date).substring(0, 10)
                    : '—'
                }
                highlight="var(--color-plum-voltage)"
              />
            </div>
          )}
        </motion.div>
      )}

      {/* AI Synthesis Summary */}
      <motion.div variants={cardAnim} className="summary-card">
        <div className="summary-card-title">
          <BookOpen size={14} />
          AI Research Summary
        </div>
        <p className="summary-text">{summary}</p>
      </motion.div>

      {/* Agent Steps */}
      <AgentStepsPanel steps={_steps} reflection={_reflection} />

      {/* Key Events located */}
      {key_events.length > 0 && (
        <motion.div variants={cardAnim}>
          <div className="section-header">
            <div className="section-title">
              <Zap size={16} style={{ color: 'var(--color-plum-voltage)' }} />
              Key Events Index
            </div>
          </div>
          <div className="events-grid">
            {key_events.map((ev, i) => (
              <motion.div
                key={i}
                className="event-card"
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08, duration: 0.35 }}
              >
                <div className="event-dot" />
                <span className="event-text">{ev}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* News Sources with Reasoning */}
      {sources.length > 0 && (
        <motion.div variants={cardAnim}>
          <div className="section-header">
            <div className="section-title">
              <Newspaper size={16} style={{ color: 'var(--color-lichen)' }} />
              Sources &amp; Why Price May Move
            </div>
            <span className="tag neutral">{sources.length} sources</span>
          </div>

          {/* Price direction rationale box */}
          {summary && (
            <div style={{
              marginBottom: 16, padding: '14px 18px',
              background: sentiment === 'bullish' ? 'rgba(128,82,255,0.07)' : sentiment === 'bearish' ? 'rgba(255,184,41,0.07)' : 'rgba(255,255,255,0.04)',
              border: `1px solid ${sentiment === 'bullish' ? 'rgba(128,82,255,0.22)' : sentiment === 'bearish' ? 'rgba(255,184,41,0.22)' : 'rgba(255,255,255,0.08)'}`,
              borderRadius: 12,
            }}>
              <div style={{ fontSize: 10, color: sentColor, fontFamily: 'var(--font-acronym)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                {getSentimentIcon(sentiment)}
                AI Rationale — Why {sentiment === 'bullish' ? '📈 Price May Rise' : sentiment === 'bearish' ? '📉 Price May Fall' : '➡️ Price May Hold'}
              </div>
              <p style={{ fontSize: 12, color: 'var(--color-ash)', lineHeight: 1.65, margin: 0, fontFamily: 'var(--font-body)' }}>
                {summary}
              </p>
              {key_events.length > 0 && (
                <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 5 }}>
                  {key_events.slice(0, 4).map((ev, i) => (
                    <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <div style={{ width: 5, height: 5, borderRadius: '50%', background: sentColor, flexShrink: 0, marginTop: 5 }} />
                      <span style={{ fontSize: 11, color: 'var(--color-smoke)', lineHeight: 1.5 }}>{ev}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="sources-grid">
            {sources.map((src, i) => {
              const isUrl = src.startsWith('http')
              const domain = isUrl ? extractDomain(src) : src
              const letter = domain[0]?.toUpperCase() || 'D'
              // Assign a key event snippet to each source (round-robin from key_events)
              const snippet = key_events.length > 0 ? key_events[i % key_events.length] : null
              return (
                <motion.a
                  key={i}
                  className="source-card"
                  href={isUrl ? src : '#'}
                  target={isUrl ? '_blank' : undefined}
                  rel="noopener noreferrer"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 8 }}
                >
                  {/* Top row: icon + domain + sentiment badge + link icon */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%' }}>
                    <div className="source-icon">{letter}</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="source-domain">{domain}</div>
                      {isUrl && <div className="source-url-hint" style={{ maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{src}</div>}
                    </div>
                    <span style={{
                      fontSize: 9, padding: '2px 7px', borderRadius: 20, fontWeight: 700,
                      fontFamily: 'var(--font-acronym)', letterSpacing: '0.05em',
                      background: sentColor + '22', color: sentColor, flexShrink: 0,
                    }}>
                      {sentiment?.toUpperCase()}
                    </span>
                    {isUrl && <ExternalLink size={11} style={{ color: 'var(--color-smoke)', flexShrink: 0 }} />}
                  </div>
                  {/* Snippet row */}
                  {snippet && (
                    <p style={{
                      margin: 0, fontSize: 11, color: 'var(--color-ash)',
                      lineHeight: 1.5, fontFamily: 'var(--font-body)',
                      borderTop: '1px solid rgba(255,255,255,0.05)',
                      paddingTop: 6, width: '100%',
                    }}>
                      {snippet.length > 110 ? snippet.slice(0, 110) + '…' : snippet}
                    </p>
                  )}
                </motion.a>
              )
            })}
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
