import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ComposedChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend
} from 'recharts'
import {
  TrendingUp, TrendingDown, Brain, AlertTriangle, RefreshCw,
  ChevronDown, ChevronUp, Activity, ExternalLink, CheckCircle,
  XCircle, Clock, BarChart2, Newspaper, FlaskConical, Zap, Sparkles
} from 'lucide-react'
import { getRandomQuote, detectMarket } from '../traderQuotes'

const API_BASE = import.meta.env.VITE_API_URL || ''

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  const hist = payload.find(p => p.dataKey === 'price')
  const pred = payload.find(p => p.dataKey === 'predicted')
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-date">{label}</div>
      {hist && (
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
          <span style={{ fontSize: 11, color: '#8052ff' }}>Actual</span>
          <span style={{ color: '#8052ff', fontWeight: 600 }}>${Number(hist.value).toFixed(2)}</span>
        </div>
      )}
      {pred && (
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
          <span style={{ fontSize: 11, color: '#ffb829' }}>AI Forecast</span>
          <span style={{ color: '#ffb829', fontWeight: 600 }}>${Number(pred.value).toFixed(2)}</span>
        </div>
      )}
    </div>
  )
}

// ─── Online Learning Badge (inline in header) ─────────────────────────────────
function OnlineLearningBadge({ ol }) {
  if (!ol) return null
  const tuned = ol.fine_tuned === true
  return (
    <motion.div
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 5,
        padding: '3px 10px', borderRadius: 20,
        background: tuned ? 'rgba(21,132,110,0.18)' : 'rgba(255,255,255,0.06)',
        border: `1px solid ${tuned ? 'rgba(21,132,110,0.4)' : 'rgba(255,255,255,0.1)'}`,
        fontSize: 10, fontFamily: 'var(--font-acronym)', letterSpacing: '0.05em',
        color: tuned ? '#15846e' : 'var(--color-smoke)',
      }}
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.5, duration: 0.35 }}
    >
      <Sparkles size={10} />
      {tuned
        ? `SELF-TUNED · ${ol.outcomes_trained} outcome${ol.outcomes_trained !== 1 ? 's' : ''}`
        : 'BASE MODEL'}
    </motion.div>
  )
}

// ─── Direction Badge ──────────────────────────────────────────────────────────
function DirectionBadge({ direction, confidence }) {
  const isRise = direction === 'rise'
  return (
    <motion.div
      className={`prediction-badge ${isRise ? 'rise' : 'fall'}`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.4, duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
    >
      {isRise ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
      <span>PREDICTED {isRise ? 'RISE' : 'FALL'}</span>
      <span className="prediction-confidence">{confidence}% confidence</span>
    </motion.div>
  )
}

// ─── Skeleton Loader ──────────────────────────────────────────────────────────
function ChartSkeleton() {
  return (
    <div className="chart-skeleton-wrapper">
      <div className="chart-skeleton-header">
        <div className="skeleton" style={{ width: 160, height: 20, borderRadius: 6 }} />
        <div className="skeleton" style={{ width: 120, height: 28, borderRadius: 20 }} />
      </div>
      <div className="skeleton chart-skeleton-body" />
    </div>
  )
}

// ─── Online Learning Details Panel ───────────────────────────────────────────
function OnlineLearningPanel({ ol }) {
  const [open, setOpen] = useState(false)
  if (!ol) return null

  const tuned = ol.fine_tuned === true
  const totalFT = ol.total_fine_tunes ?? 0
  const improvement = ol.loss_improvement_pct
  const reason = ol.reason

  return (
    <div style={{ marginTop: 16, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 14 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)',
          fontSize: 11, letterSpacing: '0.05em', textTransform: 'uppercase', padding: 0,
        }}
      >
        <Sparkles size={12} style={{ color: tuned ? '#15846e' : 'var(--color-smoke)' }} />
        Online Learning
        {tuned ? (
          <span style={{
            marginLeft: 6, padding: '2px 8px', borderRadius: 20,
            background: improvement > 0 ? 'rgba(21,132,110,0.2)' : 'rgba(255,255,255,0.08)',
            color: improvement > 0 ? '#15846e' : 'var(--color-smoke)',
            fontSize: 10, fontWeight: 700,
          }}>
            {improvement > 0 ? `↓${improvement}% loss` : 'tuned'}
          </span>
        ) : (
          <span style={{
            marginLeft: 6, padding: '2px 8px', borderRadius: 20,
            background: 'rgba(255,255,255,0.05)', color: 'var(--color-smoke)',
            fontSize: 10,
          }}>
            {totalFT > 0 ? `${totalFT} total runs` : 'waiting for outcomes'}
          </span>
        )}
        {open ? <ChevronUp size={12} style={{ marginLeft: 4 }} /> : <ChevronDown size={12} style={{ marginLeft: 4 }} />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            style={{ overflow: 'hidden' }}
          >
            <div style={{ marginTop: 14 }}>
              {tuned ? (
                <>
                  {/* Stat row */}
                  <div style={{
                    display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 14,
                    padding: '10px 14px',
                    background: 'rgba(21,132,110,0.06)',
                    borderRadius: 10, border: '1px solid rgba(21,132,110,0.18)',
                  }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: '#15846e', fontFamily: 'var(--font-acronym)' }}>
                        {ol.outcomes_trained}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Outcomes Used</div>
                    </div>
                    <div style={{ width: 1, background: 'rgba(255,255,255,0.08)' }} />
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: improvement > 0 ? '#15846e' : '#ffb829', fontFamily: 'var(--font-acronym)' }}>
                        {improvement > 0 ? `↓${improvement}%` : '—'}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Loss Reduction</div>
                    </div>
                    <div style={{ width: 1, background: 'rgba(255,255,255,0.08)' }} />
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--color-plum-voltage)', fontFamily: 'var(--font-acronym)' }}>
                        {totalFT}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Total Fine-Tunes</div>
                    </div>
                    {ol.avg_loss_before != null && (
                      <>
                        <div style={{ width: 1, background: 'rgba(255,255,255,0.08)' }} />
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--color-ash)', fontFamily: 'var(--font-acronym)' }}>
                            {ol.avg_loss_before.toFixed(5)}
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>MSE Before</div>
                        </div>
                        <div style={{ width: 1, background: 'rgba(255,255,255,0.08)' }} />
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: 14, fontWeight: 700, color: '#15846e', fontFamily: 'var(--font-acronym)' }}>
                            {ol.avg_loss_after.toFixed(5)}
                          </div>
                          <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>MSE After</div>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Explanation */}
                  <div style={{
                    padding: '10px 14px', borderRadius: 10,
                    background: 'rgba(128,82,255,0.07)',
                    border: '1px solid rgba(128,82,255,0.15)',
                    fontSize: 11, color: 'var(--color-ash)', lineHeight: 1.65,
                  }}>
                    🧠 <strong>How it worked:</strong>{' '}
                    The model loaded its cached weights and applied online learning on
                    {' '}{ol.outcomes_trained} resolved prediction outcome{ol.outcomes_trained !== 1 ? 's' : ''} —
                    comparing each past forecast to the actual close price and nudging weights to reduce that error.
                    {improvement > 0 && ` Average MSE loss dropped by ${improvement}% before generating this forecast.`}
                    {ol.last_ft_at && <div style={{ marginTop: 6, color: 'var(--color-smoke)' }}>Last tuned: {ol.last_ft_at}</div>}
                  </div>
                </>
              ) : (
                <div style={{
                  padding: '10px 14px', borderRadius: 10,
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  fontSize: 11, color: 'var(--color-ash)', lineHeight: 1.65,
                }}>
                  {reason === 'no_new_outcomes' || reason === 'no_cached_model' ? (
                    <>
                      ⏳ <strong>Waiting for outcomes:</strong>{' '}
                      Once today&apos;s forecast dates pass and real prices become available,
                      the model will automatically fine-tune itself on those verified errors.
                      {totalFT > 0 && ` It has already self-corrected ${totalFT} time${totalFT !== 1 ? 's' : ''} in total.`}
                    </>
                  ) : (
                    <>
                      🔬 <strong>Base model active.</strong>{' '}
                      No fine-tuning this cycle{reason ? ` (${reason})` : ''}. The model trained fresh on 6 months of price history.
                    </>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Reasoning + Sources Panel ────────────────────────────────────────────────
function ReasoningPanel({ reasoning = [], signals = {}, sources = [], ticker = '' }) {
  const [open, setOpen] = useState(false)
  const sigEntries = Object.entries(signals)
  const urlSources = sources.filter(s => s.startsWith('http'))
  const [quote] = useState(() => getRandomQuote(detectMarket(ticker)))

  return (
    <div style={{ marginTop: 16, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 14 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)',
          fontSize: 11, letterSpacing: '0.05em', textTransform: 'uppercase', padding: 0,
        }}
      >
        <Activity size={12} style={{ color: 'var(--color-plum-voltage)' }} />
        Why this prediction? {urlSources.length > 0 && `+ ${urlSources.length} sources`}
        {open ? <ChevronUp size={12} style={{ marginLeft: 4 }} /> : <ChevronDown size={12} style={{ marginLeft: 4 }} />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            style={{ overflow: 'hidden' }}
          >
            {/* Technical signals grid */}
            {sigEntries.length > 0 && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
                gap: 8, marginTop: 14, marginBottom: 14,
              }}>
                {sigEntries.map(([key, sig]) => (
                  <div key={key} style={{
                    background: 'rgba(128,82,255,0.06)',
                    border: '1px solid rgba(128,82,255,0.15)',
                    borderRadius: 10, padding: '8px 12px',
                  }}>
                    <div style={{ fontSize: 10, color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)', letterSpacing: '0.04em', marginBottom: 2 }}>
                      {sig.label}
                    </div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--color-bone)', fontFamily: 'var(--font-acronym)' }}>
                      {(key === 'Momentum5D' || key === 'Volatility30D')
                        ? `${sig.value}%`
                        : key === 'RSI14'
                          ? sig.value
                          : `$${Number(sig.value).toLocaleString()}`
                      }
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Reasoning bullet points */}
            {reasoning.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
                {reasoning.map((r, i) => (
                  <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                    <div style={{
                      width: 5, height: 5, borderRadius: '50%', flexShrink: 0,
                      background: i === 0 ? 'var(--color-plum-voltage)' : 'var(--color-smoke)',
                      marginTop: 6,
                    }} />
                    <span style={{ fontSize: 12, color: 'var(--color-ash)', fontFamily: 'var(--font-body)', lineHeight: 1.55 }}>
                      {r}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Source links */}
            {urlSources.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div style={{
                  fontSize: 10, color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)',
                  letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: 8,
                  display: 'flex', alignItems: 'center', gap: 6
                }}>
                  <Newspaper size={10} style={{ color: 'var(--color-lichen)' }} />
                  Supporting Sources
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {urlSources.slice(0, 5).map((src, i) => {
                    let domain = src
                    try { domain = new URL(src).hostname.replace('www.', '') } catch {}
                    return (
                      <a
                        key={i}
                        href={src}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          display: 'flex', alignItems: 'center', gap: 8,
                          padding: '6px 10px',
                          background: 'rgba(21,132,110,0.08)',
                          border: '1px solid rgba(21,132,110,0.2)',
                          borderRadius: 8,
                          textDecoration: 'none',
                          transition: 'background 0.2s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'rgba(21,132,110,0.16)'}
                        onMouseLeave={e => e.currentTarget.style.background = 'rgba(21,132,110,0.08)'}
                      >
                        <div style={{
                          width: 22, height: 22, borderRadius: 6,
                          background: 'rgba(21,132,110,0.3)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: 10, fontWeight: 700, color: '#15846e', flexShrink: 0,
                        }}>
                          {domain[0]?.toUpperCase()}
                        </div>
                        <span style={{ fontSize: 11, color: '#15846e', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'var(--font-acronym)' }}>
                          {domain}
                        </span>
                        <ExternalLink size={10} style={{ color: '#15846e', flexShrink: 0 }} />
                      </a>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Trader Quote at the bottom of the reasoning panel */}
            <div style={{
              marginTop: 18,
              padding: '12px 16px',
              background: 'rgba(255, 184, 41, 0.05)',
              border: '1px solid rgba(255, 184, 41, 0.15)',
              borderRadius: 12,
            }}>
              <div style={{
                fontSize: 9, fontFamily: 'var(--font-acronym)', letterSpacing: '0.08em',
                color: '#ffb829', marginBottom: 7,
                display: 'flex', alignItems: 'center', gap: 5,
              }}>
                <span>💡</span>
                TRADER WISDOM
              </div>
              <p style={{
                margin: 0, fontSize: 11.5, lineHeight: 1.6,
                color: 'var(--color-ash)', fontStyle: 'italic',
                fontFamily: 'var(--font-body)',
              }}>
                "{quote.quote}"
              </p>
              <div style={{ marginTop: 8, fontSize: 10, color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)' }}>
                — {quote.author}, {quote.role}
              </div>
            </div>

          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Past Predictions Panel ───────────────────────────────────────────────────
function PastPredictionsPanel({ ticker }) {
  const [open,    setOpen]    = useState(false)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ticker) return
    setLoading(true)
    fetch(`${API_BASE}/api/predict/${ticker}/history`)
      .then(r => r.json())
      .then(d => setHistory(d.history || []))
      .catch(() => setHistory([]))
      .finally(() => setLoading(false))
  }, [ticker])

  // Collect all outcomes across all prediction runs
  const allOutcomes = history.flatMap(entry =>
    (entry.outcomes || []).map(o => ({
      ...o,
      made_at:   entry.made_at,
      direction: entry.direction,
    }))
  )

  // Count pending (predictions with no outcome yet)
  const allPreds = history.flatMap(e => e.predictions || [])
  const pendingCount = allPreds.length - allOutcomes.length

  if (history.length === 0 && !loading) return null

  const correctCount = allOutcomes.filter(o => o.correct).length
  const accuracy     = allOutcomes.length > 0
    ? Math.round((correctCount / allOutcomes.length) * 100)
    : null
  const avgError = allOutcomes.length > 0
    ? Math.abs(allOutcomes.reduce((s, o) => s + o.error_pct, 0) / allOutcomes.length).toFixed(2)
    : null

  return (
    <div style={{ marginTop: 16, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 14 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)',
          fontSize: 11, letterSpacing: '0.05em', textTransform: 'uppercase', padding: 0,
        }}
      >
        <BarChart2 size={12} style={{ color: 'var(--color-amber-spark)' }} />
        Past Predictions vs Reality
        {accuracy !== null && (
          <span style={{
            marginLeft: 6, padding: '2px 8px', borderRadius: 20,
            background: accuracy >= 60 ? 'rgba(21,132,110,0.2)' : 'rgba(255,184,41,0.2)',
            color: accuracy >= 60 ? '#15846e' : '#ffb829',
            fontSize: 10, fontWeight: 700,
          }}>
            {accuracy}% accurate
          </span>
        )}
        {pendingCount > 0 && (
          <span style={{
            marginLeft: 4, padding: '2px 7px', borderRadius: 20,
            background: 'rgba(255,255,255,0.07)', color: 'var(--color-smoke)',
            fontSize: 10,
          }}>
            {pendingCount} pending
          </span>
        )}
        {open ? <ChevronUp size={12} style={{ marginLeft: 4 }} /> : <ChevronDown size={12} style={{ marginLeft: 4 }} />}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            style={{ overflow: 'hidden' }}
          >
            {loading ? (
              <div style={{ color: 'var(--color-smoke)', fontSize: 12, marginTop: 12 }}>Loading history…</div>
            ) : (
              <div style={{ marginTop: 14 }}>
                {/* Summary bar */}
                {allOutcomes.length > 0 && (
                  <div style={{
                    display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 14,
                    padding: '10px 14px',
                    background: 'rgba(255,255,255,0.03)',
                    borderRadius: 10, border: '1px solid rgba(255,255,255,0.07)',
                  }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: accuracy >= 60 ? '#15846e' : '#ffb829', fontFamily: 'var(--font-acronym)' }}>{accuracy}%</div>
                      <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Direction Accuracy</div>
                    </div>
                    <div style={{ width: 1, background: 'rgba(255,255,255,0.08)' }} />
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--color-bone)', fontFamily: 'var(--font-acronym)' }}>{avgError}%</div>
                      <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Avg Price Error</div>
                    </div>
                    <div style={{ width: 1, background: 'rgba(255,255,255,0.08)' }} />
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--color-plum-voltage)', fontFamily: 'var(--font-acronym)' }}>{allOutcomes.length}</div>
                      <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Verified Days</div>
                    </div>
                  </div>
                )}

                {/* Outcomes table */}
                {allOutcomes.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                    <div style={{
                      display: 'grid', gridTemplateColumns: '90px 90px 90px 70px 32px',
                      gap: 4, padding: '4px 8px',
                      fontSize: 9, color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)',
                      letterSpacing: '0.06em', textTransform: 'uppercase',
                    }}>
                      <span>Date</span>
                      <span>Predicted</span>
                      <span>Actual</span>
                      <span>Error</span>
                      <span>Hit?</span>
                    </div>
                    {allOutcomes.slice(-14).reverse().map((o, i) => (
                      <div key={i} style={{
                        display: 'grid', gridTemplateColumns: '90px 90px 90px 70px 32px',
                        gap: 4, padding: '6px 8px',
                        background: o.correct ? 'rgba(21,132,110,0.07)' : 'rgba(255,184,41,0.07)',
                        borderRadius: 8,
                        border: `1px solid ${o.correct ? 'rgba(21,132,110,0.15)' : 'rgba(255,184,41,0.15)'}`,
                        fontSize: 11, fontFamily: 'var(--font-acronym)',
                      }}>
                        <span style={{ color: 'var(--color-ash)' }}>{o.date}</span>
                        <span style={{ color: '#8052ff' }}>${o.predicted.toFixed(2)}</span>
                        <span style={{ color: 'var(--color-bone)' }}>${o.actual.toFixed(2)}</span>
                        <span style={{ color: Math.abs(o.error_pct) < 2 ? '#15846e' : '#ffb829' }}>
                          {o.error_pct > 0 ? '+' : ''}{o.error_pct}%
                        </span>
                        <span>
                          {o.correct
                            ? <CheckCircle size={13} style={{ color: '#15846e' }} />
                            : <XCircle size={13} style={{ color: '#ffb829' }} />
                          }
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    color: 'var(--color-smoke)', fontSize: 12, marginTop: 8,
                  }}>
                    <Clock size={13} />
                    {pendingCount > 0
                      ? `${pendingCount} prediction(s) stored — outcomes will appear when the dates arrive.`
                      : 'No completed predictions yet. Check back after the forecast dates pass.'
                    }
                  </div>
                )}

                {/* Learning insight */}
                {allOutcomes.length >= 3 && (
                  <div style={{
                    marginTop: 12, padding: '8px 12px',
                    background: 'rgba(128,82,255,0.07)',
                    border: '1px solid rgba(128,82,255,0.15)',
                    borderRadius: 8, fontSize: 11, color: 'var(--color-ash)',
                    lineHeight: 1.5,
                  }}>
                    🧠 <strong>Model Insight:</strong>{' '}
                    {accuracy >= 70
                      ? `Strong directional accuracy (${accuracy}%). The AI model is reading ${ticker}'s trend well.`
                      : accuracy >= 50
                        ? `Moderate accuracy (${accuracy}%). Price magnitude predictions are within ${avgError}% on average.`
                        : `Low accuracy (${accuracy}%) so far — market may be highly volatile or reacting to unexpected news. Average error: ${avgError}%.`
                    }
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Backtest Panel ───────────────────────────────────────────────────────────
function BacktestPanel({ ticker }) {
  const [open,       setOpen]       = useState(false)
  const [loading,    setLoading]    = useState(false)
  const [result,     setResult]     = useState(null)
  const [error,      setError]      = useState(null)
  const [daysBack,   setDaysBack]   = useState(7)
  const [hasRun,     setHasRun]     = useState(false)

  const runBacktest = async () => {
    setLoading(true); setError(null); setResult(null); setHasRun(true)
    try {
      const res  = await fetch(`${API_BASE}/api/predict/${ticker}/backtest?days_back=${daysBack}`)
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || `HTTP ${res.status}`)
      setResult(json)
    } catch (e) {
      setError(e.message || 'Backtest failed')
    } finally {
      setLoading(false)
    }
  }

  const gradeColor = (g) => {
    if (g === 'A') return '#15846e'
    if (g === 'B') return '#8052ff'
    if (g === 'C') return '#ffb829'
    return '#ff6b6b'
  }

  return (
    <div style={{ marginTop: 16, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 14 }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <button
          onClick={() => setOpen(o => !o)}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)',
            fontSize: 11, letterSpacing: '0.05em', textTransform: 'uppercase', padding: 0,
          }}
        >
          <FlaskConical size={12} style={{ color: '#15846e' }} />
          Backtest Report
          {result && (
            <span style={{
              marginLeft: 6, padding: '2px 8px', borderRadius: 20,
              background: gradeColor(result.grade) + '22',
              color: gradeColor(result.grade),
              fontSize: 10, fontWeight: 700,
            }}>
              Grade {result.grade} · {result.direction_accuracy}% accurate
            </span>
          )}
          {open ? <ChevronUp size={12} style={{ marginLeft: 4 }} /> : <ChevronDown size={12} style={{ marginLeft: 4 }} />}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28 }}
            style={{ overflow: 'hidden' }}
          >
            <div style={{ marginTop: 14 }}>
              {/* Controls */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 11, color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)' }}>Days to test:</span>
                  {[5, 7, 10, 14].map(d => (
                    <button
                      key={d}
                      onClick={() => setDaysBack(d)}
                      style={{
                        padding: '4px 10px', borderRadius: 20,
                        border: `1px solid ${daysBack === d ? 'rgba(128,82,255,0.5)' : 'rgba(255,255,255,0.1)'}`,
                        background: daysBack === d ? 'rgba(128,82,255,0.18)' : 'rgba(255,255,255,0.04)',
                        color: daysBack === d ? 'var(--color-plum-voltage)' : 'var(--color-smoke)',
                        fontSize: 11, fontFamily: 'var(--font-acronym)',
                        cursor: 'pointer', transition: 'all 0.2s',
                      }}
                    >{d}d</button>
                  ))}
                </div>
                <button
                  onClick={runBacktest}
                  disabled={loading}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '6px 16px', borderRadius: 20,
                    border: '1px solid rgba(21,132,110,0.4)',
                    background: loading ? 'rgba(21,132,110,0.08)' : 'rgba(21,132,110,0.16)',
                    color: '#15846e',
                    fontSize: 11, fontFamily: 'var(--font-acronym)', letterSpacing: '0.04em',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={e => !loading && (e.currentTarget.style.background = 'rgba(21,132,110,0.28)')}
                  onMouseLeave={e => !loading && (e.currentTarget.style.background = 'rgba(21,132,110,0.16)')}
                >
                  {loading
                    ? <><motion.span animate={{ rotate: 360 }} transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }} style={{ display: 'inline-flex' }}><RefreshCw size={11} /></motion.span> Training…</>
                    : <><Zap size={11} /> Run Backtest</>}
                </button>
              </div>

              {/* Description callout (first time, before run) */}
              {!hasRun && (
                <div style={{
                  padding: '10px 14px', borderRadius: 10,
                  background: 'rgba(128,82,255,0.06)',
                  border: '1px solid rgba(128,82,255,0.15)',
                  fontSize: 11, color: 'var(--color-ash)', lineHeight: 1.6,
                }}>
                  <span style={{ color: 'var(--color-plum-voltage)', fontWeight: 600 }}>How it works: </span>
                  The model pretends today is <strong>{daysBack} trading days ago</strong>. It trains on only the older data,
                  predicts the next {daysBack} days, then compares to actual prices we <em>already have</em> — instant accuracy!
                </div>
              )}

              {/* Loading spinner */}
              {loading && (
                <div style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center',
                  gap: 12, padding: '20px 0',
                }}>
                  <motion.div
                    style={{
                      width: 32, height: 32, borderRadius: '50%',
                      border: '2px solid rgba(21,132,110,0.2)',
                      borderTop: '2px solid #15846e',
                    }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
                  />
                  <span style={{ fontSize: 11, color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)' }}>
                    Training AI model on historical data…
                  </span>
                </div>
              )}

              {/* Error */}
              {error && (
                <div style={{
                  display: 'flex', gap: 8, alignItems: 'center',
                  padding: '10px 14px', borderRadius: 10,
                  background: 'rgba(255,107,107,0.08)',
                  border: '1px solid rgba(255,107,107,0.2)',
                  fontSize: 11, color: '#ff6b6b',
                }}>
                  <AlertTriangle size={12} /> {error}
                </div>
              )}

              {/* Results */}
              {result && !loading && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35 }}
                >
                  {/* Grade + stat summary */}
                  <div style={{
                    display: 'flex', gap: 12, flexWrap: 'wrap',
                    marginBottom: 14, padding: '12px 16px',
                    background: 'rgba(255,255,255,0.03)',
                    borderRadius: 12, border: '1px solid rgba(255,255,255,0.07)',
                    alignItems: 'center',
                  }}>
                    {/* Grade circle */}
                    <div style={{
                      width: 52, height: 52, borderRadius: '50%',
                      border: `2px solid ${gradeColor(result.grade)}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      flexShrink: 0,
                      background: gradeColor(result.grade) + '18',
                    }}>
                      <span style={{ fontSize: 20, fontWeight: 700, color: gradeColor(result.grade), fontFamily: 'var(--font-acronym)' }}>
                        {result.grade}
                      </span>
                    </div>

                    <div style={{ flex: 1, display: 'flex', gap: 20, flexWrap: 'wrap' }}>
                      <div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: gradeColor(result.grade), fontFamily: 'var(--font-acronym)' }}>
                          {result.direction_accuracy}%
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Direction Accuracy</div>
                      </div>
                      <div style={{ width: 1, background: 'rgba(255,255,255,0.08)', alignSelf: 'stretch' }} />
                      <div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--color-bone)', fontFamily: 'var(--font-acronym)' }}>
                          {result.avg_price_error}%
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Avg Price Error</div>
                      </div>
                      <div style={{ width: 1, background: 'rgba(255,255,255,0.08)', alignSelf: 'stretch' }} />
                      <div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: result.direction_correct ? '#15846e' : '#ffb829', fontFamily: 'var(--font-acronym)' }}>
                          {result.direction_correct ? '✅' : '❌'}
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Overall Direction</div>
                      </div>
                      <div style={{ width: 1, background: 'rgba(255,255,255,0.08)', alignSelf: 'stretch' }} />
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-ash)', fontFamily: 'var(--font-acronym)' }}>
                          {result.cutoff_date}
                        </div>
                        <div style={{ fontSize: 10, color: 'var(--color-smoke)' }}>Training cutoff</div>
                      </div>
                    </div>
                  </div>

                  {/* Direction verdict callout */}
                  <div style={{
                    marginBottom: 12, padding: '8px 14px', borderRadius: 10,
                    background: result.direction_correct ? 'rgba(21,132,110,0.08)' : 'rgba(255,184,41,0.08)',
                    border: `1px solid ${result.direction_correct ? 'rgba(21,132,110,0.2)' : 'rgba(255,184,41,0.2)'}`,
                    fontSize: 11, color: result.direction_correct ? '#15846e' : '#ffb829',
                    display: 'flex', alignItems: 'center', gap: 8,
                  }}>
                    {result.direction_correct
                      ? <CheckCircle size={12} />
                      : <XCircle size={12} />}
                    Overall: model predicted <strong>{result.overall_direction.toUpperCase()}</strong>,
                    actual was <strong>{result.actual_direction.toUpperCase()}</strong>
                    {result.direction_correct ? ' — Direction CORRECT ✅' : ' — Direction MISSED ❌'}
                  </div>

                  {/* Per-day table */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                    <div style={{
                      display: 'grid', gridTemplateColumns: '90px 90px 90px 68px 36px',
                      gap: 4, padding: '4px 8px',
                      fontSize: 9, color: 'var(--color-smoke)', fontFamily: 'var(--font-acronym)',
                      letterSpacing: '0.06em', textTransform: 'uppercase',
                    }}>
                      <span>Date</span>
                      <span>Predicted</span>
                      <span>Actual</span>
                      <span>Error</span>
                      <span>Hit?</span>
                    </div>
                    {result.rows.map((row, i) => (
                      <motion.div
                        key={row.date}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.05 }}
                        style={{
                          display: 'grid', gridTemplateColumns: '90px 90px 90px 68px 36px',
                          gap: 4, padding: '6px 8px',
                          background: row.hit ? 'rgba(21,132,110,0.07)' : 'rgba(255,184,41,0.07)',
                          borderRadius: 8,
                          border: `1px solid ${row.hit ? 'rgba(21,132,110,0.15)' : 'rgba(255,184,41,0.15)'}`,
                          fontSize: 11, fontFamily: 'var(--font-acronym)',
                        }}
                      >
                        <span style={{ color: 'var(--color-ash)' }}>{row.date}</span>
                        <span style={{ color: '#8052ff' }}>${row.predicted.toFixed(2)}</span>
                        <span style={{ color: 'var(--color-bone)' }}>${row.actual.toFixed(2)}</span>
                        <span style={{ color: Math.abs(row.error_pct) < 2 ? '#15846e' : '#ffb829' }}>
                          {row.error_pct > 0 ? '+' : ''}{row.error_pct}%
                        </span>
                        <span>
                          {row.hit
                            ? <CheckCircle size={13} style={{ color: '#15846e' }} />
                            : <XCircle size={13} style={{ color: '#ffb829' }} />}
                        </span>
                      </motion.div>
                    ))}
                  </div>

                  {/* Model insight */}
                  <div style={{
                    marginTop: 12, padding: '8px 12px',
                    background: 'rgba(128,82,255,0.07)',
                    border: '1px solid rgba(128,82,255,0.15)',
                    borderRadius: 8, fontSize: 11, color: 'var(--color-ash)', lineHeight: 1.5,
                  }}>
                    🧠 <strong>Backtest Insight:</strong>{' '}
                    {result.direction_accuracy >= 70
                      ? `Strong directional accuracy (${result.direction_accuracy}%) with avg price error of only ${result.avg_price_error}%. The AI model is reading ${result.ticker}'s trend well.`
                      : result.direction_accuracy >= 50
                        ? `Moderate accuracy (${result.direction_accuracy}%). Price magnitude within ${result.avg_price_error}% on average. Acceptable for volatile assets.`
                        : `Low accuracy (${result.direction_accuracy}%) in this window — market may have reacted to unexpected news. Avg error: ${result.avg_price_error}%.`
                    }
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function StockChart({ ticker, sources = [] }) {
  const [data,     setData]     = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)
  const [retrying, setRetrying] = useState(false)
  const abortRef = useRef(null)

  const load = async () => {
    if (abortRef.current) abortRef.current.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setLoading(true); setError(null); setRetrying(false)

    try {
      const res  = await fetch(`${API_BASE}/api/predict/${ticker}`, { signal: ctrl.signal })
      if (!res.ok) {
        const json = await res.json().catch(() => ({}))
        throw new Error(json.detail || `HTTP ${res.status}`)
      }
      setData(await res.json())
    } catch (e) {
      if (e.name === 'AbortError') return
      setError(e.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!ticker) return
    load()
    return () => abortRef.current?.abort()
  }, [ticker])

  const chartData = (() => {
    if (!data) return []

    // Base historical prices (purple line)
    const hist = (data.historical || []).map(h => ({ date: h.date, price: h.price }))

    // Future predictions (yellow dashed line)
    const pred = (data.predictions || []).map(p => ({ date: p.date, predicted: p.price }))
    if (hist.length && pred.length) {
      pred[0] = { ...pred[0], bridge: hist[hist.length - 1].price }
    }

    return [...hist, ...pred]
  })()

  const splitDate = data?.predictions?.[0]?.date
  const allPrices = chartData.flatMap(d => [d.price, d.predicted].filter(Boolean))
  const yMin      = allPrices.length ? Math.min(...allPrices) * 0.97 : 'auto'
  const yMax      = allPrices.length ? Math.max(...allPrices) * 1.03 : 'auto'
  const interval  = Math.max(1, Math.floor(chartData.length / 12))

  return (
    <motion.div
      className="stock-chart-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
    >
      {/* Header */}
      <div className="chart-card-header">
        <div className="chart-card-title-row">
          <div className="chart-card-icon"><Brain size={14} /></div>
          <span className="chart-card-title">AI Price Forecast</span>
          <span className="chart-ticker-label">{ticker}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {data && <OnlineLearningBadge ol={data.online_learning} />}
          {data && <DirectionBadge direction={data.direction} confidence={data.confidence} />}
          {!loading && (
            <button className="chart-refresh-btn" onClick={load} title="Re-train model" disabled={retrying}>
              <RefreshCw size={13} />
            </button>
          )}
        </div>
      </div>

      {/* Chart body */}
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div key="skeleton" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <ChartSkeleton />
            <div className="chart-training-label">
              <span className="chart-training-dot" />
              Training AI model on {ticker} history…
            </div>
            <div className="chart-loading-notice">
              <Clock size={12} />
              <span>The graph may take <strong>3–5 minutes</strong> to appear while the AI model trains on historical data. Please be patient.</span>
            </div>
          </motion.div>
        ) : error ? (
          <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="chart-error">
            <AlertTriangle size={18} style={{ color: '#ffb829' }} />
            <span>{error}</span>
            <button className="btn-secondary" style={{ padding: '6px 14px', fontSize: 12 }} onClick={load}>Retry</button>
          </motion.div>
        ) : (
          <motion.div key="chart" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData} margin={{ top: 10, right: 8, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#8052ff" stopOpacity={0.18} />
                    <stop offset="95%" stopColor="#8052ff" stopOpacity={0.0}  />
                  </linearGradient>
                  <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#ffb829" stopOpacity={0.22} />
                    <stop offset="95%" stopColor="#ffb829" stopOpacity={0.0}  />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="2 6" stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: '#5a5a6e', fontSize: 10, fontFamily: 'var(--font-acronym)' }} axisLine={false} tickLine={false} interval={interval} />
                <YAxis domain={[yMin, yMax]} tick={{ fill: '#5a5a6e', fontSize: 10, fontFamily: 'var(--font-acronym)' }} axisLine={false} tickLine={false} tickFormatter={v => `$${v >= 1000 ? (v/1000).toFixed(1)+'k' : v.toFixed(0)}`} width={52} />
                <Tooltip content={<ChartTooltip />} />
                {splitDate && (
                  <ReferenceLine x={splitDate} stroke="rgba(255,255,255,0.18)" strokeDasharray="4 4"
                    label={{ value: 'TODAY', position: 'top', fill: '#5a5a6e', fontSize: 9, fontFamily: 'var(--font-acronym)', letterSpacing: '0.08em' }}
                  />
                )}
                {/* Actual historical prices — purple */}
                <Area type="monotone" dataKey="price" stroke="#8052ff" strokeWidth={2} fill="url(#histGrad)" dot={false} activeDot={{ r: 4, fill: '#8052ff', stroke: '#000', strokeWidth: 2 }} name="Actual price" connectNulls={false} />

                {/* Future predictions — yellow dashed */}
                <Area type="monotone" dataKey="predicted" stroke="#ffb829" strokeWidth={2} strokeDasharray="5 4" fill="url(#predGrad)" dot={{ r: 3, fill: '#ffb829', stroke: '#000', strokeWidth: 1.5 }} activeDot={{ r: 5, fill: '#ffb829', stroke: '#000', strokeWidth: 2 }} name="AI Forecast" connectNulls={false} />
                <Legend verticalAlign="bottom" height={28} formatter={(value) => {
                  const colors = { 'Actual price': '#8052ff', 'AI Forecast': '#ffb829' }
                  return (
                    <span style={{ color: colors[value] || '#fff', fontSize: 11, fontFamily: 'var(--font-acronym)', letterSpacing: '0.05em' }}>
                      {value.toUpperCase()}
                    </span>
                  )
                }} />
              </ComposedChart>
            </ResponsiveContainer>

            {/* Online Learning Details */}
            <OnlineLearningPanel ol={data?.online_learning} />

            {/* Reasoning + Sources */}
            <ReasoningPanel
              reasoning={data?.reasoning || []}
              signals={data?.signals || {}}
              sources={sources}
              ticker={ticker}
            />

            {/* Past Predictions vs Reality */}
            <PastPredictionsPanel ticker={ticker} />

            {/* Backtest Report */}
            <BacktestPanel ticker={ticker} />

            {/* Disclaimer */}
            <div className="chart-disclaimer">
              <AlertTriangle size={10} />
              AI Forecast — Not Financial Advice. Model trained on historical data · Self-corrects on verified outcomes.
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
