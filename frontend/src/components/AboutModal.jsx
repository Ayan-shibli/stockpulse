import { motion, AnimatePresence } from 'framer-motion'
import { X, Brain, Cpu, Zap, Activity, Database, BarChart2, Newspaper, Target, Shield, Sparkles, ArrowRight, GitBranch } from 'lucide-react'

const backdrop = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.25 },
}
const modal = {
  initial: { opacity: 0, scale: 0.92, y: 30 },
  animate: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.35, ease: [0.4, 0, 0.2, 1] } },
  exit: { opacity: 0, scale: 0.95, y: 20, transition: { duration: 0.2 } },
}

const PIPELINE_STEPS = [
  { icon: <Target size={14} />, label: 'User Query', desc: 'Ticker or company name' },
  { icon: <Cpu size={14} />, label: 'Ticker Resolver', desc: 'Validates & resolves symbol' },
  { icon: <Zap size={14} />, label: 'Research Agent', desc: 'Multi-agent tool loop' },
  { icon: <Activity size={14} />, label: 'Synthesizer', desc: 'JSON report generation' },
  { icon: <Shield size={14} />, label: 'Reflector', desc: 'Quality gate (score ≥ 7/10)' },
  { icon: <Brain size={14} />, label: 'AI Predictor', desc: '7-day price forecast' },
]

const TOOLS = [
  { name: 'get_company_news', desc: 'News & sentiment analysis' },
  { name: 'get_stock_price', desc: 'Real-time price & fundamentals' },
  { name: 'get_technical_info', desc: 'MA, volume, beta, trend analysis' },
  { name: 'get_earnings_info', desc: 'EPS, revenue growth, margins' },
  { name: 'search_tickers', desc: 'Name → ticker resolution' },
]

export default function AboutModal({ open, onClose }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="about-modal-backdrop"
          {...backdrop}
          onClick={onClose}
        >
          <motion.div
            className="about-modal-card"
            {...modal}
            onClick={e => e.stopPropagation()}
          >
            {/* Close Button */}
            <button className="about-modal-close" onClick={onClose}>
              <X size={18} />
            </button>

            {/* Header */}
            <div className="about-modal-header">
              <div className="about-modal-logo">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
              </div>
              <div>
                <h2 className="about-modal-title">AstraQuant</h2>
                <p className="about-modal-subtitle">AI-Powered Stock Research Agent — v3.0</p>
              </div>
            </div>

            {/* Architecture Pipeline */}
            <div className="about-section">
              <div className="about-section-label">
                <GitBranch size={12} /> ARCHITECTURE PIPELINE
              </div>
              <div className="about-pipeline">
                {PIPELINE_STEPS.map((step, i) => (
                  <div key={i} className="about-pipeline-step">
                    <div className="about-pipeline-icon">{step.icon}</div>
                    <div>
                      <div className="about-pipeline-name">{step.label}</div>
                      <div className="about-pipeline-desc">{step.desc}</div>
                    </div>
                    {i < PIPELINE_STEPS.length - 1 && (
                      <div className="about-pipeline-arrow">
                        <ArrowRight size={10} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Tools */}
            <div className="about-section">
              <div className="about-section-label">
                <Zap size={12} /> RESEARCH TOOLS (5)
              </div>
              <div className="about-tools-grid">
                {TOOLS.map((tool, i) => (
                  <div key={i} className="about-tool-item">
                    <span className="about-tool-name">{tool.name}</span>
                    <span className="about-tool-desc">{tool.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Predictor Features */}
            <div className="about-section">
              <div className="about-section-label">
                <Sparkles size={12} /> AI PREDICTOR
              </div>
              <div className="about-features-list">
                <span>12-feature technical matrix</span>
                <span>·</span>
                <span>Deep learning ensemble</span>
                <span>·</span>
                <span>5-model diverse forecast</span>
                <span>·</span>
                <span>Online learning from outcomes</span>
                <span>·</span>
                <span>RSI/MACD/Bollinger/ATR signals</span>
                <span>·</span>
                <span>Regime-aware momentum decay</span>
              </div>
            </div>

            {/* Footer */}
            <div className="about-modal-footer">
              Built with ❤️ by{' '}
              <a
                href="https://ayan-shibli-portfolio-g2dci0iwp-ayanhero1859-6169s-projects.vercel.app/"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: 'var(--color-plum-voltage)', textDecoration: 'none' }}
              >
                Ayan Shibli
              </a>
              {' '}— Not financial advice. AI predictions are experimental.
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
