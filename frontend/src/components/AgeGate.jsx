import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ShieldCheck, XCircle } from 'lucide-react'

const backdrop = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.4 },
}

const card = {
  initial: { opacity: 0, scale: 0.88, y: 40 },
  animate: { opacity: 1, scale: 1, y: 0, transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] } },
  exit: { opacity: 0, scale: 0.92, y: 20, transition: { duration: 0.25 } },
}

export default function AgeGate({ onVerified }) {
  const [denied, setDenied] = useState(false)

  return (
    <AnimatePresence>
      <motion.div className="age-gate-backdrop" {...backdrop}>
        <motion.div className="age-gate-card" {...card}>
          {!denied ? (
            <>
              {/* Icon */}
              <div className="age-gate-icon">
                <ShieldCheck size={36} />
              </div>

              {/* Title */}
              <h2 className="age-gate-title">Age Verification</h2>
              <p className="age-gate-subtitle">
                This website contains financial market data and AI-driven stock analysis.
                You must be <strong>18 years or older</strong> to access this content.
              </p>

              {/* Question */}
              <p className="age-gate-question">Are you 18 years of age or older?</p>

              {/* Buttons */}
              <div className="age-gate-actions">
                <button className="age-gate-btn age-gate-btn-yes" onClick={onVerified}>
                  Yes, I'm 18+
                </button>
                <button className="age-gate-btn age-gate-btn-no" onClick={() => setDenied(true)}>
                  No, I'm under 18
                </button>
              </div>
            </>
          ) : (
            <>
              {/* Denied State */}
              <div className="age-gate-icon denied">
                <XCircle size={36} />
              </div>
              <h2 className="age-gate-title">Access Denied</h2>
              <p className="age-gate-subtitle">
                Sorry, you must be 18 years or older to access AstraQuant.
                Please close this tab or navigate away.
              </p>
              <div className="age-gate-denied-bar" />
            </>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
