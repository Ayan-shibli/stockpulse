import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unknown error' }
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', error, info)
  }

  render() {
    if (!this.state.hasError) return this.props.children

    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', minHeight: '40vh', gap: 16,
        padding: '2rem', textAlign: 'center',
      }}>
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: 'rgba(255,80,80,0.12)',
          border: '1px solid rgba(255,80,80,0.3)',
          display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: 22,
        }}>⚠</div>
        <h2 style={{
          fontFamily: 'var(--font-acronym)', fontSize: 18,
          fontWeight: 400, color: 'var(--color-bone)', margin: 0,
        }}>
          Something went wrong
        </h2>
        <p style={{
          fontSize: 13, color: 'var(--color-smoke)',
          maxWidth: '40ch', margin: 0, lineHeight: 1.6,
        }}>
          {this.state.message}
        </p>
        <button
          onClick={() => {
            this.setState({ hasError: false, message: '' })
            if (this.props.onReset) this.props.onReset()
          }}
          style={{
            marginTop: 8, padding: '8px 20px', borderRadius: 20,
            border: '1px solid rgba(255,255,255,0.15)',
            background: 'transparent', color: 'var(--color-bone)',
            fontSize: 13, cursor: 'pointer',
            fontFamily: 'var(--font-acronym)',
          }}
        >
          Try again
        </button>
      </div>
    )
  }
}
