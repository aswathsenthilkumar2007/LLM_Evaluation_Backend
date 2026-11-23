import { useEffect, useMemo, useState } from 'react'
import { API_BASE_URL, fetchProviders, runComparison } from './api'
import './App.css'

const PROVIDER_ORDER = ['gemini', 'groq', 'nvidia', 'ollama']

const PROVIDER_INFO = {
  gemini: {
    name: 'Google Gemini',
    shortName: 'Gemini',
    description: 'Google AI model',
    mark: 'G',
    theme: 'gemini',
  },
  groq: {
    name: 'Groq',
    shortName: 'Groq',
    description: 'Groq-hosted language model',
    mark: 'Gr',
    theme: 'groq',
  },
  nvidia: {
    name: 'NVIDIA',
    shortName: 'NVIDIA',
    description: 'NVIDIA NIM endpoint',
    mark: 'N',
    theme: 'nvidia',
  },
  ollama: {
    name: 'Ollama',
    shortName: 'Ollama',
    description: 'Configured Ollama endpoint',
    mark: 'oll',
    theme: 'ollama',
  },
}

const SCORE_DIMENSIONS = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'relevance', label: 'Relevance' },
  { key: 'completeness', label: 'Completeness' },
  { key: 'clarity', label: 'Clarity' },
]

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: 'grid' },
  { id: 'compare', label: 'Compare Models', icon: 'compare' },
  { id: 'history', label: 'Evaluation History', icon: 'clock' },
  { id: 'leaderboard', label: 'Leaderboard', icon: 'chart' },
  { id: 'settings', label: 'Settings', icon: 'settings' },
]

function Icon({ name, size = 18 }) {
  const paths = {
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    compare: (
      <>
        <path d="M8 4H4v16h4M16 4h4v16h-4" />
        <path d="M9 8h6M9 12h6M9 16h6" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    chart: (
      <>
        <path d="M4 19V5M4 19h17" />
        <path d="m7 15 4-4 3 2 5-6" />
        <path d="M16 7h3v3" />
      </>
    ),
    settings: (
      <>
        <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z" />
        <path d="m19.4 15 .1.1a1.8 1.8 0 0 1-2.5 2.5l-.1-.1a1.8 1.8 0 0 0-3 .9v.2a1.8 1.8 0 0 1-3.6 0v-.2a1.8 1.8 0 0 0-3-.9l-.1.1a1.8 1.8 0 0 1-2.5-2.5l.1-.1a1.8 1.8 0 0 0-.9-3h-.2a1.8 1.8 0 0 1 0-3.6h.2a1.8 1.8 0 0 0 .9-3l-.1-.1a1.8 1.8 0 0 1 2.5-2.5l.1.1a1.8 1.8 0 0 0 3-.9v-.2a1.8 1.8 0 0 1 3.6 0v.2a1.8 1.8 0 0 0 3 .9l.1-.1a1.8 1.8 0 0 1 2.5 2.5l-.1.1a1.8 1.8 0 0 0 .9 3h.2a1.8 1.8 0 0 1 0 3.6h-.2a1.8 1.8 0 0 0-.9 3Z" />
      </>
    ),
    sparkle: (
      <>
        <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
        <path d="m19 14 .9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9L19 14Z" />
      </>
    ),
    arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
    external: (
      <>
        <path d="M14 4h6v6M20 4l-9 9" />
        <path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    plus: <path d="M12 5v14M5 12h14" />,
    trash: (
      <>
        <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" />
        <path d="M10 11v5M14 11v5" />
      </>
    ),
    refresh: (
      <>
        <path d="M20 7v5h-5" />
        <path d="M4 17v-5h5" />
        <path d="M5.6 9a7 7 0 0 1 11.6-2L20 12M4 12l2.8 5a7 7 0 0 0 11.6-2" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="12" height="12" rx="2" />
        <path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" />
      </>
    ),
    terminal: (
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="m7 9 3 3-3 3M13 15h4" />
      </>
    ),
  }

  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  )
}

function providerInfo(provider) {
  return PROVIDER_INFO[provider] || {
    name: provider,
    shortName: provider,
    description: 'Backend provider',
    mark: provider.slice(0, 1).toUpperCase(),
    theme: 'other',
  }
}

function ProviderMark({ provider, large = false }) {
  const info = providerInfo(provider)
  return (
    <span className={`provider-mark theme-${info.theme} ${large ? 'provider-mark-large' : ''}`}>
      {info.mark}
    </span>
  )
}

function Sidebar({ activeSection, connected, connection, onNavigate }) {
  return (
    <aside className="sidebar">
      <a className="brand" href="#dashboard" onClick={() => onNavigate('dashboard')}>
        <span className="brand-logo"><Icon name="sparkle" size={20} /></span>
        <span className="brand-text">
          <strong>Multi-LLM</strong>
          <small>Evaluator</small>
        </span>
      </a>

      <div className="sidebar-caption">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        {NAV_ITEMS.map((item) => (
          <button
            aria-current={activeSection === item.id ? 'page' : undefined}
            className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
            key={item.id}
            onClick={() => onNavigate(item.id)}
            type="button"
          >
            <Icon name={item.icon} size={18} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className={`backend-status-card ${connected ? 'connected' : ''}`}>
          <span className={`status-light ${connected ? 'online' : ''}`} />
          <span className="backend-status-copy">
            <strong>Backend status</strong>
            <small>{connection === 'loading' ? 'Checking connection...' : connected ? 'Connected to API' : 'Not connected'}</small>
          </span>
          <span className={`backend-status-tag ${connected ? 'tag-online' : ''}`}>
            {connection === 'loading' ? '...' : connected ? 'Online' : 'Offline'}
          </span>
        </div>
        <div className="sidebar-footnote">
          <span className="footnote-mark">M</span>
          <span><strong>Model workspace</strong><small>Local development</small></span>
        </div>
      </div>
    </aside>
  )
}

function SectionTitle({ kicker, title, description, action }) {
  return (
    <div className="section-title">
      <div className="section-title-copy">
        {kicker && <span className="section-kicker">{kicker}</span>}
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  )
}

function SummaryCard({ label, value, detail, icon, accent }) {
  return (
    <article className="summary-card">
      <div className="summary-card-top">
        <span>{label}</span>
        <span className={`summary-icon accent-${accent}`}><Icon name={icon} size={17} /></span>
      </div>
      <strong className="summary-value">{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function StatusPill({ status, label }) {
  return (
    <span className={`status-pill status-${status}`}>
      <span className="status-dot" />
      {label}
    </span>
  )
}

function ProviderSelector({ providers, selectedProviders, onToggle, connected }) {
  if (!connected) {
    return (
      <div className="provider-load-empty">
        <span className="empty-icon"><Icon name="terminal" size={18} /></span>
        <div><strong>Provider list unavailable</strong><span>Reconnect the backend to load available providers.</span></div>
      </div>
    )
  }

  return (
    <div className="provider-selector-grid">
      {providers.map((provider) => {
        const info = providerInfo(provider)
        const selected = selectedProviders.includes(provider)
        return (
          <button
            aria-pressed={selected}
            className={`provider-select-card ${selected ? 'selected' : ''}`}
            key={provider}
            onClick={() => onToggle(provider)}
            type="button"
          >
            <ProviderMark provider={provider} />
            <span className="provider-select-copy">
              <strong>{info.name}</strong>
              <small>{info.description}</small>
            </span>
            <span className={`selection-box ${selected ? 'checked' : ''}`}>
              {selected && <Icon name="check" size={12} />}
            </span>
          </button>
        )
      })}
    </div>
  )
}

function ProviderResponseCard({
  provider,
  item,
  supported,
  selected,
  isRunning,
  wasSelected,
  isWinner,
}) {
  const info = providerInfo(provider)
  const isTranslation = item?.model_id?.toLowerCase().includes('riva-translate')
  let state = 'ready'
  let stateLabel = selected ? 'Ready' : 'Not selected'

  if (!supported) {
    state = 'unsupported'
    stateLabel = 'Not available'
  } else if (isRunning) {
    state = selected ? 'loading' : 'idle'
    stateLabel = selected ? 'Generating' : 'Not selected'
  } else if (wasSelected && item) {
    state = item.error ? 'failed' : 'success'
    stateLabel = item.error ? 'Failed' : 'Complete'
  } else if (wasSelected) {
    state = 'failed'
    stateLabel = 'No response'
  } else if (wasSelected === false) {
    state = 'idle'
    stateLabel = 'Not run'
  }

  const modelId = wasSelected && item
    ? item.model_id
    : wasSelected === false
      ? 'Not run in this evaluation'
      : 'Resolved by the backend when run'

  return (
    <article className={`model-card model-${state} ${isWinner ? 'is-winner' : ''}`}>
      <div className="model-card-head">
        <div className="model-identity">
          <ProviderMark provider={provider} large />
          <div className="model-title">
            <span className="provider-label">{info.name}</span>
            <span className="model-id" title={modelId}>{modelId}</span>
          </div>
        </div>
        <div className="model-card-badges">
          {isWinner && <span className="winner-tag"><Icon name="sparkle" size={12} /> Winner</span>}
          <StatusPill status={state} label={stateLabel} />
        </div>
      </div>

      <div className="model-card-meta">
        <span className={`http-code ${item?.http_status >= 200 && item.http_status < 300 ? 'http-success' : item?.http_status ? 'http-error' : ''}`}>
          {item?.http_status ? `HTTP ${item.http_status}` : 'HTTP -'}
        </span>
        <span className="response-time"><Icon name="clock" size={13} />{item ? formatTime(item.response_time_ms) : ' - '}</span>
        {item?.rank != null && <span className="rank-label">Rank #{item.rank}</span>}
        {item?.quality_score != null && <span className="card-quality">{Math.round(item.quality_score)}<small> / 100</small></span>}
      </div>

      <div className="model-response">
        <div className="response-label"><span>GENERATED RESPONSE</span><span>{item?.response ? 'Live result' : ''}</span></div>
        {isRunning && selected ? (
          <div className="response-loading">
            <span className="loading-line" /><span className="loading-line short" /><span className="loading-line medium" />
            <small>Waiting for provider response...</small>
          </div>
        ) : item?.error ? (
          <div className="response-error"><Icon name="close" size={15} /><span>{item.error}</span></div>
        ) : item?.response ? (
          <p className="response-text">{item.response}</p>
        ) : (
          <div className="response-placeholder">
            {!supported
              ? 'This provider is not listed by the backend.'
              : wasSelected === false
                ? 'This provider was not selected for the latest evaluation.'
                : 'Run an evaluation to see the provider response here.'}
          </div>
        )}
      </div>

      {item?.scores && (
        <div className="score-list">
          {SCORE_DIMENSIONS.map(({ key, label }) => (
            <div className="score-line" key={key}>
              <span>{label}</span>
              <span className="score-track"><i style={{ width: `${(item.scores[key] / 5) * 100}%` }} /></span>
              <strong>{item.scores[key]}<small>/5</small></strong>
            </div>
          ))}
        </div>
      )}

      {item?.score_rationale && (
        <details className="rationale-details">
          <summary>Evaluation rationale <span>+</span></summary>
          <p>{item.score_rationale}</p>
        </details>
      )}
      {isTranslation && <div className="specialist-badge">Translation-specialized model</div>}
    </article>
  )
}

function QualityChart({ responses }) {
  const scored = responses.filter((item) => item.quality_score != null)
  if (!scored.length) {
    return <div className="chart-placeholder">Actual quality scores will appear after evaluation.</div>
  }

  return (
    <div className="quality-chart">
      {scored.map((item) => (
        <div className="quality-chart-row" key={item.model}>
          <span className="quality-chart-name">{providerInfo(item.model).shortName}</span>
          <span className="quality-chart-track">
            <i style={{ width: `${Math.min(100, Math.max(0, item.quality_score))}%` }} />
          </span>
          <strong>{Math.round(item.quality_score)}</strong>
        </div>
      ))}
    </div>
  )
}

function formatTime(value) {
  if (value == null || !Number.isFinite(Number(value))) return '-'
  return `${Math.round(Number(value))} ms`
}

function App() {
  const [providers, setProviders] = useState([])
  const [selectedProviders, setSelectedProviders] = useState([])
  const [connection, setConnection] = useState('loading')
  const [prompt, setPrompt] = useState('')
  const [referenceAnswer, setReferenceAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [requestError, setRequestError] = useState('')
  const [latestRun, setLatestRun] = useState(null)
  const [history, setHistory] = useState([])
  const [runCount, setRunCount] = useState(0)
  const [activeSection, setActiveSection] = useState('dashboard')

  async function loadProviders() {
    setConnection('loading')
    try {
      const available = await fetchProviders()
      setProviders(available)
      setSelectedProviders((current) => (
        current.length ? current.filter((provider) => available.includes(provider)) : available
      ))
      setConnection('connected')
      setRequestError('')
    } catch (error) {
      setConnection('offline')
      setRequestError(error.message)
    }
  }

  useEffect(() => {
    let active = true
    fetchProviders()
      .then((available) => {
        if (!active) return
        setProviders(available)
        setSelectedProviders(available)
        setConnection('connected')
      })
      .catch((error) => {
        if (!active) return
        setConnection('offline')
        setRequestError(error.message)
      })
    return () => {
      active = false
    }
  }, [])

  const historyResponses = useMemo(
    () => history.flatMap((run) => run.result.responses || []),
    [history],
  )
  const historyScores = historyResponses.filter((item) => item.quality_score != null)
  const averageQuality = historyScores.length
    ? Math.round(historyScores.reduce((sum, item) => sum + item.quality_score, 0) / historyScores.length)
    : null
  const providerAverages = Object.values(historyScores.reduce((totals, item) => {
    const total = totals[item.model] || { model: item.model, score: 0, count: 0 }
    total.score += item.quality_score
    total.count += 1
    totals[item.model] = total
    return totals
  }, {})).sort((a, b) => b.score / b.count - a.score / a.count)

  const currentResponses = latestRun?.result.responses || []
  const winners = latestRun?.result.winner_models || []
  const selectedInLatest = latestRun?.providers || []
  const allCards = [...new Set([...PROVIDER_ORDER, ...providers])]
  const currentScores = currentResponses.filter((item) => item.quality_score != null)
  const latestAverage = currentScores.length
    ? Math.round(currentScores.reduce((sum, item) => sum + item.quality_score, 0) / currentScores.length)
    : null
  const bestRecentProvider = providerAverages[0]
  const supportedCount = providers.length
  const activeSectionTitle = NAV_ITEMS.find((item) => item.id === activeSection)?.label || 'Dashboard'

  function navigateTo(section) {
    setActiveSection(section)
    document.getElementById(section)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  function toggleProvider(provider) {
    setSelectedProviders((current) => (
      current.includes(provider)
        ? current.filter((value) => value !== provider)
        : [...current, provider]
    ))
  }

  function clearPrompt() {
    setPrompt('')
    setReferenceAnswer('')
    setRequestError('')
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setRequestError('')

    if (!prompt.trim()) {
      setRequestError('Enter your prompt before starting an evaluation.')
      return
    }
    if (!selectedProviders.length) {
      setRequestError('Select at least one model to compare.')
      return
    }
    if (connection !== 'connected') {
      setRequestError('The backend is not connected. Check the API URL in Settings.')
      return
    }

    setLoading(true)
    try {
      const result = await runComparison({
        prompt: prompt.trim(),
        ...(referenceAnswer.trim() ? { reference_answer: referenceAnswer.trim() } : {}),
        providers: selectedProviders,
      })
      const run = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        createdAt: new Date().toISOString(),
        prompt: prompt.trim(),
        providers: [...selectedProviders],
        result,
      }
      setLatestRun(run)
      setHistory((current) => [run, ...current].slice(0, 20))
      setRunCount((count) => count + 1)
      setActiveSection('compare')
      window.setTimeout(() => {
        document.getElementById('compare')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 50)
    } catch (error) {
      setRequestError(error.message)
    } finally {
      setLoading(false)
    }
  }

  function restoreRun(run) {
    setLatestRun(run)
    navigateTo('compare')
  }

  const connectionLabel = connection === 'loading'
    ? 'Checking'
    : connection === 'connected'
      ? 'Connected'
      : 'Offline'

  return (
    <div className="app-shell">
      <Sidebar
        activeSection={activeSection}
        connected={connection === 'connected'}
        connection={connection}
        onNavigate={navigateTo}
      />

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span><span className="breadcrumb-separator">/</span>
            <strong>{activeSectionTitle}</strong>
          </div>
          <div className="topbar-right">
            <StatusPill
              status={connection === 'connected' ? 'success' : connection === 'loading' ? 'loading' : 'failed'}
              label={`Backend ${connectionLabel.toLowerCase()}`}
            />
            <a className="docs-button" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
              <Icon name="terminal" size={15} /><span>API docs</span><Icon name="external" size={13} />
            </a>
            <span className="avatar">ML</span>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-section" id="dashboard">
            <div className="welcome-copy">
              <span className="eyebrow"><span className="eyebrow-mark" /> AI MODEL WORKSPACE</span>
              <h1>Multi-LLM Evaluation Workspace</h1>
              <p>Compare, analyze and evaluate multiple language models.</p>
            </div>
            <div className="welcome-actions">
              <span className="workspace-live"><i /> Live workspace</span>
              <button className="button button-primary" onClick={() => navigateTo('compare')} type="button">
                <Icon name="plus" size={16} /> New comparison
              </button>
            </div>
          </section>

          {connection === 'offline' && (
            <div className="alert-banner" role="alert">
              <span className="alert-symbol"><Icon name="close" size={16} /></span>
              <div><strong>Backend connection unavailable</strong><span>{requestError || `Unable to reach ${API_BASE_URL}`}</span></div>
              <button className="button button-secondary button-small" disabled={connection === 'loading'} onClick={loadProviders} type="button">
                <Icon name="refresh" size={14} /> Retry
              </button>
            </div>
          )}

          <section className="summary-grid" aria-label="Workspace overview">
            <SummaryCard
              accent="purple"
              detail="This browser session"
              icon="compare"
              label="Evaluations run"
              value={runCount}
            />
            <SummaryCard
              accent="blue"
              detail={connection === 'connected' ? 'Listed by your backend' : 'Backend unavailable'}
              icon="grid"
              label="Supported models"
              value={connection === 'connected' ? supportedCount : '-'}
            />
            <SummaryCard
              accent="cyan"
              detail={historyScores.length ? `From ${historyScores.length} scored responses` : 'Actual scores appear here'}
              icon="chart"
              label="Average quality"
              value={averageQuality == null ? '-' : `${averageQuality}%`}
            />
            <SummaryCard
              accent="violet"
              detail={bestRecentProvider ? `Recent average ${Math.round(bestRecentProvider.score / bestRecentProvider.count)}%` : 'Based on returned scores'}
              icon="sparkle"
              label="Top model"
              value={bestRecentProvider ? providerInfo(bestRecentProvider.model).shortName : '-'}
            />
          </section>

          <section className="workspace-panel panel" id="workspace">
            <SectionTitle
              action={<span className="section-step"><span>01</span> PROMPT WORKSPACE</span>}
              description="Choose one or more providers and send the same prompt for a fair comparison."
              kicker="CREATE AN EVALUATION"
              title="Set up your comparison"
            />

            <form onSubmit={handleSubmit}>
              <div className="form-block">
                <div className="field-heading">
                  <label>Select models</label>
                  <div className="provider-actions">
                    <span>{selectedProviders.length} selected</span>
                    <button
                      className="link-button"
                      disabled={!providers.length}
                      onClick={() => setSelectedProviders(selectedProviders.length === providers.length ? [] : providers)}
                      type="button"
                    >
                      {selectedProviders.length === providers.length ? 'Clear selection' : 'Select all'}
                    </button>
                  </div>
                </div>
                <ProviderSelector
                  connected={connection === 'connected'}
                  onToggle={toggleProvider}
                  providers={providers}
                  selectedProviders={selectedProviders}
                />
              </div>

              <div className="form-block prompt-block">
                <div className="field-heading">
                  <label htmlFor="prompt">Your prompt <span className="required">*</span></label>
                  <span className="field-hint">A clear prompt helps compare model quality</span>
                </div>
                <textarea
                  id="prompt"
                  maxLength={12000}
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder="Enter your prompt here..."
                  rows={5}
                  value={prompt}
                />
                <div className="prompt-toolbar">
                  <details className="reference-details">
                    <summary><Icon name="plus" size={14} /> Add reference answer <span>Optional</span></summary>
                    <textarea
                      aria-label="Reference answer"
                      maxLength={12000}
                      onChange={(event) => setReferenceAnswer(event.target.value)}
                      placeholder="Provide a trusted answer to help evaluate accuracy..."
                      rows={3}
                      value={referenceAnswer}
                    />
                  </details>
                  <span className="character-count">{prompt.length.toLocaleString()} / 12,000</span>
                </div>
              </div>

              {requestError && connection !== 'offline' && (
                <div className="form-error" role="alert"><Icon name="close" size={15} />{requestError}</div>
              )}

              <div className="form-footer">
                <button className="button button-quiet" disabled={!prompt && !referenceAnswer} onClick={clearPrompt} type="button">
                  <Icon name="trash" size={15} /> Clear prompt
                </button>
                <button className="button button-primary run-button" disabled={loading || connection !== 'connected'} type="submit">
                  {loading
                    ? <><span className="spinner" /> Evaluating models...</>
                    : <><Icon name="sparkle" size={16} /> Run Evaluation <Icon name="arrow" size={15} /></>}
                </button>
              </div>
            </form>
          </section>

          <section className="comparison-section" id="compare">
            <SectionTitle
              action={latestRun && <StatusPill status={latestRun.result.status === 'success' ? 'success' : latestRun.result.status === 'failed' ? 'failed' : 'warning'} label={latestRun.result.status.replace('_', ' ')} />}
              description="Provider results are shown as returned by the FastAPI backend."
              kicker="MODEL COMPARISON"
              title="Compare model responses"
            />
            {latestRun?.result.evaluation_error && (
              <div className="evaluation-error" role="status">
                <strong>Evaluation error</strong><span>{latestRun.result.evaluation_error}</span>
              </div>
            )}
            {latestRun?.result.evaluation_status === 'not_run' && (
              <div className="evaluation-error"><strong>Evaluation not run</strong><span>No successful model responses were available to score.</span></div>
            )}
            <div className="model-card-grid">
              {allCards.map((provider) => {
                const item = currentResponses.find((response) => response.model === provider)
                const wasSelected = latestRun ? selectedInLatest.includes(provider) : undefined
                return (
                  <ProviderResponseCard
                    isRunning={loading}
                    isWinner={winners.includes(provider)}
                    item={item}
                    key={provider}
                    provider={provider}
                    selected={selectedProviders.includes(provider)}
                    supported={providers.includes(provider)}
                    wasSelected={wasSelected}
                  />
                )
              })}
            </div>
            {!latestRun && !loading && (
              <div className="section-empty-note"><Icon name="sparkle" size={15} /> Model responses will appear here after you run an evaluation.</div>
            )}
          </section>

          <section className="evaluation-section panel" id="evaluation">
            <SectionTitle
              description="Scores, ranking and rationale are sourced from the evaluation response."
              kicker="ANALYSIS"
              title="Evaluation results"
              action={latestRun && (
                <StatusPill
                  status={latestRun.result.evaluation_status === 'success' ? 'success' : latestRun.result.evaluation_status === 'failed' ? 'failed' : 'idle'}
                  label={latestRun.result.evaluation_status.replace('_', ' ')}
                />
              )}
            />

            {latestRun ? (
              <>
                <div className="evaluation-metrics">
                  <div className="evaluation-metric">
                    <span>Evaluation status</span>
                    <strong className="evaluation-status-text">{latestRun.result.evaluation_status.replace('_', ' ')}</strong>
                  </div>
                  <div className="evaluation-metric">
                    <span>Evaluation model</span>
                    <strong className="evaluation-model-name">{latestRun.result.evaluation_model || 'Not returned'}</strong>
                  </div>
                  <div className="evaluation-metric">
                    <span>Average quality</span>
                    <strong className="evaluation-score">{latestAverage == null ? 'Not scored' : `${latestAverage}%`}</strong>
                    <small>{latestAverage == null ? 'No score returned' : `${currentScores.length} model scores`}</small>
                  </div>
                  <div className="evaluation-metric">
                    <span>Winner</span>
                    <strong className="evaluation-model-name">
                      {winners.length
                        ? winners.map((provider) => providerInfo(provider).shortName).join(' / ')
                        : 'Not determined'}
                    </strong>
                  </div>
                </div>

                {winners.length > 0 && (
                  <div className="winner-banner">
                    <span className="winner-icon"><Icon name="sparkle" size={17} /></span>
                    <div><small>{winners.length > 1 ? 'TOP-RANKED MODELS' : 'TOP-PERFORMING MODEL'}</small><strong>{winners.map((provider) => providerInfo(provider).name).join(' and ')}</strong></div>
                    <span className="winner-score">{latestAverage == null ? '' : `${Math.max(...currentScores.map((item) => item.quality_score))}% best score`}</span>
                  </div>
                )}

                <div className="quality-panel">
                  <div className="quality-panel-heading">
                    <div><strong>Quality score overview</strong><small>Only scores returned by the backend are charted.</small></div>
                    <span className="quality-chart-key"><i /> Quality score</span>
                  </div>
                  <QualityChart responses={currentResponses} />
                </div>
              </>
            ) : (
              <div className="empty-state">
                <span className="empty-state-icon"><Icon name="chart" size={21} /></span>
                <strong>Evaluation results will appear here</strong>
                <span>Run an evaluation to see the judge model, actual scores, and winner.</span>
              </div>
            )}
          </section>

          <section className="leaderboard-section panel" id="leaderboard">
            <SectionTitle
              description="Ordered by the backend's returned quality score and rank."
              kicker="MODEL PERFORMANCE"
              title="Leaderboard"
              action={latestRun && <span className="section-context">Latest evaluation</span>}
            />
            {currentScores.length ? (
              <div className="leaderboard-wrap">
                <table className="leaderboard-table">
                  <thead><tr><th>RANK</th><th>MODEL</th><th>QUALITY SCORE</th><th>RESPONSE TIME</th><th>STATUS</th></tr></thead>
                  <tbody>
                    {[...currentResponses]
                      .filter((item) => item.quality_score != null && item.rank != null)
                      .sort((a, b) => a.rank - b.rank)
                      .map((item) => (
                        <tr key={item.model}>
                          <td><span className={`rank-number ${item.rank === 1 ? 'first' : ''}`}>{String(item.rank).padStart(2, '0')}</span></td>
                          <td><span className="table-provider"><ProviderMark provider={item.model} /><span><strong>{providerInfo(item.model).name}</strong><small>{item.model_id}</small></span></span></td>
                          <td><strong className="table-quality">{Math.round(item.quality_score)}<small> / 100</small></strong><span className="table-quality-track"><i style={{ width: `${Math.min(100, Math.max(0, item.quality_score))}%` }} /></span></td>
                          <td className="table-time">{formatTime(item.response_time_ms)}</td>
                          <td><StatusPill status={item.error ? 'failed' : 'success'} label={item.error ? 'Failed' : 'Evaluated'} /></td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state compact-empty"><strong>No ranked models yet</strong><span>Ranks appear only when the backend returns evaluated scores.</span></div>
            )}
          </section>

          <section className="history-section panel" id="history">
            <SectionTitle
              action={history.length > 0 && (
                <button className="button button-quiet" onClick={() => { setHistory([]); setLatestRun(null); setRunCount(0) }} type="button">
                  <Icon name="trash" size={14} /> Clear history
                </button>
              )}
              description="Recent evaluations stored for this browser session only. The backend has no history API."
              kicker="SESSION ACTIVITY"
              title="Evaluation history"
            />
            {history.length ? (
              <div className="history-list">
                {history.map((run) => {
                  const scores = (run.result.responses || []).map((item) => item.quality_score).filter((score) => score != null)
                  const average = scores.length ? Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length) : null
                  return (
                    <button className="history-row" key={run.id} onClick={() => restoreRun(run)} type="button">
                      <span className="history-row-icon"><Icon name="clock" size={16} /></span>
                      <span className="history-row-copy"><strong>{run.prompt}</strong><small>{new Date(run.createdAt).toLocaleString()} - {run.providers.length} providers</small></span>
                      <span className={`history-eval-state eval-${run.result.evaluation_status}`}>{run.result.evaluation_status.replace('_', ' ')}</span>
                      <strong className="history-average">{average == null ? '-' : `${average}%`}</strong>
                      <Icon name="arrow" size={15} />
                    </button>
                  )
                })}
              </div>
            ) : (
              <div className="empty-state compact-empty">
                <span className="empty-state-icon"><Icon name="clock" size={19} /></span>
                <strong>No evaluations in this session</strong>
                <span>Completed comparisons will appear here until this page is refreshed.</span>
              </div>
            )}
          </section>

          <section className="settings-section panel" id="settings">
            <SectionTitle
              description="Frontend connection details. Provider credentials stay in the backend environment."
              kicker="CONFIGURATION"
              title="Settings"
            />
            <div className="settings-grid">
              <div className="setting-item">
                <span className="setting-icon"><Icon name="terminal" size={17} /></span>
                <div><strong>FastAPI base URL</strong><small>Read from VITE_API_BASE_URL</small><code>{API_BASE_URL}</code></div>
                <StatusPill status={connection === 'connected' ? 'success' : connection === 'loading' ? 'loading' : 'failed'} label={connection === 'connected' ? 'Connected' : connection === 'loading' ? 'Checking' : 'Offline'} />
              </div>
              <div className="setting-item">
                <span className="setting-icon safe-icon"><Icon name="check" size={17} /></span>
                <div><strong>Provider credentials</strong><small>API keys remain server-side and are never sent to this frontend.</small></div>
                <span className="secure-label">SERVER-SIDE</span>
              </div>
            </div>
            <div className="settings-actions">
              <button className="button button-secondary" disabled={connection === 'loading'} onClick={loadProviders} type="button">
                <Icon name="refresh" size={15} /> Test connection
              </button>
              <a className="button button-quiet" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
                Open API docs <Icon name="external" size={14} />
              </a>
            </div>
          </section>

          <footer className="page-footer">
            <span>Multi-LLM Evaluator <b>·</b> Powered by your FastAPI backend</span>
            <span>Session history is not persisted.</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
