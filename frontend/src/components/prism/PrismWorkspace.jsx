import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import ConversationWorkspace from './ConversationWorkspace'

const PROVIDER_ORDER = ['gemini', 'groq', 'nvidia', 'ollama']
const HEADER_DATE = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric' }).format(new Date())

const PROVIDER_INFO = {
  gemini: { name: 'Google Gemini', shortName: 'Gemini', icon: 'sparkle', theme: 'gemini' },
  groq: { name: 'Groq', shortName: 'Groq', icon: 'route', theme: 'groq' },
  nvidia: { name: 'NVIDIA', shortName: 'NVIDIA', icon: 'cpu', theme: 'nvidia' },
  ollama: { name: 'Ollama', shortName: 'Ollama', icon: 'drive', theme: 'ollama' },
}

const SCORE_DIMENSIONS = [
  { key: 'instruction_following', label: 'Instruction following' },
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'completeness', label: 'Completeness' },
  { key: 'formatting', label: 'Formatting' },
  { key: 'relevance', label: 'Relevance' },
  { key: 'clarity', label: 'Reasoning' },
]

const DETAILED_SCORE_DIMENSIONS = [
  { key: 'relevance', evidenceKey: 'response_quality', label: 'Response Quality / Relevance' },
  { key: 'clarity', evidenceKey: 'reasoning_clarity', label: 'Reasoning Clarity' },
  { key: 'accuracy', evidenceKey: 'factual_support', label: 'Factual Support / Potential Hallucination' },
  { key: 'instruction_following', evidenceKey: 'instruction_following', label: 'Instruction Following' },
  { key: 'formatting', evidenceKey: 'formatting', label: 'Formatting' },
  { key: 'completeness', evidenceKey: 'completeness', label: 'Completeness' },
]

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: 'grid' },
  { id: 'comparison', label: 'Model Comparison', icon: 'columns' },
  { id: 'playground', label: 'Prompt Playground', icon: 'terminal' },
  { id: 'analytics', label: 'Evaluation Analytics', icon: 'chart' },
  { id: 'leaderboard', label: 'Leaderboard', icon: 'trophy' },
  { id: 'history', label: 'Evaluation History', icon: 'history' },
  { id: 'settings', label: 'Settings', icon: 'settings' },
]

const PROMPT_SUGGESTIONS = [
  { label: 'Code review', prompt: 'Review the following code for correctness, security, and readability:\n\n' },
  { label: 'Summarization', prompt: 'Summarize the following text in three concise bullet points:\n\n' },
  { label: 'Reasoning', prompt: 'Solve this problem step by step and explain your assumptions:\n\n' },
  { label: 'Creative writing', prompt: 'Write an engaging short story about:\n\n' },
]

const ICON_PATHS = {
  grid: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  terminal: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="m7 9 3 3-3 3M13 15h4" />
    </>
  ),
  columns: (
    <>
      <rect x="3" y="4" width="8" height="16" rx="2" />
      <rect x="13" y="4" width="8" height="16" rx="2" />
    </>
  ),
  chart: (
    <>
      <path d="M4 19V5M4 19h17" />
      <path d="m7 15 4-4 3 2 5-6" />
    </>
  ),
  trophy: (
    <>
      <path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4Z" />
      <path d="M7 6H4v2a4 4 0 0 0 4 4M17 6h3v2a4 4 0 0 1-4 4" />
    </>
  ),
  history: (
    <>
      <path d="M3 12a9 9 0 1 0 2.6-6.4L3 8" />
      <path d="M3 3v5h5M12 7v5l3 2" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="m19.4 15 .1.1a1.8 1.8 0 0 1-2.5 2.5l-.1-.1a1.8 1.8 0 0 0-3 .9v.2a1.8 1.8 0 0 1-3.6 0v-.2a1.8 1.8 0 0 0-3-.9l-.1.1a1.8 1.8 0 0 1-2.5-2.5l.1-.1a1.8 1.8 0 0 0-.9-3h-.2a1.8 1.8 0 0 1 0-3.6h.2a1.8 1.8 0 0 0 .9-3l-.1-.1a1.8 1.8 0 0 1 2.5-2.5l.1.1a1.8 1.8 0 0 0 3-.9v-.2a1.8 1.8 0 0 1 3.6 0v.2a1.8 1.8 0 0 0 3 .9l.1-.1a1.8 1.8 0 0 1 2.5 2.5l-.1.1a1.8 1.8 0 0 0 .9 3h.2a1.8 1.8 0 0 1 0 3.6h-.2a1.8 1.8 0 0 0-.9 3Z" />
    </>
  ),
  sparkle: (
    <>
      <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
      <path d="m19 14 .9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9L19 14Z" />
    </>
  ),
  route: (
    <>
      <circle cx="6" cy="18" r="2" />
      <circle cx="18" cy="6" r="2" />
      <path d="M8 18h4a3 3 0 0 0 3-3V9a3 3 0 0 1 3-3M6 16V9a3 3 0 0 1 3-3h2" />
    </>
  ),
  cpu: (
    <>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 9h6v6H9zM9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4" />
    </>
  ),
  drive: (
    <>
      <rect x="3" y="5" width="18" height="14" rx="3" />
      <path d="M3 14h18" />
      <circle cx="7" cy="16.5" r=".7" fill="currentColor" />
      <circle cx="11" cy="16.5" r=".7" fill="currentColor" />
    </>
  ),
  clock: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
  external: (
    <>
      <path d="M14 4h6v6m0-6-9 9" />
      <path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6" />
    </>
  ),
  check: <path d="m5 12 4 4L19 6" />,
  plus: <path d="M12 5v14m-7-7h14" />,
  trash: (
    <>
      <path d="M3 6h18M8 6V4h8v2m3 0-1 14H6L5 6" />
      <path d="M10 11v5m4-5v5" />
    </>
  ),
  refresh: (
    <>
      <path d="M20 7v5h-5M4 17v-5h5" />
      <path d="M5.6 9a7 7 0 0 1 11.6-2L20 12M4 12l2.8 5a7 7 0 0 0 11.6-2" />
    </>
  ),
  command: (
    <>
      <path d="M9 7V5a2 2 0 1 0-2 2h10a2 2 0 1 0-2-2v14a2 2 0 1 0 2-2H7a2 2 0 1 0 2 2V7Z" />
    </>
  ),
  bell: (
    <>
      <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
      <path d="M10 21h4" />
    </>
  ),
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  message: <path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-5.5A7.5 7.5 0 1 1 20 11.5Z" />,
  wand: (
    <>
      <path d="m15 4 5 5M4 20 17 7M7 4v3m-3 1h3m12 7v3m-3-3h3" />
    </>
  ),
  activity: <path d="M3 12h4l3-8 4 16 3-8h4" />,
  gauge: (
    <>
      <path d="M20 13a8 8 0 1 0-16 0" />
      <path d="m12 13 4-4M5 17h14" />
    </>
  ),
  server: (
    <>
      <rect x="3" y="4" width="18" height="7" rx="2" />
      <rect x="3" y="13" width="18" height="7" rx="2" />
      <path d="M7 7.5h.01M7 16.5h.01M11 7.5h6M11 16.5h6" />
    </>
  ),
}

function Icon({ name, size = 16, className = '' }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICON_PATHS[name]}
    </svg>
  )
}

function providerInfo(provider) {
  return PROVIDER_INFO[provider] || {
    name: provider,
    shortName: provider,
    icon: 'server',
    theme: 'other',
  }
}

function ProviderIcon({ provider, large = false }) {
  const info = providerInfo(provider)
  return (
    <span className={`provider-icon provider-${info.theme} ${large ? 'provider-icon-large' : ''}`}>
      <Icon name={info.icon} size={large ? 19 : 16} />
    </span>
  )
}

function GlassCard({ children, className = '', id, ...props }) {
  return (
    <article id={id} className={`glass-card ${className}`} {...props}>
      {children}
    </article>
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

function StatusPill({ status, label }) {
  return (
    <span className={`status-pill status-${status}`} role="status">
      <span className="status-dot" />
      {label}
    </span>
  )
}

function ScoreRing({ value, size = 64 }) {
  const radius = 26
  const circumference = 2 * Math.PI * radius
  const normalized = value == null ? 0 : Math.min(100, Math.max(0, Number(value)))
  return (
    <div className="score-ring" style={{ width: size, height: size }}>
      <svg viewBox="0 0 64 64" aria-hidden="true">
        <circle className="score-ring-track" cx="32" cy="32" r={radius} />
        <circle
          className="score-ring-value"
          cx="32"
          cy="32"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - normalized / 100)}
        />
      </svg>
      <span className="score-ring-label">{value == null ? '—' : Math.round(normalized)}</span>
    </div>
  )
}

function Navigation({ activeSection, onNavigate, onClose }) {
  return (
    <nav className="prism-nav" aria-label="Main">
      {NAV_ITEMS.map((item) => (
        <button
          key={item.id}
          type="button"
          className={`nav-item ${activeSection === item.id ? 'active' : ''}`}
          aria-current={activeSection === item.id ? 'page' : undefined}
          onClick={() => {
            onNavigate(item.id)
            onClose?.()
          }}
        >
          <Icon name={item.icon} size={16} />
          <span>{item.label}</span>
        </button>
      ))}
    </nav>
  )
}

function PrismSidebar({ activeSection, onNavigate, connection, onDocs }) {
  const connected = connection === 'connected'
  return (
    <aside className="prism-sidebar">
      <div className="sidebar-glass">
        <a className="prism-brand" href="#dashboard" onClick={(event) => {
          event.preventDefault()
          onNavigate('dashboard')
        }}>
          <span className="brand-icon"><Icon name="sparkle" size={17} /></span>
          <span className="brand-copy">
            <strong>Multi-LLM</strong>
            <small>LLM Evaluation</small>
          </span>
        </a>
        <Navigation activeSection={activeSection} onNavigate={onNavigate} />
        <div className="sidebar-promo">
          <div className="promo-icon"><Icon name="sparkle" size={16} /></div>
          <strong>Compare your models</strong>
          <p>Run a multi-turn evaluation with persistent history and analytics.</p>
          <button type="button" className="promo-button" onClick={() => onNavigate('comparison')}>
            Open Model Comparison <Icon name="arrow" size={13} />
          </button>
        </div>
        <div className={`sidebar-connection ${connected ? 'is-connected' : ''}`}>
          <span className="connection-indicator" />
          <span className="connection-copy">
            <strong>Backend status</strong>
            <small>{connection === 'loading' ? 'Checking connection' : connected ? 'Connected to API' : 'Not connected'}</small>
          </span>
          <span className="connection-state">{connection === 'loading' ? '···' : connected ? 'Live' : 'Offline'}</span>
        </div>
        <button type="button" className="sidebar-docs" onClick={onDocs}>
          <Icon name="external" size={14} /> Open API documentation
        </button>
      </div>
    </aside>
  )
}

function MobileNavigation({ open, activeSection, onNavigate, onClose }) {
  useEffect(() => {
    if (!open) return undefined
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open, onClose])

  if (!open) return null
  return (
    <div className="mobile-nav-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose()
    }}>
      <aside id="mobile-navigation" className="mobile-nav-sheet" role="dialog" aria-modal="true" aria-label="Workspace navigation">
        <div className="mobile-nav-heading">
          <div className="prism-brand">
            <span className="brand-icon"><Icon name="sparkle" size={17} /></span>
            <span className="brand-copy"><strong>Multi-LLM</strong><small>LLM Evaluation</small></span>
          </div>
          <button type="button" className="icon-button" aria-label="Close navigation" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <Navigation activeSection={activeSection} onNavigate={onNavigate} onClose={onClose} />
      </aside>
    </div>
  )
}

function PrismHeader({ title, connection, onQuickAction, onOpenSettings, menuOpen, onToggleMenu }) {
  return (
    <header className="prism-header">
      <button
        type="button"
        className="icon-button mobile-menu-button"
        aria-label="Open navigation"
        aria-expanded={menuOpen}
        aria-controls="mobile-navigation"
        onClick={onToggleMenu}
      >
        <Icon name="menu" />
      </button>
      <div className="header-title">
        <span>{HEADER_DATE}</span>
        <strong>{title}</strong>
      </div>
      <button type="button" className="quick-action-button" onClick={onQuickAction}>
        <Icon name="command" size={14} />
        Quick actions
        <kbd>Ctrl K</kbd>
      </button>
      <button
        type="button"
        className="icon-button backend-icon-button"
        aria-label={`Backend status: ${connection}. Open settings`}
        title={`Backend ${connection}`}
        onClick={onOpenSettings}
      >
        <Icon name="bell" />
        <span className={`header-status-dot ${connection === 'connected' ? 'online' : connection === 'offline' ? 'offline' : ''}`} />
      </button>
      <span className="header-avatar" aria-label="Multi-LLM Evaluator">ML</span>
    </header>
  )
}

function ProviderSelector({ providers, selectedProviders, onToggle, connected }) {
  const visibleProviders = providers.length ? providers : PROVIDER_ORDER
  return (
    <div className="provider-picker">
      <div className="provider-avatar-stack" role="group" aria-label="Providers to evaluate">
        {visibleProviders.map((provider) => {
          const selected = selectedProviders.includes(provider)
          return (
            <button
              key={provider}
              type="button"
              className={`provider-avatar-toggle ${selected ? 'selected' : ''}`}
              aria-pressed={selected}
              aria-label={providerInfo(provider).name}
              title={providerInfo(provider).name}
              disabled={!connected}
              onClick={() => onToggle(provider)}
            >
              <ProviderIcon provider={provider} />
            </button>
          )
        })}
      </div>
      <span className="provider-count">{selectedProviders.length} of {visibleProviders.length} models</span>
    </div>
  )
}

function PrismPrompt({
  providers,
  selectedProviders,
  onToggleProvider,
  connected,
  prompt,
  onPromptChange,
  referenceAnswer,
  onReferenceChange,
  onSubmit,
  onClear,
  loading,
  error,
}) {
  const promptRef = useRef(null)
  const [referenceOpen, setReferenceOpen] = useState(Boolean(referenceAnswer))

  useEffect(() => {
    const handleShortcut = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        document.getElementById('playground')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        promptRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [])

  function handlePromptKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.nativeEvent.isComposing) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <form className="prompt-frame" onSubmit={onSubmit}>
      <div className="prompt-inner">
        <label className="sr-only" htmlFor="prism-prompt">Prompt</label>
        <textarea
          ref={promptRef}
          id="prism-prompt"
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          onKeyDown={handlePromptKeyDown}
          rows={3}
          maxLength={12000}
          placeholder="What would you like the models to evaluate?"
          aria-describedby="prompt-counter"
        />
        <div className="prompt-meta">
          <span id="prompt-counter">{prompt.length.toLocaleString()} / 12,000</span>
          <span>Ctrl + Enter to evaluate</span>
        </div>
        {referenceOpen && (
          <div className="reference-field">
            <label htmlFor="reference-answer">Reference answer <span>Optional</span></label>
            <textarea
              id="reference-answer"
              value={referenceAnswer}
              onChange={(event) => onReferenceChange(event.target.value)}
              rows={2}
              maxLength={12000}
              placeholder="Add a trusted answer to help evaluate accuracy..."
            />
          </div>
        )}
        {error && <div className="prompt-error" role="alert"><Icon name="close" size={14} />{error}</div>}
        <div className="prompt-toolbar">
          <ProviderSelector
            connected={connected}
            onToggle={onToggleProvider}
            providers={providers}
            selectedProviders={selectedProviders}
          />
          <div className="prompt-actions">
            <button
              type="button"
              className="prompt-optional-button"
              aria-expanded={referenceOpen}
              onClick={() => setReferenceOpen((open) => !open)}
            >
              <Icon name="plus" size={14} />
              {referenceOpen ? 'Hide reference' : 'Add reference'}
            </button>
            <button type="button" className="clear-prompt-button" disabled={!prompt && !referenceAnswer} onClick={onClear}>
              <Icon name="trash" size={14} />
              Clear
            </button>
            <button className="evaluate-button" type="submit" disabled={loading || !connected}>
              {loading ? <span className="spinner" /> : <Icon name="sparkle" size={15} />}
              {loading ? 'Evaluating…' : 'Evaluate'}
              {!loading && <Icon name="arrow" size={15} className="evaluate-arrow" />}
            </button>
          </div>
        </div>
      </div>
      <div className="prompt-suggestions" aria-label="Prompt templates">
        {PROMPT_SUGGESTIONS.map((suggestion) => (
          <button key={suggestion.label} type="button" className="suggestion-chip" onClick={() => {
            onPromptChange(suggestion.prompt)
            promptRef.current?.focus()
          }}>
            <Icon name="wand" size={12} />
            {suggestion.label}
          </button>
        ))}
      </div>
    </form>
  )
}

function KpiCard({ label, value, detail, icon, tint, index }) {
  return (
    <GlassCard className="kpi-card animate-fade-up" style={{ animationDelay: `${index * 80}ms` }}>
      <span className="kpi-glow" style={{ backgroundColor: tint }} />
      <Icon name={icon} size={20} className="kpi-icon" />
      <span className="kpi-label">{label}</span>
      <strong className="kpi-value">{value}</strong>
      <span className="kpi-detail">{detail}</span>
    </GlassCard>
  )
}

function ModelCard({ provider, item, supported, selected, isRunning, wasSelected, isWinner }) {
  const info = providerInfo(provider)
  const [responseExpanded, setResponseExpanded] = useState(false)
  const isTranslation = item?.model_id?.toLowerCase().includes('riva-translate')

  let state = 'idle'
  let stateLabel = 'Not run'

  if (!supported) {
    state = 'unsupported'
    stateLabel = 'Not available'
  } else if (isRunning) {
    state = selected ? 'loading' : 'idle'
    stateLabel = selected ? 'Generating' : 'Not selected'
  } else if (item) {
    state = item.error ? 'failed' : 'success'
    stateLabel = item.error ? 'Failed' : 'Complete'
  } else if (wasSelected) {
    state = 'failed'
    stateLabel = 'No response'
  } else if (wasSelected === undefined) {
    state = 'ready'
    stateLabel = selected ? 'Ready' : 'Not selected'
  }

  const modelId = item?.model_id || (wasSelected === false ? 'Not run in this evaluation' : 'Resolved by the backend')
  return (
    <GlassCard className={`model-card model-${state} provider-card-${info.theme} ${isWinner ? 'is-winner' : ''}`}>
      <div className="model-card-head">
        <div className="model-identity">
          <ProviderIcon provider={provider} large />
          <div className="model-title">
            <h3>{info.name}{isWinner && <span className="winner-medal" aria-label="Best response">✦</span>}</h3>
            <span className="model-id" title={modelId}>{modelId}</span>
          </div>
          {item?.quality_score != null && <ScoreRing value={item.quality_score} />}
        </div>
        <StatusPill status={state} label={stateLabel} />
      </div>
      <div className="model-meta-row">
        <span className={`http-code ${item?.http_status >= 200 && item.http_status < 300 ? 'http-success' : item?.http_status ? 'http-error' : ''}`}>
          HTTP {item?.http_status ?? '—'}
        </span>
        <span className="response-time"><Icon name="clock" size={13} />{formatTime(item?.response_time_ms)}</span>
        {item?.rank != null && <span className="rank-label">Rank #{item.rank}</span>}
        {item?.quality_score != null && <span className="card-quality">{Math.round(item.quality_score)}<small> / 100</small></span>}
      </div>
      <div className="model-response-well">
        <span className="response-label">GENERATED RESPONSE</span>
        {isRunning && selected ? (
          <div className="response-loading" aria-label="Waiting for provider response">
            <i /><i /><i />
          </div>
        ) : item?.error ? (
          <p className="response-error">{item.error}</p>
        ) : item?.response ? (
          <>
            <div
              id={`response-content-${provider}`}
              className={`response-copy ${responseExpanded ? 'is-expanded' : ''}`}
            >
              <ReactMarkdown remarkPlugins={[remarkBreaks]}>{item.response}</ReactMarkdown>
            </div>
            <button
              type="button"
              className="response-toggle"
              aria-expanded={responseExpanded}
              aria-controls={`response-content-${provider}`}
              onClick={() => setResponseExpanded((expanded) => !expanded)}
            >
              {responseExpanded ? 'Collapse response' : 'Expand response'}
            </button>
          </>
        ) : (
          <p className="response-placeholder">
            {!supported
              ? 'This provider is not listed by the backend.'
              : wasSelected === false
                ? 'This provider was not selected for the latest evaluation.'
                : 'Run an evaluation to see the provider response here.'}
          </p>
        )}
      </div>
      {item?.scores && (
        <div className="model-score-summary">
          <span className="score-origin">AI-generated rubric ratings · not human-verified</span>
          <div className="metric-chips">
            {DETAILED_SCORE_DIMENSIONS.map(({ key, label }) => {
              const score = item.scores[key]
              return (
                <span className="metric-chip" key={key}>
                  <span>{label}</span>
                  <strong>{score == null ? 'N/A' : `${score}/5`}</strong>
                </span>
              )
            })}
          </div>
        </div>
      )}
      {(item?.score_rationale || item?.score_evidence) && (
        <details className="rationale-details">
          <summary>Evaluation evidence <span>+</span></summary>
          <div className="evaluation-evidence-list">
            {item.score_rationale && <p className="evaluation-overall-rationale">{item.score_rationale}</p>}
            {DETAILED_SCORE_DIMENSIONS.map(({ key, evidenceKey, label }) => {
              const evidence = item.score_evidence?.[evidenceKey]
              if (!evidence) return null
              const score = item.scores?.[key]
              return (
                <section className="evaluation-evidence-item" key={evidenceKey}>
                  <strong>{label} · {Number.isInteger(score) ? `${score}/5` : 'N/A'}</strong>
                  <p>{evidence.reason}</p>
                  <blockquote>{evidence.evidence}</blockquote>
                  {evidence.missing_requirements?.length > 0 && (
                    <p className="evaluation-evidence-gaps">
                      Missing: {evidence.missing_requirements.join(', ')}
                    </p>
                  )}
                </section>
              )
            })}
          </div>
        </details>
      )}
      {item?.score_error && (
        <div className="evaluation-error model-score-error" role="status">
          <strong>Scoring unavailable</strong>
          <span>{item.score_error}</span>
        </div>
      )}
      {isTranslation && <span className="specialist-badge">Translation-specialized model</span>}
    </GlassCard>
  )
}

function EmptyState({ children, className = '' }) {
  return <div className={`prism-empty-state ${className}`}>{children}</div>
}

function formatEstimatedCost(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return 'N/A'
  if (value === 0) return '$0.0000'

  const fixed = value.toFixed(4)
  return fixed === '0.0000' ? `$${value.toExponential(4)}` : `$${fixed}`
}

function DetailedComparison({ responses }) {
  if (!responses.length) return null

  function renderValue(response, metric) {
    if (metric.type === 'score') {
      const score = response.scores?.[metric.key]
      if (!Number.isInteger(score) || score < 1 || score > 5) return 'N/A'
      return (
        <span className="comparison-score" aria-label={`${score} out of 5`}>
          <span className="comparison-score-track" aria-hidden="true">
            <span style={{ width: `${score * 20}%` }} />
          </span>
          <strong>{score}/5</strong>
        </span>
      )
    }

    if (metric.key === 'simple_checks') {
      if (!response.simple_checks) return 'N/A'
      const { word_count: wordCount, ending_check: endingCheck } = response.simple_checks
      const checks = []
      if (Number.isInteger(wordCount)) checks.push(`${wordCount} words`)
      if (typeof endingCheck === 'boolean') {
        checks.push(`Ending check ${endingCheck ? 'passed' : 'needs review'}`)
      }
      return checks.length ? checks.join(' · ') : 'N/A'
    }

    if (metric.key === 'latency') {
      return Number.isFinite(response.response_time_ms)
        ? `${(response.response_time_ms / 1000).toFixed(2)} s`
        : 'N/A'
    }

    if (metric.key === 'token_usage') {
      const promptTokens = Number.isInteger(response.prompt_tokens) ? response.prompt_tokens : 'N/A'
      const completionTokens = Number.isInteger(response.completion_tokens) ? response.completion_tokens : 'N/A'
      return promptTokens === 'N/A' && completionTokens === 'N/A'
        ? 'N/A'
        : `${promptTokens} / ${completionTokens}`
    }

    return formatEstimatedCost(response.estimated_cost_usd)
  }

  const metrics = [
    ...DETAILED_SCORE_DIMENSIONS.map((metric) => ({ ...metric, type: 'score' })),
    { key: 'simple_checks', label: 'Simple Checks', type: 'value' },
    { key: 'latency', label: 'Latency', type: 'value' },
    { key: 'token_usage', label: 'Token Usage', type: 'value' },
    { key: 'estimated_cost_usd', label: 'Estimated Cost', type: 'value' },
  ]

  return (
    <section className="detailed-comparison" aria-label="Detailed model metrics">
      <div className="detailed-comparison-heading">
        <div>
          <h3>Detailed comparison</h3>
          <p>
            AI-generated rubric ratings are not human ratings. Simple checks are deterministic.
            Estimated cost is N/A until exact-model pricing is configured.
          </p>
        </div>
      </div>
      <div className="detailed-comparison-scroll">
        <table
          className="detailed-comparison-table"
          style={{ minWidth: `${Math.max(680, 260 + 150 * responses.length)}px` }}
        >
          <thead>
            <tr>
              <th scope="col">Metric</th>
              {responses.map((response) => (
                <th scope="col" key={response.model}>{providerInfo(response.model).name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => (
              <tr key={metric.key}>
                <th scope="row">{metric.label}</th>
                {responses.map((response) => (
                  <td key={response.model}>{renderValue(response, metric)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function ScoreOrbit({ responses }) {
  const scored = responses.filter((item) => item.quality_score != null)
  if (!scored.length) {
    return <EmptyState>No returned quality scores to chart yet.</EmptyState>
  }
  const rings = [...scored].reverse().slice(0, 6)
  return (
    <div className="orbit-layout">
      <svg className="orbit-chart" viewBox="0 0 200 200" role="img" aria-label="Overall quality scores by model">
        {rings.map((item, index) => {
          const radius = 88 - index * 12
          const circumference = 2 * Math.PI * radius
          const score = Math.max(0, Math.min(100, Number(item.quality_score)))
          return (
            <g key={item.analytics_key || item.model}>
              <circle className="orbit-track" cx="100" cy="100" r={radius} />
              <circle
                className={`orbit-value provider-stroke-${providerInfo(item.model).theme}`}
                cx="100"
                cy="100"
                r={radius}
                strokeDasharray={circumference}
                strokeDashoffset={circumference * (1 - score / 100)}
              />
            </g>
          )
        })}
        <text className="orbit-center-label" x="100" y="96" textAnchor="middle">QUALITY</text>
        <text className="orbit-center-value" x="100" y="119" textAnchor="middle">
          {formatAverage([Math.max(...scored.map((item) => Number(item.quality_score)))])}
        </text>
      </svg>
      <ul className="orbit-legend">
        {scored.map((item) => (
          <li key={item.analytics_key || item.model}>
            <span className={`legend-dot provider-bg-${providerInfo(item.model).theme}`} />
            <span>{providerInfo(item.model).shortName}</span>
            <strong>{formatAverage([item.quality_score])}</strong>
          </li>
        ))}
      </ul>
    </div>
  )
}

function MetricRadar({ responses }) {
  const scored = responses.filter((item) => (
    item.scores
    && SCORE_DIMENSIONS.every(({ key }) => (
      typeof item.scores[key] === 'number' && Number.isFinite(item.scores[key])
    ))
  ))
  const available = responses.filter((item) => (
    item.scores
    && SCORE_DIMENSIONS.some(({ key }) => (
      typeof item.scores[key] === 'number' && Number.isFinite(item.scores[key])
    ))
  ))
  if (!available.length) return <EmptyState>No evaluation metrics to compare yet.</EmptyState>

  const center = 120
  const radius = 76
  const angles = SCORE_DIMENSIONS.map((_, index) => (-Math.PI / 2) + (2 * Math.PI * index / SCORE_DIMENSIONS.length))
  const point = (angle, amount) => `${center + Math.cos(angle) * amount},${center + Math.sin(angle) * amount}`
  return (
    <div className="radar-wrap">
      <svg viewBox="0 0 240 240" role="img" aria-label="Radar comparison for models with all six available evaluation dimensions">
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <polygon
            key={scale}
            className="radar-grid"
            points={angles.map((angle) => point(angle, radius * scale)).join(' ')}
          />
        ))}
        {angles.map((angle) => (
          <line key={angle} className="radar-axis" x1={center} y1={center} x2={center + Math.cos(angle) * radius} y2={center + Math.sin(angle) * radius} />
        ))}
        {scored.map((item) => {
          const theme = providerInfo(item.model).theme
          const points = SCORE_DIMENSIONS.map(({ key }, index) => {
            const value = Number(item.scores[key]) || 0
            return point(angles[index], radius * Math.min(1, Math.max(0, value / 5)))
          }).join(' ')
          return <polygon key={item.analytics_key || item.model} className={`radar-series provider-radar-${theme}`} points={points} />
        })}
      </svg>
      <ul className="radar-labels">
        {SCORE_DIMENSIONS.map(({ key, label }) => (
          <li key={key}>
            <span>{label}</span>
            <strong>{formatAverage(available.map((item) => item.scores?.[key])) ?? '—'}/5</strong>
          </li>
        ))}
      </ul>
      <div className="radar-legend">
        {scored.map((item) => (
          <span key={item.analytics_key || item.model}><i className={`legend-dot provider-bg-${providerInfo(item.model).theme}`} />{providerInfo(item.model).shortName}</span>
        ))}
      </div>
      {available.length > scored.length && (
        <p className="radar-note">Radar shapes require all six scores; partial metrics remain visible in the turn details and leaderboard.</p>
      )}
    </div>
  )
}

function ScoreTrend({ history, scope = 'this browser session' }) {
  const runs = [...history].reverse()
  const allValues = runs.map((run, index) => ({
    index,
    scores: (run.result.responses || []).filter((item) => item.quality_score != null),
  }))
  const values = allValues.filter((entry) => entry.scores.length)
  if (!values.length) return <EmptyState>No scored evaluations to plot yet.</EmptyState>

  const width = 360
  const height = 150
  const padding = 18
  const x = (index) => allValues.length === 1
    ? width / 2
    : padding + index * (width - 2 * padding) / (allValues.length - 1)
  const y = (score) => height - padding - Math.min(100, Math.max(0, Number(score))) * (height - 2 * padding) / 100
  return (
    <div className="trend-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Quality score trend across ${scope}`}>
        {[0, 50, 100].map((score) => (
          <line key={score} className="trend-gridline" x1={padding} y1={y(score)} x2={width - padding} y2={y(score)} />
        ))}
        {PROVIDER_ORDER.map((provider) => {
          const series = values.flatMap((entry) => {
            const item = entry.scores.find((response) => response.model === provider)
            return item ? [{ x: x(entry.index), y: y(item.quality_score), key: `${entry.index}-${provider}` }] : []
          })
          if (!series.length) return null
          return (
            <g key={provider} className={`trend-series provider-trend-${providerInfo(provider).theme}`}>
              {series.length > 1 && <polyline points={series.map((point) => `${point.x},${point.y}`).join(' ')} />}
              {series.map((point) => <circle key={point.key} cx={point.x} cy={point.y} r="3" />)}
            </g>
          )
        })}
      </svg>
      <div className="trend-labels"><span>0</span><span>{values.length} scored turn{values.length === 1 ? '' : 's'} · {scope}</span><span>100</span></div>
    </div>
  )
}

function LatencyChart({ responses }) {
  const rows = responses.filter((item) => item.response_time_ms != null)
  if (!rows.length) return <EmptyState>No response-time values were returned.</EmptyState>
  const max = Math.max(1, ...rows.map((item) => Number(item.response_time_ms) || 0))
  return (
    <div className="latency-chart">
      {rows.map((item) => (
        <div className="latency-row" key={item.analytics_key || item.model}>
          <span className={`latency-provider provider-text-${providerInfo(item.model).theme}`}>{providerInfo(item.model).shortName}</span>
          <span className="latency-track"><i className={`provider-fill-${providerInfo(item.model).theme}`} style={{ width: `${Math.min(100, (Number(item.response_time_ms) / max) * 100)}%` }} /></span>
          <strong>{formatTime(item.response_time_ms)}</strong>
        </div>
      ))}
    </div>
  )
}

function Analytics({ responses, history }) {
  return (
    <section id="analytics" className="analytics-section scroll-anchor">
      <SectionTitle
        kicker="EVALUATION ANALYTICS"
        title="See how the models compare"
        description="Charts use only scores and response times returned by your backend."
      />
      <div className="analytics-grid">
        <GlassCard className="analytics-card">
          <SectionTitle title="Score orbit" description="Overall quality per model" />
          <ScoreOrbit responses={responses} />
        </GlassCard>
        <GlassCard className="analytics-card">
          <SectionTitle title="Metric Share" description="Available evaluation dimensions per model" />
          <MetricRadar responses={responses} />
        </GlassCard>
        <div className="analytics-stack">
          <GlassCard className="analytics-card trend-panel">
            <SectionTitle title="Quality trend" description="Evaluations in this browser session" />
            <ScoreTrend history={history} />
          </GlassCard>
          <GlassCard className="analytics-card latency-panel">
            <SectionTitle title="Response time" description="Latest evaluation, milliseconds" />
            <LatencyChart responses={responses} />
          </GlassCard>
        </div>
      </div>
    </section>
  )
}

function Leaderboard({ responses }) {
  const ranked = responses
    .filter((item) => item.quality_score != null && item.rank != null)
    .sort((a, b) => a.rank - b.rank || b.quality_score - a.quality_score)
  if (!ranked.length) {
    return (
      <GlassCard id="leaderboard" className="section-card scroll-anchor">
        <SectionTitle title="Leaderboard" description="Ranked by the scores returned in the latest evaluation." />
        <EmptyState>No ranked models yet. Run an evaluation to populate the leaderboard.</EmptyState>
      </GlassCard>
    )
  }

  const podium = ranked.filter((item) => item.rank <= 3).slice(0, 3)
  const remaining = ranked.filter((item) => !podium.includes(item))
  return (
    <GlassCard id="leaderboard" className="section-card scroll-anchor">
      <SectionTitle title="Leaderboard" description="Ranked by the scores returned in the latest evaluation." />
      <ol className="podium">
        {podium.map((item) => (
          <li key={item.model} className={`podium-entry podium-rank-${Math.min(item.rank, 3)}`}>
            <ProviderIcon provider={item.model} large />
            <strong>{providerInfo(item.model).name}</strong>
            <span className="podium-score">{Math.round(item.quality_score)} / 100</span>
            <div className={`podium-bar provider-podium-${providerInfo(item.model).theme}`}>
              <span className="podium-place"><span className="sr-only">Place </span>{item.rank}</span>
            </div>
          </li>
        ))}
      </ol>
      {remaining.map((item) => (
        <div className="leaderboard-row" key={item.model}>
          <span className="leaderboard-rank">{String(item.rank).padStart(2, '0')}</span>
          <ProviderIcon provider={item.model} />
          <span className="leaderboard-model">
            <strong>{providerInfo(item.model).name}</strong>
            <small>{item.model_id}</small>
          </span>
          <span className="leaderboard-time">{formatTime(item.response_time_ms)}</span>
          <strong className="leaderboard-score">{Math.round(item.quality_score)}<small> / 100</small></strong>
        </div>
      ))}
    </GlassCard>
  )
}

function History({ history, onRestore, onClear }) {
  return (
    <GlassCard id="history" className="section-card scroll-anchor">
      <SectionTitle
        title="Evaluation history"
        description="Recent evaluations from this browser session. They are not saved to the backend."
        action={history.length > 0 && (
          <button type="button" className="text-button" onClick={onClear}><Icon name="trash" size={14} /> Clear</button>
        )}
      />
      {history.length ? (
        <ol className="history-timeline">
          {history.map((run) => {
            const items = run.result.responses || []
            const score = averageScore(items.map((item) => item.quality_score))
            const winner = (run.result.winner_models || [])[0] || items.find((item) => item.rank === 1)?.model
            const info = winner ? providerInfo(winner) : null
            return (
              <li key={run.id} className="history-item">
                <span className={`timeline-dot ${info ? `provider-bg-${info.theme}` : ''}`} />
                <button type="button" className="history-entry" onClick={() => onRestore(run)}>
                  <span className="history-prompt">{run.prompt}</span>
                  <span className="history-meta">
                    <time dateTime={run.createdAt}>{new Date(run.createdAt).toLocaleString()}</time>
                    <span>·</span>
                    <span>{run.providers.length} provider{run.providers.length === 1 ? '' : 's'}</span>
                    {score != null && <><span>·</span><span>Avg {score}</span></>}
                    {info && <><span>·</span><span>{info.shortName} top ranked</span></>}
                  </span>
                </button>
                <Icon name="arrow" size={14} />
              </li>
            )
          })}
        </ol>
      ) : (
        <EmptyState>No evaluations in this session yet.</EmptyState>
      )}
    </GlassCard>
  )
}

function Connections({ providers, allCards, connection, apiBaseUrl, onRetry, onDocs }) {
  return (
    <GlassCard id="settings" className="section-card scroll-anchor">
      <SectionTitle title="Connections" description={`FastAPI backend · ${apiBaseUrl}`} />
      <ul className="connection-grid">
        {allCards.map((provider) => {
          const available = providers.includes(provider)
          return (
            <li className="connection-tile" key={provider}>
              <ProviderIcon provider={provider} />
              <span className="connection-tile-copy">
                <strong>{providerInfo(provider).name}</strong>
                <small>{available ? 'Listed by backend' : 'Not listed by backend'}</small>
              </span>
              <span
                className={`connection-tile-dot ${available ? 'available' : ''}`}
                aria-label={available ? 'Listed by backend' : 'Not listed by backend'}
              />
            </li>
          )
        })}
      </ul>
      <div className="settings-footer">
        <div className="settings-note">
          <span className={`connection-indicator ${connection === 'connected' ? 'online' : ''}`} />
          <span>{connection === 'connected' ? 'Backend connected' : connection === 'loading' ? 'Checking backend connection' : 'Backend unavailable'}</span>
          <span className="settings-separator">·</span>
          <span>Provider credentials stay server-side.</span>
        </div>
        <div className="settings-actions">
          <button type="button" className="secondary-button" disabled={connection === 'loading'} onClick={onRetry}>
            <Icon name="refresh" size={14} /> Test connection
          </button>
          <button type="button" className="secondary-button" onClick={onDocs}>
            API docs <Icon name="external" size={13} />
          </button>
        </div>
      </div>
    </GlassCard>
  )
}

function ModelComparison({ providers, latestRun, loading, selectedProviders, winners }) {
  const responses = latestRun?.result.responses || []
  const allCards = [...new Set([...PROVIDER_ORDER, ...providers])]
  const selectedInLatest = latestRun?.providers || []
  return (
    <section id="playground-results" className="comparison-section">
      <SectionTitle
        kicker="PROMPT PLAYGROUND RESULT"
        title="One-off comparison results"
        description="Results from the separate Prompt Playground evaluation."
        action={latestRun && (
          <StatusPill
            status={latestRun.result.status === 'success' ? 'success' : latestRun.result.status === 'failed' ? 'failed' : 'warning'}
            label={latestRun.result.status.replace('_', ' ')}
          />
        )}
      />
      {latestRun?.result.evaluation_error && (
        <div
          className={`evaluation-error ${latestRun.result.evaluation_status === 'partial_failure' ? 'evaluation-partial-error' : ''}`}
          role={latestRun.result.evaluation_status === 'partial_failure' ? 'status' : 'alert'}
        >
          <strong>{latestRun.result.evaluation_status === 'partial_failure' ? 'Partial evaluation' : 'Evaluation error'}</strong>
          <span>{latestRun.result.evaluation_error}</span>
        </div>
      )}
      {latestRun?.result.evaluation_status === 'not_run' && (
        <div className="evaluation-error" role="status"><strong>Evaluation not run</strong><span>No successful model responses were available to score.</span></div>
      )}
      <div className="model-card-grid">
        {allCards.map((provider) => (
          <ModelCard
            key={provider}
            provider={provider}
            item={responses.find((response) => response.model === provider)}
            supported={providers.includes(provider)}
            selected={selectedProviders.includes(provider)}
            isRunning={loading}
            wasSelected={latestRun ? selectedInLatest.includes(provider) : undefined}
            isWinner={winners.includes(provider)}
          />
        ))}
      </div>
      {latestRun && <DetailedComparison responses={responses} />}
      {!latestRun && !loading && <EmptyState className="comparison-empty">Your model responses will appear here after an evaluation.</EmptyState>}
    </section>
  )
}

function EvaluationResults({ latestRun, responses, winners }) {
  if (!latestRun) {
    return (
      <GlassCard className="evaluation-panel section-card">
        <SectionTitle kicker="ANALYSIS" title="Evaluation results" description="Scores, ranking, and rationale come from the backend response." />
        <EmptyState>Run an evaluation to see the evaluation model, actual scores, and winner.</EmptyState>
      </GlassCard>
    )
  }
  const scored = responses.filter((item) => item.quality_score != null)
  const average = averageScore(scored.map((item) => item.quality_score))
  return (
    <GlassCard className="evaluation-panel section-card">
      <SectionTitle
        kicker="ANALYSIS"
        title="Evaluation results"
        description="Scores, ranking, and rationale come from the backend response."
        action={<StatusPill
          status={latestRun.result.evaluation_status === 'success' ? 'success' : latestRun.result.evaluation_status === 'failed' ? 'failed' : latestRun.result.evaluation_status === 'partial_failure' ? 'warning' : 'idle'}
          label={latestRun.result.evaluation_status.replace('_', ' ')}
        />}
      />
      <div className="evaluation-metrics">
        <div><span>Evaluation status</span><strong>{latestRun.result.evaluation_status.replace('_', ' ')}</strong></div>
        <div><span>Evaluation model</span><strong className="evaluation-model-name">{latestRun.result.evaluation_model || 'Not returned'}</strong></div>
        <div><span>Average quality</span><strong>{average == null ? 'Not scored' : `${average} / 100`}</strong></div>
        <div><span>Winner</span><strong>{winners.length ? winners.map((provider) => providerInfo(provider).shortName).join(' / ') : 'Not determined'}</strong></div>
      </div>
      {winners.length > 0 && (
        <div className="winner-banner">
          <span className="winner-icon"><Icon name="sparkle" size={16} /></span>
          <span><small>{winners.length > 1 ? 'TOP-RANKED MODELS' : 'TOP-PERFORMING MODEL'}</small><strong>{winners.map((provider) => providerInfo(provider).name).join(' and ')}</strong></span>
          {scored.length > 0 && <strong className="winner-score">{Math.max(...scored.map((item) => item.quality_score))} / 100</strong>}
        </div>
      )}
    </GlassCard>
  )
}

function averageScore(values) {
  const valid = values.map(Number).filter(Number.isFinite)
  return valid.length ? Math.round(valid.reduce((sum, value) => sum + value, 0) / valid.length) : null
}

function meanValue(values) {
  const valid = values.filter((value) => typeof value === 'number' && Number.isFinite(value))
  return valid.length
    ? valid.reduce((sum, value) => sum + value, 0) / valid.length
    : null
}

function formatAverage(values) {
  const average = meanValue(values)
  return average == null ? null : Number(average.toFixed(1))
}

function formatTime(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  return `${Math.round(Number(value))} ms`
}

function aggregateConversationResponses(turns) {
  const groups = new Map()
  for (const turn of turns) {
    for (const response of turn.result?.responses || []) {
      const analyticsKey = `${response.model}:${response.model_id || ''}`
      if (!groups.has(analyticsKey)) {
        groups.set(analyticsKey, { analyticsKey, responses: [] })
      }
      groups.get(analyticsKey).responses.push(response)
    }
  }

  return [...groups.values()].map(({ analyticsKey, responses }) => {
    const latest = responses[responses.length - 1]
    const scores = Object.fromEntries(
      DETAILED_SCORE_DIMENSIONS.map(({ key }) => [
        key,
        meanValue(responses.map((response) => response.scores?.[key])),
      ]),
    )
    const hasScores = Object.values(scores).some((score) => score != null)
    return {
      ...latest,
      analytics_key: analyticsKey,
      quality_score: meanValue(responses.map((response) => response.quality_score)),
      response_time_ms: meanValue(responses.map((response) => response.response_time_ms)),
      scores: hasScores ? scores : null,
      scored_turn_count: responses.filter((response) => response.quality_score != null).length,
      rank: null,
    }
  })
}

function ConversationLeaderboard({ responses }) {
  const ranked = responses
    .filter((response) => response.quality_score != null)
    .sort((left, right) => (
      right.quality_score - left.quality_score
      || (left.response_time_ms ?? Number.POSITIVE_INFINITY)
        - (right.response_time_ms ?? Number.POSITIVE_INFINITY)
      || left.model.localeCompare(right.model)
    ))

  return (
    <GlassCard className="section-card conversation-leaderboard">
      <SectionTitle
        title="Final leaderboard"
        description="Mean backend quality scores, with mean latency as the tie-breaker used by the per-turn evaluator."
      />
      {ranked.length ? (
        <div className="conversation-leaderboard-scroll">
          <table className="conversation-leaderboard-table">
            <thead>
              <tr>
                <th scope="col">Position</th>
                <th scope="col">Model</th>
                <th scope="col">Mean quality</th>
                <th scope="col">Mean latency</th>
                {SCORE_DIMENSIONS.map(({ key, label }) => (
                  <th scope="col" key={key}>{label}</th>
                ))}
                <th scope="col">Scored turns</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((response, index) => (
                <tr key={response.analytics_key} className={index === 0 ? 'is-conversation-leader' : ''}>
                  <td>#{index + 1}{index === 0 && <span className="conversation-winner-tag">Leader</span>}</td>
                  <th scope="row">
                    <strong>{providerInfo(response.model).name}</strong>
                    <small>{response.model_id}</small>
                  </th>
                  <td>{formatAverage([response.quality_score])} / 100</td>
                  <td>{formatTime(response.response_time_ms)}</td>
                  {SCORE_DIMENSIONS.map(({ key }) => (
                    <td key={key}>
                      {response.scores?.[key] == null
                        ? 'N/A'
                        : `${formatAverage([response.scores[key]])} / 5`}
                    </td>
                  ))}
                  <td>{response.scored_turn_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState>No conversation-wide leaderboard is available until responses have complete overall scores.</EmptyState>
      )}
    </GlassCard>
  )
}

function ConversationAnalytics({ turns }) {
  const currentTurn = turns[turns.length - 1]
  const currentResponses = currentTurn?.result?.responses || []
  const conversationAverages = aggregateConversationResponses(turns)
  const trendHistory = [...turns].reverse()

  return (
    <>
      <section className="analytics-section conversation-analytics">
        <SectionTitle
          kicker="TURN AND CONVERSATION ANALYTICS"
          title="Evaluation analytics"
          description="Current-turn results are separated from conversation-wide averages. Missing evaluations are excluded, not replaced."
        />
        <div className="conversation-analytics-scope">
          <span>Current turn</span>
          <small>{currentTurn ? `Turn ${turns.length} · values from this turn only` : 'No evaluated turns yet'}</small>
        </div>
        <div className="analytics-grid">
          <GlassCard className="analytics-card">
            <SectionTitle title="Score Orbit" description="Current-turn overall quality per model" />
            <ScoreOrbit responses={currentResponses} />
          </GlassCard>
          <GlassCard className="analytics-card">
            <SectionTitle title="Metric Share" description="Current-turn scores across available evaluation dimensions" />
            <MetricRadar responses={currentResponses} />
          </GlassCard>
          <div className="analytics-stack">
            <GlassCard className="analytics-card trend-panel">
              <SectionTitle title="Quality Trend" description={`All ${turns.length} turns in this conversation`} />
              <ScoreTrend history={trendHistory} scope="this conversation" />
            </GlassCard>
            <GlassCard className="analytics-card latency-panel">
              <SectionTitle title="Response Time" description="Current-turn provider latency" />
              <LatencyChart responses={currentResponses} />
            </GlassCard>
          </div>
        </div>

        <div className="conversation-analytics-scope">
          <span>Conversation-wide</span>
          <small>Means across available evaluated turns; per-turn ranks remain unchanged.</small>
        </div>
        <div className="analytics-grid conversation-average-grid">
          <GlassCard className="analytics-card">
            <SectionTitle title="Score Orbit" description="Mean overall quality per model" />
            <ScoreOrbit responses={conversationAverages} />
          </GlassCard>
          <GlassCard className="analytics-card">
            <SectionTitle title="Metric Share" description="Mean of each available rubric metric" />
            <MetricRadar responses={conversationAverages} />
          </GlassCard>
          <GlassCard className="analytics-card">
            <SectionTitle title="Response Time" description="Mean provider latency across conversation turns" />
            <LatencyChart responses={conversationAverages} />
          </GlassCard>
        </div>
      </section>
      <ConversationLeaderboard responses={conversationAverages} />
    </>
  )
}

export default function PrismWorkspace({
  apiBaseUrl,
  providers,
  selectedProviders,
  onToggleProvider,
  connection,
  onRetry,
  prompt,
  onPromptChange,
  referenceAnswer,
  onReferenceChange,
  onSubmit,
  onClearPrompt,
  loading,
  requestError,
  latestRun,
  history,
  runCount,
  activeSection,
  onNavigate,
  onClearHistory,
  onRestoreRun,
}) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      document.getElementById('comparison')?.scrollIntoView({ block: 'start' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [])

  const allCards = useMemo(() => [...new Set([...PROVIDER_ORDER, ...providers])], [providers])
  const currentResponses = latestRun?.result.responses || []
  const winners = latestRun?.result.winner_models || []
  const historyResponses = history.flatMap((run) => run.result.responses || [])
  const historyScores = historyResponses.filter((item) => item.quality_score != null)
  const historyLatencies = historyResponses
    .map((item) => item.response_time_ms)
    .filter((value) => value != null && Number.isFinite(Number(value)))
  const averageQuality = averageScore(historyScores.map((item) => item.quality_score))
  const averageLatency = historyLatencies.length
    ? Math.round(historyLatencies.reduce((sum, value) => sum + Number(value), 0) / historyLatencies.length)
    : null
  const activeSectionTitle = NAV_ITEMS.find((item) => item.id === activeSection)?.label || 'Dashboard'
  const connected = connection === 'connected'

  function navigateTo(section) {
    setMobileNavOpen(false)
    onNavigate(section)
  }

  function openDocs() {
    window.open(`${apiBaseUrl}/docs`, '_blank', 'noopener,noreferrer')
  }

  function quickAction() {
    navigateTo('comparison')
    window.setTimeout(() => document.getElementById('conversation-prompt')?.focus(), 100)
  }

  return (
    <div className="theme-prism">
      <div className="prism-backdrop" aria-hidden="true">
        <span className="glow glow-blue" />
        <span className="glow glow-violet" />
        <span className="glow glow-cyan" />
      </div>
      <div className="prism-layout">
        <PrismSidebar
          activeSection={activeSection}
          onNavigate={navigateTo}
          connection={connection}
          onDocs={openDocs}
        />
        <div className="prism-content-column">
          <PrismHeader
            title={activeSectionTitle}
            connection={connection}
            onQuickAction={quickAction}
            onOpenSettings={() => navigateTo('settings')}
            menuOpen={mobileNavOpen}
            onToggleMenu={() => setMobileNavOpen((open) => !open)}
          />
          <main className="prism-main">
            <section id="dashboard" className="hero-section scroll-anchor">
              <p className="hero-eyebrow">{providers.length} provider{providers.length === 1 ? '' : 's'} · persistent multi-turn evaluation</p>
              <h1>Compare models.<br /><span>Turn by turn.</span></h1>
              <p className="hero-subtitle">One conversation workspace for model responses, evaluation analytics, and rankings.</p>
              {connection === 'offline' && (
                <div className="backend-error" role="alert">
                  <Icon name="close" size={15} />
                  <span><strong>Backend connection unavailable.</strong> {requestError || `Unable to reach ${apiBaseUrl}`}</span>
                  <button type="button" className="text-button" disabled={connection === 'loading'} onClick={onRetry}>Retry</button>
                </div>
              )}
              <button type="button" className="secondary-button" onClick={() => navigateTo('comparison')}>
                Open Model Comparison <Icon name="arrow" size={14} />
              </button>
            </section>

            <section className="kpi-grid" aria-label="Workspace metrics">
              <KpiCard index={0} label="Evaluations" value={runCount.toLocaleString()} detail="This browser session" icon="activity" tint="var(--chart-1)" />
              <KpiCard index={1} label="Avg. quality" value={averageQuality == null ? '—' : `${averageQuality}`} detail={historyScores.length ? `${historyScores.length} scored responses` : 'Actual scores appear here'} icon="gauge" tint="var(--chart-2)" />
              <KpiCard index={2} label="Avg. latency" value={averageLatency == null ? '—' : formatTime(averageLatency)} detail={historyLatencies.length ? `${historyLatencies.length} responses in session` : 'Response time from backend'} icon="clock" tint="var(--chart-3)" />
              <KpiCard index={3} label="Providers" value={connected ? providers.length : '—'} detail={connected ? 'Listed by your backend' : 'Backend unavailable'} icon="server" tint="var(--chart-4)" />
            </section>

            <ConversationWorkspace
              providers={providers}
              selectedProviders={selectedProviders}
              onToggleProvider={onToggleProvider}
              connection={connection}
              renderAnalytics={(turns) => <ConversationAnalytics turns={turns} />}
            />

            <section id="playground" className="playground-section scroll-anchor">
              <SectionTitle
                kicker="PROMPT PLAYGROUND"
                title="Single-run evaluation"
                description="A focused one-off comparison tool. For persistent multi-turn conversations, use Model Comparison."
              />
              <PrismPrompt
                providers={providers}
                selectedProviders={selectedProviders}
                onToggleProvider={onToggleProvider}
                connected={connected}
                prompt={prompt}
                onPromptChange={onPromptChange}
                referenceAnswer={referenceAnswer}
                onReferenceChange={onReferenceChange}
                onSubmit={onSubmit}
                onClear={onClearPrompt}
                loading={loading}
                error={requestError && connection !== 'offline' ? requestError : ''}
              />
              <ModelComparison
                providers={providers}
                latestRun={latestRun}
                loading={loading}
                selectedProviders={selectedProviders}
                winners={winners}
              />
              <EvaluationResults latestRun={latestRun} responses={currentResponses} winners={winners} />
            </section>
            <Analytics responses={currentResponses} history={history} />

            <div className="lower-grid">
              <Leaderboard responses={currentResponses} />
              <History history={history} onRestore={onRestoreRun} onClear={onClearHistory} />
            </div>

            <Connections
              providers={providers}
              allCards={allCards}
              connection={connection}
              apiBaseUrl={apiBaseUrl}
              onRetry={onRetry}
              onDocs={openDocs}
            />
            <footer className="prism-footer">
              <span>Multi-LLM Evaluator <b>·</b> Powered by your FastAPI backend</span>
              <span>Conversations persist in SQLite; Prompt Playground history is session-only.</span>
            </footer>
          </main>
        </div>
      </div>
      <MobileNavigation
        open={mobileNavOpen}
        activeSection={activeSection}
        onNavigate={navigateTo}
        onClose={() => setMobileNavOpen(false)}
      />
    </div>
  )
}
