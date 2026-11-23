import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import {
  createConversation,
  deleteConversation,
  fetchConversation,
  fetchConversations,
  renameConversation,
  submitConversationTurn,
} from '../../api'

const PROVIDER_NAMES = {
  gemini: 'Google Gemini',
  groq: 'Groq',
  nvidia: 'NVIDIA NIM',
  ollama: 'Ollama',
}

const SCORE_DIMENSIONS = [
  ['instruction_following', 'Instruction following'],
  ['accuracy', 'Accuracy'],
  ['completeness', 'Completeness'],
  ['formatting', 'Formatting'],
  ['relevance', 'Relevance'],
  ['clarity', 'Reasoning'],
]

function providerName(provider) {
  return PROVIDER_NAMES[provider] || provider
}

function formatTimestamp(value) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString()
}

function formatEstimatedCost(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return 'N/A'
  if (value === 0) return '$0.0000'
  const fixed = value.toFixed(4)
  return fixed === '0.0000' ? `$${value.toExponential(4)}` : `$${fixed}`
}

function ConversationResponse({ response, winner }) {
  const evidence = response.score_evidence || {}
  return (
    <article className={`conversation-provider-card ${winner ? 'is-winner' : ''}`}>
      <header className="conversation-provider-heading">
        <div>
          <strong>{providerName(response.model)}</strong>
          <small>{response.model_id}</small>
        </div>
        <div className="conversation-provider-rank">
          {winner && <span className="conversation-winner-tag">Winner</span>}
          {response.rank != null && <span>Rank #{response.rank}</span>}
        </div>
      </header>

      {response.error ? (
        <p className="conversation-provider-error" role="status">{response.error}</p>
      ) : (
        <div className="conversation-provider-answer">
          <ReactMarkdown remarkPlugins={[remarkBreaks]}>
            {response.response || ''}
          </ReactMarkdown>
        </div>
      )}

      <dl className="conversation-provider-metrics">
        <div><dt>Overall score</dt><dd>{response.quality_score == null ? 'Not scored' : `${response.quality_score}/100`}</dd></div>
        <div><dt>Latency</dt><dd>{Number.isFinite(response.response_time_ms) ? `${response.response_time_ms} ms` : 'N/A'}</dd></div>
        <div><dt>Tokens</dt><dd>{response.prompt_tokens ?? 'N/A'} / {response.completion_tokens ?? 'N/A'}</dd></div>
        <div><dt>Estimated cost</dt><dd>{formatEstimatedCost(response.estimated_cost_usd)}</dd></div>
      </dl>

      {response.scores && (
        <div className="conversation-score-grid" aria-label={`${providerName(response.model)} evaluation scores`}>
          {SCORE_DIMENSIONS.map(([key, label]) => (
            <div key={key}>
              <span>{label}</span>
              <strong>{response.scores[key] == null ? 'N/A' : `${response.scores[key]}/5`}</strong>
            </div>
          ))}
        </div>
      )}

      {response.score_error && (
        <p className="conversation-provider-error" role="status">{response.score_error}</p>
      )}

      {(response.score_rationale || Object.keys(evidence).length > 0) && (
        <details className="conversation-evidence">
          <summary>Evaluation evidence</summary>
          {response.score_rationale && <p>{response.score_rationale}</p>}
          {Object.entries(evidence).map(([key, item]) => (
            <div className="conversation-evidence-item" key={key}>
              <strong>{key.replaceAll('_', ' ')}</strong>
              <p>{item.reason}</p>
              {item.evidence && <blockquote>{item.evidence}</blockquote>}
              {item.missing_requirements?.length > 0 && (
                <small>Unmet: {item.missing_requirements.join(', ')}</small>
              )}
            </div>
          ))}
        </details>
      )}
    </article>
  )
}

function ConversationTurnView({ turn }) {
  const result = turn.result
  const winners = result.winner_models || []
  return (
    <article className="conversation-turn">
      <div className="conversation-user-message">
        <span>You</span>
        <p>{turn.prompt}</p>
        {turn.reference_answer && (
          <details>
            <summary>Reference answer</summary>
            <p>{turn.reference_answer}</p>
          </details>
        )}
        <small>{formatTimestamp(turn.created_at)}</small>
      </div>

      <section className="conversation-turn-evaluation" aria-label="Turn evaluation">
        <header className="conversation-turn-summary">
          <div>
            <strong>Comparison and evaluation</strong>
            <span>Judged by {result.evaluation_model || 'the configured evaluation models'}</span>
          </div>
          <span className={`status-pill status-${result.evaluation_status === 'success' ? 'success' : result.evaluation_status === 'partial_failure' ? 'warning' : result.evaluation_status === 'failed' ? 'failed' : 'idle'}`}>
            {String(result.evaluation_status || 'not_run').replaceAll('_', ' ')}
          </span>
        </header>

        {result.evaluation_error && (
          <p className="conversation-provider-error" role="status">{result.evaluation_error}</p>
        )}
        {winners.length > 0 && (
          <div className="conversation-turn-winner">
            Winner{winners.length === 1 ? '' : 's'}: {winners.map(providerName).join(' · ')}
          </div>
        )}

        <div className="conversation-response-grid">
          {(result.responses || []).map((response) => (
            <ConversationResponse
              key={response.model}
              response={response}
              winner={winners.includes(response.model)}
            />
          ))}
        </div>
      </section>
    </article>
  )
}

export default function ConversationWorkspace({
  providers,
  selectedProviders,
  onToggleProvider,
  connection,
  renderAnalytics,
}) {
  const [conversations, setConversations] = useState([])
  const [activeConversationId, setActiveConversationId] = useState(null)
  const [turns, setTurns] = useState([])
  const [prompt, setPrompt] = useState('')
  const [referenceAnswer, setReferenceAnswer] = useState('')
  const [referenceOpen, setReferenceOpen] = useState(false)
  const [loadingList, setLoadingList] = useState(true)
  const [loadingConversation, setLoadingConversation] = useState(false)
  const [creating, setCreating] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [requestError, setRequestError] = useState('')
  const [pendingPrompt, setPendingPrompt] = useState('')
  const messageListRef = useRef(null)
  const busy = loadingConversation || creating || submitting
  const activeConversation = conversations.find(
    (conversation) => conversation.id === activeConversationId,
  )

  useEffect(() => {
    let active = true
    async function loadInitialConversations() {
      try {
        const available = await fetchConversations()
        if (!active) return
        setConversations(available)
        if (available.length) {
          setLoadingConversation(true)
          const recent = await fetchConversation(available[0].id)
          if (!active) return
          setActiveConversationId(recent.id)
          setTurns(recent.turns)
        }
      } catch (error) {
        if (active) {
          setRequestError(error instanceof Error ? error.message : String(error))
        }
      } finally {
        if (active) {
          setLoadingList(false)
          setLoadingConversation(false)
        }
      }
    }
    loadInitialConversations()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight
    }
  }, [turns, pendingPrompt])

  async function refreshConversations() {
    const available = await fetchConversations()
    setConversations(available)
  }

  async function handleNewConversation() {
    setRequestError('')
    setCreating(true)
    try {
      const created = await createConversation()
      setConversations((current) => [created, ...current])
      setActiveConversationId(created.id)
      setTurns([])
      setPrompt('')
      setReferenceAnswer('')
      setReferenceOpen(false)
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : String(error))
    } finally {
      setCreating(false)
    }
  }

  async function handleOpenConversation(conversationId) {
    if (busy || conversationId === activeConversationId) return
    setRequestError('')
    setLoadingConversation(true)
    try {
      const conversation = await fetchConversation(conversationId)
      setActiveConversationId(conversation.id)
      setTurns(conversation.turns)
      setPrompt('')
      setReferenceAnswer('')
      setReferenceOpen(false)
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : String(error))
    } finally {
      setLoadingConversation(false)
    }
  }

  async function handleRename() {
    if (!activeConversation || busy) return
    const title = window.prompt('Rename conversation', activeConversation.title)
    if (title == null) return
    setRequestError('')
    try {
      const updated = await renameConversation(activeConversation.id, title)
      setConversations((current) => current.map((item) => (
        item.id === updated.id ? updated : item
      )))
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : String(error))
    }
  }

  async function handleDelete(conversation) {
    if (busy || !window.confirm(`Delete "${conversation.title}" and all its turns?`)) return
    setRequestError('')
    try {
      await deleteConversation(conversation.id)
      setConversations((current) => current.filter((item) => item.id !== conversation.id))
      if (conversation.id === activeConversationId) {
        setActiveConversationId(null)
        setTurns([])
      }
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : String(error))
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const currentPrompt = prompt.trim()
    if (!currentPrompt) return
    if (connection !== 'connected') {
      setRequestError('The backend is not connected.')
      return
    }
    if (!selectedProviders.length) {
      setRequestError('Select at least one provider to compare.')
      return
    }

    setRequestError('')
    setSubmitting(true)
    setPendingPrompt(currentPrompt)
    try {
      let conversationId = activeConversationId
      if (!conversationId) {
        const created = await createConversation()
        conversationId = created.id
        setActiveConversationId(created.id)
        setConversations((current) => [created, ...current])
      }

      const turn = await submitConversationTurn(conversationId, {
        prompt: currentPrompt,
        ...(referenceAnswer.trim() ? { reference_answer: referenceAnswer.trim() } : {}),
        providers: selectedProviders,
      })
      setTurns((current) => [...current, turn])
      setPrompt('')
      setReferenceAnswer('')
      setReferenceOpen(false)
      await refreshConversations()
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : String(error))
    } finally {
      setPendingPrompt('')
      setSubmitting(false)
    }
  }

  function handlePromptKeyDown(event) {
    if (
      event.key === 'Enter'
      && !event.shiftKey
      && !event.nativeEvent.isComposing
    ) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <section id="comparison" className="conversation-section scroll-anchor">
      <div className="conversation-section-heading">
        <div>
          <span className="section-kicker">MULTI-TURN EVALUATION</span>
          <h2>Model Comparison</h2>
          <p>Compare every turn, inspect its evaluation, and track performance across this conversation.</p>
        </div>
      </div>

      <div className="conversation-workspace">
        <aside className="conversation-sidebar" aria-label="Conversation history">
          <button
            type="button"
            className="conversation-new-button"
            disabled={busy}
            onClick={handleNewConversation}
          >
            <span aria-hidden="true">+</span>
            New Chat
          </button>
          <div className="conversation-sidebar-label">Your conversations</div>
          <div className="conversation-list">
            {loadingList && <p className="conversation-list-empty">Loading history…</p>}
            {!loadingList && conversations.length === 0 && (
              <p className="conversation-list-empty">Your saved conversations will appear here.</p>
            )}
            {conversations.map((conversation) => (
              <div
                className={`conversation-list-item ${conversation.id === activeConversationId ? 'active' : ''}`}
                key={conversation.id}
              >
                <button
                  type="button"
                  className="conversation-open-button"
                  disabled={busy}
                  onClick={() => handleOpenConversation(conversation.id)}
                  title={conversation.title}
                >
                  <span>{conversation.title}</span>
                  <small>{conversation.turn_count} turn{conversation.turn_count === 1 ? '' : 's'}</small>
                </button>
                <button
                  type="button"
                  className="conversation-delete-button"
                  aria-label={`Delete ${conversation.title}`}
                  title="Delete conversation"
                  disabled={busy}
                  onClick={() => handleDelete(conversation)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </aside>

        <div className="conversation-chat-panel">
          <header className="conversation-chat-header">
            <div>
              <strong>{activeConversation?.title || 'New conversation'}</strong>
              <small>{activeConversation ? `${turns.length} evaluated turn${turns.length === 1 ? '' : 's'}` : 'Start a new multi-model evaluation'}</small>
            </div>
            {activeConversation && (
              <button
                type="button"
                className="conversation-rename-button"
                disabled={busy}
                onClick={handleRename}
              >
                Rename
              </button>
            )}
          </header>

          <div className="conversation-message-list" ref={messageListRef} aria-live="polite">
            {loadingConversation && <p className="conversation-list-empty">Opening conversation…</p>}
            {!loadingConversation && turns.length === 0 && !pendingPrompt && (
              <div className="conversation-welcome">
                <span className="conversation-welcome-icon" aria-hidden="true">✦</span>
                <h3>Compare a prompt across your models</h3>
                <p>Previous turns are supplied as provider-specific context. Each new turn is independently scored and ranked.</p>
              </div>
            )}
            {turns.map((turn) => <ConversationTurnView key={turn.id} turn={turn} />)}
            {pendingPrompt && (
              <article className="conversation-turn">
                <div className="conversation-user-message">
                  <span>You</span>
                  <p>{pendingPrompt}</p>
                </div>
                <div className="conversation-pending" role="status">
                  <span className="spinner" />
                  <span>Generating responses and evaluating each provider…</span>
                </div>
              </article>
            )}
            <div aria-hidden="true" />
          </div>

          <form className="conversation-composer" onSubmit={handleSubmit}>
            {requestError && <p className="conversation-error" role="alert">{requestError}</p>}
            <div className="conversation-provider-selection">
              <span>Compare:</span>
              {providers.map((provider) => (
                <label key={provider} title={providerName(provider)}>
                  <input
                    type="checkbox"
                    checked={selectedProviders.includes(provider)}
                    disabled={connection !== 'connected' || busy}
                    onChange={() => onToggleProvider(provider)}
                  />
                  {providerName(provider)}
                </label>
              ))}
            </div>
            {referenceOpen && (
              <label className="conversation-reference-field">
                Reference answer <span>Optional</span>
                <textarea
                  value={referenceAnswer}
                  maxLength={12000}
                  rows={2}
                  disabled={busy}
                  onChange={(event) => setReferenceAnswer(event.target.value)}
                  placeholder="Add a trusted answer for factual evaluation..."
                />
              </label>
            )}
            <div className="conversation-input-row">
              <textarea
                id="conversation-prompt"
                aria-label="Message to compare"
                value={prompt}
                maxLength={12000}
                rows={2}
                disabled={busy}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={handlePromptKeyDown}
                placeholder="Message your models..."
              />
              <button
                type="submit"
                className="conversation-send-button"
                disabled={busy || !prompt.trim() || connection !== 'connected'}
                aria-label={submitting ? 'Evaluating prompt' : 'Send prompt for evaluation'}
              >
                {submitting ? <span className="spinner" /> : <span aria-hidden="true">↑</span>}
              </button>
            </div>
            <div className="conversation-composer-footer">
              <button
                type="button"
                className="conversation-reference-toggle"
                disabled={busy}
                onClick={() => setReferenceOpen((open) => !open)}
              >
                {referenceOpen ? 'Remove reference' : 'Add reference answer'}
              </button>
              <span>Enter to evaluate · Shift+Enter for a new line</span>
            </div>
          </form>
        </div>
      </div>
      {renderAnalytics?.(turns)}
    </section>
  )
}
