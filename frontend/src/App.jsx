import { useEffect, useState } from 'react'
import { API_BASE_URL, fetchProviders, runComparison } from './api'
import PrismWorkspace from './components/prism/PrismWorkspace'
import './App.css'

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
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
  const [activeSection, setActiveSection] = useState('comparison')

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
        setRequestError(errorMessage(error))
      })
    return () => {
      active = false
    }
  }, [])

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
      setRequestError(errorMessage(error))
    }
  }

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
      setActiveSection('comparison')
      window.setTimeout(() => {
        document.getElementById('comparison')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 50)
    } catch (error) {
      setRequestError(errorMessage(error))
    } finally {
      setLoading(false)
    }
  }

  function restoreRun(run) {
    setLatestRun(run)
    navigateTo('comparison')
  }

  function clearHistory() {
    setHistory([])
    setLatestRun(null)
    setRunCount(0)
  }

  return (
    <PrismWorkspace
      apiBaseUrl={API_BASE_URL}
      activeSection={activeSection}
      connection={connection}
      history={history}
      latestRun={latestRun}
      loading={loading}
      onClearHistory={clearHistory}
      onClearPrompt={clearPrompt}
      onNavigate={navigateTo}
      onPromptChange={setPrompt}
      onReferenceChange={setReferenceAnswer}
      onRestoreRun={restoreRun}
      onRetry={loadProviders}
      onSubmit={handleSubmit}
      onToggleProvider={toggleProvider}
      prompt={prompt}
      providers={providers}
      referenceAnswer={referenceAnswer}
      requestError={requestError}
      runCount={runCount}
      selectedProviders={selectedProviders}
    />
  )
}

export default App
