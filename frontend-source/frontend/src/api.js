const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

async function readResponse(response) {
  let payload
  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    const detail = payload?.detail
    const message =
      typeof detail === 'string'
        ? detail
        : detail?.message || `Request failed with status ${response.status}`
    throw new Error(message)
  }

  return payload
}

export async function fetchProviders() {
  const response = await fetch(`${API_BASE_URL}/models`)
  const payload = await readResponse(response)
  if (!Array.isArray(payload?.models)) {
    throw new Error('The backend returned an unexpected provider list.')
  }
  return payload.models
}

export async function runComparison(request) {
  const response = await fetch(`${API_BASE_URL}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return readResponse(response)
}

export { API_BASE_URL }
