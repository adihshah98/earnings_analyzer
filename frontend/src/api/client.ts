import type {
  AgentResponse,
  ConversationHistoryEntry,
  ConversationSessionSummary,
  QueryRequest,
  TranscriptResponse,
} from '../types'
import { getStoredToken } from '../context/AuthContext'

const API_BASE = import.meta.env.VITE_API_URL || ''

let _lastQueryAt: number | null = null
export function getLastQueryTime(): number | null { return _lastQueryAt }

function getAuthHeaders(): HeadersInit {
  const token = getStoredToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

const JSON_HEADERS: HeadersInit = {
  'Content-Type': 'application/json',
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Request failed with status ${res.status}`)
  }
  return (await res.json()) as T
}

export type StreamEvent =
  | { type: 'delta'; text: string }
  | { type: 'done'; payload: AgentResponse }

/**
 * Stream agent response as SSE. Yields delta events then a done event with full payload.
 */
export async function* queryAgentStream(
  body: QueryRequest,
): AsyncGenerator<StreamEvent, void, unknown> {
  _lastQueryAt = Date.now()
  const res = await fetch(`${API_BASE}/agent/query`, {
    method: 'POST',
    headers: { ...JSON_HEADERS, ...getAuthHeaders() },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Request failed with status ${res.status}`)
  }
  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6)) as { type: string; text?: string; payload?: AgentResponse }
          if (data.type === 'delta' && typeof data.text === 'string') {
            yield { type: 'delta', text: data.text }
          } else if (data.type === 'done' && data.payload) {
            yield { type: 'done', payload: data.payload }
          }
        } catch {
          // ignore malformed
        }
      }
    }
  }
  if (buffer.trim()) {
    try {
      const data = JSON.parse(buffer.startsWith('data: ') ? buffer.slice(6) : buffer) as {
        type: string
        payload?: AgentResponse
      }
      if (data.type === 'done' && data.payload) {
        yield { type: 'done', payload: data.payload }
      }
    } catch {
      // ignore
    }
  }
}

export async function getTranscriptByChunkId(
  chunkId: string,
): Promise<TranscriptResponse> {
  const res = await fetch(
    `${API_BASE}/rag/chunks/${encodeURIComponent(chunkId)}/transcript`,
    { method: 'GET', headers: getAuthHeaders() },
  )
  return handleResponse<TranscriptResponse>(res)
}

/** Fetch list of conversation sessions from backend (newest first). */
export async function getConversationSessions(): Promise<
  ConversationSessionSummary[]
> {
  const res = await fetch(`${API_BASE}/conversations/sessions`, {
    method: 'GET',
    headers: getAuthHeaders(),
  })
  return handleResponse<ConversationSessionSummary[]>(res)
}

/** Fetch conversation history for a session. */
export async function getConversationHistory(
  sessionId: string,
): Promise<ConversationHistoryEntry[]> {
  const res = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(sessionId)}/history`,
    { method: 'GET', headers: getAuthHeaders() },
  )
  return handleResponse<ConversationHistoryEntry[]>(res)
}

/** Check if the backend is reachable. Returns true if healthy. */
export async function checkHealth(timeoutMs = 8000): Promise<boolean> {
  try {
    const controller = new AbortController()
    const id = setTimeout(() => controller.abort(), timeoutMs)
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal })
    clearTimeout(id)
    return res.ok
  } catch {
    return false
  }
}

/**
 * Warm up backend caches (company list + embedding client) and return health status.
 * Use on initial mount so the first real query doesn't pay the cold-cache penalty.
 */
export async function warmupBackend(timeoutMs = 15000): Promise<boolean> {
  try {
    const controller = new AbortController()
    const id = setTimeout(() => controller.abort(), timeoutMs)
    const res = await fetch(`${API_BASE}/warmup`, { signal: controller.signal })
    clearTimeout(id)
    return res.ok
  } catch {
    return false
  }
}

export async function deleteConversation(sessionId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/conversations/${encodeURIComponent(sessionId)}`,
    { method: 'DELETE', headers: getAuthHeaders() },
  )
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `Delete failed with status ${res.status}`)
  }
}

