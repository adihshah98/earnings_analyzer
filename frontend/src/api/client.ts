import type {
  AgentResponse,
  QueryRequest,
  TranscriptResponse,
} from '../types'

const API_BASE = import.meta.env.VITE_API_URL || ''

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
  const res = await fetch(`${API_BASE}/agent/query`, {
    method: 'POST',
    headers: JSON_HEADERS,
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
    { method: 'GET' },
  )
  return handleResponse<TranscriptResponse>(res)
}

