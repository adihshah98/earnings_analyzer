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

export async function queryAgent(body: QueryRequest): Promise<AgentResponse> {
  const res = await fetch(`${API_BASE}/agent/query`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  })
  return handleResponse<AgentResponse>(res)
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

