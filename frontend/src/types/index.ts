export type SearchModeType = 'vector' | 'keyword' | 'hybrid'

export interface QueryRequest {
  query: string
  session_id?: string | null
  search_mode?: SearchModeType | null
  retrieval_threshold?: number | null
}

export interface CitedSpan {
  start: number
  end: number
}

export interface SourceDocument {
  chunk_id: string
  content: string
  similarity: number
  metadata: Record<string, unknown>
  /** Character ranges in content that were cited in the answer (for highlighting). */
  cited_spans?: CitedSpan[]
  /** 1-based index matching [Source N] in the answer text. */
  source_index?: number
}

export interface AgentResponse {
  answer: string
  sources: SourceDocument[]
  reasoning?: string | null
  tool_calls_made: string[]
}

export interface HistoryEntry {
  role: 'user' | 'assistant'
  content: string
  created_at?: string | null
}

/** Backend conversation session list item */
export interface ConversationSessionSummary {
  session_id: string
  updated_at: string | null
}

/** Backend conversation history entry (same shape as HistoryEntry) */
export type ConversationHistoryEntry = HistoryEntry

export type Role = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: Role
  content: string
  createdAt: string
  sources?: SourceDocument[]
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
}

/** Single chunk in a full-transcript response */
export interface TranscriptChunk {
  id: string
  content: string
  metadata: Record<string, unknown>
  chunk_index: number
}

/** Full transcript for a document, returned when user clicks a chunk */
export interface TranscriptResponse {
  doc_id: string
  title: string
  metadata: Record<string, unknown>
  chunks: TranscriptChunk[]
  requested_chunk_id: string
  /** Overlap-free full text when stored (for new ingestions); falls back to chunked view */
  full_transcript?: string | null
}

