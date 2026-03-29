import { useState } from 'react'
import { useChatContext } from '../context/ChatContext'
import { SourceCard } from './SourceCard'
import { TranscriptModal } from './TranscriptModal'

export function SourcesPanel() {
  const { activeChat, sourcesMessageId } = useChatContext()
  const [transcriptChunkId, setTranscriptChunkId] = useState<string | null>(null)

  if (!activeChat || sourcesMessageId == null) {
    return null
  }

  const targetMessage = activeChat.messages.find((m) => m.id === sourcesMessageId)
  const rawSources = targetMessage?.sources ?? []

  // Sort chronologically by call_date, then by source_index within the same date
  const sources = [...rawSources].sort((a, b) => {
    const da = typeof a.metadata.call_date === 'string' ? a.metadata.call_date : ''
    const db = typeof b.metadata.call_date === 'string' ? b.metadata.call_date : ''
    if (da !== db) return da < db ? -1 : 1
    return (a.source_index ?? 0) - (b.source_index ?? 0)
  })

  return (
    <>
      <aside className="sources-panel">
        <header className="sources-header">
          <div>
            <div className="sources-title">Sources</div>
            <div className="sources-subtitle">
              {sources.length > 0
                ? `${sources.length} source${sources.length === 1 ? '' : 's'} — click to view transcript`
                : 'No sources for this message'}
            </div>
          </div>
        </header>
        <div className="sources-body">
          {sources.length === 0 ? (
            <div className="empty-state">
              When the model cites transcript chunks, they will appear here with raw content and
              metadata.
            </div>
          ) : (
            sources.map((s, idx) => (
              <SourceCard
                key={s.chunk_id}
                source={s}
                index={idx}
                onViewTranscript={setTranscriptChunkId}
              />
            ))
          )}
        </div>
      </aside>
      {transcriptChunkId && (
        <TranscriptModal
          chunkId={transcriptChunkId}
          onClose={() => setTranscriptChunkId(null)}
        />
      )}
    </>
  )
}

