import { useState } from 'react'
import { useChatContext } from '../context/ChatContext'
import { SourceCard } from './SourceCard'
import { TranscriptModal } from './TranscriptModal'

export function SourcesPanel() {
  const { activeChat, sourcesMessageId, setSourcesTarget } = useChatContext()
  const [transcriptChunkId, setTranscriptChunkId] = useState<string | null>(null)

  if (!activeChat || sourcesMessageId == null) {
    return null
  }

  const targetMessage = activeChat.messages.find((m) => m.id === sourcesMessageId)
  const sources = targetMessage?.sources ?? []

  return (
    <>
      <aside className="sources-panel">
        <header className="sources-header">
          <div>
            <div className="sources-title">Sources</div>
            <div className="sources-subtitle">
              {sources.length > 0
                ? `${sources.length} source${sources.length === 1 ? '' : 's'} — click to see full transcript`
                : 'No sources for this message'}
            </div>
          </div>
          <button
            type="button"
            className="sources-hide-button"
            onClick={() => setSourcesTarget(null)}
          >
            Hide sources
          </button>
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

