import { useState } from 'react'
import { useChatContext } from '../context/ChatContext'
import { SourceCard } from './SourceCard'
import { TranscriptModal } from './TranscriptModal'

export function SourcesPanel() {
  const { activeChat, sourcesMessageId } = useChatContext()
  const [transcriptChunkId, setTranscriptChunkId] = useState<string | null>(null)

  if (!activeChat) {
    return <aside className="sources-panel" />
  }

  const targetMessage =
    activeChat.messages.find((m) => m.id === sourcesMessageId) ??
    [...activeChat.messages].reverse().find((m) => m.role === 'assistant' && m.sources?.length)

  const sources = targetMessage?.sources ?? []

  return (
    <>
      <aside className="sources-panel">
        <header className="sources-header">
          <div>
            <div className="sources-title">Sources</div>
            <div className="sources-subtitle">
              {sources.length > 0
                ? `Showing ${sources.length} chunk${sources.length === 1 ? '' : 's'} — click to see full transcript`
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

