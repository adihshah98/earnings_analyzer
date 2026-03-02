import { useEffect, useState } from 'react'
import { getTranscriptByChunkId } from '../api/client'
import type { TranscriptChunk, TranscriptResponse } from '../types'

type Props = {
  chunkId: string
  onClose: () => void
}

export function TranscriptModal({ chunkId, onClose }: Props) {
  const [data, setData] = useState<TranscriptResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    getTranscriptByChunkId(chunkId)
      .then((res) => {
        if (!cancelled) setData(res)
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load transcript')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [chunkId])

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose()
  }

  return (
    <div className="transcript-modal-backdrop" onClick={handleBackdropClick} role="dialog" aria-modal="true" aria-label="Full transcript">
      <div className="transcript-modal">
        <header className="transcript-modal-header">
          <h2 className="transcript-modal-title">
            {loading ? 'Loading…' : error ? 'Error' : data ? data.title : 'Transcript'}
          </h2>
          <button type="button" className="transcript-modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="transcript-modal-body">
          {loading && <div className="transcript-modal-loading">Loading full transcript…</div>}
          {error && <div className="error-banner">{error}</div>}
          {data && (
            <div className="transcript-full-chunks">
              <div className="transcript-meta">
                {data.metadata.company_ticker && (
                  <span>Company: {String(data.metadata.company_ticker)}</span>
                )}
                {data.metadata.call_date && (
                  <span>Call date: {String(data.metadata.call_date)}</span>
                )}
                {data.full_transcript ? (
                  <span>Full transcript</span>
                ) : (
                  <span>{data.chunks.length} chunk{data.chunks.length === 1 ? '' : 's'}</span>
                )}
              </div>
              {data.full_transcript ? (
                <>
                  <div className="transcript-full-text">{data.full_transcript}</div>
                  {data.chunks.some((c) => c.id === data.requested_chunk_id) && (
                    <div className="transcript-cited-chunk">
                      <span className="transcript-chunk-block-badge">Cited in answer</span>
                      <div className="transcript-chunk-block-content">
                        {data.chunks.find((c) => c.id === data.requested_chunk_id)?.content}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                data.chunks.map((chunk) => (
                  <TranscriptChunkBlock
                    key={chunk.id}
                    chunk={chunk}
                    isHighlighted={chunk.id === data.requested_chunk_id}
                  />
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function TranscriptChunkBlock({
  chunk,
  isHighlighted,
}: {
  chunk: TranscriptChunk
  isHighlighted: boolean
}) {
  return (
    <div
      className={
        'transcript-chunk-block' + (isHighlighted ? ' transcript-chunk-block-highlighted' : '')
      }
      data-chunk-id={chunk.id}
    >
      <div className="transcript-chunk-block-header">
        <span className="transcript-chunk-block-index">Chunk {chunk.chunk_index + 1}</span>
        {isHighlighted && <span className="transcript-chunk-block-badge">Cited in answer</span>}
      </div>
      <div className="transcript-chunk-block-content">{chunk.content}</div>
    </div>
  )
}
