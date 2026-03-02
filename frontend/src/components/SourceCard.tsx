import type { SourceDocument } from '../types'

type Props = {
  source: SourceDocument
  index: number
  onViewTranscript?: (chunkId: string) => void
}

export function SourceCard({ source, index, onViewTranscript }: Props) {
  const title =
    (typeof source.metadata.title === 'string' && source.metadata.title) ||
    (typeof source.metadata.company_ticker === 'string' && source.metadata.company_ticker) ||
    'Source'

  const similarityPercent = (source.similarity * 100).toFixed(1)

  const company =
    typeof source.metadata.company_ticker === 'string' ? source.metadata.company_ticker : null
  const callDate =
    typeof source.metadata.call_date === 'string' ? source.metadata.call_date : null

  const handleClick = () => onViewTranscript?.(source.chunk_id)

  return (
    <article
      className={'source-card' + (onViewTranscript ? ' source-card-clickable' : '')}
      onClick={onViewTranscript ? handleClick : undefined}
      role={onViewTranscript ? 'button' : undefined}
      tabIndex={onViewTranscript ? 0 : undefined}
      onKeyDown={
        onViewTranscript
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                handleClick()
              }
            }
          : undefined
      }
    >
      <div className="source-card-header">
        <div className="source-title">
          #{index + 1} {title}
        </div>
        <div className="similarity-pill">{similarityPercent}% match</div>
      </div>
      <div className="source-content">{source.content}</div>
      <div className="source-meta">
        {company && <span>Company: {company}</span>}
        {callDate && <span>Date: {callDate}</span>}
        <span>Chunk: {source.chunk_id}</span>
      </div>
      {onViewTranscript && (
        <div className="source-card-view-transcript">View full transcript →</div>
      )}
    </article>
  )
}

