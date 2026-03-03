import type { CitedSpan, SourceDocument } from '../types'

type Props = {
  source: SourceDocument
  index: number
  onViewTranscript?: (chunkId: string) => void
}

/** Merge overlapping/adjacent spans and return sorted by start. */
function mergeSpans(spans: CitedSpan[]): CitedSpan[] {
  if (spans.length === 0) return []
  const sorted = [...spans].sort((a, b) => a.start - b.start)
  const out: CitedSpan[] = [sorted[0]]
  for (let i = 1; i < sorted.length; i++) {
    const cur = sorted[i]
    const last = out[out.length - 1]
    if (cur.start <= last.end) {
      last.end = Math.max(last.end, cur.end)
    } else {
      out.push(cur)
    }
  }
  return out
}

/** Split content into segments: { text, highlighted }. */
function contentWithHighlights(content: string, spans: CitedSpan[]): { text: string; highlighted: boolean }[] {
  const merged = mergeSpans(spans)
  if (merged.length === 0) return [{ text: content, highlighted: false }]
  const segments: { text: string; highlighted: boolean }[] = []
  let pos = 0
  for (const span of merged) {
    if (span.start > pos) {
      segments.push({ text: content.slice(pos, span.start), highlighted: false })
    }
    segments.push({ text: content.slice(span.start, span.end), highlighted: true })
    pos = span.end
  }
  if (pos < content.length) {
    segments.push({ text: content.slice(pos), highlighted: false })
  }
  return segments
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

  const hasHighlights = source.cited_spans && source.cited_spans.length > 0
  const segments = hasHighlights
    ? contentWithHighlights(source.content, source.cited_spans!)
    : [{ text: source.content, highlighted: false }]

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
      <div className="source-content">
        {segments.map((seg, i) =>
          seg.highlighted ? (
            <mark key={i} className="source-cited" title="Cited in answer">
              {seg.text}
            </mark>
          ) : (
            <span key={i}>{seg.text}</span>
          )
        )}
      </div>
      <div className="source-meta">
        {company && <span>Company: {company}</span>}
        {callDate && <span>Date: {callDate}</span>}
      </div>
      {onViewTranscript && (
        <div className="source-card-view-transcript">View full transcript →</div>
      )}
    </article>
  )
}

