import { useMemo } from 'react'
import type { Components } from 'react-markdown'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage } from '../types'
import { prepareAssistantMarkdown } from '../utils/markdown'

type Props = {
  message: ChatMessage
  /** While true, skip Markdown parsing so partial tables are not broken mid-stream. */
  isStreaming?: boolean
  onToggleSources?: () => void
  isSourcesShown?: boolean
  /** Called when a [Source N] citation badge in the answer is clicked. */
  onSourceClick?: (sourceIndex: number) => void
}

export function MessageBubble({
  message,
  isStreaming = false,
  onToggleSources,
  isSourcesShown,
  onSourceClick,
}: Props) {
  const isUser = message.role === 'user'
  const hasSources = !!message.sources && message.sources.length > 0

  const markdownComponents: Components = useMemo(() => ({
    table: ({ children }) => (
      <div className="markdown-table-wrap">
        <table>{children}</table>
      </div>
    ),
    a: ({ href, children }) => {
      const match = href?.match(/^#source-(\d+)$/)
      if (match) {
        const idx = parseInt(match[1], 10)
        return (
          <button
            type="button"
            className="source-ref-badge"
            onClick={() => onSourceClick?.(idx)}
          >
            {children}
          </button>
        )
      }
      return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
    },
  }), [onSourceClick])

  const markdownSource = useMemo(
    () => prepareAssistantMarkdown(message.content),
    [message.content],
  )

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
        <div className="message-content">
          {isUser ? (
            message.content
          ) : message.content === '' ? (
            <div className="thinking-dots">
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              <span className="thinking-dot" />
            </div>
          ) : isStreaming ? (
            <div className="message-content-streaming">{message.content}</div>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {markdownSource}
            </ReactMarkdown>
          )}
        </div>
        {!isUser && (
          <div className="message-meta">
            {hasSources && onToggleSources && (
              <button type="button" className="view-sources-button" onClick={onToggleSources}>
                {isSourcesShown ? 'Hide sources' : `View sources (${message.sources!.length})`}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
