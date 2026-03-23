import { useMemo } from 'react'
import type { Components } from 'react-markdown'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage } from '../types'
import { prepareAssistantMarkdown } from '../utils/markdown'

const MARKDOWN_COMPONENTS: Components = {
  table: ({ children }) => (
    <div className="markdown-table-wrap">
      <table>{children}</table>
    </div>
  ),
}

type Props = {
  message: ChatMessage
  /** While true, skip Markdown parsing so partial tables are not broken mid-stream. */
  isStreaming?: boolean
  onToggleSources?: () => void
  isSourcesShown?: boolean
}

export function MessageBubble({
  message,
  isStreaming = false,
  onToggleSources,
  isSourcesShown,
}: Props) {
  const isUser = message.role === 'user'
  const hasSources = !!message.sources && message.sources.length > 0

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
              components={MARKDOWN_COMPONENTS}
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
