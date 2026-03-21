import ReactMarkdown from 'react-markdown'
import type { ChatMessage } from '../types'

type Props = {
  message: ChatMessage
  onToggleSources?: () => void
  isSourcesShown?: boolean
}

export function MessageBubble({ message, onToggleSources, isSourcesShown }: Props) {
  const isUser = message.role === 'user'
  const hasSources = !!message.sources && message.sources.length > 0

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
          ) : (
            <ReactMarkdown>{message.content}</ReactMarkdown>
          )}
        </div>
        {!isUser && (
          <div className="message-meta">
            {typeof message.confidence === 'number' && (
              <span className="confidence-pill">
                Confidence {(message.confidence * 100).toFixed(0)}%
              </span>
            )}
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

