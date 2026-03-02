import type { ChatMessage } from '../types'

type Props = {
  message: ChatMessage
  onViewSources?: () => void
}

export function MessageBubble({ message, onViewSources }: Props) {
  const isUser = message.role === 'user'
  const hasSources = !!message.sources && message.sources.length > 0

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
        <div>{message.content}</div>
        {!isUser && (
          <div className="message-meta">
            {typeof message.confidence === 'number' && (
              <span className="confidence-pill">
                Confidence {(message.confidence * 100).toFixed(0)}%
              </span>
            )}
            {hasSources && onViewSources && (
              <button type="button" className="view-sources-button" onClick={onViewSources}>
                View sources ({message.sources!.length})
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

