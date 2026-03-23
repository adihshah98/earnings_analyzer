import { useEffect, useRef } from 'react'
import { useChatContext } from '../context/ChatContext'
import { MessageBubble } from './MessageBubble'
import { ChatInput } from './ChatInput'

export function ChatView() {
  const { activeChat, sourcesMessageId, setSourcesTarget, streamingAssistantId } =
    useChatContext()
  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const el = listRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [activeChat?.messages.length])

  if (!activeChat) {
    return (
      <main className="main-panel">
        <header className="chat-header">
          <div>
            <div className="chat-header-title">Earnings Analyzer</div>
            <div className="chat-header-subtitle">
              Start a conversation to analyze earnings call transcripts.
            </div>
          </div>
        </header>
        <div className="empty-state">
          Create a new chat from the left to begin. Ask about revenue, guidance, margin trends, or
          anything else you care about.
        </div>
      </main>
    )
  }

  return (
    <main className="main-panel">
      <header className="chat-header">
        <div>
          <div className="chat-header-title">{activeChat.title || 'Conversation'}</div>
          <div className="chat-header-subtitle">
            {activeChat.messages.length} message
            {activeChat.messages.length === 1 ? '' : 's'}
          </div>
        </div>
      </header>

      <div className="chat-messages" ref={listRef}>
        {activeChat.messages.map((m) => (
          <MessageBubble
            key={m.id}
            message={m}
            isStreaming={m.role === 'assistant' && streamingAssistantId === m.id}
            onToggleSources={
              m.role === 'assistant' && m.sources && m.sources.length > 0
                ? () => setSourcesTarget(sourcesMessageId === m.id ? null : m.id)
                : undefined
            }
            isSourcesShown={sourcesMessageId === m.id}
          />
        ))}
        {activeChat.messages.length === 0 && (
          <div className="empty-state" style={{ alignItems: 'flex-start' }}>
            Ask a question like “What did ACME guide for revenue in Q3 2024?” to get started.
          </div>
        )}
      </div>

      <ChatInput />
    </main>
  )
}

