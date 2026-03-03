import type { FormEvent, KeyboardEvent } from 'react'
import { useChatContext } from '../context/ChatContext'

export function ChatInput() {
  const {
    sendMessage,
    error,
    activeChatId,
    sendingChatId,
    draftForActiveChat,
    setDraft,
  } = useChatContext()

  const isActiveChatSending = activeChatId != null && sendingChatId === activeChatId

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const trimmed = draftForActiveChat.trim()
    if (!trimmed) return
    if (!activeChatId) return
    await sendMessage(trimmed)
  }

  const handleKeyDown = async (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      await handleSubmit(e as unknown as FormEvent)
    }
  }

  return (
    <form className="chat-input-bar" onSubmit={handleSubmit}>
      {error && <div className="error-banner">{error}</div>}
      <div className="chat-input-row">
        <textarea
          className="chat-input-textarea"
          placeholder="Ask about earnings calls, guidance, and more…"
          value={draftForActiveChat}
          onChange={(e) => activeChatId && setDraft(activeChatId, e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isActiveChatSending}
        />
        <button
          type="submit"
          className="send-button"
          disabled={isActiveChatSending || !draftForActiveChat.trim()}
          title="Send"
        >
          <span className="send-icon" aria-hidden>↑</span>
        </button>
      </div>
      <div className="input-hint">Press Enter to send, Shift+Enter for a new line.</div>
    </form>
  )
}

