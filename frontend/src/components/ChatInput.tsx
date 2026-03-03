import { useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { useChatContext } from '../context/ChatContext'

export function ChatInput() {
  const { sendMessage, isSending, error } = useChatContext()
  const [value, setValue] = useState('')

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed) return
    await sendMessage(trimmed)
    setValue('')
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
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSending}
        />
        <button type="submit" className="send-button" disabled={isSending || !value.trim()}>
          {isSending ? 'Thinking…' : 'Send'}
        </button>
      </div>
      <div className="input-hint">Press Enter to send, Shift+Enter for a new line.</div>
    </form>
  )
}

