export function createSessionId(): string {
  // Lightweight UUID-ish generator sufficient for client-side session IDs
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export function truncateTitle(text: string, maxLength = 40): string {
  const trimmed = text.trim()
  if (trimmed.length <= maxLength) return trimmed || 'New chat'
  return `${trimmed.slice(0, maxLength - 1)}…`
}

const STORAGE_KEY = 'earnings-analyzer-chats-v1'

export function saveToStorage<T>(value: T): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value))
  } catch {
    // ignore
  }
}

export function loadFromStorage<T>(): T | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

