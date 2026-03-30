import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react'
import {
  deleteConversation,
  getConversationHistory,
  getConversationSessions,
  queryAgentStream,
} from '../api/client'
import type { AgentResponse, ChatMessage, ChatSession } from '../types'
import {
  createSessionId,
  loadFromStorage,
  saveToStorage,
  truncateTitle,
} from '../utils/helpers'

type ChatState = {
  chats: ChatSession[]
  activeChatId: string | null
  sendingChatId: string | null
  streamingAssistantId: string | null
  drafts: Record<string, string>
  error: string | null
  sourcesMessageId: string | null
}

type ChatAction =
  | { type: 'INIT_FROM_STORAGE'; payload: { chats: ChatSession[]; activeChatId: string | null } }
  | { type: 'INIT_FROM_BACKEND'; payload: { chats: ChatSession[]; activeChatId: string | null } }
  | { type: 'CREATE_CHAT'; payload: { chat: ChatSession } }
  | { type: 'SET_ACTIVE_CHAT'; payload: { chatId: string } }
  | {
      type: 'SET_CHAT_MESSAGES'
      payload: { chatId: string; messages: ChatMessage[]; title?: string }
    }
  | { type: 'DELETE_CHAT'; payload: { chatId: string } }
  | { type: 'SET_ERROR'; payload: { error: string | null } }
  | { type: 'SET_DRAFT'; payload: { chatId: string; value: string } }
  | { type: 'SEND_START'; payload: { chatId: string } }
  | {
      type: 'ADD_STREAMING_MESSAGES'
      payload: {
        chatId: string
        userMessage: ChatMessage
        assistantMessage: ChatMessage
      }
    }
  | { type: 'STREAM_DELTA'; payload: { chatId: string; text: string } }
  | {
      type: 'SEND_SUCCESS_STREAM'
      payload: { chatId: string; messageId: string; payload: AgentResponse }
    }
  | {
      type: 'SEND_SUCCESS'
      payload: {
        chatId: string
        userMessage: ChatMessage
        assistantMessage: ChatMessage
      }
    }
  | {
      type: 'SEND_ERROR'
      payload: {
        chatId: string
        userMessageId: string
        error: string
        /** When streaming, remove this assistant message (placeholder) on error */
        assistantMessageId?: string
      }
    }
  | { type: 'SET_SOURCES_TARGET'; payload: { messageId: string | null } }

const initialState: ChatState = {
  chats: [],
  activeChatId: null,
  sendingChatId: null,
  streamingAssistantId: null,
  drafts: {},
  error: null,
  sourcesMessageId: null,
}

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'INIT_FROM_STORAGE': {
      return {
        ...state,
        chats: action.payload.chats,
        activeChatId: action.payload.activeChatId,
      }
    }
    case 'INIT_FROM_BACKEND': {
      return {
        ...state,
        chats: action.payload.chats,
        activeChatId: action.payload.activeChatId,
      }
    }
    case 'SET_CHAT_MESSAGES': {
      const { chatId, messages, title } = action.payload
      return {
        ...state,
        chats: state.chats.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                messages,
                messageCount: messages.length,
                ...(title !== undefined ? { title } : {}),
              }
            : chat,
        ),
      }
    }
    case 'DELETE_CHAT': {
      const { chatId } = action.payload
      const nextChats = state.chats.filter((c) => c.id !== chatId)
      const nextActive =
        state.activeChatId === chatId
          ? nextChats[0]?.id ?? null
          : state.activeChatId
      return {
        ...state,
        chats: nextChats,
        activeChatId: nextActive,
        drafts: (() => {
          const { [chatId]: _, ...rest } = state.drafts
          return rest
        })(),
      }
    }
    case 'CREATE_CHAT': {
      return {
        ...state,
        chats: [action.payload.chat, ...state.chats],
        activeChatId: action.payload.chat.id,
        error: null,
        sourcesMessageId: null,
      }
    }
    case 'SET_ACTIVE_CHAT': {
      return {
        ...state,
        activeChatId: action.payload.chatId,
        error: null,
        sourcesMessageId: null,
      }
    }
    case 'SET_ERROR': {
      return {
        ...state,
        error: action.payload.error,
      }
    }
    case 'SET_DRAFT': {
      const { chatId, value } = action.payload
      return {
        ...state,
        drafts: { ...state.drafts, [chatId]: value },
      }
    }
    case 'SEND_START': {
      return {
        ...state,
        sendingChatId: action.payload.chatId,
        drafts: { ...state.drafts, [action.payload.chatId]: '' },
        error: null,
        sourcesMessageId: null,
      }
    }
    case 'ADD_STREAMING_MESSAGES': {
      const { chatId, userMessage, assistantMessage } = action.payload
      return {
        ...state,
        sendingChatId: chatId,
        streamingAssistantId: assistantMessage.id,
        chats: state.chats.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                title:
                  chat.messages.length === 0 && userMessage.content
                    ? truncateTitle(userMessage.content)
                    : chat.title,
                messages: [...chat.messages, userMessage, assistantMessage],
              }
            : chat,
        ),
      }
    }
    case 'STREAM_DELTA': {
      const { chatId, text } = action.payload
      return {
        ...state,
        chats: state.chats.map((chat) =>
          chat.id === chatId && state.streamingAssistantId
            ? {
                ...chat,
                messages: chat.messages.map((msg) =>
                  msg.id === state.streamingAssistantId
                    ? { ...msg, content: msg.content + text }
                    : msg,
                ),
              }
            : chat,
        ),
      }
    }
    case 'SEND_SUCCESS_STREAM': {
      const { chatId, messageId, payload: res } = action.payload
      return {
        ...state,
        sendingChatId: null,
        streamingAssistantId: null,
        drafts: { ...state.drafts, [chatId]: '' },
        chats: state.chats.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                messages: chat.messages.map((msg) =>
                  msg.id === messageId
                    ? {
                        ...msg,
                        content: res.answer,
                        sources: res.sources,
                      }
                    : msg,
                ),
              }
            : chat,
        ),
      }
    }
    case 'SEND_SUCCESS': {
      const { chatId, userMessage, assistantMessage } = action.payload
      const nextDrafts = { ...state.drafts, [chatId]: '' }
      return {
        ...state,
        sendingChatId: null,
        drafts: nextDrafts,
        chats: state.chats.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                title:
                  chat.messages.length === 0 && userMessage.content
                    ? truncateTitle(userMessage.content)
                    : chat.title,
                messages: [...chat.messages, userMessage, assistantMessage],
              }
            : chat,
        ),
      }
    }
    case 'SEND_ERROR': {
      const { chatId, userMessageId, error, assistantMessageId } = action.payload
      const idToRemove = assistantMessageId ?? userMessageId
      return {
        ...state,
        sendingChatId: null,
        streamingAssistantId: null,
        error,
        chats: state.chats.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                messages: chat.messages.filter((m) => m.id !== idToRemove),
              }
            : chat,
        ),
      }
    }
    case 'SET_SOURCES_TARGET': {
      return {
        ...state,
        sourcesMessageId: action.payload.messageId,
      }
    }
    default:
      return state
  }
}

type ChatContextValue = {
  chats: ChatSession[]
  activeChatId: string | null
  activeChat: ChatSession | null
  sendingChatId: string | null
  /** Assistant message id currently receiving streamed tokens; null when idle. */
  streamingAssistantId: string | null
  draftForActiveChat: string
  error: string | null
  sourcesMessageId: string | null
  createNewChat: () => void
  setActiveChat: (chatId: string) => void
  setDraft: (chatId: string, value: string) => void
  sendMessage: (content: string) => Promise<void>
  setSourcesTarget: (messageId: string | null) => void
  deleteChat: (chatId: string) => Promise<void>
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined)

type StoredState = {
  chats: ChatSession[]
  activeChatId: string | null
}

function persistState(state: ChatState): void {
  const toStore: StoredState = {
    chats: state.chats,
    activeChatId: state.activeChatId,
  }
  saveToStorage(toStore)
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, initialState)
  /** Avoid writing empty initial state to localStorage before session list/hydration runs. */
  const [initialSyncDone, setInitialSyncDone] = useState(false)
  const chatsRef = useRef(state.chats)
  chatsRef.current = state.chats

  // Load from backend on first mount; fall back to localStorage on failure
  useEffect(() => {
    let cancelled = false
    getConversationSessions()
      .then((sessions) => {
        if (cancelled) return
        const chats: ChatSession[] = sessions.map((s) => ({
          id: s.session_id,
          title: s.title ? truncateTitle(s.title) : 'Chat',
          messages: [],
          messageCount: s.message_count ?? 0,
        }))
        const stored = loadFromStorage<StoredState>()
        const activeChatId =
          stored?.activeChatId && chats.some((c) => c.id === stored.activeChatId)
            ? stored.activeChatId
            : chats[0]?.id ?? null
        dispatch({
          type: 'INIT_FROM_BACKEND',
          payload: { chats, activeChatId },
        })
      })
      .catch(() => {
        if (cancelled) return
        const stored = loadFromStorage<StoredState>()
        if (stored) {
          dispatch({
            type: 'INIT_FROM_STORAGE',
            payload: {
              chats: stored.chats ?? [],
              activeChatId: stored.activeChatId ?? null,
            },
          })
        }
      })
      .finally(() => {
        if (!cancelled) setInitialSyncDone(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Hydrate every chat that has no messages yet (sidebar counts + open tab) in parallel.
  useEffect(() => {
    if (!initialSyncDone) return
    const need = state.chats.filter((c) => c.messages.length === 0)
    if (need.length === 0) return
    void Promise.all(
      need.map((chat) =>
        getConversationHistory(chat.id).then((entries) => {
          const latest = chatsRef.current.find((c) => c.id === chat.id)
          if (!latest || latest.messages.length > 0) return
          const messages: ChatMessage[] = entries.map((e) => ({
            id: createSessionId(),
            role: e.role as 'user' | 'assistant',
            content: e.content,
            createdAt: e.created_at ?? new Date().toISOString(),
            sources: e.sources ?? undefined,
          }))
          const firstUser = entries.find((e) => e.role === 'user')
          const title = firstUser?.content
            ? truncateTitle(firstUser.content)
            : undefined
          dispatch({
            type: 'SET_CHAT_MESSAGES',
            payload: { chatId: chat.id, messages, title },
          })
        }),
      ),
    ).catch(() => {
      // leave empty; user can retry by switching chats
    })
  }, [state.chats, initialSyncDone])

  // Persist on change (cache for offline / fallback) — never clobber storage with pre-sync empty state
  useEffect(() => {
    if (!initialSyncDone) return
    persistState(state)
  }, [state, initialSyncDone])

  const createNewChat = useCallback(() => {
    const id = createSessionId()
    const newChat: ChatSession = {
      id,
      title: 'New chat',
      messages: [],
    }
    dispatch({ type: 'CREATE_CHAT', payload: { chat: newChat } })
  }, [])

  const setActiveChat = useCallback((chatId: string) => {
    dispatch({ type: 'SET_ACTIVE_CHAT', payload: { chatId } })
  }, [])

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim()
      if (!trimmed) return

      let chatId = state.activeChatId
      if (!chatId) {
        chatId = createSessionId()
        const newChat: ChatSession = {
          id: chatId,
          title: 'New chat',
          messages: [],
        }
        dispatch({ type: 'CREATE_CHAT', payload: { chat: newChat } })
      }

      const userMessage: ChatMessage = {
        id: createSessionId(),
        role: 'user',
        content: trimmed,
        createdAt: new Date().toISOString(),
      }

      dispatch({ type: 'SEND_START', payload: { chatId } })

      const assistantMessage: ChatMessage = {
        id: createSessionId(),
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
      }
      dispatch({
        type: 'ADD_STREAMING_MESSAGES',
        payload: { chatId, userMessage, assistantMessage },
      })

      try {
        let sawDone = false
        for await (const event of queryAgentStream({
          query: trimmed,
          session_id: chatId,
        })) {
          if (event.type === 'delta') {
            dispatch({
              type: 'STREAM_DELTA',
              payload: { chatId, text: event.text },
            })
          } else if (event.type === 'done') {
            sawDone = true
            dispatch({
              type: 'SEND_SUCCESS_STREAM',
              payload: {
                chatId,
                messageId: assistantMessage.id,
                payload: event.payload,
              },
            })
          }
        }
        if (!sawDone) {
          dispatch({
            type: 'SEND_ERROR',
            payload: {
              chatId,
              userMessageId: userMessage.id,
              assistantMessageId: assistantMessage.id,
              error: 'The response ended before completion. Please try again.',
            },
          })
        }
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Something went wrong while sending your message.'
        dispatch({
          type: 'SEND_ERROR',
          payload: {
            chatId,
            userMessageId: userMessage.id,
            error: errorMessage,
            assistantMessageId: assistantMessage.id,
          },
        })
      }
    },
    [state.activeChatId],
  )

  const setSourcesTarget = useCallback((messageId: string | null) => {
    dispatch({ type: 'SET_SOURCES_TARGET', payload: { messageId } })
  }, [])

  const setDraft = useCallback((chatId: string, value: string) => {
    dispatch({ type: 'SET_DRAFT', payload: { chatId, value } })
  }, [])

  const deleteChat = useCallback(async (chatId: string) => {
    try {
      await deleteConversation(chatId)
      dispatch({ type: 'DELETE_CHAT', payload: { chatId } })
    } catch (err) {
      dispatch({
        type: 'SET_ERROR',
        payload: {
          error:
            err instanceof Error ? err.message : 'Failed to delete conversation',
        },
      })
    }
  }, [])

  const activeChat = useMemo(
    () => state.chats.find((c) => c.id === state.activeChatId) ?? null,
    [state.chats, state.activeChatId],
  )

  const draftForActiveChat = state.activeChatId ? (state.drafts[state.activeChatId] ?? '') : ''

  const value: ChatContextValue = {
    chats: state.chats,
    activeChatId: state.activeChatId,
    activeChat,
    sendingChatId: state.sendingChatId,
    streamingAssistantId: state.streamingAssistantId,
    draftForActiveChat,
    error: state.error,
    sourcesMessageId: state.sourcesMessageId,
    createNewChat,
    setActiveChat,
    setDraft,
    sendMessage,
    setSourcesTarget,
    deleteChat,
  }

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export function useChatContext(): ChatContextValue {
  const ctx = useContext(ChatContext)
  if (!ctx) {
    throw new Error('useChatContext must be used within a ChatProvider')
  }
  return ctx
}

