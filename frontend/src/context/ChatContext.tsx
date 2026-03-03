import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from 'react'
import { queryAgentStream } from '../api/client'
import type { AgentResponse, ChatMessage, ChatSession } from '../types'
import { createSessionId, loadFromStorage, saveToStorage, truncateTitle } from '../utils/helpers'

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
  | { type: 'CREATE_CHAT'; payload: { chat: ChatSession } }
  | { type: 'SET_ACTIVE_CHAT'; payload: { chatId: string } }
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
        error: null,
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
                        confidence: res.confidence,
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
  draftForActiveChat: string
  error: string | null
  sourcesMessageId: string | null
  createNewChat: () => void
  setActiveChat: (chatId: string) => void
  setDraft: (chatId: string, value: string) => void
  sendMessage: (content: string) => Promise<void>
  setSourcesTarget: (messageId: string | null) => void
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

  // Load from localStorage on first mount
  useEffect(() => {
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
  }, [])

  // Persist on change
  useEffect(() => {
    persistState(state)
  }, [state])

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
    draftForActiveChat,
    error: state.error,
    sourcesMessageId: state.sourcesMessageId,
    createNewChat,
    setActiveChat,
    setDraft,
    sendMessage,
    setSourcesTarget,
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

