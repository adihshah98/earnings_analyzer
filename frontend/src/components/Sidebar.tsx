import { useChatContext } from '../context/ChatContext'

export function Sidebar() {
  const { chats, activeChatId, createNewChat, setActiveChat, deleteChat } =
    useChatContext()

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div>
          <div className="sidebar-title">Earnings Analyzer</div>
        </div>
        <button type="button" className="new-chat-button" onClick={createNewChat} title="New chat">
          +
        </button>
      </div>

      <div className="chat-list">
        {chats.length === 0 ? (
          <div className="empty-state" style={{ fontSize: 13, padding: '16px 8px' }}>
            Start a new chat to analyze earnings transcripts.
          </div>
        ) : (
          chats.map((chat) => (
            <div
              key={chat.id}
              className={
                'chat-list-item-wrapper' +
                (chat.id === activeChatId ? ' active' : '')
              }
            >
              <button
                type="button"
                className="chat-list-item"
                onClick={() => setActiveChat(chat.id)}
              >
                <div className="chat-list-title">{chat.title || 'Untitled chat'}</div>
                <div className="chat-list-subtitle">
                  {chat.messages.length} message{chat.messages.length === 1 ? '' : 's'}
                </div>
              </button>
              <button
                type="button"
                className="chat-list-item-delete"
                onClick={(e) => {
                  e.stopPropagation()
                  deleteChat(chat.id)
                }}
                title="Delete chat"
                aria-label="Delete chat"
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  )
}

