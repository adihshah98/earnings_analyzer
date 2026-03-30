import { useAuth } from '../context/AuthContext'
import { useChatContext } from '../context/ChatContext'
import { displayMessageCount } from '../utils/helpers'

export function Sidebar() {
  const { chats, activeChatId, createNewChat, setActiveChat, deleteChat } =
    useChatContext()
  const { user, logout } = useAuth()

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
          chats.map((chat) => {
            const n = displayMessageCount(chat)
            return (
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
                    {n} message{n === 1 ? '' : 's'}
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
            )
          })
        )}
      </div>

      {user && (
        <div className="sidebar-user">
          {user.avatar_url ? (
            <img src={user.avatar_url} alt={user.name} className="sidebar-user-avatar" />
          ) : (
            <div className="sidebar-user-avatar-placeholder">
              {user.name.charAt(0).toUpperCase()}
            </div>
          )}
          <span className="sidebar-user-name">{user.name}</span>
          <button type="button" className="sidebar-logout-btn" onClick={logout} title="Sign out">
            Sign out
          </button>
        </div>
      )}
    </aside>
  )
}

