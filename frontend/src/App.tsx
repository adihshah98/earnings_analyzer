import { AuthProvider, useAuth } from './context/AuthContext'
import { ChatProvider, useChatContext } from './context/ChatContext'
import { LoginPage } from './components/LoginPage'
import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import { SourcesPanel } from './components/SourcesPanel'

function AppContent() {
  const { sourcesMessageId } = useChatContext()
  return (
    <div className="app-shell">
      <Sidebar />
      <ChatView />
      {sourcesMessageId != null && <SourcesPanel />}
    </div>
  )
}

function AuthGate() {
  const { user, isLoading } = useAuth()
  if (isLoading) return null
  if (!user) return <LoginPage />
  return (
    <ChatProvider>
      <AppContent />
    </ChatProvider>
  )
}

function App() {
  return (
    <AuthProvider>
      <AuthGate />
    </AuthProvider>
  )
}

export default App
