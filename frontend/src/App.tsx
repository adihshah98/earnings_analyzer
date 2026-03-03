import { ChatProvider, useChatContext } from './context/ChatContext'
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

function App() {
  return (
    <ChatProvider>
      <AppContent />
    </ChatProvider>
  )
}

export default App
