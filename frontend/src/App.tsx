import { ChatProvider } from './context/ChatContext'
import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import { SourcesPanel } from './components/SourcesPanel'

function App() {
  return (
    <ChatProvider>
      <div className="app-shell">
        <Sidebar />
        <ChatView />
        <SourcesPanel />
      </div>
    </ChatProvider>
  )
}

export default App
