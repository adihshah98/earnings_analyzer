import { useEffect, useRef, useState } from 'react'
import { AuthProvider, useAuth } from './context/AuthContext'
import { ChatProvider, useChatContext } from './context/ChatContext'
import { LoginPage } from './components/LoginPage'
import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'
import { SourcesPanel } from './components/SourcesPanel'
import { checkHealth, warmupBackend, getLastQueryTime } from './api/client'

const IDLE_THRESHOLD = 15 * 60_000 // 15 minutes
const POLL_INTERVAL  = 10_000       // 10s while warming

function useBackendReady(): boolean {
  const [ready, setReady] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    function stopPolling() {
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
    }

    function startPolling() {
      stopPolling()
      intervalRef.current = setInterval(async () => {
        const ok = await checkHealth()
        if (ok) { setReady(true); stopPolling() }
      }, POLL_INTERVAL)
    }

    // Initial check on mount — use /warmup to also prime company + embedding caches
    warmupBackend().then(ok => {
      setReady(ok)
      if (!ok) startPolling()
    })

    // On tab focus: re-check if idle >15 min
    function handleVisibility() {
      if (document.visibilityState !== 'visible') return
      const last = getLastQueryTime()
      const idle = last === null ? Date.now() - performance.timeOrigin : Date.now() - last
      if (idle > IDLE_THRESHOLD) {
        checkHealth().then(ok => {
          if (!ok) { setReady(false); startPolling() }
        })
      }
    }

    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibility)
      stopPolling()
    }
  }, [])

  return ready
}

function WarmingBanner({ ready }: { ready: boolean }) {
  if (ready) return null
  return (
    <div className="warming-banner">
      <span className="warming-banner__dot" />
      Backend warming up. Will take roughly 2 minutes because of free tier limitations…
    </div>
  )
}

function AppContent() {
  const { sourcesMessageId } = useChatContext()
  const backendReady = useBackendReady()
  return (
    <div className="app-shell-wrapper">
      <WarmingBanner ready={backendReady} />
      <div className="app-shell">
        <Sidebar />
        <ChatView />
        {sourcesMessageId != null && <SourcesPanel />}
      </div>
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
