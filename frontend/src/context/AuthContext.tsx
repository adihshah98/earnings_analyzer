import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react'

const TOKEN_KEY = 'ea_auth_token'
const API_BASE = import.meta.env.VITE_API_URL || ''

export interface AuthUser {
  sub: string
  email: string
  name: string
  avatar_url?: string
}

interface AuthContextValue {
  user: AuthUser | null
  token: string | null
  isLoading: boolean
  authError: string | null
  login: () => void
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

function decodeJwt(token: string): AuthUser | null {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const payload = JSON.parse(atob(base64)) as Record<string, unknown>
    return {
      sub: payload.sub as string,
      email: payload.email as string,
      name: payload.name as string,
      avatar_url: payload.avatar_url as string | undefined,
    }
  } catch {
    return null
  }
}

function isTokenExpired(token: string): boolean {
  try {
    const base64Url = token.split('.')[1]
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
    const payload = JSON.parse(atob(base64)) as { exp?: number }
    if (!payload.exp) return false
    return Date.now() / 1000 > payload.exp
  } catch {
    return true
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [authError, setAuthError] = useState<string | null>(null)

  useEffect(() => {
    // Check if Google just redirected back with a token or error in the URL
    const params = new URLSearchParams(window.location.search)
    const urlToken = params.get('token')
    const urlError = params.get('auth_error')

    if (urlToken) {
      localStorage.setItem(TOKEN_KEY, urlToken)
      window.history.replaceState({}, '', window.location.pathname)
      const decoded = decodeJwt(urlToken)
      setToken(urlToken)
      setUser(decoded)
    } else if (urlError) {
      window.history.replaceState({}, '', window.location.pathname)
      setAuthError(urlError)
    } else {
      const stored = localStorage.getItem(TOKEN_KEY)
      if (stored && !isTokenExpired(stored)) {
        const decoded = decodeJwt(stored)
        setToken(stored)
        setUser(decoded)
      } else if (stored) {
        // Expired — clear it
        localStorage.removeItem(TOKEN_KEY)
      }
    }
    setIsLoading(false)
  }, [])

  const login = useCallback(() => {
    window.location.href = `${API_BASE}/auth/google`
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem('earnings-analyzer-chats-v1')
    setToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, isLoading, authError, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}

/** Returns the stored JWT token for use in API calls. */
export function getStoredToken(): string | null {
  const token = localStorage.getItem(TOKEN_KEY)
  if (!token || isTokenExpired(token)) return null
  return token
}
