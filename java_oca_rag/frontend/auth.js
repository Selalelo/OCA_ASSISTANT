import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'

let supabase = null

async function initSupabase() {
  if (supabase) return supabase  // already initialized

  const res = await fetch('/config')
  const CONFIG = await res.json()

  supabase = createClient(CONFIG.supabase_url, CONFIG.supabase_anon_key)
  return supabase
}

export async function loginWithGoogle() {
  const client = await initSupabase()
  const { error } = await client.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: 'https://vigilant-pancake-qr746q4rv6924ww5-8000.app.github.dev/app'
    }
  })
  if (error) console.error('Login error:', error)
}

export async function logout() {
  const client = await initSupabase()
  await client.auth.signOut()
}

export async function getSession() {
  const client = await initSupabase()
  const { data: { session } } = await client.auth.getSession()
  return session
}

export async function getToken() {
  const session = await getSession()
  return session?.access_token || null
}

window.loginWithGoogle = loginWithGoogle
window.logout = logout