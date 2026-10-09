import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:5000'
const INGEST_URL = `${API_BASE}/ingest`
const QUERY_URL = `${API_BASE}/query`
const DOCUMENTS_URL = `${API_BASE}/documents`

function App() {
  const [file, setFile] = useState(null)
  const [ingestStatus, setIngestStatus] = useState('idle') // idle | uploading | success | error
  const [ingestMessage, setIngestMessage] = useState('')

  const [documents, setDocuments] = useState([])
  const [documentsError, setDocumentsError] = useState('')

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  // Each page load starts a new conversation, so earlier questions are not shown.
  const conversationIdRef = useRef(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isAsking])

  const loadDocuments = async () => {
    try {
      const response = await fetch(DOCUMENTS_URL)
      if (!response.ok) throw new Error('Could not load ingested files')
      const data = await response.json()
      setDocuments(data.documents || [])
      setDocumentsError('')
    } catch (err) {
      setDocumentsError(err.message)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null)
    setIngestStatus('idle')
    setIngestMessage('')
  }

  const handleIngestSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setIngestStatus('uploading')
    setIngestMessage('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(INGEST_URL, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed')
      }

      setIngestStatus('success')
      setIngestMessage(`Ingested "${data.source}" — ${data.chunks_added} chunk(s) added.`)
      setFile(null)
      e.target.reset()
      loadDocuments()
    } catch (err) {
      setIngestStatus('error')
      setIngestMessage(err.message)
    }
  }

  const handleChatSubmit = async (e) => {
    e.preventDefault()
    const query = input.trim()
    if (!query || isAsking) return

    const assistantId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user', content: query },
      { id: assistantId, role: 'assistant', content: '', sources: [], streaming: true },
    ])
    setInput('')
    setIsAsking(true)

    const updateAssistant = (updater) => {
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? updater(m) : m)))
    }

    try {
      const response = await fetch(QUERY_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          k: 4,
          conversation_id: conversationIdRef.current,
        }),
      })

      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.error || 'Query failed')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.trim()) continue
          const event = JSON.parse(line)

          if (event.type === 'conversation_id') {
            conversationIdRef.current = event.conversation_id
          } else if (event.type === 'sources') {
            const sourceNames = [...new Set(event.sources.map((s) => s.source).filter(Boolean))]
            updateAssistant((m) => ({ ...m, sources: sourceNames }))
          } else if (event.type === 'token') {
            updateAssistant((m) => ({ ...m, content: m.content + event.content }))
          } else if (event.type === 'done') {
            updateAssistant((m) => ({ ...m, streaming: false }))
          }
        }
      }
    } catch (err) {
      updateAssistant((m) => ({
        ...m,
        content: `Error: ${err.message}`,
        isError: true,
        streaming: false,
      }))
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <div className="container">
      <h1>RAG Knowledge Base</h1>

      <section className="ingest-section">
        <h2>Ingest a Document</h2>
        <p className="subtitle">Upload a PDF, image, or audio file to add it to the knowledge base.</p>

        <form onSubmit={handleIngestSubmit}>
          <input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.webp,.gif,.mp3,.wav,.m4a,.webm,.mp4,.mpga,.mpeg"
            onChange={handleFileChange}
          />
          <button type="submit" disabled={!file || ingestStatus === 'uploading'}>
            {ingestStatus === 'uploading' ? 'Uploading...' : 'Ingest File'}
          </button>
        </form>

        {ingestMessage && (
          <p className={ingestStatus === 'error' ? 'message error' : 'message success'}>
            {ingestMessage}
          </p>
        )}
      </section>

      <section className="documents-section">
        <h2>Ingested Files ({documents.length})</h2>
        {documentsError ? (
          <p className="message error">{documentsError}</p>
        ) : documents.length === 0 ? (
          <p className="subtitle">No files have been ingested yet.</p>
        ) : (
          <ul className="documents-list">
            {documents.map((doc) => (
              <li key={doc.source}>
                <span className="document-name">{doc.source}</span>
                <span className="document-chunks">
                  {doc.chunks} chunk(s) · {new Date(doc.ingested_at).toLocaleDateString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="chat-section">
        <h2>Ask a Question</h2>

        <div className="chat-window">
          {messages.length === 0 && (
            <p className="chat-placeholder">Ask something about the documents you've ingested.</p>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`chat-bubble ${msg.role} ${msg.isError ? 'error' : ''}`}>
              <p>
                {msg.streaming && !msg.content ? (
                  <span className="thinking">Thinking...</span>
                ) : (
                  msg.content
                )}
                {msg.streaming && msg.content && <span className="cursor" />}
              </p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  {msg.sources.map((name, j) => (
                    <span key={j} className="source-badge">
                      {name}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}

          <div ref={bottomRef} />
        </div>

        <form className="chat-input-row" onSubmit={handleChatSubmit}>
          <input
            type="text"
            placeholder="Type your question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isAsking}
          />
          <button type="submit" disabled={!input.trim() || isAsking}>
            Send
          </button>
        </form>
      </section>
    </div>
  )
}

export default App
