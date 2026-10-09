import base64
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".mp4", ".mpga", ".mpeg"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 4
CHAT_MODEL = "gpt-4o-mini"
HISTORY_DB_PATH = "./chat_history.db"

openai_client = OpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)


def get_db():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL UNIQUE,
            content_hash TEXT UNIQUE,
            chunks INTEGER NOT NULL,
            ingested_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def find_document(source: str, content_hash: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT source, chunks, ingested_at FROM documents "
        "WHERE source = ? OR content_hash = ?",
        (source, content_hash),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def add_document(source: str, content_hash: str | None, chunks: int):
    conn = get_db()
    conn.execute(
        "INSERT INTO documents (source, content_hash, chunks, ingested_at) VALUES (?, ?, ?, ?)",
        (source, content_hash, chunks, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def list_documents() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT source, chunks, ingested_at FROM documents ORDER BY ingested_at ASC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def backfill_documents_from_vector_store():
    """Record files that were ingested into Chroma before the documents table existed."""
    metadatas = vector_store.get(include=["metadatas"])["metadatas"]
    chunk_counts: dict[str, int] = {}
    for metadata in metadatas:
        source = (metadata or {}).get("source")
        if source:
            chunk_counts[source] = chunk_counts.get(source, 0) + 1

    conn = get_db()
    known = {row["source"] for row in conn.execute("SELECT source FROM documents")}
    conn.close()
    for source, count in chunk_counts.items():
        if source not in known:
            add_document(source, None, count)


def file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def create_conversation() -> str:
    conversation_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
        (conversation_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    conn.close()
    return row is not None


def add_message(conversation_id: str, role: str, content: str, sources: list | None = None):
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            conversation_id,
            role,
            content,
            json.dumps(sources) if sources else None,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_messages(conversation_id: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, sources, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]) if row["sources"] else [],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


init_history_db()
backfill_documents_from_vector_store()


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def extract_text_from_image(file_path: str) -> str:
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    mime_type = "jpeg" if ext == "jpg" else ext

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe all visible text in this image. "
                        "If there is no text, describe the image content in detail.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime_type};base64,{encoded_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def extract_text_from_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    return transcription.text


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in PDF_EXTENSIONS:
        return extract_text_from_pdf(file_path)
    if ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(file_path)
    if ext in AUDIO_EXTENSIONS:
        return extract_text_from_audio(file_path)

    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def ingest(file_path: str, source_name: str | None = None) -> int:
    source_name = source_name or os.path.basename(file_path)

    text = extract_text(file_path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from '{source_name}'")

    chunks = chunk_text(text)
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve(query: str, k: int = TOP_K):
    return vector_store.similarity_search_with_score(query, k=k)


def build_context(results) -> str:
    blocks = []
    for doc, _score in results:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def stream_answer(query: str, context: str):
    stream = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions using only "
                "the provided context. If the answer is not contained in the context, "
                "say you don't know instead of guessing.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_query_response(query: str, conversation_id: str, k: int = TOP_K):
    yield json.dumps({"type": "conversation_id", "conversation_id": conversation_id}) + "\n"
    add_message(conversation_id, "user", query)

    results = retrieve(query, k=k)

    if not results:
        answer = "No relevant documents found."
        yield json.dumps({"type": "sources", "sources": []}) + "\n"
        yield json.dumps({"type": "token", "content": answer}) + "\n"
        add_message(conversation_id, "assistant", answer, sources=[])
        yield json.dumps({"type": "done"}) + "\n"
        return

    sources = [
        {
            "source": doc.metadata.get("source"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "score": score,
        }
        for doc, score in results
    ]
    yield json.dumps({"type": "sources", "sources": sources}) + "\n"

    context = build_context(results)
    full_answer = ""
    for delta in stream_answer(query, context):
        full_answer += delta
        yield json.dumps({"type": "token", "content": delta}) + "\n"

    add_message(conversation_id, "assistant", full_answer, sources=sources)
    yield json.dumps({"type": "done"}) + "\n"


app = Flask(__name__)
CORS(app)


@app.route("/ingest", methods=["POST"])
def ingest_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send it as multipart form field 'file'."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(uploaded_file.filename)[1].lower()
    if ext not in PDF_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        uploaded_file.save(tmp.name)
        content_hash = file_sha256(tmp.name)

        existing = find_document(uploaded_file.filename, content_hash)
        if existing:
            return jsonify(
                {
                    "error": f"'{uploaded_file.filename}' is already ingested "
                    f"(stored as '{existing['source']}').",
                    "document": existing,
                }
            ), 409

        try:
            chunk_count = ingest(tmp.name, source_name=uploaded_file.filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    add_document(uploaded_file.filename, content_hash, chunk_count)

    return jsonify(
        {
            "source": uploaded_file.filename,
            "chunks_added": chunk_count,
        }
    ), 201


@app.route("/documents", methods=["GET"])
def list_documents_endpoint():
    return jsonify({"documents": list_documents()}), 200


@app.route("/query", methods=["POST"])
def query_endpoint():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' field."}), 400

    try:
        k = int(data.get("k", TOP_K))
    except (TypeError, ValueError):
        return jsonify({"error": "'k' must be an integer."}), 400

    conversation_id = data.get("conversation_id")
    if not conversation_id or not conversation_exists(conversation_id):
        conversation_id = create_conversation()

    return Response(
        stream_with_context(stream_query_response(query, conversation_id, k=k)),
        mimetype="application/x-ndjson",
    )


@app.route("/conversations/<conversation_id>/messages", methods=["GET"])
def get_conversation_messages_endpoint(conversation_id):
    if not conversation_exists(conversation_id):
        return jsonify({"error": "Conversation not found."}), 404

    return jsonify(
        {
            "conversation_id": conversation_id,
            "messages": get_messages(conversation_id),
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
