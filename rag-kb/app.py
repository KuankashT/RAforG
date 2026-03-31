"""
RAG Knowledge Base — FastAPI Backend
Полностью офлайн: PDF, Word, Excel, OCR, TF-IDF поиск, генерация тестов.

Запуск:
    pip install -r requirements.txt
    # Для OCR: sudo apt install tesseract-ocr tesseract-ocr-rus
    uvicorn app:app --reload --port 8000
"""

import os
import re
import uuid
import json
import math
import random
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import Counter

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# ─── Document extractors ───
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from openpyxl import load_workbook
from PIL import Image

# OCR (optional — graceful fallback)
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# ═══════════════════════════════════════
# App setup
# ═══════════════════════════════════════

app = FastAPI(title="RAG Knowledge Base", version="1.0")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ═══════════════════════════════════════
# In-memory knowledge base
# ═══════════════════════════════════════

documents: dict[str, dict] = {}  # doc_id -> {name, type, content, chunks, ts, path?}
chunks: list[dict] = []          # [{id, doc_id, doc_name, text}]
vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix = None

knowledge_graph: nx.Graph = nx.Graph()
SIMILARITY_THRESHOLD = 0.15


# ═══════════════════════════════════════
# Text extraction
# ═══════════════════════════════════════

def extract_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"[Стр. {i+1}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages)


def extract_docx(path: str) -> str:
    """Extract text from Word .docx."""
    doc = DocxDocument(path)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    # Also extract tables
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_xlsx(path: str) -> str:
    """Extract text from Excel .xlsx."""
    wb = load_workbook(path, read_only=True, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                rows.append(",".join(cells))
        if rows:
            parts.append(f"[Лист: {sheet_name}]\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(parts)


def extract_image(path: str) -> str:
    """OCR image using Tesseract."""
    if not HAS_TESSERACT:
        return "[OCR недоступен — установите tesseract-ocr]"
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="rus+eng")
        return text.strip() or "[Изображение без распознанного текста]"
    except Exception as e:
        return f"[Ошибка OCR: {e}]"


def extract_text_file(path: str) -> str:
    """Read plain text file."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="cp1251", errors="replace")


# ═══════════════════════════════════════
# Chunking
# ═══════════════════════════════════════

STOP_WORDS = set(
    "и в на с по для из к от за не что как это но а о у же бы да нет "
    "он она они мы вы его её их был была было были быть есть то все так "
    "уже ну вот тоже только ли или если при до после через между "
    "the a an is are was were be been have has had do does did will would "
    "could should may might can shall of in to for with on at from by as "
    "into about than its this that these those it and but or not no so".split()
)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 60) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    result = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 15:
            result.append(chunk)

    return result if result else [text]


# ═══════════════════════════════════════
# TF-IDF vectorization
# ═══════════════════════════════════════

def rebuild_vectors():
    """Rebuild TF-IDF matrix from all chunks."""
    global vectorizer, tfidf_matrix

    if not chunks:
        vectorizer = None
        tfidf_matrix = None
        rebuild_graph()
        return

    texts = [ch["text"] for ch in chunks]
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words=list(STOP_WORDS),
        token_pattern=r"(?u)\b\w{2,}\b",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    rebuild_graph()


def add_doc_to_graph(doc: dict, doc_chunks: list[str]):
    """Add document and its chunks as nodes to the knowledge graph."""
    doc_id = doc["id"]
    knowledge_graph.add_node(
        doc_id,
        node_type="document",
        name=doc["name"],
        doc_type=doc["type"],
        chunk_count=doc["chunk_count"],
        char_count=doc["char_count"],
        ts=doc["ts"],
    )
    for i, text in enumerate(doc_chunks):
        chunk_id = f"{doc_id}-{i}"
        knowledge_graph.add_node(chunk_id, node_type="chunk", doc_id=doc_id)
        knowledge_graph.add_edge(doc_id, chunk_id, edge_type="CONTAINS", weight=1.0)


def rebuild_graph():
    """Rebuild SIMILAR_TO edges between documents based on TF-IDF similarity."""
    # Remove all existing SIMILAR_TO edges
    edges_to_remove = [
        (u, v) for u, v, d in knowledge_graph.edges(data=True)
        if d.get("edge_type") == "SIMILAR_TO"
    ]
    knowledge_graph.remove_edges_from(edges_to_remove)

    if tfidf_matrix is None or len(documents) < 2:
        return

    # Build doc_id -> list of chunk indices mapping
    doc_ids = list(documents.keys())
    doc_chunk_indices: dict[str, list[int]] = {did: [] for did in doc_ids}
    for i, ch in enumerate(chunks):
        if ch["doc_id"] in doc_chunk_indices:
            doc_chunk_indices[ch["doc_id"]].append(i)

    # Compute per-document average TF-IDF vector
    doc_vectors = {}
    for did in doc_ids:
        idxs = doc_chunk_indices[did]
        if idxs:
            doc_vectors[did] = np.asarray(tfidf_matrix[idxs].mean(axis=0))
        else:
            doc_vectors[did] = None

    valid_docs = [did for did in doc_ids if doc_vectors[did] is not None]
    if len(valid_docs) < 2:
        return

    # Stack into matrix and compute pairwise similarity
    matrix = np.vstack([doc_vectors[did] for did in valid_docs])
    sim_matrix = cosine_similarity(matrix)

    for i in range(len(valid_docs)):
        for j in range(i + 1, len(valid_docs)):
            sim = float(sim_matrix[i, j])
            if sim >= SIMILARITY_THRESHOLD:
                knowledge_graph.add_edge(
                    valid_docs[i], valid_docs[j],
                    edge_type="SIMILAR_TO",
                    weight=round(sim, 4),
                )


def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Search chunks by TF-IDF cosine similarity."""
    if vectorizer is None or tfidf_matrix is None:
        return []

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        score = float(scores[idx])
        if score > 0.01:
            results.append({**chunks[idx], "score": round(score, 4)})
    return results


# ═══════════════════════════════════════
# Answer generation (offline — extractive)
# ═══════════════════════════════════════

def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sents = re.split(r'(?<=[.!?。])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 20]


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer by extracting most relevant sentences."""
    if not context_chunks:
        return "По данному вопросу ничего не найдено в базе знаний."

    query_words = set(w.lower() for w in re.findall(r'\w{3,}', query) if w.lower() not in STOP_WORDS)

    scored_sents = []
    for ch in context_chunks:
        for sent in split_sentences(ch["text"]):
            sent_words = set(w.lower() for w in re.findall(r'\w{3,}', sent))
            overlap = len(query_words & sent_words)
            scored_sents.append({
                "text": sent,
                "score": overlap / max(len(query_words), 1),
                "source": ch["doc_name"],
            })

    scored_sents.sort(key=lambda x: x["score"], reverse=True)
    best = scored_sents[:6]

    if best and best[0]["score"] > 0:
        lines = []
        for i, s in enumerate(best, 1):
            lines.append(f"{i}. {s['text']}")
            lines.append(f"   — Источник: {s['source']}")
        return "На основе найденных документов:\n\n" + "\n\n".join(lines)
    else:
        lines = []
        for i, ch in enumerate(context_chunks[:3], 1):
            snippet = ch["text"][:300] + ("..." if len(ch["text"]) > 300 else "")
            lines.append(f"[{i}] ({ch['doc_name']}):\n{snippet}")
        return "Наиболее релевантные фрагменты:\n\n" + "\n\n".join(lines)


# ═══════════════════════════════════════
# Quiz generation (offline)
# ═══════════════════════════════════════

def extract_keywords(text: str, top_n: int = 25) -> list[str]:
    """Extract top keywords by frequency."""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS and not w.isdigit()]
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


def generate_quiz(relevant_chunks: list[dict], count: int = 5) -> list[dict]:
    """Generate quiz questions from chunks (fill-in-blank + true/false)."""
    all_text = " ".join(ch["text"] for ch in relevant_chunks)
    sentences = split_sentences(all_text)
    keywords = extract_keywords(all_text, 30)

    if len(sentences) < 3:
        return []

    questions = []
    used_kw = set()

    # 1) Fill-in-the-blank
    for sent in sentences:
        if len(questions) >= count:
            break
        sent_lower = sent.lower()
        sent_words = set(re.findall(r'\b\w{3,}\b', sent_lower))

        for kw in keywords:
            if kw in used_kw:
                continue
            if kw in sent_words:
                blanked = re.sub(
                    rf'\b{re.escape(kw)}\b',
                    '______',
                    sent,
                    count=1,
                    flags=re.IGNORECASE,
                )
                distractors = [k for k in keywords if k != kw and k not in sent_words][:3]
                if len(distractors) < 3:
                    continue

                options = [kw] + distractors
                random.shuffle(options)
                correct_idx = options.index(kw)

                questions.append({
                    "question": f"Заполните пропуск:\n«{blanked}»",
                    "options": options,
                    "correct": correct_idx,
                })
                used_kw.add(kw)
                break

    # 2) True/False
    if len(questions) < count:
        for sent in sentences:
            if len(questions) >= count:
                break
            if 30 < len(sent) < 250 and sent not in used_kw:
                is_true = random.random() > 0.35
                displayed = sent

                if not is_true:
                    sent_words_list = re.findall(r'\b\w{3,}\b', sent.lower())
                    kw_in_sent = next((w for w in sent_words_list if w in keywords), None)
                    replacement = next(
                        (k for k in keywords if k != kw_in_sent and k not in sent_words_list),
                        None,
                    )
                    if kw_in_sent and replacement:
                        displayed = re.sub(
                            rf'\b{re.escape(kw_in_sent)}\b',
                            replacement,
                            sent,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                    else:
                        continue

                questions.append({
                    "question": f"Верно ли утверждение?\n«{displayed}»",
                    "options": ["Верно", "Неверно"],
                    "correct": 0 if is_true else 1,
                })
                used_kw.add(sent)

    return questions[:count]


# ═══════════════════════════════════════
# API Routes
# ═══════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats")
async def get_stats():
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "has_tesseract": HAS_TESSERACT,
    }


@app.get("/api/documents")
async def list_documents():
    return list(documents.values())


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file."""
    doc_id = str(uuid.uuid4())[:8]
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else "txt"
    save_path = UPLOAD_DIR / f"{doc_id}.{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Determine type and extract
    ftype = "text"
    content = ""

    if ext == "pdf":
        ftype = "pdf"
        content = extract_pdf(str(save_path))
    elif ext in ("docx", "doc"):
        ftype = "word"
        content = extract_docx(str(save_path))
    elif ext in ("xlsx", "xls", "xlsm"):
        ftype = "excel"
        content = extract_xlsx(str(save_path))
    elif ext in ("png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"):
        ftype = "image"
        content = extract_image(str(save_path))
    elif ext in ("csv", "tsv"):
        ftype = "csv"
        content = extract_text_file(str(save_path))
    else:
        ftype = "text"
        content = extract_text_file(str(save_path))

    if not content.strip():
        content = "[Не удалось извлечь текст]"

    # Chunk
    doc_chunks = chunk_text(content)

    # Store
    doc = {
        "id": doc_id,
        "name": file.filename,
        "type": ftype,
        "content_preview": content[:500],
        "chunk_count": len(doc_chunks),
        "char_count": len(content),
        "ts": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }
    documents[doc_id] = doc

    for i, text in enumerate(doc_chunks):
        chunks.append({
            "id": f"{doc_id}-{i}",
            "doc_id": doc_id,
            "doc_name": file.filename,
            "text": text,
        })

    add_doc_to_graph(doc, doc_chunks)
    rebuild_vectors()

    return {"status": "ok", "document": doc}


@app.post("/api/add-text")
async def add_text(text: str = Form(...), name: str = Form("")):
    """Add raw text to knowledge base."""
    doc_id = str(uuid.uuid4())[:8]
    if not name:
        name = f"Заметка #{len(documents) + 1}"

    doc_chunks = chunk_text(text)

    doc = {
        "id": doc_id,
        "name": name,
        "type": "text",
        "content_preview": text[:500],
        "chunk_count": len(doc_chunks),
        "char_count": len(text),
        "ts": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }
    documents[doc_id] = doc

    for i, chunk_text_ in enumerate(doc_chunks):
        chunks.append({
            "id": f"{doc_id}-{i}",
            "doc_id": doc_id,
            "doc_name": name,
            "text": chunk_text_,
        })

    add_doc_to_graph(doc, doc_chunks)
    rebuild_vectors()
    return {"status": "ok", "document": doc}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document and its chunks."""
    global chunks
    if doc_id in documents:
        del documents[doc_id]
        chunks = [c for c in chunks if c["doc_id"] != doc_id]
        if knowledge_graph.has_node(doc_id):
            knowledge_graph.remove_node(doc_id)
        rebuild_vectors()
        # Clean up file
        for f in UPLOAD_DIR.glob(f"{doc_id}.*"):
            f.unlink(missing_ok=True)
    return {"status": "ok"}


@app.post("/api/search")
async def api_search(query: str = Form(...), top_k: int = Form(5)):
    """Vector search across chunks."""
    results = search_chunks(query, top_k)
    return {"results": results}


@app.post("/api/ask")
async def api_ask(question: str = Form(...)):
    """Ask a question — returns extracted answer + context."""
    results = search_chunks(question, 5)
    answer = generate_answer(question, results)
    return {"answer": answer, "context": results}


@app.post("/api/quiz")
async def api_quiz(topic: str = Form(""), count: int = Form(5)):
    """Generate quiz questions."""
    if topic.strip():
        relevant = search_chunks(topic, 10)
    else:
        relevant = chunks[:10] if chunks else []

    if not relevant:
        return {"questions": [], "error": "Недостаточно материала"}

    questions = generate_quiz(relevant, count)
    return {"questions": questions}


@app.get("/api/graph")
async def api_graph(threshold: float = SIMILARITY_THRESHOLD):
    """Return graph data for D3.js visualization."""
    doc_nodes = [
        (n, d) for n, d in knowledge_graph.nodes(data=True)
        if d.get("node_type") == "document"
    ]

    centrality = nx.degree_centrality(knowledge_graph) if len(knowledge_graph) > 0 else {}

    nodes = []
    for node_id, data in doc_nodes:
        nodes.append({
            "id": node_id,
            "name": data.get("name", node_id),
            "type": data.get("doc_type", "text"),
            "chunk_count": data.get("chunk_count", 0),
            "char_count": data.get("char_count", 0),
            "ts": data.get("ts", ""),
            "degree": knowledge_graph.degree(node_id),
            "centrality": round(centrality.get(node_id, 0), 4),
        })

    links = []
    for u, v, data in knowledge_graph.edges(data=True):
        if data.get("edge_type") == "SIMILAR_TO" and data.get("weight", 0) >= threshold:
            links.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 0),
                "edge_type": "SIMILAR_TO",
            })

    doc_graph = knowledge_graph.subgraph([n for n, _ in doc_nodes])
    stats = {
        "node_count": len(nodes),
        "edge_count": len(links),
        "density": round(nx.density(doc_graph), 4) if len(nodes) > 1 else 0,
        "components": nx.number_connected_components(doc_graph) if len(nodes) > 0 else 0,
    }

    return {"nodes": nodes, "links": links, "stats": stats}


@app.get("/api/graph/neighbors/{doc_id}")
async def api_graph_neighbors(doc_id: str):
    """Return neighbors and keywords for a document node."""
    if not knowledge_graph.has_node(doc_id):
        return {"doc_id": doc_id, "neighbors": [], "top_keywords": []}

    neighbors = []
    for nbr in knowledge_graph.neighbors(doc_id):
        edge_data = knowledge_graph.edges[doc_id, nbr]
        if edge_data.get("edge_type") == "SIMILAR_TO":
            nbr_data = knowledge_graph.nodes[nbr]
            neighbors.append({
                "id": nbr,
                "name": nbr_data.get("name", nbr),
                "similarity": edge_data.get("weight", 0),
            })

    neighbors.sort(key=lambda x: x["similarity"], reverse=True)

    doc_chunks_text = " ".join(ch["text"] for ch in chunks if ch["doc_id"] == doc_id)
    top_keywords = extract_keywords(doc_chunks_text, 10) if doc_chunks_text else []

    return {
        "doc_id": doc_id,
        "neighbors": neighbors,
        "top_keywords": top_keywords,
    }


@app.get("/api/graph/stats")
async def api_graph_stats():
    """Return global graph analytics."""
    doc_nodes = [n for n, d in knowledge_graph.nodes(data=True) if d.get("node_type") == "document"]
    doc_graph = knowledge_graph.subgraph(doc_nodes)

    sim_edges = [
        (u, v, d) for u, v, d in knowledge_graph.edges(data=True)
        if d.get("edge_type") == "SIMILAR_TO"
    ]

    most_connected = None
    if doc_nodes:
        top_node = max(doc_nodes, key=lambda n: doc_graph.degree(n))
        top_data = knowledge_graph.nodes[top_node]
        most_connected = {
            "id": top_node,
            "name": top_data.get("name", top_node),
            "degree": doc_graph.degree(top_node),
        }

    avg_sim = round(sum(d.get("weight", 0) for _, _, d in sim_edges) / len(sim_edges), 4) if sim_edges else 0

    return {
        "node_count": len(doc_nodes),
        "edge_count": len(sim_edges),
        "density": round(nx.density(doc_graph), 4) if len(doc_nodes) > 1 else 0,
        "components": nx.number_connected_components(doc_graph) if doc_nodes else 0,
        "most_connected": most_connected,
        "avg_similarity": avg_sim,
        "threshold": SIMILARITY_THRESHOLD,
    }


# ═══════════════════════════════════════
# Run
# ═══════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
