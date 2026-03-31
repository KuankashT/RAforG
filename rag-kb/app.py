import uuid
import base64
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from openpyxl import load_workbook

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ═══════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════

app = FastAPI(title="Multimodal VRAG Knowledge Base")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRAMES_DIR = BASE_DIR / "frames"
DB_DIR = BASE_DIR / "chroma_db"

for d in [UPLOAD_DIR, FRAMES_DIR, DB_DIR]:
    d.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

os.environ["ANONYMIZED_TELEMETRY"] = "False"
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
collection = chroma_client.get_or_create_collection(
    name="multimodal_kb",
    metadata={"hnsw:space": "cosine"}  # явно cosine → distance в [0, 2], score = 1 - dist/2
)

print("--- Загрузка модели CLIP... ---")
text_embedder = SentenceTransformer('intfloat/multilingual-e5-large')
img_embedder = SentenceTransformer('clip-ViT-B-32')

# ── Настройки Ollama ──────────────────────────────────────────────
# Для текстовых ответов и перевода (может быть удалённый мощный сервер)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "deepseek-llm:7b")

# Для визуальных вопросов (vision-capable модель, должна поддерживать изображения)
OLLAMA_VISION_HOST = os.environ.get("OLLAMA_VISION_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava")

ollama_client = ollama.Client(host=OLLAMA_HOST)
ollama_vision_client = ollama.Client(host=OLLAMA_VISION_HOST)

print(f"--- Текст: {OLLAMA_HOST} | {OLLAMA_TEXT_MODEL} ---")
print(f"--- Vision: {OLLAMA_VISION_HOST} | {OLLAMA_VISION_MODEL} ---")
print("--- Система готова! ---")

# ═══════════════════════════════════════
# Вспомогательные функции
# ═══════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 400, overlap_paras: int = 1) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks, current, current_words = [], [], 0
    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paras:]
            current_words = sum(len(p.split()) for p in current)
        current.append(para)
        current_words += para_words
    if current:
        chunks.append("\n\n".join(current))
    return chunks

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ═══════════════════════════════════════
# API: Загрузка и Обработка
# ═══════════════════════════════════════

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4()).replace("-", "")
    ext = file.filename.split(".")[-1].lower()
    save_path = UPLOAD_DIR / f"{doc_id}.{ext}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    print(f">> Обработка: {file.filename}")

    # 1. ИЗОБРАЖЕНИЯ (JPG, PNG)
    if ext in ("png", "jpg", "jpeg"):
        emb = img_embedder.encode(Image.open(save_path)).tolist()
        collection.add(
            ids=[f"{doc_id}-img"],
            embeddings=[emb],
            metadatas=[{
                "type": "image",
                "content": "",
                "frame_path": str(save_path),
                "doc_name": file.filename,
                "doc_id": doc_id
            }]
        )

    # 2. ВИДЕО (Нарезка кадров)
    elif ext in ("mp4", "avi", "mov"):
        cap = cv2.VideoCapture(str(save_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = int(fps * 5)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if count % interval == 0:
                p = str(FRAMES_DIR / f"{doc_id}_v_{count}.jpg")
                cv2.imwrite(p, frame)
                emb = img_embedder.encode(Image.open(p)).tolist()
                collection.add(
                    ids=[f"{doc_id}-v-{count}"],
                    embeddings=[emb],
                    metadatas=[{
                        "type": "frame",
                        "content": "",
                        "frame_path": p,
                        "doc_name": file.filename,
                        "doc_id": doc_id
                    }]
                )
            count += 1
        cap.release()

    # 3. PDF — текст постранично, картинка страницы как ссылка в метадате
    elif ext == "pdf":
        doc = fitz.open(save_path)

        # Сначала рендерим все страницы в файлы (без векторизации)
        page_frame_paths = {}
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
            p = str(FRAMES_DIR / f"{doc_id}_p_{i}.jpg")
            pix.save(p)
            page_frame_paths[i] = p

        # Текстовые чанки с привязкой к картинке своей страницы
        chunk_global = 0
        for page_idx in range(len(doc)):
            page_text = doc[page_idx].get_text().strip()
            if not page_text:
                continue
            frame_path = page_frame_paths.get(page_idx, "")
            for txt in chunk_text(page_text):
                emb = text_embedder.encode(f"passage: {txt}").tolist()
                collection.add(
                    ids=[f"{doc_id}-txt-{chunk_global}"],
                    embeddings=[emb],
                    metadatas=[{
                        "type": "text",
                        "content": txt,
                        "frame_path": frame_path,  # путь к картинке этой страницы
                        "doc_name": file.filename,
                        "doc_id": doc_id,
                        "page_num": page_idx
                    }]
                )
                chunk_global += 1

        doc.close()

    # 4. ДРУГИЕ ДОКУМЕНТЫ
    else:
        content = ""
        if ext in ("docx", "doc"):
            content = "\n".join(p.text for p in DocxDocument(save_path).paragraphs)
        elif ext in ("xlsx", "xls"):
            wb = load_workbook(save_path, read_only=True, data_only=True)
            content = "\n".join([", ".join([str(c) for c in r]) for s in wb.sheetnames for r in wb[s].iter_rows(values_only=True)])
        else:
            content = Path(save_path).read_text(errors="replace")

        for i, txt in enumerate(chunk_text(content)):
            emb = text_embedder.encode(f"passage: {txt}").tolist()
            collection.add(
                ids=[f"{doc_id}-t-{i}"],
                embeddings=[emb],
                metadatas=[{
                    "type": "text",
                    "content": txt,
                    "frame_path": "",
                    "doc_name": file.filename,
                    "doc_id": doc_id
                }]
            )

    return {"status": "ok", "filename": file.filename}

# ═══════════════════════════════════════
# API: Поиск и Ответ
# ═══════════════════════════════════════

@app.post("/api/ask")
async def api_ask(question: str = Form(...)):
    # Переводим вопрос на английский для лучшего поиска по англоязычным документам
    try:
        tr = ollama_client.chat(model=OLLAMA_TEXT_MODEL, messages=[{
            'role': 'user',
            'content': f'Translate to English, output only the translation, no explanations: {question}'
        }])
        question_en = tr['message']['content'].strip()
        print(f"  Перевод: {question_en}")
    except:
        question_en = question

    q_emb = text_embedder.encode(f"query: {question_en}").tolist()

    # Ищем топ-5 текстовых чанков
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        where={"type": "text"}
    )

    texts, sources = [], []
    seen_frames = set()
    frame_urls, imgs_for_ollama = [], []

    if res['metadatas'][0]:
        for m, dist in zip(res['metadatas'][0], res['distances'][0]):
            score = 1.0 - (dist / 2.0)  # cosine distance [0,2] → score [0,1]
            print(f"  chunk score={score:.3f} dist={dist:.3f} doc={m.get('doc_name')} page={m.get('page_num')}")
            if score < 0.4:
                continue

            texts.append(m.get("content", ""))
            sources.append({
                "doc_name": m.get("doc_name"),
                "score": round(score, 3),
                "text": m.get("content", "")[:200]
            })

            # Берём картинку страницы прямо из метадаты — без доп. запроса к БД
            frame_path = m.get("frame_path", "")
            if frame_path and frame_path not in seen_frames and Path(frame_path).exists():
                seen_frames.add(frame_path)
                frame_urls.append(f"/frames/{Path(frame_path).name}")
                if len(imgs_for_ollama) < 1:  # только топ-1 картинка в LLaVA
                    imgs_for_ollama.append(image_to_base64(frame_path))

    # Формируем промпт
    context_block = "\n\n---\n".join(texts) if texts else "(контекст не найден)"
    has_images = bool(imgs_for_ollama)

    visual_keywords = ("схем", "график", "диаграмм", "рисун", "покажи", "где находи", "изображен", "фото", "чертёж", "чертеж")
    is_visual_question = any(kw in question.lower() for kw in visual_keywords)

    if is_visual_question and has_images:
        prompt = f"""Ты — ассистент по анализу технических документов.

ВОПРОС: {question}

ТЕКСТОВЫЙ КОНТЕКСТ СО СТРАНИЦЫ:
{context_block}

ИНСТРУКЦИЯ: Вопрос касается визуального содержимого. На изображении показана страница документа — изучи её внимательно и опиши что видишь применительно к вопросу. Используй текстовый контекст как дополнение.
Ответь на русском языке."""
    else:
        prompt = f"""Ты — ассистент по анализу технических документов.

ВОПРОС: {question}

КОНТЕКСТ ИЗ ДОКУМЕНТА:
{context_block}

ИНСТРУКЦИЯ: Ответь на вопрос строго на основе контекста выше. Не придумывай факты. Если информации недостаточно — скажи об этом явно.
Ответь на русском языке."""

    try:
        if is_visual_question and has_images:
            # Визуальный вопрос — vision модель (llava или qwen2.5vl)
            chat_res = ollama_vision_client.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[{'role': 'user', 'content': prompt, 'images': imgs_for_ollama}]
            )
        else:
            # Текстовый вопрос — быстрая текстовая модель
            chat_res = ollama_client.chat(
                model=OLLAMA_TEXT_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
        answer = chat_res['message']['content']
    except Exception as e:
        answer = f"Ошибка Ollama: {str(e)}"

    return {
        "answer": answer,
        "images": frame_urls,
        "context": sources
    }

# ═══════════════════════════════════════
# API: Управление данными и Граф
# ═══════════════════════════════════════

@app.delete("/api/documents/all")
async def delete_all():
    chroma_client.delete_collection("multimodal_kb")
    global collection
    collection = chroma_client.get_or_create_collection(
        name="multimodal_kb",
        metadata={"hnsw:space": "cosine"}
    )
    return {"status": "ok", "message": "База очищена"}

@app.get("/api/documents")
async def get_documents():
    data = collection.get(include=["metadatas"])
    if not data or not data["metadatas"]: return []
    docs = {}
    for m in data["metadatas"]:
        n = m.get("doc_name", "Unknown")
        if n not in docs: docs[n] = {"id": n, "name": n, "type": m.get("type"), "chunk_count": 0, "ts": "На диске"}
        docs[n]["chunk_count"] += 1
    return list(docs.values())

@app.get("/api/graph")
async def api_graph(threshold: float = 0.65):
    data = collection.get(include=["embeddings", "metadatas"])
    if not data or not data["embeddings"]: return {"nodes": [], "links": []}

    doc_embs = {}
    for i, m in enumerate(data["metadatas"]):
        name = m.get("doc_name", "Unknown")
        if name not in doc_embs: doc_embs[name] = []
        doc_embs[name].append(data["embeddings"][i])

    names = list(doc_embs.keys())
    avg_embs = [np.mean(doc_embs[n], axis=0) for n in names]
    nodes = [{"id": n, "name": n, "type": "doc"} for n in names]
    links = []

    if len(names) > 1:
        sims = cosine_similarity(avg_embs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if sims[i, j] >= threshold:
                    links.append({"source": names[i], "target": names[j], "weight": float(sims[i, j])})

    return {"nodes": nodes, "links": links}

@app.get("/api/stats")
async def get_stats():
    all_meta = collection.get(include=["metadatas"])
    unique_docs = len(set(
        m.get("doc_name") for m in all_meta["metadatas"] if m.get("doc_name")
    )) if all_meta["metadatas"] else 0
    return {"documents": unique_docs, "chunks": collection.count()}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
