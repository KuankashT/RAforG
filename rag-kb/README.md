# RAG Knowledge Base

Полностью офлайн система управления знаниями с векторным поиском и генерацией тестов.

## Возможности

- **PDF** — извлечение текста через PyMuPDF (включая многостраничные документы)
- **Word (.docx)** — извлечение текста и таблиц через python-docx
- **Excel (.xlsx)** — все листы конвертируются в текст через openpyxl
- **CSV / TSV** — прямое чтение
- **Изображения (OCR)** — распознавание текста через Tesseract (рус + англ)
- **Текстовые файлы** — .txt, .md, .json, .log и др.

### Поиск
TF-IDF векторизация через scikit-learn + cosine similarity.
Автоматический чанкинг документов (300 слов, overlap 60).

### Q&A
Извлекательный (extractive) ответ: находит top-5 релевантных чанков,
ранжирует предложения по пересечению с вопросом, выдаёт с указанием источника.

### Контрольные вопросы
Два типа вопросов, генерируются из материала:
1. **Fill-in-the-blank** — пропуск ключевого слова, 4 варианта ответа
2. **True/False** — утверждение из текста (верное или с подменой термина)

## Установка

```bash
# 1. Python 3.10+
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Зависимости
pip install -r requirements.txt

# 3. Tesseract OCR (опционально — для распознавания картинок)
# Ubuntu/Debian:
sudo apt install tesseract-ocr tesseract-ocr-rus
# macOS:
brew install tesseract tesseract-lang
# Windows: скачать установщик с https://github.com/UB-Mannheim/tesseract/wiki

# 4. Запуск
python app.py
# или
uvicorn app:app --reload --port 8000
```

Открыть в браузере: **http://localhost:8000**

## Структура проекта

```
rag-kb/
├── app.py              # FastAPI бэкенд (API + логика)
├── requirements.txt    # Зависимости
├── README.md           # Документация
├── templates/
│   └── index.html      # Фронтенд (SPA)
├── static/             # Статика (пока пусто)
└── uploads/            # Загруженные файлы
```

## API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Главная страница |
| GET | `/api/stats` | Статистика базы |
| GET | `/api/documents` | Список документов |
| POST | `/api/upload` | Загрузка файла (multipart) |
| POST | `/api/add-text` | Добавление текста |
| DELETE | `/api/documents/{id}` | Удаление документа |
| POST | `/api/search` | Векторный поиск |
| POST | `/api/ask` | Вопрос-ответ |
| POST | `/api/quiz` | Генерация теста |

## Без API-ключей

Система полностью автономна. Не требует OpenAI, Claude или других внешних API.
Все операции (OCR, извлечение текста, поиск, генерация вопросов) выполняются локально.
