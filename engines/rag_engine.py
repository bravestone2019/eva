"""Local, file-backed retrieval for EVA's user-provided text documents."""

from collections import Counter
from pathlib import Path
import re
import uuid

from app.config import DOCUMENT_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUFFIXES = {".md", ".txt"}
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 160


class RAGEngine:
    """Index private Markdown/text files and retrieve relevant chunks locally."""

    def __init__(self, document_dir=None):
        configured_dir = Path(document_dir or DOCUMENT_DIR)
        self.document_dir = configured_dir if configured_dir.is_absolute() else PROJECT_ROOT / configured_dir
        self.documents = []
        self.reload_documents()

    @staticmethod
    def _split_text(text):
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        chunks = []
        start = 0
        while start < len(cleaned):
            end = min(start + CHUNK_SIZE, len(cleaned))
            if end < len(cleaned):
                boundary = cleaned.rfind(" ", start + CHUNK_SIZE // 2, end)
                if boundary > start:
                    end = boundary
            chunks.append(cleaned[start:end].strip())
            if end >= len(cleaned):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
        return chunks

    def add_document(self, text, metadata=None):
        metadata = dict(metadata or {})
        chunks = self._split_text(text)
        for index, chunk in enumerate(chunks):
            self.documents.append({
                "text": chunk,
                "metadata": {**metadata, "chunk": index + 1, "chunks": len(chunks)},
            })
        return len(chunks)

    def reload_documents(self):
        self.documents = []
        self.document_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.document_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                try:
                    text = path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError):
                    continue
                doc_id, separator, display_name = path.name.partition("__")
                self.add_document(text, {
                    "id": doc_id if separator else path.stem,
                    "source": display_name if separator else path.name,
                })

    def save_document(self, filename, text):
        original_name = Path(filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("Only .txt and .md documents are supported.")
        safe_stem = re.sub(r"[^\w.-]+", "_", Path(original_name).stem, flags=re.UNICODE).strip("._")
        if not safe_stem:
            safe_stem = "document"
        safe_name = f"{uuid.uuid4().hex}__{safe_stem[:80]}{suffix}"
        path = self.document_dir / safe_name
        path.write_text(text, encoding="utf-8")
        self.reload_documents()
        return {
            "id": path.name.split("__", 1)[0],
            "name": original_name,
            "chunks": len(self._split_text(text)),
        }

    def list_documents(self):
        if not self.document_dir.exists():
            return []
        entries = []
        for path in sorted(self.document_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            doc_id, separator, display_name = path.name.partition("__")
            try:
                text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError):
                continue
            entries.append({
                "id": doc_id if separator else path.stem,
                "name": display_name if separator else path.name,
                "chunks": len(self._split_text(text)),
            })
        return entries

    def delete_document(self, document_id):
        if not re.fullmatch(r"[0-9a-f]{32}", document_id):
            return False
        matches = list(self.document_dir.glob(f"{document_id}__*"))
        if not matches:
            return False
        for path in matches:
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                path.unlink()
        self.reload_documents()
        return True

    @staticmethod
    def _fallback_score(query_terms, text):
        words = re.findall(r"[\w']+", text.lower())
        counts = Counter(words)
        return sum(counts.get(term, 0) for term in query_terms)

    def retrieve(self, query, top_k=3):
        if not self.documents or not query.strip():
            return []
        texts = [document["text"] for document in self.documents]
        scores = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(texts + [query])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        except (ImportError, ValueError):
            terms = re.findall(r"[\w']+", query.lower())
            scores = [self._fallback_score(terms, text) for text in texts]

        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        results = []
        for index, score in ranked:
            if float(score) < 0.025:
                continue
            results.append({
                **self.documents[index],
                "score": round(float(score), 4),
            })
            if len(results) >= top_k:
                break
        return results
