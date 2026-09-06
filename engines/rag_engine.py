class RAGEngine:

    def __init__(self):

        self.documents = []

    def add_document(
        self,
        text,
        metadata=None
    ):

        self.documents.append({

            "text":
            text,

            "metadata":
            metadata or {}
        })

    def retrieve(
        self,
        query,
        top_k=3
    ):

        # Temporary implementation.
        #
        # Phase 2 will replace this with:
        #
        # SentenceTransformer
        #       ↓
        # Embeddings
        #       ↓
        # FAISS / Qdrant
        #       ↓
        # Semantic retrieval

        return self.documents[:top_k]