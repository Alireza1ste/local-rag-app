"""Document processing and vector store management."""

from pathlib import Path
from typing import Optional, Sequence

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import EmbeddingsConfig, VectorstoreConfig
from .utils import build_embeddings


def load_documents_from_uploads(
    uploaded_files: Optional[Sequence[str]],
) -> list[Document]:
    """Load documents from uploaded file paths.
    
    Supports .txt and .pdf files. Skips unsupported formats with warning.
    
    Args:
        uploaded_files: Sequence of file paths to load.
    
    Returns:
        List of LangChain Document objects.
    """
    if not uploaded_files:
        return []

    uploaded_files = (
        [uploaded_files]
        if isinstance(uploaded_files, (str, Path))
        else uploaded_files
    )
    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        path = Path(uploaded_file)
        if not path.exists():
            continue

        ext = path.suffix.lower()
        try:
            if ext == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                documents.extend(loader.load())
            elif ext == ".pdf":
                loader = PyPDFLoader(str(path), extract_images=True)
                documents.extend(loader.load())
            else:
                print(f"⚠️ Skipping unsupported format: {ext}")
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

    return documents


def split_documents(
    documents: list[Document],
    config: VectorstoreConfig | None = None,
) -> list[Document]:
    """Split documents into chunks using recursive text splitter.
    
    Args:
        documents: List of documents to split.
        config: Vector store configuration with chunk settings.
    
    Returns:
        List of split Document objects.
    """
    if not documents:
        return []

    cfg = config or VectorstoreConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vectorstore(
    documents: list[Document],
    embeddings_config: EmbeddingsConfig | None = None,
    vectorstore_config: VectorstoreConfig | None = None,
) -> QdrantVectorStore:
    """Create a Qdrant vector store with hybrid retrieval.
    
    Combines dense embeddings with sparse BM25 embeddings for hybrid search.
    
    Args:
        documents: List of documents to index.
        embeddings_config: Configuration for embeddings model.
        vectorstore_config: Configuration for vector store.
    
    Returns:
        Initialized QdrantVectorStore instance.
    """
    emb_cfg = embeddings_config or EmbeddingsConfig()
    vs_cfg = vectorstore_config or VectorstoreConfig()

    embeddings = build_embeddings(emb_cfg.model_name)
    sparse_embeddings = FastEmbedSparse(model_name=emb_cfg.sparse_model)

    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=vs_cfg.collection_name,
        location=":memory:",
        retrieval_mode=RetrievalMode.HYBRID,
    )
