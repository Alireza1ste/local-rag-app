"""Data models and configuration for local-rag-chat."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RAGConfig:
    """Configuration for RAG retrieval parameters.
    
    Attributes:
        k: Number of documents to retrieve.
        fetch_k: Number of documents to fetch before ranking.
        score_threshold: Minimum similarity score threshold.
        search_type: Type of search to perform ('similarity', 'mmr', 'similarity_score_threshold').
    """

    k: int = 2
    fetch_k: int = 5
    score_threshold: float = 0.6
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity"


@dataclass
class ChatMessage:
    """Represents a single chat message.
    
    Attributes:
        role: Either 'user' or 'assistant'.
        content: The message text content.
    """

    role: Literal["user", "assistant"]
    content: str

    def to_dict(self) -> dict:
        """Convert message to dictionary format."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """Create ChatMessage from dictionary."""
        return cls(role=data.get("role", "user"), content=data.get("content", ""))


@dataclass
class LLMConfig:
    """Configuration for LLM parameters.
    
    Attributes:
        model_name: Name of the Ollama model to use.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = random).
    """

    model_name: str = "gemma4:26b"
    temperature: float = 0.0


@dataclass
class EmbeddingsConfig:
    """Configuration for embeddings model.
    
    Attributes:
        model_name: Name of the Ollama embeddings model.
        sparse_model: Name of the sparse embedding model.
    """

    model_name: str = "embeddinggemma"
    sparse_model: str = "Qdrant/bm25"


@dataclass
class VectorstoreConfig:
    """Configuration for vector database.
    
    Attributes:
        collection_name: Name of the Qdrant collection.
        chunk_size: Size of text chunks.
        chunk_overlap: Overlap between chunks.
    """

    collection_name: str = "gradio_workshop_docs"
    chunk_size: int = 1000
    chunk_overlap: int = 200
