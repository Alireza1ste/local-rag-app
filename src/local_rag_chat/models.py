"""Data models and configuration for local-rag-chat."""

from dataclasses import dataclass
from typing import Literal

# here the big problem is the "super generig name" 'models' turning the file into a random bag
# first step: One class -> one file
# 2nd step ... check where it is used and move some convenience functionality INTO the corresponding class
# Nothing prevents a DatatClass from having some utility
# Typical thing for eg. Configs:
# - validate self
# - to_<whatever> so that you can use it easily

@dataclass
class RAGConfig:
    """Configuration for RAG retrieval parameters."""

    k: int = 20
    fetch_k: int = 50
    score_threshold: float = 0.0
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity"


@dataclass
class ChatMessage:
    """Represents a single chat message."""

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
    """Configuration for LLM parameters."""

    model_name: str = "gemma4:26b"
    temperature: float = 0.0


@dataclass
class EmbeddingsConfig:
    """Configuration for embeddings model."""

    model_name: str = "embeddinggemma"
    sparse_model: str = "Qdrant/bm25"


@dataclass
class VectorstoreConfig:
    """Configuration for vector database."""

    collection_name: str = "gradio_workshop_docs"