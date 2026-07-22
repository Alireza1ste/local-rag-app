"""Utility functions for LLM and embeddings initialization."""

import re
from typing import Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from .models import LLMConfig


# OPTIMIZATION: Pre-compile regex patterns at module level
QUESTION_PREFIX_RE = re.compile(
    r"^\s*(what(?:'s|\s+is|\s+are)?|who(?:\s+is)?|how(?:\s+do|\s+does|\s+is)?|why|explain|describe|define|tell me about|give me|show me|list)\b\s*",
    flags=re.I,
)
PRONOUN_PATTERN = re.compile(
    r"\b(it|they|them|that|this|those|these|its|their|he|she)\b",
    flags=re.I,
)


def build_llm(config: LLMConfig | None = None) -> ChatOllama:
    """Initialize and return a ChatOllama LLM instance.
    
    Args:
        config: LLM configuration. Uses defaults if None.
    
    Returns:
        Configured ChatOllama instance.
    """
    cfg = config or LLMConfig()
    return ChatOllama(model=cfg.model_name, temperature=cfg.temperature)


def build_embeddings(model_name: str = "embeddinggemma") -> OllamaEmbeddings:
    """Initialize and return an OllamaEmbeddings instance.
    
    Args:
        model_name: Name of the embeddings model.
    
    Returns:
        Configured OllamaEmbeddings instance.
    """
    return OllamaEmbeddings(model=model_name)


def build_system_prompt(role: str, default_role: str = "You are a patent attorney.") -> str:
    """Build a system prompt for the RAG chain.
    
    Args:
        role: User-provided role description.
        default_role: Default role if none provided.
    
    Returns:
        Formatted system prompt string.
    """
    selected_role = role.strip() or default_role
    return (
        f"<|think|> {selected_role}. Use the following context to answer the question. "
        "If the context does not contain relevant information, say: "
        "'This information is not contained in my documents.' "
        "Keep the answer short and concise.\n\n"
        "context:\n\n{context}"
    )


def extract_question_prefix(text: str) -> str:
    """Remove common question prefixes from text.
    
    Args:
        text: Input text.
    
    Returns:
        Text with question prefix removed.
    """
    return QUESTION_PREFIX_RE.sub("", text).strip(" ?.")


def replace_pronouns_with_context(text: str, context: str) -> str:
    """Replace pronouns in text with context word.
    
    Args:
        text: Text containing pronouns.
        context: Word to replace pronouns with.
    
    Returns:
        Text with pronouns replaced.
    """
    if not context:
        return text

    def _replace(match: re.Match) -> str:
        pron = match.group(0)
        if pron.isupper():
            return context.upper()
        if pron[0].isupper():
            return context.capitalize()
        return context

    return PRONOUN_PATTERN.sub(_replace, text, count=1)


def format_documents(docs: list) -> str:
    """Format document list into concatenated string.
    
    Args:
        docs: List of Document objects.
    
    Returns:
        Concatenated page content with double newline separator.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_retriever(
    vectorstore: QdrantVectorStore,
    k: int = 2,
    fetch_k: int = 5,
    score_threshold: float = 0.6,
    search_type: str = "similarity",
):
    """Create a retriever from vectorstore with specified parameters.
    
    Args:
        vectorstore: Qdrant vector store instance.
        k: Number of documents to retrieve.
        fetch_k: Number to fetch before ranking.
        score_threshold: Minimum similarity threshold.
        search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold').
    
    Returns:
        Configured retriever object.
    """
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )
    if search_type == "similarity_score_threshold":
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": k},
        )
    return vectorstore.as_retriever(search_kwargs={"k": k})
