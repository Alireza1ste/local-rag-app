"""RAG chain construction and query processing."""

from operator import itemgetter
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .models import ChatMessage
from .utils import (
    build_llm,
    build_system_prompt,
    extract_question_prefix,
    format_documents,
    replace_pronouns_with_context,
)
from langchain_core.output_parsers import StrOutputParser


def build_rag_chain(
    retriever,
    llm: BaseLanguageModel,
    role: str = "You are a helpful assistant",
) -> Runnable:
    """Build a RAG chain for question answering.
    
    Combines document retrieval with LLM generation using semantic search.
    
    Args:
        retriever: LangChain retriever for document lookup.
        llm: Language model for generation.
        role: System role prompt.
    
    Returns:
        Runnable RAG chain (input: dict with search_query, input_text).
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_system_prompt(role)),
            ("human", "{input}"),
        ]
    )

    # Decouple semantic search query from LLM text prompt via itemgetter
    return (
        {
            "context": itemgetter("search_query") | retriever | format_documents,
            "input": itemgetter("input_text"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def normalize_chat_history(chat_history: list) -> list[ChatMessage]:
    """Normalize chat history into consistent ChatMessage format.
    
    Handles various input formats: ChatMessage objects, dicts, tuples/lists.
    Filters for valid user/assistant roles and non-empty content.
    
    Args:
        chat_history: Chat history in any supported format.
    
    Returns:
        List of ChatMessage objects.
    """
    normalized: list[ChatMessage] = []

    for entry in chat_history or []:
        try:
            if hasattr(entry, "role") and hasattr(entry, "content"):
                # Handle ChatMessage-like objects
                msg = ChatMessage(
                    role=str(entry.role).lower().strip(),
                    content=str(entry.content).strip(),
                )
                if msg.role in {"user", "assistant"} and msg.content:
                    normalized.append(msg)
            elif isinstance(entry, dict) and "role" in entry and "content" in entry:
                # Handle dictionary format
                msg = ChatMessage(
                    role=str(entry["role"]).lower().strip(),
                    content=str(entry["content"]).strip(),
                )
                if msg.role in {"user", "assistant"} and msg.content:
                    normalized.append(msg)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                # Handle tuple format (user, assistant) pairs
                if entry[0]:
                    normalized.append(ChatMessage(role="user", content=str(entry[0]).strip()))
                if len(entry) >= 2 and entry[1]:
                    normalized.append(
                        ChatMessage(role="assistant", content=str(entry[1]).strip())
                    )
        except (AttributeError, ValueError, KeyError):
            continue

    return normalized


def condense_question(history: list, message: str) -> str:
    """Condense current question using chat history context.
    
    If message contains pronouns, replaces them with recent context.
    Otherwise, appends context from previous turns.
    
    Args:
        history: Chat message history.
        message: Current user message.
    
    Returns:
        Condensed question string.
    """
    if not message or not message.strip():
        return ""

    message = message.strip()
    normalized = normalize_chat_history(history)
    candidate: Optional[str] = None

    # Find previous user question
    for turn in reversed(normalized):
        if turn.role != "user":
            continue
        text = extract_question_prefix(turn.content).strip()
        if text and text.lower() != message.lower():
            candidate = text
            break

    # If no user question, find previous assistant response
    if not candidate:
        for turn in reversed(normalized):
            if turn.role == "assistant":
                candidate = turn.content.split("\n")[0][:200].strip(" .")
                break

    # Replace pronouns if needed
    if candidate and PRONOUN_PATTERN.search(message):
        condensed = replace_pronouns_with_context(message, candidate)
    else:
        condensed = f"{message} (context: {candidate})" if candidate else message

    return " ".join(condensed.split())


# Import pattern check after functions to avoid circular deps
from .utils import PRONOUN_PATTERN
