"""RAG chain construction and query processing."""

import re
from operator import itemgetter
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from .models import ChatMessage
from .utils import (
    PRONOUN_PATTERN,
    build_system_prompt,
    extract_question_prefix,
    format_documents,
    replace_pronouns_with_context,
)


def parse_reasoning_and_content(msg: AIMessage) -> str:
    """Extracts both the reasoning blocks and the final answer from ChatOllama."""
    content = msg.content
    reasoning = msg.additional_kwargs.get("reasoning_content", "")
    
    if reasoning:
        return f"🤔 **Thinking Process:**\n{reasoning}\n\n---\n\n✅ **Answer:**\n{content}"
    return content


def build_rag_chain(
    retriever,
    llm: BaseLanguageModel,
    role: str = "You are a patent attorney.",
) -> Runnable:
    """Build a RAG chain that returns both answer and retrieved source documents in one pass."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_system_prompt(role)),
            ("human", "{input}"),
        ]
    )

    return RunnablePassthrough.assign(
        docs=itemgetter("search_query") | retriever
    ).assign(
        context=lambda x: format_documents(x["docs"]),
        answer=(
            {
                "context": lambda x: format_documents(x["docs"]),
                "input": itemgetter("input_text"),
            }
            | prompt
            | llm
            | parse_reasoning_and_content
        )
    )


def normalize_chat_history(chat_history: list) -> list[ChatMessage]:
    """Normalize chat history into consistent ChatMessage format."""
    normalized: list[ChatMessage] = []

    for entry in chat_history or []:
        try:
            if hasattr(entry, "role") and hasattr(entry, "content"):
                msg = ChatMessage(
                    role=str(entry.role).lower().strip(),
                    content=str(entry.content).strip(),
                )
                if msg.role in {"user", "assistant"} and msg.content:
                    normalized.append(msg)
            elif isinstance(entry, dict) and "role" in entry and "content" in entry:
                msg = ChatMessage(
                    role=str(entry["role"]).lower().strip(),
                    content=str(entry["content"]).strip(),
                )
                if msg.role in {"user", "assistant"} and msg.content:
                    normalized.append(msg)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
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
    """Condense current question using chat history context."""
    if not message or not message.strip():
        return ""

    message = message.strip()
    normalized = normalize_chat_history(history)
    candidate: Optional[str] = None

    for turn in reversed(normalized):
        if turn.role != "user":
            continue
        text = extract_question_prefix(turn.content).strip()
        if text and text.lower() != message.lower():
            candidate = text
            break

    if not candidate:
        for turn in reversed(normalized):
            if turn.role == "assistant":
                candidate = turn.content.split("\n")[0][:200].strip(" .")
                break

    if candidate and PRONOUN_PATTERN.search(message):
        condensed = replace_pronouns_with_context(message, candidate)
    else:
        condensed = f"{message} (context: {candidate})" if candidate else message

    return " ".join(condensed.split())