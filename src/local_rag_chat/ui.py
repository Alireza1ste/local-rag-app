"""Gradio UI orchestration and event handlers."""

from typing import Optional

import gradio as gr
from langchain_qdrant import QdrantVectorStore

from .models import RAGConfig
from .document_processor import (
    create_vectorstore,
    load_documents_from_uploads,
    split_documents,
)
from .rag_chain import build_rag_chain, condense_question, normalize_chat_history
from .utils import build_llm, create_retriever


DEFAULT_ROLE = "You are a helpful assistant"


def process_uploaded_documents(uploaded_files: list) -> tuple[Optional[QdrantVectorStore], str]:
    """Process uploaded documents and create vector store.
    
    Args:
        uploaded_files: List of uploaded file paths.
    
    Returns:
        Tuple of (vectorstore or None, status_message).
    """
    if not uploaded_files:
        return None, "Keine Dokumente geladen."

    try:
        raw_documents = load_documents_from_uploads(uploaded_files)

        if not raw_documents:
            return None, "❌ Es konnten keine Inhalte aus den Dateien extrahiert werden."

        documents = split_documents(raw_documents)

        if not documents:
            return None, "❌ Es konnten keine Textblöcke erzeugt werden."

        vectorstore = create_vectorstore(documents)
        return (
            vectorstore,
            f"✅ Erfolgreich verarbeitet: {len(documents)} Textblöcke bereit für Anfragen.",
        )

    except Exception as exc:
        return None, f"❌ Fehler bei der Verarbeitung: {exc}"


def handle_query(
    question: str,
    vectorstore: Optional[QdrantVectorStore],
    role: str,
    k: int,
    fetch_k: int,
    score_threshold: float,
    search_type: str,
    chat_history: list,
) -> tuple[str, str, list]:
    """Process user query and generate response using RAG chain.
    
    Args:
        question: User question.
        vectorstore: Current document vector store.
        role: System role prompt.
        k: Number of documents to retrieve.
        fetch_k: Number to fetch before ranking.
        score_threshold: Similarity threshold.
        search_type: Type of retrieval search.
        chat_history: Previous chat messages.
    
    Returns:
        Tuple of (answer, sources, updated_chat_history).
    """
    output_chat = chat_history if chat_history else []

    if not question or not question.strip():
        return "", "", output_chat

    # No vectorstore loaded
    if not vectorstore:
        msg = "Bitte laden Sie zuerst ein Dokument (TXT oder PDF) hoch."
        output_chat.append({"role": "user", "content": question})
        output_chat.append({"role": "assistant", "content": msg})
        return msg, "", output_chat

    try:
        # Normalize history for RAG chain
        normalized_history = normalize_chat_history(output_chat)
        condensed = condense_question(normalized_history, question)

        # Build chain and invoke
        retriever = create_retriever(vectorstore, k, fetch_k, score_threshold, search_type)
        llm = build_llm()
        rag_chain = build_rag_chain(retriever, llm, role)

        answer = rag_chain.invoke({
            "search_query": condensed,
            "input_text": question,
        })

        # Get source documents
        sources = retriever.invoke(condensed)
        source_text = "\n\n".join(
            f"{idx}. {doc.page_content[:400].replace(chr(10), ' ')}..."
            for idx, doc in enumerate(sources, 1)
        )

        # Update chat history
        output_chat.append({"role": "user", "content": question})
        output_chat.append({"role": "assistant", "content": answer})

        return answer, source_text, output_chat

    except Exception as exc:
        error_msg = f"Fehler bei der Anfrage: {exc}"
        output_chat.append({"role": "user", "content": question})
        output_chat.append({"role": "assistant", "content": error_msg})
        return error_msg, "", output_chat


def build_interface() -> gr.Blocks:
    """Build Gradio interface for RAG application.
    
    Returns:
        Configured Gradio Blocks interface.
    """
    with gr.Blocks(title="RAG Workshop UI") as demo:
        vs_state = gr.State(None)

        gr.Markdown("# RAG-Dokumente mit Gradio")
        gr.Markdown(
            "Laden Sie Dokumente (TXT, PDF) hoch, wählen Sie Retrieval-Parameter "
            "und stellen Sie eine Frage. (Läuft zu 100% lokal via Ollama inkl. OCR für PDFs)"
        )

        with gr.Row():
            with gr.Column():
                files = gr.File(
                    label="Dokumente hochladen (TXT, PDF)",
                    file_types=[".txt", ".pdf"],
                    file_count="multiple",
                )
                status_text = gr.Markdown("⏳ Warte auf Dokumente...")

            with gr.Column():
                question = gr.Textbox(label="Frage", lines=3)
                role = gr.Textbox(label="Rollenbeschreibung", value=DEFAULT_ROLE, lines=1)

        chat = gr.Chatbot(label="Chatverlauf")

        with gr.Row():
            k = gr.Slider(minimum=1, maximum=1000, value=2, step=1, label="k")
            fetch_k = gr.Slider(minimum=1, maximum=2000, value=5, step=1, label="fetch_k")
            score_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="score_threshold"
            )
            search_type = gr.Dropdown(
                ["similarity", "mmr", "similarity_score_threshold"],
                value="similarity",
                label="Suchtyp",
            )

        submit = gr.Button("Frage beantworten")

        with gr.Row():
            answer_output = gr.Textbox(label="Antwort", lines=10)
            source_output = gr.Textbox(label="Quellen", lines=10)

        files.upload(
            fn=process_uploaded_documents,
            inputs=[files],
            outputs=[vs_state, status_text],
        )

        files.clear(
            fn=lambda: (None, "🗑️ Dokumente entfernt. Bitte neue hochladen."),
            outputs=[vs_state, status_text],
        )

        submit.click(
            fn=handle_query,
            inputs=[question, vs_state, role, k, fetch_k, score_threshold, search_type, chat],
            outputs=[answer_output, source_output, chat],
        )

    return demo
