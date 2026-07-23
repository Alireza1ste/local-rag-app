"""Gradio UI orchestration and event handlers."""

from pathlib import Path
from typing import Optional

import gradio as gr
from langchain_qdrant import QdrantVectorStore

from .document_processor import (
    create_vectorstore,
    load_documents_from_uploads,
)
from .rag_chain import build_rag_chain, condense_question, normalize_chat_history
from .utils import build_llm, create_retriever


DEFAULT_ROLE = "You are a patent attorney."


def process_uploaded_documents(
    uploaded_files: list,
    enable_vision: bool,
) -> tuple[Optional[QdrantVectorStore], str]:
    """Process uploaded documents and create vector store."""
    if not uploaded_files:
        return None, "❌ Bitte laden Sie zuerst mindestens ein Dokument hoch."

    try:
        file_paths = [f.name for f in uploaded_files]
        # Pass the vision toggle state to the document loader
        raw_documents = load_documents_from_uploads(file_paths, enable_vision=enable_vision)

        if not raw_documents:
            return None, "❌ Es konnten keine Inhalte aus den Dateien extrahiert werden."

        # Pass whole pages directly to the vector store
        vectorstore = create_vectorstore(raw_documents)
        return (
            vectorstore,
            f"✅ Erfolgreich verarbeitet: {len(raw_documents)} vollständige Seiten bereit für Anfragen.",
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
    """Process user query and generate response using RAG chain."""
    output_chat = chat_history if chat_history else []

    if not question or not question.strip():
        return "", "", output_chat

    if not vectorstore:
        msg = "Bitte laden Sie zuerst ein Dokument (TXT oder PDF) hoch und verarbeiten Sie es."
        output_chat.append({"role": "user", "content": question})
        output_chat.append({"role": "assistant", "content": msg})
        return msg, "", output_chat

    try:
        normalized_history = normalize_chat_history(output_chat)
        condensed = condense_question(normalized_history, question)

        retriever = create_retriever(vectorstore, k, fetch_k, score_threshold, search_type)
        llm = build_llm()
        rag_chain = build_rag_chain(retriever, llm, role)

        chain_res = rag_chain.invoke({
            "search_query": condensed,
            "input_text": question,
        })

        if isinstance(chain_res, dict):
            answer = chain_res.get("answer", "")
            sources = chain_res.get("docs", retriever.invoke(condensed))
        else:
            answer = chain_res
            sources = retriever.invoke(condensed)

        # Display full untruncated sources
        source_text = "\n\n---\n\n".join(
            f"### Source {idx}\n{doc.page_content}"
            for idx, doc in enumerate(sources, 1)
        )

        output_chat.append({"role": "user", "content": question})
        output_chat.append({"role": "assistant", "content": answer})

        return answer, source_text, output_chat

    except Exception as exc:
        error_msg = f"Fehler bei der Anfrage: {exc}"
        output_chat.append({"role": "user", "content": question})
        output_chat.append({"role": "assistant", "content": error_msg})
        return error_msg, "", output_chat


def build_interface() -> gr.Blocks:
    """Build Gradio interface for RAG application."""
    with gr.Blocks(title="RAG Workshop UI") as demo:
        vs_state = gr.State(None)

        gr.Markdown("# RAG-Dokumente mit Gradio")
        gr.Markdown("Wählen Sie Dokumente (TXT, PDF) aus und klicken Sie auf **Dokumente verarbeiten**.")

        with gr.Row():
            with gr.Column():
                files = gr.File(
                    label="1. Dokumente auswählen (TXT, PDF)",
                    file_types=[".txt", ".pdf"],
                    file_count="multiple",
                )
                enable_vision = gr.Checkbox(
                    label="👁️ Enable Vision OCR/Analysis (Langsamer, extrahiert Bilder & Diagramme)",
                    value=False,
                )
                process_btn = gr.Button("⚙️ 2. Dokumente verarbeiten", variant="primary")
                status_text = gr.Markdown("⏳ Warte auf Dokumenten-Upload...")

            with gr.Column():
                question = gr.Textbox(label="Frage", lines=3)
                role = gr.Textbox(label="Rollenbeschreibung", value=DEFAULT_ROLE, lines=1)

        chat = gr.Chatbot(label="Chatverlauf")

        with gr.Row():
            k = gr.Slider(minimum=1, maximum=1000, value=20, step=1, label="k")
            fetch_k = gr.Slider(minimum=1, maximum=2000, value=50, step=1, label="fetch_k")
            score_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="score_threshold"
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

        process_btn.click(
            fn=process_uploaded_documents,
            inputs=[files, enable_vision],
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