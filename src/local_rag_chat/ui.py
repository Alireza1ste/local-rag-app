"""Gradio UI orchestration and event handlers."""

from pathlib import Path
from typing import Optional

import gradio as gr
from langchain_qdrant import QdrantVectorStore

from .document_processor import (
    create_vectorstore,
    load_documents_from_uploads,
    split_documents,
)
from .rag_chain import build_rag_chain, condense_question, normalize_chat_history
from .utils import build_llm, create_retriever


DEFAULT_ROLE = "You are a helpful assistant"

ALL_LANGUAGES = [
    "afr", "amh", "ara", "asm", "aze", "aze_cyrl", "bel", "ben", "bod", "bos", 
    "bre", "bul", "cat", "ceb", "ces", "chi_sim", "chi_sim_vert", "chi_tra", 
    "chi_tra_vert", "chr", "cos", "cym", "dan", "deu", "deu_latf", "div", "dzo", 
    "ell", "eng", "enm", "epo", "equ", "est", "eus", "fao", "fas", "fil", "fin", 
    "fra", "frk", "frm", "fry", "gla", "gle", "glg", "grc", "guj", "hat", "heb", 
    "hin", "hrv", "hun", "hye", "iku", "ind", "isl", "ita", "ita_old", "jav", 
    "jpn", "jpn_vert", "kan", "kat", "kat_old", "kaz", "khm", "kir", "kmr", 
    "kor", "kor_vert", "lao", "lat", "lav", "lit", "ltz", "mal", "mar", "mkd", 
    "mlt", "mon", "mri", "msa", "mya", "nep", "nld", "nor", "oci", "ori", "osd", 
    "pan", "pol", "por", "pus", "que", "ron", "rus", "san", "sin", "slk", "slv", 
    "snd", "spa", "spa_old", "sqi", "srp", "srp_latn", "sun", "swa", "swe", "syr", 
    "tam", "tat", "tel", "tgk", "tha", "tir", "ton", "tur", "uig", "ukr", "urd", 
    "uzb", "uzb_cyrl", "vie", "yid", "yor"
]


def process_uploaded_documents(
    uploaded_files: list, 
    file_lang_map: dict
) -> tuple[Optional[QdrantVectorStore], str]:
    """Process uploaded documents and create vector store using assigned languages."""
    if not uploaded_files:
        return None, "❌ Bitte laden Sie zuerst mindestens ein Dokument hoch."

    try:
        file_paths = [f.name for f in uploaded_files]
        raw_documents = load_documents_from_uploads(file_paths, file_lang_map)

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

        answer = rag_chain.invoke({
            "search_query": condensed,
            "input_text": question,
        })

        sources = retriever.invoke(condensed)
        source_text = "\n\n".join(
            f"{idx}. {doc.page_content[:400].replace(chr(10), ' ')}..."
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
        file_lang_map = gr.State({})

        gr.Markdown("# RAG-Dokumente mit Gradio")
        gr.Markdown(
            "Wählen Sie Dokumente (TXT, PDF) aus, stellen Sie die OCR-Sprache für jedes Dokument ein, "
            "und klicken Sie auf **Dokumente verarbeiten**."
        )

        with gr.Row():
            with gr.Column():
                files = gr.File(
                    label="1. Dokumente auswählen (TXT, PDF)",
                    file_types=[".txt", ".pdf"],
                    file_count="multiple",
                )
                
                # Render ONLY when 'files' changes (prevents re-render loops)
                @gr.render(inputs=[files])
                def render_per_file_languages(uploaded_files):
                    if not uploaded_files:
                        return
                    
                    gr.Markdown("### 🌐 2. OCR-Sprache(n) pro Dokument auswählen")
                    for file_obj in uploaded_files:
                        file_path = file_obj.name
                        file_name = Path(file_path).name

                        # Explicitly set interactive=True so it can be clicked & edited
                        dd = gr.Dropdown(
                            choices=ALL_LANGUAGES,
                            value=["eng", "deu"],
                            multiselect=True,
                            interactive=True,
                            label=f"Sprache(n) für: {file_name}",
                            info="Klicken Sie ins Feld oder tippen Sie, um weitere Sprachen zu suchen."
                        )

                        # Update state without triggering render_per_file_languages again
                        def update_language_for_file(selected_langs, current_map, path=file_path):
                            updated_map = dict(current_map or {})
                            updated_map[path] = selected_langs
                            return updated_map

                        dd.change(
                            fn=update_language_for_file,
                            inputs=[dd, file_lang_map],
                            outputs=[file_lang_map],
                        )

                process_btn = gr.Button("⚙️ 3. Dokumente verarbeiten", variant="primary")
                status_text = gr.Markdown("⏳ Warte auf Dokumenten-Upload...")

            with gr.Column():
                question = gr.Textbox(label="Frage", lines=3)
                role = gr.Textbox(label="Rollenbeschreibung", value=DEFAULT_ROLE, lines=1)

        chat = gr.Chatbot(label="Chatverlauf")

        with gr.Row():
            k = gr.Slider(minimum=1, maximum=1000, value=20, step=1, label="k")
            fetch_k = gr.Slider(minimum=1, maximum=2000, value=50, step=1, label="fetch_k")
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

        # Pre-populate map defaults when files uploaded
        def initialize_file_map(uploaded_files):
            if not uploaded_files:
                return {}
            return {f.name: ["eng", "deu"] for f in uploaded_files}

        files.upload(
            fn=initialize_file_map,
            inputs=[files],
            outputs=[file_lang_map],
        )

        process_btn.click(
            fn=process_uploaded_documents,
            inputs=[files, file_lang_map],
            outputs=[vs_state, status_text],
        )

        files.clear(
            fn=lambda: (None, {}, "🗑️ Dokumente entfernt. Bitte neue hochladen."),
            outputs=[vs_state, file_lang_map, status_text],
        )

        submit.click(
            fn=handle_query,
            inputs=[question, vs_state, role, k, fetch_k, score_threshold, search_type, chat],
            outputs=[answer_output, source_output, chat],
        )

    return demo