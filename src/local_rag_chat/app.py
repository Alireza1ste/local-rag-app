"""Main entry point for local-rag-chat application."""

from dotenv import load_dotenv

from .ui import build_interface


def main() -> None:
    """Launch the RAG application with Gradio UI."""
    # The 'load_dotenv' is a bit "alone"
    # basically you want to wrap the "config stuff" a bit
    # Mainly to check "is the config all right"
    # and probably fail to start - with a UI Popup "ups this and that is wrong, do this ..."
    # 
    load_dotenv()
    #

    # Add one more step here - check if Ollama is "up and running"

    app = build_interface()
    app.launch(share=False)


if __name__ == "__main__":
    main()