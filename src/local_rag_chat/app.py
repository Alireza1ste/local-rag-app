"""Main entry point for local-rag-chat application."""

from dotenv import load_dotenv

from .ui import build_interface

load_dotenv()


def main() -> None:
    """Launch the RAG application with Gradio UI.
    
    Initializes the Gradio interface and starts the web server.
    Loads environment variables from .env file if present.
    """
    app = build_interface()
    app.launch(share=False)


if __name__ == "__main__":
    main()