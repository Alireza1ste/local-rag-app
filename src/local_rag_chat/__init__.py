"""Local RAG Chat - A completely local RAG application using Gradio, LangChain, Qdrant, and Ollama."""

__version__ = "0.3.1"
__author__ = "Dr. Alireza Bayat"
__email__ = "dr.bayat.alireza@gmail.com"

from .app import main

__all__ = ["main"]