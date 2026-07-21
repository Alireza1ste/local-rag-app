"""Document processing and vector store management."""

import os
from pathlib import Path
from typing import Optional, Sequence

import pytesseract
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import EmbeddingsConfig, VectorstoreConfig
from .utils import build_embeddings

# Safe Tesseract OCR path configuration for Windows
TESSERACT_WIN_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESSERACT_WIN_PATH):
    pytesseract.pytesseract.pytesseract_cmd = TESSERACT_WIN_PATH

# Safe Poppler path configuration for Windows
POPPLER_CANDIDATE_PATHS = [
    r'C:\Program Files\poppler\bin',
    r'C:\Program Files\poppler\Library\bin',
    r'C:\Program Files\Poppler\bin',
    r'C:\Program Files\Poppler\Library\bin',
]

local_appdata = os.environ.get("LOCALAPPDATA", "")
if local_appdata:
    winget_packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.exists():
        for pdftoppm_exe in winget_packages.rglob("pdftoppm.exe"):
            if "Poppler" in str(pdftoppm_exe):
                POPPLER_CANDIDATE_PATHS.append(str(pdftoppm_exe.parent))

for poppler_path in POPPLER_CANDIDATE_PATHS:
    if os.path.exists(poppler_path) and poppler_path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + poppler_path


def load_documents_from_uploads(
    uploaded_files: Optional[Sequence[str]],
    ocr_lang_map: Optional[dict[str, list[str]]] = None,
) -> list[Document]:
    """Load documents from uploaded file paths with per-file OCR language configuration.
    
    Args:
        uploaded_files: Sequence of file paths to load.
        ocr_lang_map: Mapping of {file_path: [language_codes]}.
    
    Returns:
        List of LangChain Document objects.
    """
    if not uploaded_files:
        return []

    uploaded_files = (
        [uploaded_files]
        if isinstance(uploaded_files, (str, Path))
        else uploaded_files
    )
    documents: list[Document] = []
    ocr_lang_map = ocr_lang_map or {}

    for uploaded_file in uploaded_files:
        path = Path(uploaded_file)
        if not path.exists():
            continue

        ext = path.suffix.lower()
        # Fallback search by exact path string or filename
        file_langs = ocr_lang_map.get(str(path)) or ocr_lang_map.get(path.name) or ["eng", "deu"]

        try:
            if ext == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                documents.extend(loader.load())
            elif ext == ".pdf":
                loader = UnstructuredPDFLoader(
                    str(path),
                    mode="single",
                    strategy="auto",
                    languages=file_langs,
                )
                documents.extend(loader.load())
            else:
                print(f"⚠️ Skipping unsupported format: {ext}")
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

    return documents


def split_documents(
    documents: list[Document],
    config: VectorstoreConfig | None = None,
) -> list[Document]:
    """Split documents into chunks using recursive text splitter."""
    if not documents:
        return []

    cfg = config or VectorstoreConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vectorstore(
    documents: list[Document],
    embeddings_config: EmbeddingsConfig | None = None,
    vectorstore_config: VectorstoreConfig | None = None,
) -> QdrantVectorStore:
    """Create a Qdrant vector store with hybrid retrieval."""
    emb_cfg = embeddings_config or EmbeddingsConfig()
    vs_cfg = vectorstore_config or VectorstoreConfig()

    embeddings = build_embeddings(emb_cfg.model_name)
    sparse_embeddings = FastEmbedSparse(model_name=emb_cfg.sparse_model)

    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=vs_cfg.collection_name,
        location=":memory:",
        retrieval_mode=RetrievalMode.HYBRID,
    )