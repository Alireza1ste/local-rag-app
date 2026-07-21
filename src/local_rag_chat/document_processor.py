"""Document processing and vector store management."""

import base64
import io
import os
from pathlib import Path
from typing import Optional, Sequence

import fitz  # PyMuPDF
import pymupdf4llm
import pytesseract
from PIL import Image
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
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


UNIVERSAL_VISION_PROMPT = (
    "Analyze this image concisely for a document search database:\n"
    "1. **If it is a logo, icon, webpage banner, or header text**: Reply ONLY with the single word 'SKIP'.\n"
    "2. **If it is a mathematical or scientific formula**: Transcribe it completely and accurately using LaTeX formatting.\n"
    "3. **If it shows a person/portrait**: Describe their outfit in detail (jacket/suit color, shirt type/color, tie pattern/color, dress, etc.).\n"
    "4. **If it is a patent drawing / technical schematic**: List all reference numbers/letters and the exact mechanical components they point to.\n"
    "5. **If it is a chart/graph**: Summarize the key data and axes."
)


def prepare_image_for_ollama(image_bytes: bytes, max_dim: int = 1024) -> Optional[str]:
    """Downscale, compress, and filter out only tiny artifacts, leaving formulas and portraits to the LLM."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        
        if w < 50 or h < 50:
            return None

        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠️ Could not decode image format: {e}")
        return None


def get_pdf_page_visual_analyses(pdf_path: str, vision_model: str = "gemma4:26b") -> dict[int, list[str]]:
    """Runs vision analysis page-by-page and maps descriptions to page indices."""
    page_analyses = {}
    file_name = Path(pdf_path).name
    
    try:
        doc = fitz.open(pdf_path)
        llm = ChatOllama(model=vision_model, temperature=0.0)
    except Exception as e:
        print(f"⚠️ Could not initialize image processor: {e}")
        return page_analyses

    print(f"🖼️ Running Universal Image Analysis on {file_name} with {vision_model}...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        drawings = page.get_drawings()
        text_content = page.get_text("text").strip()
        
        page_analyses[page_num] = []
        has_extracted_raster = False

        # Tier 1: Raster images (Portraits, figures)
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                image_b64 = prepare_image_for_ollama(image_bytes, max_dim=1024)
                if not image_b64:
                    continue

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": UNIVERSAL_VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                )
                
                response = llm.invoke([message])
                analysis_text = response.content.strip()

                if "SKIP" in analysis_text.upper() and len(analysis_text) < 15:
                    continue

                page_analyses[page_num].append(f"[Visual Analysis | Image {img_index + 1}]: {analysis_text}")
                has_extracted_raster = True
            except Exception as e:
                pass

        # Tier 2: Page drawings/schematics
        if not has_extracted_raster and (len(drawings) > 0 or len(text_content) < 300):
            try:
                pix = page.get_pixmap(dpi=150)
                image_bytes = pix.tobytes("png")
                
                image_b64 = prepare_image_for_ollama(image_bytes, max_dim=1024)
                if image_b64:
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": UNIVERSAL_VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    )
                    response = llm.invoke([message])
                    analysis_text = response.content.strip()

                    if not ("SKIP" in analysis_text.upper() and len(analysis_text) < 15):
                        page_analyses[page_num].append(f"[Page Schematic Analysis]: {analysis_text}")
            except Exception as e:
                pass

    return page_analyses


def load_documents_from_uploads(
    uploaded_files: Optional[Sequence[str]],
    ocr_lang_map: Optional[dict[str, list[str]]] = None,
) -> list[Document]:
    """Load documents, extracting markdown text per page and merging visual analyses directly."""
    if not uploaded_files:
        return []

    uploaded_files = (
        [uploaded_files]
        if isinstance(uploaded_files, (str, Path))
        else uploaded_files
    )
    documents: list[Document] = []
    
    for uploaded_file in uploaded_files:
        path = Path(uploaded_file)
        if not path.exists():
            continue

        ext = path.suffix.lower()

        try:
            if ext == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                documents.extend(loader.load())
            elif ext == ".pdf":
                print(f"📝 Extracting markdown and merging visual content for {path.name}...")
                
                # Get layout-aware markdown page-by-page
                page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
                # Run vision analysis mapped by page
                visual_analyses = get_pdf_page_visual_analyses(str(path))

                for chunk in page_chunks:
                    page_num = chunk["metadata"]["page_number"] - 1  # 0-based index
                    page_text = chunk["text"]

                    # Directly append visual analysis to the exact same page chunk text
                    if page_num in visual_analyses and visual_analyses[page_num]:
                        visual_block = "\n\n".join(visual_analyses[page_num])
                        page_text += f"\n\n--- VISUAL / IMAGE ANALYSIS FOR THIS PAGE ---\n{visual_block}"

                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": str(path),
                                "file_name": path.name,
                                "page": page_num,
                                "type": "page_markdown_combined"
                            }
                        )
                    )
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