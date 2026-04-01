"""Multi-format document loader for HR-RAG pipeline.

Supports:
- Documents: PDF, EPUB, TXT, TEXT
- Markdown: MD, MDX, Markdown, MDown, MKD, MKDN
- Web Content: HTM, HTML, XHTML
"""

import re
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> str:
        """Load and extract text content from a document."""
        pass

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> list[str]:
        """Return list of supported file extensions (without dot)."""
        pass


class TextLoader(DocumentLoader):
    """Loader for plain text files."""

    def load(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return ["txt", "text"]


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files - normalizes content."""

    def load(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove YAML frontmatter if present
        content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)

        # Remove HTML comments
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        return content.strip()

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return ["md", "mdx", "markdown", "mdown", "mkd", "mkdn"]


class HTMLLoader(DocumentLoader):
    """Loader for HTML files - extracts text content."""

    def load(self, file_path: Path) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTML parsing. "
                "Install with: pip install beautifulsoup4"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        return text

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return ["htm", "html", "xhtml"]


class PDFLoader(DocumentLoader):
    """Loader for PDF files using PyMuPDF."""

    def load(self, file_path: Path) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF parsing. "
                "Install with: pip install pymupdf"
            )

        doc = fitz.open(file_path)
        texts = []

        for page in doc:
            text = page.get_text("text")
            if text.strip():
                texts.append(f"[Page {page.number + 1}]\n{text}")

        doc.close()
        return "\n\n".join(texts)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return ["pdf"]


class EPUBLoader(DocumentLoader):
    """Loader for EPUB files using EbookLib."""

    def load(self, file_path: Path) -> str:
        try:
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "ebooklib and beautifulsoup4 are required for EPUB parsing. "
                "Install with: pip install ebooklib beautifulsoup4"
            )

        book = epub.read_epub(file_path)
        texts = []

        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "lxml")
                text = soup.get_text(separator="\n", strip=True)
                if text.strip():
                    texts.append(text)

        return "\n\n".join(texts)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return ["epub"]


# Registry of all loaders
LOADERS: dict[str, DocumentLoader] = {}

for loader_class in [
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    PDFLoader,
    EPUBLoader,
]:
    loader = loader_class()
    for ext in loader_class.supported_extensions():
        LOADERS[ext.lower()] = loader


def get_loader_for_file(file_path: Path) -> Optional[DocumentLoader]:
    """Get the appropriate loader for a file based on its extension."""
    ext = file_path.suffix.lstrip(".").lower()
    return LOADERS.get(ext)


def is_supported_file(file_path: Path) -> bool:
    """Check if a file has a supported extension."""
    ext = file_path.suffix.lstrip(".").lower()
    return ext in LOADERS


def get_supported_extensions() -> list[str]:
    """Return list of all supported extensions."""
    return sorted(set(LOADERS.keys()))


def load_document(file_path: Path) -> str:
    """
    Load a document from file, auto-detecting the format.

    Args:
        file_path: Path to the document file

    Returns:
        Extracted text content

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = get_loader_for_file(file_path)
    if loader is None:
        supported = ", ".join(get_supported_extensions())
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. "
            f"Supported formats: {supported}"
        )

    return loader.load(file_path)
