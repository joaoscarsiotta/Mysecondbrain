"""Extração de texto de múltiplos formatos de arquivo."""

import os
from pathlib import Path


def extract_text(file_path: str) -> dict:
    """Extrai texto de PDF, DOCX, TXT, MD ou EPUB.

    Retorna dict com 'text' e 'metadata'.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    extractors = {
        ".pdf": _extract_pdf,
        ".docx": _extract_docx,
        ".txt": _extract_plain,
        ".md": _extract_plain,
        ".epub": _extract_epub,
    }

    extractor = extractors.get(ext)
    if not extractor:
        raise ValueError(f"Formato não suportado: {ext}")

    text = extractor(file_path)

    return {
        "text": text,
        "metadata": {
            "filename": path.name,
            "file_type": ext,
            "num_chars": len(text),
        },
    }


def _extract_pdf(file_path: str) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    from docx import Document

    doc = Document(file_path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_plain(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_epub(file_path: str) -> str:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(file_path)
    texts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            texts.append(text.strip())
    return "\n\n".join(texts)


SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".epub"]
