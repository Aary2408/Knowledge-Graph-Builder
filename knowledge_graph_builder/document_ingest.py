from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def extract_document_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type {extension or '(none)'}. Upload TXT, Markdown, PDF, or DOCX.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    if extension in {".txt", ".md"}:
        try:
            return content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise ValueError("Text files must use UTF-8 encoding.") from error
    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF text extraction requires pypdf. Install the project requirements and retry.") from error
        try:
            reader = PdfReader(BytesIO(content), strict=True)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as error:
            raise ValueError(f"Could not read the uploaded PDF: {error}") from error
    try:
        with ZipFile(BytesIO(content)) as archive:
            xml_content = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml_content)
    except (BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise ValueError(f"Could not read the uploaded DOCX document: {error}") from error

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        pieces = []
        for node in paragraph.iter():
            if node.tag.endswith("}t") and node.text:
                pieces.append(node.text)
            elif node.tag.endswith("}tab"):
                pieces.append("\t")
            elif node.tag.endswith("}br"):
                pieces.append("\n")
        paragraph_text = "".join(pieces).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return "\n".join(paragraphs).strip()
