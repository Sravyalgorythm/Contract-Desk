from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


@dataclass(frozen=True)
class SourceRef:
    filename: str
    locator: str


@dataclass(frozen=True)
class DocumentBlock:
    text: str
    source: SourceRef


@dataclass(frozen=True)
class TextChunk:
    text: str
    sources: tuple[SourceRef, ...]


class DocumentError(ValueError):
    """Raised when an uploaded document cannot be processed safely."""


def extract_blocks(content: bytes, filename: str) -> list[DocumentBlock]:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return _extract_pdf(content, filename)
    if extension == ".docx":
        return _extract_docx(content, filename)
    raise DocumentError("Upload a PDF or DOCX contract.")


def _extract_pdf(content: bytes, filename: str) -> list[DocumentBlock]:
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise DocumentError("This PDF is encrypted. Upload an unlocked copy.")
        blocks = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append(
                    DocumentBlock(text, SourceRef(filename, f"PDF page {page_number}"))
                )
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Could not read this PDF: {exc}") from exc

    if not blocks:
        raise DocumentError(
            "No selectable text was found in this PDF. Scanned documents need OCR "
            "before they can be analyzed."
        )
    return blocks


def _extract_docx(content: bytes, filename: str) -> list[DocumentBlock]:
    try:
        document = Document(BytesIO(content))
        blocks = []
        current_heading = "Document start"
        paragraph_number = 0
        table_number = 0

        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                paragraph_number += 1
                text = paragraph.text.strip()
                if not text:
                    continue
                if paragraph.style and paragraph.style.name.startswith("Heading"):
                    current_heading = text
                locator = f"{current_heading} / paragraph {paragraph_number}"
                blocks.append(DocumentBlock(text, SourceRef(filename, locator)))
            elif child.tag.endswith("}tbl"):
                table_number += 1
                table = Table(child, document)
                for row_number, row in enumerate(table.rows, start=1):
                    cells = [cell.text.strip() for cell in row.cells]
                    text = " | ".join(cell for cell in cells if cell)
                    if text:
                        locator = (
                            f"{current_heading} / table {table_number}, row {row_number}"
                        )
                        blocks.append(DocumentBlock(text, SourceRef(filename, locator)))
    except Exception as exc:
        raise DocumentError(f"Could not read this DOCX: {exc}") from exc

    if not blocks:
        raise DocumentError("No readable text was found in this DOCX.")
    return blocks
