from typing import Sequence

from bs4 import BeautifulSoup
import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import pdfkit
from pathlib import Path
from PyPDF2 import PdfMerger


def html_to_pdf(htmls: Sequence[str],
                pdf_file: Path
                ) -> None:

    merger = PdfMerger()

    for html in htmls:
        pdf_buffer = io.BytesIO(pdfkit.from_string(html, False))
        merger.append(pdf_buffer)

    with pdf_file.open("wb") as fout:
        merger.write(fout)
