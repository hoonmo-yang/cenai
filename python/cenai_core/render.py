from typing import Sequence

import logging
from pathlib import Path
from weasyprint import HTML


weasy_logger = logging.getLogger("weasyprint")
weasy_logger.propagate = False


def html_to_pdf(pdf_file: Path,
                htmls: Sequence[str]
                ) -> None:

    pdfs = [HTML(string=html).render() for html in htmls]

    pages = []
    for pdf in pdfs:
        pages.extend(pdf.pages)

    document = pdfs[0].copy()
    document.pages = pages
    document.write_pdf(str(pdf_file))
