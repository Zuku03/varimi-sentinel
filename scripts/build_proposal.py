"""Build the Track-3 proposal PDF from Markdown (offline, deterministic).

Typography matches the POTRAZ spec: Arial/Helvetica 11pt, 1.15 line spacing,
1-inch margins, A4. Filename: [ProjectID]_AI4I_Proposal_Development.pdf.

Run: python scripts/build_proposal.py [PROJECT_ID]
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from pypdf import PdfReader
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "proposal" / "VaRimi_Sentinel_Proposal_Development.md"

CSS = """
@page { size: A4; margin: 1in; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.15;
       color: #111; }
h1 { font-size: 22pt; margin: 0 0 4pt 0; }
h2 { font-size: 13pt; margin: 12pt 0 4pt 0; border-bottom: 1px solid #999;
     padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 8pt 0 2pt 0; }
p, li { font-size: 11pt; line-height: 1.15; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; }
th, td { border: 1px solid #888; padding: 3pt 5pt; font-size: 10pt; text-align: left; }
th { background: #eee; }
pre { font-family: Courier, monospace; font-size: 8pt; line-height: 1.05;
      background: #f5f5f5; padding: 6pt; }
em { color: #444; }
"""


def build(project_id: str = "VARIMI") -> Path:
    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html_body = html_body.replace(
        "<!-- PAGEBREAK -->", '<div style="page-break-after: always;"></div>'
    )
    html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"

    out = ROOT / "proposal" / f"{project_id}_AI4I_Proposal_Development.pdf"
    with out.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh)
    if result.err:
        raise RuntimeError(f"PDF generation reported {result.err} error(s)")

    pages = len(PdfReader(str(out)).pages)
    print(f"Wrote {out.name}: {pages} page(s)")
    if pages > 10:
        print(f"WARNING: {pages} pages exceeds the 10-page limit - trim content.")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "VARIMI")