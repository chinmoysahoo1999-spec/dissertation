"""
Build a thesis-style PDF of survey_chapter.md using reportlab Platypus.

Usage:
    python _build_pdf.py
Reads:  survey_chapter.md      (sibling)
Writes: Literature_Survey_Chapter.pdf
"""
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
)

HERE = Path(__file__).resolve().parent
SRC = HERE / "survey_chapter.md"
OUT = HERE / "Literature_Survey_Chapter.pdf"

# -----------------------------------------------------------------------------
# Styles
# -----------------------------------------------------------------------------
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleStyle", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=18, leading=22,
    spaceAfter=14, alignment=TA_LEFT, textColor=colors.HexColor("#16263a"),
)
subtitle_style = ParagraphStyle(
    "SubtitleStyle", parent=styles["Italic"],
    fontName="Helvetica-Oblique", fontSize=10.5, leading=14,
    spaceAfter=20, alignment=TA_LEFT, textColor=colors.HexColor("#4d5870"),
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=14, leading=18,
    spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#16263a"),
    keepWithNext=True,
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=12, leading=15,
    spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#22324b"),
    keepWithNext=True,
)
h3_style = ParagraphStyle(
    "H3", parent=styles["Heading3"],
    fontName="Helvetica-Bold", fontSize=10.5, leading=13,
    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2a3a55"),
    keepWithNext=True,
)
body_style = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontName="Helvetica", fontSize=10, leading=14,
    spaceAfter=8, alignment=TA_JUSTIFY, firstLineIndent=0,
)
quote_style = ParagraphStyle(
    "Quote", parent=body_style,
    leftIndent=18, rightIndent=18, fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#3a4960"), spaceBefore=4, spaceAfter=10,
    borderColor=colors.HexColor("#9eb1cc"), borderPadding=6, borderWidth=0,
)
ref_style = ParagraphStyle(
    "Ref", parent=body_style,
    leftIndent=14, firstLineIndent=-14,
    fontSize=9, leading=12, spaceAfter=4,
)
hr_style = ParagraphStyle(
    "HR", parent=body_style, alignment=TA_CENTER,
    textColor=colors.HexColor("#9eb1cc"), spaceBefore=4, spaceAfter=12,
)

# -----------------------------------------------------------------------------
# Inline-formatting (markdown → reportlab XML)
# -----------------------------------------------------------------------------
def inline(md: str) -> str:
    # Order: escape <, then markdown markup
    s = md.replace("&", "&amp;")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    # bold
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    # italic (single asterisk or single underscore not adjacent to word char)
    s = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"\*([^*]+?)\*", r"<i>\1</i>", s)
    # code → courier
    s = re.sub(r"`([^`]+?)`", r"<font face='Courier' color='#1a1a1a'>\1</font>", s)
    return s


# -----------------------------------------------------------------------------
# Parse markdown into a Platypus story
# -----------------------------------------------------------------------------
def parse_markdown(text: str):
    story = []
    lines = text.splitlines()
    i = 0
    in_table = False
    table_buf = []

    def flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        # Strip alignment row (---), parse rows split by |
        rows = []
        for r in table_buf:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            rows.append([Paragraph(inline(c), ParagraphStyle(
                "td", parent=body_style, fontSize=8.5, leading=11, spaceAfter=0))
                for c in cells])
        if not rows:
            table_buf.clear()
            return
        col_count = max(len(r) for r in rows)
        for r in rows:
            while len(r) < col_count:
                r.append(Paragraph("", body_style))
        # uniform column widths fitting A4 textwidth
        usable = A4[0] - 4 * cm
        col_w = [usable / col_count] * col_count
        t = Table(rows, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16263a")),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9eb1cc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(Spacer(1, 6))
        story.append(t)
        story.append(Spacer(1, 10))
        table_buf.clear()

    while i < len(lines):
        ln = lines[i].rstrip()
        # table detection (heuristic): line containing "|" and next line is | --- |
        if ln.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|") and set(lines[i + 1].replace(" ", "").replace("|", "").replace(":", "")) <= {"-"}:
            in_table = True
            table_buf.append(ln)
            i += 1
            continue
        if in_table:
            if ln.startswith("|"):
                table_buf.append(ln)
                i += 1
                continue
            else:
                flush_table()
                in_table = False
                # fall through to handle current line

        if ln.strip() == "":
            i += 1
            continue
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), title_style))
        elif ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), h1_style))
        elif ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), h2_style))
        elif ln.startswith("#### "):
            story.append(Paragraph(inline(ln[5:]), h3_style))
        elif ln.startswith(">"):
            # blockquote — accumulate consecutive '>' lines
            buf = [ln.lstrip(">").strip()]
            while i + 1 < len(lines) and lines[i + 1].lstrip().startswith(">"):
                i += 1
                buf.append(lines[i].lstrip(">").strip())
            story.append(Paragraph(inline(" ".join(buf)), quote_style))
        elif ln.startswith("- ") or ln.startswith("* "):
            buf = ["• " + ln[2:]]
            while i + 1 < len(lines) and (lines[i + 1].startswith("- ") or lines[i + 1].startswith("* ")):
                i += 1
                buf.append("• " + lines[i][2:])
            for b in buf:
                story.append(Paragraph(inline(b), ParagraphStyle(
                    "li", parent=body_style, leftIndent=14, firstLineIndent=-10,
                    spaceAfter=2)))
        elif re.match(r"^\d+\.\s", ln):
            buf = [ln]
            while i + 1 < len(lines) and re.match(r"^\d+\.\s", lines[i + 1]):
                i += 1
                buf.append(lines[i])
            for b in buf:
                story.append(Paragraph(inline(b), ParagraphStyle(
                    "ol", parent=body_style, leftIndent=18, firstLineIndent=-18,
                    spaceAfter=2)))
        elif ln.strip() == "---":
            story.append(Paragraph("• • •", hr_style))
        else:
            # body paragraph — accumulate until blank line / next markdown structure
            buf = [ln]
            while i + 1 < len(lines):
                nxt = lines[i + 1]
                if (
                    nxt.strip() == ""
                    or nxt.startswith("#")
                    or nxt.startswith(">")
                    or nxt.startswith("- ")
                    or nxt.startswith("* ")
                    or nxt.startswith("|")
                    or re.match(r"^\d+\.\s", nxt)
                    or nxt.strip() == "---"
                ):
                    break
                i += 1
                buf.append(lines[i])
            # detect if this looks like a reference (starts with '- ' would already have been caught above)
            story.append(Paragraph(inline(" ".join(buf)), body_style))
        i += 1

    if in_table:
        flush_table()
    return story


# -----------------------------------------------------------------------------
# Build doc
# -----------------------------------------------------------------------------
def main():
    md = SRC.read_text(encoding="utf-8")
    story = parse_markdown(md)

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2.2 * cm,
        title="Literature Survey — Hallucination Detection in LLMs",
        author="Chinmoy Sahoo — M.Tech Dissertation",
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6e7a90"))
        canvas.drawString(2 * cm, 1.2 * cm,
                          "MIND-style Hallucination Detection — Chapter 2 (Related Work)")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
