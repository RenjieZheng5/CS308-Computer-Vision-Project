from __future__ import annotations

import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "defense_script_and_qa.md"
OUTPUT = ROOT / "defense_script_and_qa.pdf"
FONT_REGULAR = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\simhei.ttf")


def strip_inline_markdown(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    return text


def wrap_line(text: str, max_units: float) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""

    def units(s: str) -> float:
        total = 0.0
        for char in s:
            total += 1.0 if ord(char) > 127 else 0.55
        return total

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if units(candidate) <= max_units:
            current = candidate
            continue

        if current:
            lines.append(current)
        current = word

        while units(current) > max_units:
            cut = max(1, int(max_units))
            lines.append(current[:cut])
            current = current[cut:]

    if current:
        lines.append(current)
    return lines or [""]


def add_page(doc: fitz.Document) -> fitz.Page:
    return doc.new_page(width=595, height=842)


def render() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    doc = fitz.open()
    page = add_page(doc)

    regular_name = "noto"
    bold_name = "bold"
    page.insert_font(fontname=regular_name, fontfile=str(FONT_REGULAR))
    page.insert_font(fontname=bold_name, fontfile=str(FONT_BOLD))

    margin_x = 50
    margin_top = 48
    margin_bottom = 48
    y = margin_top
    max_units = 70
    in_code = False

    def ensure_space(height: float) -> None:
        nonlocal page, y
        if y + height <= page.rect.height - margin_bottom:
            return
        page = add_page(doc)
        page.insert_font(fontname=regular_name, fontfile=str(FONT_REGULAR))
        page.insert_font(fontname=bold_name, fontfile=str(FONT_BOLD))
        y = margin_top

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            ensure_space(12)
            y += 6
            continue

        if not line:
            ensure_space(10)
            y += 8
            continue

        if in_code:
            size = 9.2
            font = regular_name
            color = (0.12, 0.12, 0.12)
            indent = 14
            max_for_line = 74
        elif line.startswith("# "):
            line = line[2:]
            size = 20
            font = bold_name
            color = (0.05, 0.18, 0.42)
            indent = 0
            max_for_line = 42
        elif line.startswith("## "):
            line = line[3:]
            size = 15
            font = bold_name
            color = (0.05, 0.18, 0.42)
            indent = 0
            max_for_line = 54
        elif line.startswith("### "):
            line = line[4:]
            size = 12.5
            font = bold_name
            color = (0.05, 0.18, 0.42)
            indent = 0
            max_for_line = 61
        elif line.startswith("- "):
            line = f"• {line[2:]}"
            size = 10.5
            font = regular_name
            color = (0, 0, 0)
            indent = 14
            max_for_line = max_units - 2
        elif re.match(r"^\d+\. ", line):
            size = 10.5
            font = regular_name
            color = (0, 0, 0)
            indent = 14
            max_for_line = max_units - 2
        else:
            size = 10.5
            font = regular_name
            color = (0, 0, 0)
            indent = 0
            max_for_line = max_units

        line = strip_inline_markdown(line)
        wrapped = wrap_line(line, max_for_line)
        line_height = size * 1.45
        ensure_space(line_height * len(wrapped) + 2)

        for part in wrapped:
            page.insert_text(
                fitz.Point(margin_x + indent, y),
                part,
                fontsize=size,
                fontname=font,
                color=color,
            )
            y += line_height

    metadata = {
        "title": "CS308 Final Project Defense Script and QA",
        "author": "Group 8",
        "subject": "Open-Vocabulary Object Detection and Visual Grounding",
    }
    doc.set_metadata(metadata)
    doc.subset_fonts()
    doc.ez_save(OUTPUT)
    doc.close()


if __name__ == "__main__":
    render()
