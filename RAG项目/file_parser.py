"""文件解析工具，支持 TXT、CSV、Word 文档内容提取"""

import csv
import io
import os


def parse_file(file_bytes: bytes, filename: str) -> str:
    """根据文件扩展名解析文件内容，返回纯文本"""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return _parse_txt(file_bytes)
    elif ext == ".csv":
        return _parse_csv(file_bytes)
    elif ext == ".docx":
        return _parse_docx(file_bytes)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


def _parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def _parse_csv(file_bytes: bytes) -> str:
    content = file_bytes.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(content))
    lines = []
    for row in reader:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _parse_docx(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)